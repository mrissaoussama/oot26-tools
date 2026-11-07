"""OoT collision viewer: terrain collision, invisible walls, void zones and Link, in 3D."""
import json, math, os, sys, threading, time
import numpy as np
from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QColor, QPen, QFont, QMatrix4x4, QPainter, QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLShaderProgram, QOpenGLShader, QOpenGLBuffer, QOpenGLVertexArrayObject
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (QApplication, QWidget, QHBoxLayout, QVBoxLayout, QCheckBox, QLabel, QPushButton,
                               QSlider, QRadioButton, QButtonGroup, QFrame, QFileDialog, QMessageBox, QProgressDialog)

import romfs, havok, build_data
from game import Game

HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = [  # key, label, rgb, alpha, shown by default
    ('terrain', 'Terrain / visible collision', (150, 160, 150), 255, True),
    ('player', 'Invisible walls (block Link)', (255, 60, 60), 90, True),
    ('climb', 'Climb-only walls', (255, 176, 0), 90, True),
    ('push', 'Push walls (doors etc.)', (64, 160, 255), 80, False),
    ('other', 'Enemy / horse / camera walls', (160, 160, 160), 60, False),
    ('void', 'Void-out zones', (210, 60, 255), 80, True),
    ('norespawn', 'No-respawn areas', (64, 224, 208), 50, False),
    ('whirlpool', 'Whirlpools', (32, 96, 255), 60, False),
]
KCOL = {k: (c, a) for k, _, c, a, _ in KINDS}
GL_FLOAT, GL_TRIANGLES, GL_LINES = 0x1406, 0x0004, 0x0001

VS = """
attribute vec3 p; attribute vec3 n; attribute vec4 c;
uniform mat4 mvp; uniform float lit;
varying vec4 vc;
void main() {
    gl_Position = mvp * vec4(p, 1.0);
    float s = 0.4 + 0.6 * abs(dot(normalize(n), normalize(vec3(0.4, 0.8, 0.45))));
    vc = vec4(c.rgb * mix(1.0, s, lit), c.a);
}"""
FS = """
varying vec4 vc;
void main() { gl_FragColor = vc; }"""


class Data:
    def __init__(self, geom):
        self.scene_names = sorted({o['scene'] for o in geom})
        sid = {s: i for i, s in enumerate(self.scene_names)}
        self.kind_names = [k for k, *_ in KINDS]
        kid = {k: i for i, k in enumerate(self.kind_names)}
        tris, tk, ts, segs, sk, ss = [], [], [], [], [], []
        self.scene_items = {}
        for o in geom:
            self.scene_items.setdefault(o['scene'], set()).add(int(o['hash']))
            k, s = kid[o['kind']], sid[o['scene']]
            for poly in o['polys']:
                p = np.asarray(poly, np.float32)
                t = np.stack([np.repeat(p[:1], len(p) - 2, 0), p[1:-1], p[2:]], 1)
                tris.append(t); tk += [k] * len(t); ts += [s] * len(t)
                e = np.stack([p, np.roll(p, -1, 0)], 1)
                segs.append(e); sk += [k] * len(e); ss += [s] * len(e)
        self.wt, self.wk, self.ws = np.concatenate(tris), np.array(tk), np.array(ts)
        self.wc = self.wt.mean(1)
        self.ls, self.lk, self.lss = np.concatenate(segs), np.array(sk), np.array(ss)
        self.lc = self.ls.mean(1)
        self.terrain, self.loading, self.tiles, self.bounds = {}, set(), {}, {}

    def terrain_near(self, scenes, x, z, rng):
        out, pending = [], 0
        if not romfs.ROOT: return out, pending
        for sc in scenes:
            if sc not in self.tiles: self.tiles[sc] = havok.tiles_for(sc)
            tiles = self.tiles[sc]
            for key, path in tiles.items():
                if path not in self.bounds and len(tiles) > 1:
                    o = np.array([(key[0] - max(k[0] for k in tiles) / 2 - 1.5) * havok.TILE, 0, (key[1] - max(k[1] for k in tiles) / 2 - 1.5) * havok.TILE])
                    lo, hi = o, o + 3 * havok.TILE
                    if x + rng < lo[0] or x - rng > hi[0] or z + rng < lo[2] or z - rng > hi[2]: continue
                if path in self.bounds:
                    lo, hi = self.bounds[path]
                    if x + rng < lo[0] or x - rng > hi[0] or z + rng < lo[2] or z - rng > hi[2]: continue
                if path in self.terrain: out.append(self.terrain[path])
                else:
                    pending += 1
                    if path not in self.loading:
                        self.loading.add(path)
                        threading.Thread(target=self._load, args=(path, key, tiles), daemon=True).start()
        return out, pending

    def _load(self, path, key, tiles):
        t = havok.load_tile(path, key, tiles)
        c = t.mean(1) if len(t) else np.zeros((0, 3), np.float32)
        self.bounds[path] = (c.min(0), c.max(0)) if len(t) else (np.full(3, 1e9), np.full(3, -1e9))
        self.terrain[path] = (t, c)


def vertices(tris, rgb, a):
    n = np.repeat(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])[:, None], 3, 1)
    c = np.broadcast_to(np.array([*rgb, a], np.float32) / 255, (len(tris), 3, 4))
    return np.concatenate([tris, n, c], 2).reshape(-1, 10).astype(np.float32)


def line_vertices(segs, rgb):
    n = np.zeros_like(segs); n[..., 1] = 1
    c = np.broadcast_to(np.array([*rgb, 255], np.float32) / 255, (len(segs), 2, 4))
    return np.concatenate([segs, n, c], 2).reshape(-1, 10).astype(np.float32)


class GLView(QOpenGLWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.mode = 'game'
        self.orbit = [0.0, 0.5, 14.0]
        self.top_h = 60.0
        self.drag = None
        self.buffers, self.build_key = {}, None
        self.counts = {'opaque': 0, 'trans': 0, 'lines': 0}
        self.setMinimumSize(800, 600)

    def initializeGL(self):
        self.prog = QOpenGLShaderProgram(self)
        self.prog.addShaderFromSourceCode(QOpenGLShader.Vertex, VS)
        self.prog.addShaderFromSourceCode(QOpenGLShader.Fragment, FS)
        for i, n in enumerate('pnc'): self.prog.bindAttributeLocation(n, i)
        self.prog.link()
        self.u_mvp, self.u_lit = self.prog.uniformLocation('mvp'), self.prog.uniformLocation('lit')
        self.vao = QOpenGLVertexArrayObject(self); self.vao.create()
        for k in self.counts:
            b = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer); b.create(); self.buffers[k] = b

    def upload(self, name, arr):
        b = self.buffers[name]; b.bind()
        data = arr.tobytes() if len(arr) else b'\0' * 40
        b.allocate(data, len(data))
        self.counts[name] = len(arr)

    def wheelEvent(self, e):
        f = 1.15 if e.angleDelta().y() < 0 else 1 / 1.15
        if self.mode == 'orbit': self.orbit[2] = min(300, max(2, self.orbit[2] * f))
        elif self.mode == 'top': self.top_h = min(400, max(5, self.top_h * f))

    def mousePressEvent(self, e): self.drag = e.position()
    def mouseReleaseEvent(self, e): self.drag = None

    def mouseMoveEvent(self, e):
        if self.drag is not None and self.mode == 'orbit':
            d = e.position() - self.drag; self.drag = e.position()
            self.orbit[0] -= d.x() * 0.01
            self.orbit[1] = min(1.55, max(-1.55, self.orbit[1] + d.y() * 0.01))

    def matrices(self, s, W, H):
        L = np.array([s['x'], s['y'] + 1.0, s['z']], np.float32)
        asp = W / max(H, 1)
        def look(eye, tgt, up=(0, 1, 0)):
            f = tgt - eye; f /= np.linalg.norm(f)
            r = np.cross(f, up); r /= np.linalg.norm(r)
            V = np.eye(4, dtype=np.float32); V[0, :3], V[1, :3], V[2, :3] = r, np.cross(r, f), -f
            V[:3, 3] = -V[:3, :3] @ eye
            return V
        def persp(fovy, near=0.1, far=2000.0):
            f = 1 / math.tan(fovy / 2)
            return np.array([[f / asp, 0, 0, 0], [0, f, 0, 0], [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)], [0, 0, -1, 0]], np.float32)
        if self.mode == 'game' and s['cam']:
            V = np.eye(4, dtype=np.float32); V[:3] = s['cam']['view']
            return V, persp(s['cam']['fovy'])
        if self.mode == 'top':
            return look(L + np.array([0, self.top_h, 0.001], np.float32), L, up=(0, 0, -1)), persp(math.radians(60))
        yaw, pitch, dist = self.orbit
        eye = L + dist * np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch), math.cos(yaw) * math.cos(pitch)], np.float32)
        return look(eye, L), persp(math.radians(55))

    def rebuild(self, s):
        app, d = self.app, self.app.data
        rng = app.range.value()
        lx, ly, lz = s['x'], s['y'], s['z']
        show = {k: cb.isChecked() for k, cb in app.showk.items()}
        height = app.only_height.isChecked()
        ceil = app.no_ceiling.isChecked() and self.mode != 'game'
        key = (round(lx / (rng / 4)), round(lz / (rng / 4)), round(ly / 2) if height or ceil else 0, ceil, rng, tuple(sorted(show.items())),
               app.only_zone.isChecked(), height, tuple(sorted(app.active or ())), len(d.terrain))
        if key == self.build_key: return
        self.build_key = key
        zone_ids = [i for i, n in enumerate(d.scene_names) if n in (app.active or ())]
        def sel(c, ks, ss, k):
            m = (np.abs(c[:, 0] - lx) < rng) & (np.abs(c[:, 2] - lz) < rng) & (ks == k)
            if app.only_zone.isChecked() and app.active: m &= np.isin(ss, zone_ids)
            return m
        trans, lines, opaque = [], [], []
        for i, k in enumerate(d.kind_names):
            if k == 'terrain' or not show[k]: continue
            m = sel(d.wc, d.wk, d.ws, i)
            if height: m &= (d.wt[:, :, 1].max(1) > ly - 4) & (d.wt[:, :, 1].min(1) < ly + 12)
            if m.any(): trans.append(vertices(d.wt[m], *KCOL[k]))
            ml = sel(d.lc, d.lk, d.lss, i)
            if ml.any(): lines.append(line_vertices(d.ls[ml], KCOL[k][0]))
        pending = 0
        if show['terrain'] and app.active:
            tl, pending = d.terrain_near(sorted(app.active), lx, lz, rng)
            for t, c in tl:
                m = (np.abs(c[:, 0] - lx) < rng) & (np.abs(c[:, 2] - lz) < rng) if len(t) else None
                if m is not None and ceil: m &= t[:, :, 1].min(1) < ly + 3.5
                if m is not None and m.any(): opaque.append(vertices(t[m], *KCOL['terrain']))
        cat = lambda L: np.concatenate(L) if L else np.zeros((0, 10), np.float32)
        self.upload('opaque', cat(opaque)); self.upload('trans', cat(trans)); self.upload('lines', cat(lines))
        if pending: self.build_key = None
        note = '\nloading %d terrain tiles...' % pending if pending else ''
        if show['terrain'] and not romfs.ROOT: note = '\nTerrain needs the romfs folder (button above).'
        app.info.setText('%d terrain + %d wall triangles%s' % (self.counts['opaque'] // 3, self.counts['trans'] // 3, note))

    def draw(self, name, mode, lit):
        if not self.counts[name]: return
        self.buffers[name].bind()
        for i, (off, size) in enumerate(((0, 3), (12, 3), (24, 4))):
            self.prog.enableAttributeArray(i); self.prog.setAttributeBuffer(i, GL_FLOAT, off, size, 40)
        self.prog.setUniformValue1f(self.u_lit, float(lit))
        self.f.glDrawArrays(mode, 0, self.counts[name])

    def paintGL(self):
        f = self.f = self.context().functions()
        f.glClearColor(0.06, 0.08, 0.1, 1); f.glClear(0x4000 | 0x100)
        s = self.app.state
        if not s: return self.hud(None, None)
        self.rebuild(s)
        V, P = self.matrices(s, self.width(), self.height())
        M = P @ V
        self.prog.bind(); self.vao.bind()
        self.prog.setUniformValue(self.u_mvp, QMatrix4x4(*M.flatten().tolist()))
        f.glEnable(0x0B71); f.glDisable(0x0BE2)  # depth test on, blend off
        self.draw('opaque', GL_TRIANGLES, 1)
        f.glEnable(0x0BE2); f.glBlendFunc(0x0302, 0x0303); f.glDepthMask(False)
        self.draw('trans', GL_TRIANGLES, 1)
        f.glDepthMask(True)
        self.draw('lines', GL_LINES, 0)
        self.vao.release(); self.prog.release()
        f.glDisable(0x0B71)
        self.hud(s, M)

    def hud(self, s, M):
        p = QPainter(self)
        p.setFont(QFont('Consolas', 9))
        W, H = self.width(), self.height()
        if s is None:
            p.setPen(QColor('white')); p.drawText(20, 30, 'Waiting for Link: ' + (self.app.game.error or 'reading...')); p.end(); return
        def scr(x, y, z):
            v = M @ np.array([x, y, z, 1], np.float32)
            return QPointF((v[0] / v[3] + 1) * W / 2, (1 - v[1] / v[3]) * H / 2) if v[3] > 0.1 else None
        for off, col, r in ((0.9, QColor(64, 255, 96), 7), (0.0, QColor(30, 140, 50), 3)):
            q = scr(s['x'], s['y'] + off, s['z'])
            if q: p.setBrush(col); p.setPen(QPen(QColor('black'), 1)); p.drawEllipse(q, r, r)
        if s['face']:
            a = scr(s['x'], s['y'] + 0.9, s['z'])
            b = scr(s['x'] + s['face'][0] * 1.5, s['y'] + 0.9, s['z'] + s['face'][1] * 1.5)
            if a and b: p.setPen(QPen(QColor(64, 255, 96), 2)); p.drawLine(a, b)
        p.setPen(QColor('white'))
        p.drawText(8, 16, 'Link %.1f %.1f %.1f   %s' % (s['x'], s['y'], s['z'], self.app.zone_msg))
        p.drawText(8, H - 8, {'game': 'Game camera', 'orbit': 'Orbit: drag to rotate, wheel to zoom', 'top': 'Top-down: wheel to zoom'}[self.mode])
        p.end()


class App(QWidget):
    def __init__(self, geom):
        super().__init__()
        self.setWindowTitle('OoT collision viewer')
        self.data = Data(geom)
        self.game = Game()
        self.state, self.active, self.zone_msg, self.scanning, self.last_pos = None, None, 'zone: detecting...', False, None
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        side = QFrame(); side.setFixedWidth(250); side.setStyleSheet('background:#1b2026; color:#d0d8e0')
        sl = QVBoxLayout(side)
        sl.addWidget(QLabel('<b>View</b>'))
        self.modes = QButtonGroup(self)
        for i, t in enumerate(['Game camera', 'Orbit around Link (drag)', 'Top-down']):
            rb = QRadioButton(t); rb.setChecked(i == 0); self.modes.addButton(rb, i); sl.addWidget(rb)
        self.modes.idClicked.connect(lambda i: setattr(self.view, 'mode', ['game', 'orbit', 'top'][i]))
        sl.addWidget(QLabel('<b>Show</b>'))
        self.showk = {}
        for k, label, rgb, a, on in KINDS:
            cb = QCheckBox(label); cb.setChecked(on); cb.setStyleSheet('color: rgb(%d,%d,%d)' % rgb)
            self.showk[k] = cb; sl.addWidget(cb)
        sl.addWidget(QLabel('<b>Filters</b>'))
        self.only_zone = QCheckBox('Only the loaded zone'); self.only_zone.setChecked(True); sl.addWidget(self.only_zone)
        self.only_height = QCheckBox("Walls near Link's height only"); sl.addWidget(self.only_height)
        self.no_ceiling = QCheckBox('Hide ceilings (for dungeons)'); self.no_ceiling.setChecked(True); sl.addWidget(self.no_ceiling)
        b = QPushButton('Re-detect zone'); b.clicked.connect(self.detect); sl.addWidget(b)
        b = QPushButton('Set romfs folder (for terrain)'); b.clicked.connect(self.pick_romfs); sl.addWidget(b)
        self.range = QSlider(Qt.Horizontal); self.range.setRange(15, 250); self.range.setValue(60)
        self.range_lbl = QLabel('Draw distance: 60 units'); sl.addWidget(self.range_lbl); sl.addWidget(self.range)
        self.range.valueChanged.connect(lambda v: self.range_lbl.setText('Draw distance: %d units' % v))
        self.info = QLabel(); self.info.setWordWrap(True); sl.addWidget(self.info)
        sl.addStretch(1)
        lay.addWidget(side)
        self.view = GLView(self); lay.addWidget(self.view, 1)
        self.timer = QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(33)
        self.resize(1300, 820)

    def pick_romfs(self):
        d = QFileDialog.getExistingDirectory(self, 'Game romfs folder (contains Banc, Phive, Pack...)')
        if not d: return
        if not os.path.isdir(os.path.join(d, 'Phive')):
            QMessageBox.warning(self, 'OoT collision viewer', 'That folder has no Phive subfolder.'); return
        romfs.save_config(d)
        self.view.build_key = None

    def detect(self):
        """Loaded zones = zones whose placement IDs are present in the actor heap."""
        if self.scanning or not self.game.addr.get('gb'): return
        self.scanning = True
        keys = np.array(sorted({h for s in self.data.scene_items.values() for h in s}), np.uint64)
        def work():
            try:
                t = time.time()
                hit = self.game.scan_u64(keys, 0x2140000000, 0x2160000000,
                                         lambda f: setattr(self, 'zone_msg', 'zone: scanning %d%%' % (f * 100)))
                act = {s for s, hs in self.data.scene_items.items() if len(hs & hit) >= max(1, 0.9 * len(hs))}
                self.active = act or None
                self.zone_msg = 'zone: %s (%.0fs)' % (', '.join(sorted(act)) if act else 'not found, showing all', time.time() - t)
            except Exception as e:
                self.zone_msg = 'zone scan failed: %r' % e
            finally:
                self.scanning = False
        threading.Thread(target=work, daemon=True).start()

    def tick(self):
        try:
            self.state = self.game.state()
        except Exception:
            self.state = None
        s = self.state
        if s:
            if self.last_pos is None or abs(s['x'] - self.last_pos[0]) + abs(s['z'] - self.last_pos[1]) > 15:
                QTimer.singleShot(2000, self.detect)
            self.last_pos = (s['x'], s['z'])
        self.view.update()


def setup(qa):
    romfs.load_config()
    gpath = os.path.join(HERE, 'geom.json')
    if not os.path.exists(gpath):
        while not romfs.load_config():
            QMessageBox.information(None, 'OoT collision viewer', 'geom.json is missing. Select the game\'s dumped "romfs" folder to build it.')
            d = QFileDialog.getExistingDirectory(None, 'romfs folder')
            if not d: sys.exit(0)
            romfs.save_config(d)
        dlg = QProgressDialog('Building collision data...', None, 0, 0)
        dlg.setWindowTitle('OoT collision viewer'); dlg.setMinimumWidth(520); dlg.show()
        def step(msg): dlg.setLabelText(msg); qa.processEvents()
        build_data.build(step)
        dlg.close()
    return json.load(open(gpath))


if __name__ == '__main__':
    fmt = QSurfaceFormat(); fmt.setDepthBufferSize(24); fmt.setSamples(4); QSurfaceFormat.setDefaultFormat(fmt)
    qa = QApplication(sys.argv)
    w = App(setup(qa)); w.show()
    sys.exit(qa.exec())
