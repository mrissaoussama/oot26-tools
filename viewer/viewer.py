"""OoT collision viewer: terrain collision, invisible walls, void zones and Link, in 3D."""
import json, math, os, re, sys, threading, time
import numpy as np
from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QColor, QPen, QFont, QMatrix4x4, QPainter, QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLShaderProgram, QOpenGLShader, QOpenGLBuffer, QOpenGLVertexArrayObject
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (QApplication, QWidget, QHBoxLayout, QVBoxLayout, QCheckBox, QLabel, QPushButton, QComboBox, QMenu,
                               QSlider, QRadioButton, QButtonGroup, QFrame, QFileDialog, QMessageBox, QProgressDialog)

import romfs, havok, build_data
from byml import Byml
from game import Game

HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = [  # key, label, rgb, alpha, shown by default
    ('terrain', 'Terrain / visible collision', (150, 160, 150), 255, True),
    ('player', 'Invisible walls (block Link)', (255, 50, 50), 90, True),
    ('climb', 'Climb-only walls', (255, 220, 0), 90, True),
    ('push', 'Push walls (doors etc.)', (60, 140, 255), 80, False),
    ('other', 'Enemy / horse / camera walls', (150, 150, 150), 60, False),
    ('object', 'Objects (pots, chests, blocks...)', (245, 245, 245), 150, True),
    ('conditional', 'Story-dependent terrain pieces', (255, 150, 200), 80, True),
    ('void', 'Fall-down areas (void-out or hole into a grotto)', (170, 60, 255), 80, True),
    ('loading', 'Loading zones', (140, 255, 60), 70, True),
    ('norespawn', 'No-respawn areas', (40, 220, 200), 60, False),
    ('whirlpool', 'Whirlpools', (30, 60, 200), 70, False),
]
KCOL = {k: (c, a) for k, _, c, a, _ in KINDS}
MAT_NAMES = ['Undefined', 'Soil', 'Grass', 'Sand', 'HeavySand', 'Snow', 'HeavySnow', 'Stone', 'StoneSlip', 'StoneNoSlip',
             'Metal', 'MetalSlip', 'MetalNoSlip', 'WireNet', 'Wood', 'Ice', 'Cloth', 'Glass', 'Bone', 'Rope', 'Character',
             'Ragdoll', 'Meat', 'Vegetable', 'Bomb', 'MagicBall', 'Barrier', 'AirWall', 'Water', 'HotWater', 'IceWater',
             'Lava', 'Bog', 'ContaminatedWater', 'Gas', 'InvalidateRestartPos', 'HorseSpeedLimit', 'ForbidDynamicCutting',
             'ForbidHorseReturn', 'WaterSlip', 'HorseDelete', 'DropItem', 'Skulltula', 'PoisonBog', 'WoodSlip', 'IceNoSlip']
SUB_NAMES = ['', 'Soft', 'Hard', 'Light', 'Heavy', 'Marble', 'DgnLight', 'DgnHeavy', 'Shallow', 'Deep', 'Moss', 'Long',
             'WithMud', 'Leaf', 'Straw', 'Thatch', 'Thin', 'Thick', 'Ice', 'Hot', 'Poison', 'StoneBottom', 'Sherbet',
             'Carpet', 'Leather', 'Pile', 'Spike', 'SpikeWeak', 'Black', 'Yellow', 'HorseReturnToSafePos', 'Spore',
             'Terrain', 'TerrainShallow', 'WithStone', 'WitheredLeaf', 'SpiderWeb', 'FloatingCarpet', 'GanonCastleBridge',
             'SmallAmount', 'MediumAmount', 'LargeAmount', 'FakeWall', 'Vine', 'WithSoil', 'ZeldaBarrier', 'MagicBeanLeaf',
             'SoilBottom', 'StoneLightBottom', 'SandBottom', 'HorseForbidToSlide', 'HammerNoThrough', 'GanonSword', 'KeepAwayDrop']
FLAG_NAMES = [(0x10, 'Abyss'), (0x10000, 'DisableRespawn'), (0x8000, 'EdgeOfTheWorld'), (0x4, 'Slip'), (0x2, 'Ladder'),
              (0x10000000000, 'Climb'), (0x20000000000, 'NoClamber'), (0x100000000000, 'StickHookshot'), (0x4000, 'Wet')]
MAT_COL = {1: (140, 100, 60), 2: (80, 160, 60), 3: (225, 205, 130), 4: (205, 180, 100), 5: (250, 250, 255),
           6: (230, 235, 250), 7: (130, 130, 130), 8: (130, 130, 130), 9: (110, 110, 110), 10: (95, 115, 145),
           11: (95, 115, 145), 12: (80, 95, 120), 14: (175, 115, 50), 15: (200, 240, 255), 17: (210, 235, 235),
           28: (60, 120, 230), 29: (90, 140, 230), 30: (120, 170, 240), 31: (255, 110, 0), 32: (95, 85, 55),
           43: (130, 170, 40), 44: (175, 115, 50), 45: (200, 240, 255)}
HAZARD_COL = {'Lava': (255, 110, 0), 'Abyss (void)': (90, 0, 140), 'No respawn': (40, 220, 200), 'Slippery': (190, 235, 255)}
NOT_FOR_LINK = (27, 41)  # AirWall faces are drawn as invisible walls; DropItem only catches items


def surface_colors(m, f):
    """Per-triangle RGBA for the surface view: material colour, with hazards overriding it."""
    c = np.full((len(m), 4), (160, 160, 160, 255), np.float32)
    for k, rgb in MAT_COL.items(): c[m == k, :3] = rgb
    c[(f & 0x4) > 0, :3] = HAZARD_COL['Slippery']
    c[(f & 0x10000) > 0, :3] = HAZARD_COL['No respawn']
    c[(f & 0x10) > 0, :3] = HAZARD_COL['Abyss (void)']
    c[m == 31, :3] = HAZARD_COL['Lava']
    return c


def surface_name(m, s, f):
    name = MAT_NAMES[m] if m < len(MAT_NAMES) else str(m)
    if 0 < s < len(SUB_NAMES): name += ' (%s)' % SUB_NAMES[s]
    flags = [n for bit, n in FLAG_NAMES if f & bit]
    return name + (' [' + ', '.join(flags) + ']' if flags else '')


ACTOR_COL = {'Enemy': QColor(255, 0, 200), 'Npc': QColor(255, 190, 70), 'Animal': QColor(90, 220, 255), 'Obj': QColor(0, 230, 120), 'Item': QColor(0, 230, 120)}
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
        tris, tk, ts, th, segs, sk, ss, sh = [], [], [], [], [], [], [], []
        self.scene_items = {}
        self.obj_pos = {int(o['hash']): o['pos'] for o in geom if o['kind'] == 'object' and 'pos' in o}
        for o in geom:
            self.scene_items.setdefault(o['scene'], set()).add(int(o['hash']))
            k, s = kid[o['kind']], sid[o['scene']]
            for poly in o['polys']:
                p = np.asarray(poly, np.float32)
                t = np.stack([np.repeat(p[:1], len(p) - 2, 0), p[1:-1], p[2:]], 1)
                tris.append(t); tk += [k] * len(t); ts += [s] * len(t); th += [int(o['hash'])] * len(t)
                e = np.stack([p, np.roll(p, -1, 0)], 1)
                segs.append(e); sk += [k] * len(e); ss += [s] * len(e); sh += [int(o['hash'])] * len(e)
        self.wt, self.wk, self.ws, self.wh = np.concatenate(tris), np.array(tk), np.array(ts), np.array(th, np.uint64)
        self.lh = np.array(sh, np.uint64)
        apath = os.path.join(HERE, 'actors.json')
        aj = json.load(open(apath)) if os.path.exists(apath) else {}
        self.actors = {int(h): n for h, n in aj.get('actors', {}).items()}
        self.actor_boxes = {n: b for n, b in aj.get('boxes', {}).items() if b}
        self.shapes = {}
        for n, polys in aj.get('shapes', {}).items():
            tris = [fan(p) for p in polys if len(p) >= 3]
            if tris: self.shapes[n] = np.concatenate(tris)
        self.obj_info = {int(o['hash']): (o['name'], o.get('scale', [1, 1, 1])) for o in geom if o['kind'] == 'object'}
        self.hash_name = {int(o['hash']): o['name'] for o in geom}
        self.loading_zones = [(np.concatenate([np.array(p) for p in o['polys']]).mean(0), o['name'], o['scene'], int(o['hash']))
                        for o in geom if o['kind'] == 'loading']
        self.groups = sorted({re.sub(r'_(Static|Dynamic)$', '', s) for s in self.scene_names})
        self.wc = self.wt.mean(1)
        self.ls, self.lk, self.lss = np.concatenate(segs), np.array(sk), np.array(ss)
        self.lc = self.ls.mean(1)
        self.terrain, self.loading, self.tiles, self.bounds, self.names = {}, set(), {}, {}, {}
        self.conditional = np.zeros(0, np.uint64)

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
                        threading.Thread(target=self._load, args=(path, key, tiles, sc), daemon=True).start()
        return out, pending

    def name_of(self, h):
        for d in self.names.values():
            if h in d: return d[h]
        return None

    def names_for(self, scene):
        """Placement hash -> actor name for every layer of the scene's area (Static/Dynamic, both ages).
        Placements with a story-flag condition are collected in self.conditional."""
        area = scene.split('_')[0]
        if area not in self.names:
            d, cond = {}, []
            folder = romfs.path('Banc', area)
            for f in os.listdir(folder) if os.path.isdir(folder) else []:
                if f.endswith('.bcett.byml.zs'):
                    for a in Byml(romfs.read('Banc', area, f)).root.get('Actors', []):
                        d[a['Hash']] = a.get('Gyaml', '')
                        if a.get('Presence'): cond.append(a['Hash'])
            self.names[area] = d
            self.conditional = np.union1d(self.conditional, np.array(cond, np.uint64))
        return self.names[area]

    def _load(self, path, key, tiles, scene):
        t, h, m, sub, f = havok.load_tile(path, key, tiles, self.names_for(scene))
        c = t.mean(1) if len(t) else np.zeros((0, 3), np.float32)
        self.bounds[path] = (c.min(0), c.max(0)) if len(t) else (np.full(3, 1e9), np.full(3, -1e9))
        self.terrain[path] = (t, c, h, m, sub, f)


def fan(poly):
    p = np.asarray(poly, np.float32)
    return np.stack([np.repeat(p[:1], len(p) - 2, 0), p[1:-1], p[2:]], 1)


def vertices(tris, rgb, a):
    n = np.repeat(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])[:, None], 3, 1)
    c = np.broadcast_to(np.array([*rgb, a], np.float32) / 255, (len(tris), 3, 4))
    return np.concatenate([tris, n, c], 2).reshape(-1, 10).astype(np.float32)


def vertices_rgba(tris, rgba):
    n = np.repeat(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])[:, None], 3, 1)
    c = np.repeat((rgba / 255)[:, None, :], 3, 1)
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
        self.counts = {'opaque': 0, 'trans': 0, 'lines': 0, 'dyn': 0}
        self.dyn_version = -1
        self.free = None  # [x, y, z, yaw, pitch]
        self.floor, self.picks, self.mvp = [], [], None
        self.keys, self.last_t = set(), time.time()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(800, 600)

    def set_mode(self, m):
        self.mode = m
        if m == 'free':
            s = self.app.state
            cam = s and s['cam']
            if cam:
                R = cam['view'][:, :3]; eye = -R.T @ cam['view'][:, 3]; f = -R[2]
            elif s:
                eye = np.array([s['x'], s['y'] + 6, s['z'] + 10], np.float32); f = np.array([0, -0.4, -1], np.float32)
            else:
                eye, f = np.zeros(3, np.float32), np.array([0, 0, -1], np.float32)
            f = f / np.linalg.norm(f)
            self.free = [*map(float, eye), math.atan2(-f[0], -f[2]), math.asin(max(-1, min(1, f[1])))]
            self.setFocus()

    def keyPressEvent(self, e): self.keys.add(e.key())
    def keyReleaseEvent(self, e): self.keys.discard(e.key())

    def step_free(self):
        now = time.time(); dt = min(0.1, now - self.last_t); self.last_t = now
        if self.mode != 'free' or not self.free: return
        x, y, z, yaw, pitch = self.free
        sp = (40 if Qt.Key_Shift in self.keys else 12) * dt
        fwd = np.array([-math.sin(yaw) * math.cos(pitch), math.sin(pitch), -math.cos(yaw) * math.cos(pitch)])
        right = np.array([math.cos(yaw), 0, -math.sin(yaw)])
        mv = np.zeros(3)
        if Qt.Key_W in self.keys: mv += fwd
        if Qt.Key_S in self.keys: mv -= fwd
        if Qt.Key_D in self.keys: mv += right
        if Qt.Key_A in self.keys: mv -= right
        if Qt.Key_E in self.keys or Qt.Key_Space in self.keys: mv[1] += 1
        if Qt.Key_Q in self.keys or Qt.Key_Control in self.keys: mv[1] -= 1
        self.free[:3] = list(np.array([x, y, z]) + mv * sp)

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

    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton and self.mode != 'free':
            self.pick_menu(e)
            return
        self.drag = e.position()

    def pick(self, x, y):
        """Placement hash of the nearest picked triangle under screen point (x, y)."""
        if self.mvp is None: return None
        W, H = self.width(), self.height()
        best = None
        sets = list(self.picks) + list(self.app.dyn_picks)
        for tris, hs in sets:
            if not len(tris): continue
            v = np.concatenate([tris, np.ones(tris.shape[:2] + (1,), np.float32)], 2) @ self.mvp.T
            w = v[:, :, 3]
            ok = np.all(w > 0.1, 1)
            sx = (v[:, :, 0] / np.where(w > 0.1, w, 1) + 1) * W / 2
            sy = (1 - v[:, :, 1] / np.where(w > 0.1, w, 1)) * H / 2
            a, b, c = np.stack([sx[:, 0], sy[:, 0]], 1), np.stack([sx[:, 1], sy[:, 1]], 1), np.stack([sx[:, 2], sy[:, 2]], 1)
            def cr(p, q): return (q[:, 0] - p[:, 0]) * (y - p[:, 1]) - (q[:, 1] - p[:, 1]) * (x - p[:, 0])
            d1, d2, d3 = cr(a, b), cr(b, c), cr(c, a)
            inside = ok & (((d1 >= 0) & (d2 >= 0) & (d3 >= 0)) | ((d1 <= 0) & (d2 <= 0) & (d3 <= 0)))
            idx = np.where(inside)[0]
            if len(idx):
                j = idx[np.argmin(w[idx].mean(1))]
                depth = float(w[j].mean())
                if best is None or depth < best[0]: best = (depth, int(hs[j]))
        return best[1] if best else None

    def pick_menu(self, e):
        h = self.pick(e.position().x(), e.position().y())
        menu = QMenu(self)
        if h:
            name = self.app.data.hash_name.get(h) or self.app.data.name_of(h) or 'object %x' % h
            menu.addAction('%s' % name).setEnabled(False)
            menu.addAction('Hide this in the viewer', lambda: self.app.hide(h))
        else:
            menu.addAction('Nothing pickable here').setEnabled(False)
        if len(self.app.hidden_keys):
            menu.addAction('Show everything hidden (%d)' % len(self.app.hidden_keys), self.app.unhide_all)
        menu.exec(e.globalPosition().toPoint())
    def mouseReleaseEvent(self, e): self.drag = None

    def mouseMoveEvent(self, e):
        if self.drag is not None and self.mode == 'free':
            d = e.position() - self.drag; self.drag = e.position()
            self.free[3] -= d.x() * 0.005
            self.free[4] = min(1.55, max(-1.55, self.free[4] - d.y() * 0.005))
            return
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
        if self.mode == 'free' and self.free:
            x, y, z, yaw, pitch = self.free
            eye = np.array([x, y, z], np.float32)
            f = np.array([-math.sin(yaw) * math.cos(pitch), math.sin(pitch), -math.cos(yaw) * math.cos(pitch)], np.float32)
            return look(eye, eye + f), persp(math.radians(60))
        if self.mode == 'top':
            return look(L + np.array([0, self.top_h, 0.001], np.float32), L, up=(0, 0, -1)), persp(math.radians(60))
        yaw, pitch, dist = self.orbit
        eye = L + dist * np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch), math.cos(yaw) * math.cos(pitch)], np.float32)
        return look(eye, L), persp(math.radians(55))

    def rebuild(self, s):
        app, d = self.app, self.app.data
        rng = app.range.value()
        lx, ly, lz = s['x'], s['y'], s['z']
        if self.mode == 'free' and self.free: lx, lz = self.free[0], self.free[2]
        show = {k: cb.isChecked() for k, cb in app.showk.items()}
        height = app.only_height.isChecked()
        ceil = app.no_ceiling.isChecked() and self.mode != 'game' and any(
            len(d.tiles.get(sc, {})) == 1 for sc in (app.active or ()))
        key = (round(lx / (rng / 4)), round(lz / (rng / 4)), round(ly / 2) if height or ceil else 0, ceil, rng, tuple(sorted(show.items())),
               app.only_zone.isChecked(), height, tuple(sorted(app.active or ())), len(d.terrain),
               app.only_live.isChecked(), app.live_version, app.surface_mode.isChecked(), app.hidden_version,
               None if app.obj_live_keys is None else len(app.obj_live_keys))
        key = key + (len(d.conditional),)
        if key == self.build_key: return
        self.build_key = key
        zone_ids = [i for i, n in enumerate(d.scene_names) if n in (app.active or ())]
        live = app.only_live.isChecked() and app.found_keys is not None and len(app.found_keys) > 0
        obj = d.kind_names.index('object')
        def near_link(g):
            """Shapes that reach Link's height (-2..+4) and come within 12 units of him horizontally."""
            y0, y1 = g[:, :, 1].min(1), g[:, :, 1].max(1)
            dx = np.clip(s['x'], g[:, :, 0].min(1), g[:, :, 0].max(1)) - s['x']
            dz = np.clip(s['z'], g[:, :, 2].min(1), g[:, :, 2].max(1)) - s['z']
            return (y1 > ly - 2) & (y0 < ly + 4) & (dx * dx + dz * dz < 144)
        def sel(c, ks, ss, k, hs):
            m = (np.abs(c[:, 0] - lx) < rng) & (np.abs(c[:, 2] - lz) < rng) & (ks == k)
            if app.only_zone.isChecked() and app.active: m &= np.isin(ss, zone_ids)
            # Objects are actors: draw only existing ones. Walls can be baked, so only drop those absent from memory.
            if live: m &= np.isin(hs, app.live_keys if k == obj else app.found_keys)
            if k == obj and app.obj_live_keys is not None: m &= ~np.isin(hs, app.obj_live_keys)
            m &= ~np.isin(hs, app.hidden_keys)
            return m
        trans, lines, opaque, picks = [], [], [], []
        for i, k in enumerate(d.kind_names):
            if k in ('terrain', 'conditional') or not show[k]: continue
            m = sel(d.wc, d.wk, d.ws, i, d.wh)
            if height: m &= near_link(d.wt)
            if m.any():
                trans.append(vertices(d.wt[m], *KCOL[k]))
                picks.append((d.wt[m], d.wh[m]))
            ml = sel(d.lc, d.lk, d.lss, i, d.lh)
            if height: ml &= near_link(d.ls)
            if ml.any(): lines.append(line_vertices(d.ls[ml], KCOL[k][0]))
        pending = 0
        if show['terrain'] and app.active:
            tl, pending = d.terrain_near(sorted(app.active), lx, lz, rng)
            floor = []
            for t, c, h, mat, sub, fl in tl:
                m = (np.abs(c[:, 0] - lx) < rng) & (np.abs(c[:, 2] - lz) < rng) if len(t) else None
                if m is not None and ceil: m &= t[:, :, 1].min(1) < ly + 3.5
                if m is None or not m.any(): continue
                m &= ~np.isin(h, app.hidden_keys)
                airwall, dropitem = m & (mat == 27), m & (mat == 41)
                m &= ~airwall & ~dropitem
                cond = np.isin(h, d.conditional) & m
                solid = m & ~cond
                if solid.any():
                    if app.surface_mode.isChecked():
                        opaque.append(vertices_rgba(t[solid], surface_colors(mat[solid], fl[solid])))
                    else:
                        opaque.append(vertices(t[solid], *KCOL['terrain']))
                    floor.append((t[solid], mat[solid], sub[solid], fl[solid]))
                    baked = solid & (h != 0)
                    if baked.any(): picks.append((t[baked], h[baked]))
                if show['conditional'] and cond.any(): trans.append(vertices(t[cond], *KCOL['conditional']))
                if show['player'] and airwall.any(): trans.append(vertices(t[airwall], *KCOL['player']))
            self.floor = floor
        self.picks = picks
        cat = lambda L: np.concatenate(L) if L else np.zeros((0, 10), np.float32)
        self.upload('opaque', cat(opaque)); self.upload('trans', cat(trans)); self.upload('lines', cat(lines))
        if pending: self.build_key = None
        note = '\nloading %d terrain tiles...' % pending if pending else ''
        if show['terrain'] and not romfs.ROOT: note = '\nTerrain needs the romfs folder (button above).'
        app.info.setText('%d terrain + %d wall triangles%s' % (self.counts['opaque'] // 3, self.counts['trans'] // 3, note))

    def floor_under(self, s):
        """Surface of the highest triangle below Link's feet."""
        best = None
        x, y, z = s['x'], s['y'], s['z']
        for t, m, sub, f in self.floor:
            lo, hi = t[:, :, [0, 2]].min(1), t[:, :, [0, 2]].max(1)
            cand = np.where((lo[:, 0] <= x) & (hi[:, 0] >= x) & (lo[:, 1] <= z) & (hi[:, 1] >= z))[0]
            for i in cand:
                a, b, c = t[i]
                d = (b[2] - c[2]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[2] - c[2])
                if abs(d) < 1e-9: continue
                w1 = ((b[2] - c[2]) * (x - c[0]) + (c[0] - b[0]) * (z - c[2])) / d
                w2 = ((c[2] - a[2]) * (x - c[0]) + (a[0] - c[0]) * (z - c[2])) / d
                w3 = 1 - w1 - w2
                if min(w1, w2, w3) < -1e-4: continue
                fy = w1 * a[1] + w2 * b[1] + w3 * c[1]
                if fy <= y + 0.6 and (best is None or fy > best[0]): best = (fy, int(m[i]), int(sub[i]), int(f[i]))
        return surface_name(*best[1:]) if best else None

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
        self.mvp = M
        self.prog.bind(); self.vao.bind()
        self.prog.setUniformValue(self.u_mvp, QMatrix4x4(*M.flatten().tolist()))
        f.glEnable(0x0B71); f.glDisable(0x0BE2)  # depth test on, blend off
        self.draw('opaque', GL_TRIANGLES, 1)
        if self.dyn_version != self.app.actor_version:
            self.dyn_version = self.app.actor_version
            self.upload('dyn', self.app.actor_mesh())
        f.glEnable(0x0BE2); f.glBlendFunc(0x0302, 0x0303); f.glDepthMask(False)
        self.draw('trans', GL_TRIANGLES, 1)
        self.draw('dyn', GL_TRIANGLES, 1)
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
        far = self.app.manual is not None and not (self.app.manual & self.app.detected)
        for off, col, r in (() if far else ((0.9, QColor(64, 255, 96), 7), (0.0, QColor(30, 140, 50), 3))):
            q = scr(s['x'], s['y'] + off, s['z'])
            if q: p.setBrush(col); p.setPen(QPen(QColor('black'), 1)); p.drawEllipse(q, r, r)
        if s['face'] and not far:
            a = scr(s['x'], s['y'] + 0.9, s['z'])
            b = scr(s['x'] + s['face'][0] * 1.5, s['y'] + 0.9, s['z'] + s['face'][1] * 1.5)
            if a and b: p.setPen(QPen(QColor(64, 255, 96), 2)); p.drawLine(a, b)
        if self.app.showk['loading'].isChecked():
            p.setFont(QFont('Consolas', 8)); p.setPen(QColor(255, 240, 80))
            act = self.app.active
            for c, name, scene, h in self.app.data.loading_zones:
                if act and self.app.only_zone.isChecked() and scene not in act: continue
                if abs(c[0] - s['x']) > self.app.range.value() or abs(c[2] - s['z']) > self.app.range.value(): continue
                q = scr(*c)
                if q and 0 <= q.x() <= W and 0 <= q.y() <= H: p.drawText(q, name)
        if self.app.show_actors.isChecked():
            p.setFont(QFont('Consolas', 8))
            for name, (x, y, z) in self.app.actor_pos:
                q = scr(x, y + 0.5, z)
                if not q or not (0 <= q.x() <= W and 0 <= q.y() <= H): continue
                col = ACTOR_COL.get(name.split('_')[0], QColor('white'))
                p.setBrush(col); p.setPen(QPen(QColor('black'), 1))
                p.drawPolygon([q + QPointF(0, -6), q + QPointF(6, 0), q + QPointF(0, 6), q + QPointF(-6, 0)])
                if self.app.show_names.isChecked():
                    p.setPen(col); p.drawText(q + QPointF(8, 4), name)
            p.setFont(QFont('Consolas', 9))
        p.setPen(QColor('white'))
        fl = self.floor_under(s)
        if fl: p.drawText(8, 32, 'Floor: ' + fl)
        p.drawText(8, 16, 'Link %.1f %.1f %.1f   %s' % (s['x'], s['y'], s['z'], self.app.zone_msg))
        p.drawText(8, H - 8, {'game': 'Game camera', 'orbit': 'Orbit: drag to rotate, wheel to zoom', 'top': 'Top-down: wheel to zoom',
                              'free': 'Free camera: drag to look, WASD move, E/Q or Space/Ctrl up/down, Shift faster (click the view first)'}[self.mode])
        p.end()


class App(QWidget):
    def __init__(self, geom):
        super().__init__()
        self.setWindowTitle('OoT collision viewer')
        self.data = Data(geom)
        self.game = Game()
        self.state, self.active, self.zone_msg, self.scanning, self.last_pos = None, None, 'zone: detecting...', False, None
        self.found_keys = self.live_keys = None
        self.live_version, self.actor_objs, self.actor_pos, self.last_scan, self.last_actor_read = 0, {}, [], 0, 0
        self.obj_actors = {}
        self.manual, self.actor_version, self.detected = None, 0, set()
        self.hidden_version, self.obj_live_keys, self.dyn_picks, self.dyn = 0, None, [], []
        try:
            self.hidden_keys = np.array(json.load(open(os.path.join(HERE, 'hidden.json'))), np.uint64)
        except Exception:
            self.hidden_keys = np.zeros(0, np.uint64)
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        side = QFrame(); side.setFixedWidth(250); side.setStyleSheet('background:#1b2026; color:#d0d8e0')
        sl = QVBoxLayout(side)
        sl.addWidget(QLabel('<b>View</b>'))
        self.modes = QButtonGroup(self)
        for i, t in enumerate(['Game camera', 'Orbit around Link (drag)', 'Top-down', 'Free camera (drag + WASD)']):
            rb = QRadioButton(t); rb.setChecked(i == 0); self.modes.addButton(rb, i); sl.addWidget(rb)
        self.modes.idClicked.connect(lambda i: self.view.set_mode(['game', 'orbit', 'top', 'free'][i]))
        sl.addWidget(QLabel('<b>Show</b>'))
        self.showk = {}
        for k, label, rgb, a, on in KINDS:
            cb = QCheckBox(label); cb.setChecked(on); cb.setStyleSheet('color: rgb(%d,%d,%d)' % rgb)
            self.showk[k] = cb; sl.addWidget(cb)
        sl.addWidget(QLabel('<b>Filters</b>'))
        self.show_actors = QCheckBox('Enemies / NPCs / animals (live)'); self.show_actors.setChecked(True); sl.addWidget(self.show_actors)
        self.show_names = QCheckBox('Actor names'); sl.addWidget(self.show_names)
        self.only_zone = QCheckBox('Only the loaded zone'); self.only_zone.setChecked(True); sl.addWidget(self.only_zone)
        self.only_live = QCheckBox('Only things that exist right now'); self.only_live.setChecked(True); sl.addWidget(self.only_live)
        self.only_height = QCheckBox("Only walls right around Link"); sl.addWidget(self.only_height)
        self.surface_mode = QCheckBox('Colour terrain by surface (lava, abyss, slip...)'); sl.addWidget(self.surface_mode)
        self.legend = QLabel(); self.legend.setWordWrap(True); self.legend.setVisible(False); sl.addWidget(self.legend)
        sq = lambda rgb, t: '<span style="color:rgb(%d,%d,%d)">&#9632;</span> %s' % (*rgb, t)
        mats = [(MAT_COL[k], MAT_NAMES[k]) for k in (2, 1, 7, 3, 5, 14, 10, 15, 28)]
        self.legend.setText('<small>' + ' &nbsp;'.join(sq(c, n) for c, n in mats) + '<br><b>Hazards:</b> ' +
                            ' &nbsp;'.join(sq(c, n) for n, c in HAZARD_COL.items()) + '<br><b>Actors:</b> ' +
                            ' &nbsp;'.join(sq((c.red(), c.green(), c.blue()), n) for n, c in ACTOR_COL.items() if n != 'Item') + '</small>')
        self.surface_mode.toggled.connect(self.legend.setVisible)
        self.no_ceiling = QCheckBox('Hide ceilings in dungeons (orbit / top / free views)'); sl.addWidget(self.no_ceiling)
        self.zone_pick = QComboBox(); self.zone_pick.addItem('Zone: follow the game')
        for g in self.data.groups: self.zone_pick.addItem(g)
        self.zone_pick.currentIndexChanged.connect(self.pick_zone); sl.addWidget(self.zone_pick)
        b = QPushButton('Re-detect zone'); b.clicked.connect(self.detect); sl.addWidget(b)
        b = QPushButton('Set romfs folder (for terrain)'); b.clicked.connect(self.pick_romfs); sl.addWidget(b)
        self.range = QSlider(Qt.Horizontal); self.range.setRange(15, 300); self.range.setValue(120)
        self.range_lbl = QLabel('Draw distance: 120 units'); sl.addWidget(self.range_lbl); sl.addWidget(self.range)
        self.range.valueChanged.connect(lambda v: self.range_lbl.setText('Draw distance: %d units' % v))
        self.info = QLabel(); self.info.setWordWrap(True); sl.addWidget(self.info)
        sl.addStretch(1)
        lay.addWidget(side)
        self.view = GLView(self); lay.addWidget(self.view, 1)
        self.timer = QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(33)
        self.resize(1300, 820)

    def pick_zone(self, i):
        if i == 0:
            self.manual = None; self.detect()
        else:
            g = self.zone_pick.currentText()
            self.manual = {s for s in self.data.scene_names if re.sub(r'_(Static|Dynamic)$', '', s) == g}
            self.active = self.manual
            self.zone_msg = 'zone (manual): ' + g
            s = self.state
            ids = [i for i, n in enumerate(self.data.scene_names) if n in self.manual]
            c = self.data.wc[np.isin(self.data.ws, ids)]
            if len(c) and not (self.detected and self.manual & self.detected):
                cx, cy, cz = np.median(c, 0)
                self.view.set_mode('free')
                self.modes.button(3).setChecked(True)
                self.view.free = [float(cx), float(cy) + 60, float(cz) + 60, 0.0, -0.75]

    def hide(self, h):
        self.hidden_keys = np.union1d(self.hidden_keys, np.array([h], np.uint64)); self.save_hidden()

    def unhide_all(self):
        self.hidden_keys = np.zeros(0, np.uint64); self.save_hidden()

    def save_hidden(self):
        self.hidden_version += 1; self.actor_version += 1
        json.dump([int(x) for x in self.hidden_keys], open(os.path.join(HERE, 'hidden.json'), 'w'))

    def actor_mesh(self):
        """Live objects and actors: their real collision shape at the live position and rotation."""
        out, picks = [], []
        hidden = set(int(x) for x in self.hidden_keys)
        for h, name, pos, R, scale, rgb, alpha in self.dyn:
            if h in hidden: continue
            t = self.data.shapes.get(name)
            if t is None: continue
            w = (t * np.array(scale, np.float32)) @ R.T + np.array(pos, np.float32)
            out.append(vertices(w, rgb, alpha)); picks.append((w, np.full(len(w), h, np.uint64)))
        self.dyn_picks = picks
        return np.concatenate(out) if out else np.zeros((0, 10), np.float32)

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
        keys = np.array(sorted({h for s in self.data.scene_items.values() for h in s} | set(self.data.actors)), np.uint64)
        first = self.active is None
        def work():
            try:
                t = time.time()
                found = self.game.scan_u64(keys, 0x2140000000, 0x2160000000,
                                           (lambda f: setattr(self, 'zone_msg', 'zone: scanning %d%%' % (f * 100))) if first else None)
                hit = set(found)
                ratio = {s: len(hs & hit) / len(hs) for s, hs in self.data.scene_items.items() if hs}
                act = {s for s, r in ratio.items() if r >= 0.9}
                if not act and ratio and max(ratio.values()) >= 0.5:
                    best = max(ratio, key=ratio.get)
                    grp = re.sub(r'_(Static|Dynamic)$', '', best)
                    act = {s for s in ratio if re.sub(r'_(Static|Dynamic)$', '', s) == grp}
                live = {}
                for h, recs in found.items():
                    for r in recs:
                        o = self.game.actor_of(r)
                        if o: live[h] = o; break
                self.detected = act
                if self.manual is None: self.active = act or None
                self.found_keys = np.array(sorted(hit), np.uint64)
                self.live_keys = np.array(sorted(live), np.uint64)
                self.actor_objs = {h: o for h, o in live.items() if h in self.data.actors}
                self.obj_actors = {h: o for h, o in live.items() if h in self.data.obj_pos}
                self.live_version += 1
                self.last_scan = time.time()
                if self.manual is None: self.zone_msg = 'zone: %s (%.0fs)' % (', '.join(sorted(act)) if act else 'not found, showing all', time.time() - t)
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
            if time.time() - self.last_scan > 20 and not self.scanning: self.detect()
            elsewhere = self.manual is not None and not (self.manual & self.detected)
            if elsewhere and self.actor_pos:
                self.actor_pos = []; self.actor_version += 1
            if not elsewhere and time.time() - self.last_actor_read > 0.15:
                self.last_actor_read = time.time()
                rng = self.range.value()
                cx, cz = (self.view.free[0], self.view.free[2]) if self.view.mode == 'free' and self.view.free else (s['x'], s['z'])
                markers, dyn = [], []
                show_obj = self.showk['object'].isChecked()
                for h, obj in list(self.actor_objs.items()) + list(self.obj_actors.items()):
                    is_obj = h in self.data.obj_info
                    if is_obj and not show_obj: continue
                    if not is_obj and not self.show_actors.isChecked(): continue
                    pos = self.game.actor_pos(obj)
                    if not pos or sum(map(abs, pos)) < 0.01 or abs(pos[0] - cx) > rng or abs(pos[2] - cz) > rng: continue
                    if is_obj:
                        name, scale = self.data.obj_info[h]
                        rgb, alpha = KCOL['object']
                    else:
                        name, scale = self.data.actors[h], [1, 1, 1]
                        c = ACTOR_COL.get(name.split('_')[0], QColor('white')); rgb, alpha = (c.red(), c.green(), c.blue()), 110
                        markers.append((name, pos))
                    R = self.game.actor_rot(obj)
                    dyn.append((h, name, pos, R if R is not None else np.eye(3, dtype=np.float32), scale, rgb, alpha))
                self.actor_pos, self.dyn = markers, dyn
                self.obj_live_keys = np.array(sorted(h for h, *_ in dyn if h in self.data.obj_info), np.uint64)
                self.actor_version += 1
        self.view.step_free()
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


def log_errors(kind, value, tb):
    import traceback
    with open(os.path.join(HERE, 'viewer_error.log'), 'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + ''.join(traceback.format_exception(kind, value, tb)) + '\n')


if __name__ == '__main__':
    sys.excepthook = log_errors
    fmt = QSurfaceFormat(); fmt.setDepthBufferSize(24); fmt.setSamples(4); QSurfaceFormat.setDefaultFormat(fmt)
    qa = QApplication(sys.argv)
    w = App(setup(qa)); w.show()
    sys.exit(qa.exec())
