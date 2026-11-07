"""Builds geom.json (invisible walls and death zones, world space) from the romfs dump."""
import json, math, os, re
import numpy as np
import romfs, havok
from byml import Byml

HERE = os.path.dirname(os.path.abspath(__file__))
WALL = re.compile(r'AirWall|Airwall')
ZONE_LOGIC = [('Gimmick_AreaContact_FallDown_Item', None), ('Gimmick_AreaContact_FallDown', 'void'), ('ItemAbyss', 'void'),
              ('InvalidateRestartArea', 'norespawn'), ('Whirlpool', 'whirlpool')]
ACTOR_KINDS = ('Enemy_', 'Npc_', 'Animal_', 'Item_', 'Obj_Rupee', 'Obj_Heart', 'Obj_FairySpirit')
OBJ_KEEP = ('Obj_', 'DgnObj_', 'FldObj_', 'TBox_')
OBJ_SKIP = ('Npc_', 'Enemy_', 'Animal_', 'Item_', 'Dm_', 'EftLoc', 'Area', 'DestinationAnchor', 'Event', 'Sound',
            'Snd', 'BGM', 'Navi', 'StartPos', 'Dungeon', 'Camera', 'Env', 'Volume', 'Forbid', 'Drop', 'Horse', 'Player')
NON_PLAYER = ('Enemy', 'Horse', 'Animal', 'Cucco', 'Npc', 'Camera', 'Sight', 'Lure', 'Bomb', 'Hookshot', 'RollingRock',
              'Dodongo', 'Poe', 'Gohma', 'HitTarget', 'ObjectAndSmall', 'CrawlHole', 'HidingHole')


def gyml(files, p, cache=None):
    """Loads a Work/ path from an actor pack, applying $parent inheritance."""
    key = p.replace('Work/', '').replace('.gyml', '.bgyml')
    if key not in files: return {}
    d = Byml(files[key]).root or {}
    base = gyml(files, d['$parent']) if d.get('$parent') else {}
    base.update({k: v for k, v in d.items() if k != '$parent'})
    return base


_shapes = {}
def actor_shapes(name):
    if name in _shapes: return _shapes[name]
    res = []
    f = romfs.path('Pack', 'Actor', name + '.pack.zs')
    if os.path.exists(f):
        files = romfs.sarc(romfs.read('Pack', 'Actor', name + '.pack.zs'))
        actor = gyml(files, 'Work/Actor/%s.engine__actor__ActorParam.gyml' % name)
        ref = actor.get('Components', {}).get('PhysicsRef')
        phys = gyml(files, ref) if ref else {}
        cset = gyml(files, phys['ControllerSetPath']) if phys.get('ControllerSetPath') else {}
        res = [gyml(files, sp['FilePath']) for sp in cset.get('ShapeNamePathAry', [])]
        if not res:
            res = [Byml(v).root or {} for k, v in files.items() if 'ShapeParam' in k]
    _shapes[name] = res
    return res


def main_box(name):
    """AABB (min, max) of an object's main collision shape, from the shape's AutoCalc block."""
    best = None
    for s in actor_shapes(name):
        a = s.get('AutoCalc')
        if not a or not any(isinstance(v, list) and v for k, v in s.items() if k != 'AutoCalc'): continue
        if best is None or a.get('Volume', 0) > best.get('Volume', 0): best = a
    if not best: return None
    return [[best['Min'][k] for k in 'XYZ'], [best['Max'][k] for k in 'XYZ']]


def rot_matrix(r):
    rx, ry, rz = r
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    X = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Z @ Y @ X


def v3(d, default=0.0):
    return np.array([d.get('X', default), d.get('Y', default), d.get('Z', default)]) if d else np.zeros(3) + default


def box(b):
    c = v3(b.get('Center')) + v3(b.get('OffsetTranslation'))
    h = v3(b.get('HalfExtents'), 0.5)
    k = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * h + c
    return [k[list(f)] for f in [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]]


def cylinder(c, n=16):
    a, b, r = v3(c.get('CenterA')), v3(c.get('CenterB')), c.get('Radius', 1.0)
    ring = [np.array([math.cos(2 * math.pi * i / n) * r, 0, math.sin(2 * math.pi * i / n) * r]) for i in range(n)]
    return [np.array([b + ring[i], b + ring[(i + 1) % n], a + ring[(i + 1) % n], a + ring[i]]) for i in range(n)]


_meshes = {}
def mesh(p):
    if p not in _meshes:
        f = romfs.path('Phive', 'Shape', 'Dcc', os.path.basename(p).replace('.phsh', '.Nin_Ounce_NVN2.bphsh.zs'))
        _meshes[p] = list(havok.file_meshes(romfs.dec(open(f, 'rb').read()))) if os.path.exists(f) else []
    return _meshes[p]


def kind_of(name):
    if 'Climb' in name: return 'climb'
    if name.startswith('PushAirWall'): return 'push'
    if any(k in name for k in NON_PLAYER): return 'other'
    return 'player'


def place(polys, a):
    R = rot_matrix(a.get('Rotate', [0, 0, 0]))
    S, T = np.array(a.get('Scale', [1, 1, 1]), float), np.array(a.get('Translate', [0, 0, 0]), float)
    return [((np.asarray(p) * S) @ R.T + T).round(3).tolist() for p in polys]


def build(progress=print):
    out, live_actors = [], {}
    files = [(dp, f) for dp, _, fs in os.walk(romfs.path('Banc')) for f in fs if f.endswith('.bcett.byml.zs')]
    for i, (dp, f) in enumerate(files):
        progress('Building collision data: %d/%d %s' % (i + 1, len(files), f))
        root = Byml(romfs.dec(open(os.path.join(dp, f), 'rb').read())).root
        if not isinstance(root, dict): continue
        scene = f.replace('.bcett.byml.zs', '')
        actors = {a['Hash']: a for a in root.get('Actors', [])}
        for h, a in actors.items():
            if a.get('Gyaml', '').startswith(ACTOR_KINDS): live_actors[str(h)] = a['Gyaml']
        for a in actors.values():
            name = a.get('Gyaml', '')
            if not WALL.search(name):
                if a.get('Bakeable') or not name.startswith(OBJ_KEEP) or name.startswith(OBJ_SKIP): continue
                bb = main_box(name)
                if bb and np.prod(np.subtract(bb[1], bb[0])) > 1e-4:
                    lo, hi = np.array(bb[0]), np.array(bb[1])
                    c, h = (lo + hi) / 2, (hi - lo) / 2
                    polys = box({'HalfExtents': dict(zip('XYZ', h)), 'Center': dict(zip('XYZ', c))})
                    out.append({'scene': scene, 'name': name, 'hash': a['Hash'], 'kind': 'object', 'polys': place(polys, a)})
                continue
            polys = []
            for s in actor_shapes(name):
                for b in s.get('Box', []): polys += box(b)
                for c in s.get('Cylinder', []): polys += cylinder(c)
                for m in s.get('PhshMesh', []): polys += mesh(m['PhshMeshPath'])
            if polys:
                out.append({'scene': scene, 'name': name, 'hash': a['Hash'], 'kind': kind_of(name), 'polys': place(polys, a)})
        # Death zones: Area actors referenced by fall-down / no-respawn logic. Shape type 3 is a kill plane.
        for g in root.get('AiGroups', []):
            logic = g.get('Logic', '')
            kind = next((k for pat, k in ZONE_LOGIC if pat in logic), False)
            if not kind: continue
            for ref in g.get('References', []):
                a = actors.get(ref.get('Reference'))
                if not a or a.get('Gyaml') != 'Area': continue
                if (a.get('Dynamic') or {}).get('RigidBodyShapeType', 1) == 3:
                    sq = np.array([[-60, 0, -60], [60, 0, -60], [60, 0, 60], [-60, 0, 60]], float)
                    polys = place([sq], {**a, 'Scale': [1, 1, 1]})
                else:
                    polys = place(box({'OffsetTranslation': {'X': 0, 'Y': 0.5, 'Z': 0}}), a)
                out.append({'scene': scene, 'name': re.search(r'Module/([^/]+)/', logic).group(1), 'hash': a['Hash'],
                            'kind': kind, 'polys': polys})
    json.dump(out, open(os.path.join(HERE, 'geom.json'), 'w'))
    json.dump(live_actors, open(os.path.join(HERE, 'actors.json'), 'w'))
    return out


if __name__ == '__main__':
    if not romfs.load_config(): raise SystemExit('Set the romfs path first (run the viewer once).')
    print(len(build(lambda m: None)), 'objects')
