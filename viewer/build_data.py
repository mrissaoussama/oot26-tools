"""Builds geom.json (invisible walls and death zones, world space) from the romfs dump."""
import json, math, os, re
import numpy as np
import romfs, havok
from byml import Byml

HERE = os.path.dirname(os.path.abspath(__file__))
WALL = re.compile(r'AirWall|Airwall')
# FallDown_Item only detects objects/animals; FallDown (incl. Mountain/GanonCastle variants) detects the player.
ZONE_LOGIC = [('Gimmick_AreaContact_FallDown_Item', None), ('Gimmick_AreaContact_FallDown', 'void'),
              ('InvalidateRestartArea', 'norespawn'), ('Whirlpool', 'whirlpool')]
# Scene-specific area modules (e.g. Haunted Wasteland quicksand) are classified by the nodes inside the logic file.
MODULE_RULES = [(b'GiveAbyssItem', None), (b'FallDown_Item', None), (b'SandAbyss', 'void', 'Quicksand'),
                (b'OutAreaCheck', 'safepath', 'Safe path (leaving every one of these sends you back)'), (b'GiveAbyssActor', 'void', 'Void (scripted)'),
                (b'InvalidateRestartPos', 'norespawn', 'No respawn here')]
_modules = None


def module_kind(logic):
    """(kind, label) for a logic module path by scanning the AINB for known node names."""
    global _modules
    if _modules is None:
        _modules = {}
        for d, _, fs in os.walk(romfs.ROOT):
            for f in fs:
                if '.logic.module.ainb' in f: _modules[f.split('.')[0]] = os.path.join(d, f)
    f = _modules.get(logic.split('/')[-1].split('.')[0])
    if not f: return None
    b = open(f, 'rb').read()
    if f.endswith('.zs'): b = romfs.dec(b)
    for rule in MODULE_RULES:
        if rule[0] in b and (rule[1] is None or b'Player' in b):
            return (rule[1], rule[2]) if rule[1] else None
    return None


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


def main_shape(name):
    """Local-space polygons of an actor's main (largest) collision shape, or None."""
    best = None
    for sp in actor_shapes(name):
        a = sp.get('AutoCalc') or {}
        if not any(isinstance(v, list) and v for k, v in sp.items() if k != 'AutoCalc'): continue
        if best is None or a.get('Volume', 0) > (best.get('AutoCalc') or {}).get('Volume', 0): best = sp
    if not best: return None
    polys = []
    for b in best.get('Box', []): polys += box(b)
    for c in best.get('Cylinder', []): polys += cylinder(c)
    for c in best.get('Capsule', []): polys += cylinder(c)
    for c in best.get('Sphere', []):
        r = c.get('Radius', 0.5); ce = v3(c.get('Center')) + v3(c.get('OffsetTranslation'))
        polys += cylinder({'CenterA': {'X': ce[0], 'Y': ce[1] + r, 'Z': ce[2]}, 'CenterB': {'X': ce[0], 'Y': ce[1] - r, 'Z': ce[2]}, 'Radius': r}, 10)
    for m in best.get('PhshMesh', []): polys += mesh(m['PhshMeshPath'])
    if not polys and best.get('AutoCalc'):
        a = best['AutoCalc']
        lo, hi = np.array([a['Min'][k] for k in 'XYZ']), np.array([a['Max'][k] for k in 'XYZ'])
        polys = box({'HalfExtents': dict(zip('XYZ', (hi - lo) / 2)), 'Center': dict(zip('XYZ', (hi + lo) / 2))})
    return [np.round(np.asarray(p, float), 3).tolist() for p in polys] if polys else None


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
    out, live_actors, boxes, shapes = [], {}, {}, {}
    files = [(dp, f) for dp, _, fs in os.walk(romfs.path('Banc')) for f in fs if f.endswith('.bcett.byml.zs')]
    for i, (dp, f) in enumerate(files):
        progress('Building collision data: %d/%d %s' % (i + 1, len(files), f))
        root = Byml(romfs.dec(open(os.path.join(dp, f), 'rb').read())).root
        if not isinstance(root, dict): continue
        scene = f.replace('.bcett.byml.zs', '')
        actors = {a['Hash']: a for a in root.get('Actors', [])}
        for h, a in actors.items():
            if a.get('Gyaml', '').startswith(ACTOR_KINDS):
                live_actors[str(h)] = a['Gyaml']
                if a['Gyaml'] not in boxes: boxes[a['Gyaml']] = main_box(a['Gyaml'])
                if a['Gyaml'] not in shapes: shapes[a['Gyaml']] = main_shape(a['Gyaml'])
        for a in actors.values():
            name = a.get('Gyaml', '')
            if not WALL.search(name):
                if a.get('Bakeable') or not name.startswith(OBJ_KEEP) or name.startswith(OBJ_SKIP): continue
                if name not in shapes: shapes[name] = main_shape(name)
                polys = shapes[name]
                if polys:
                    out.append({'scene': scene, 'name': name, 'hash': a['Hash'], 'kind': 'object', 'polys': place(polys, a),
                                'pos': a.get('Translate', [0, 0, 0]), 'scale': a.get('Scale', [1, 1, 1])})
                continue
            polys = []
            for s in actor_shapes(name):
                for b in s.get('Box', []): polys += box(b)
                for c in s.get('Cylinder', []): polys += cylinder(c)
                for m in s.get('PhshMesh', []): polys += mesh(m['PhshMeshPath'])
            if polys:
                out.append({'scene': scene, 'name': name, 'hash': a['Hash'], 'kind': kind_of(name), 'polys': place(polys, a)})
        # Death zones: Area actors referenced by fall-down / no-respawn logic.
        # RigidBodyShapeType: 0 sphere, 1 box, 3 cylinder; unit shapes sit on their origin (y 0..1) like AirWall cubes.
        for g in root.get('AiGroups', []):
            logic = g.get('Logic', '')
            if 'FieldDungeonEntranceSet' in logic:
                refs = [actors.get(r.get('Reference')) for r in g.get('References', [])]
                dest = next(((a.get('Dynamic') or {}) for a in refs if a and a.get('Gyaml') == 'FieldAndDungeonEntranceEventRequestActor'), {})
                label = dest.get('NextBancParamNameChild') or dest.get('NextBancParamNameAdult') or '?'
                if dest.get('NextBancParamNameAdult') and dest.get('NextBancParamNameAdult') != dest.get('NextBancParamNameChild'):
                    label += ' / ' + dest['NextBancParamNameAdult']
                for a in refs:
                    if a and a.get('Gyaml') == 'Area':
                        out.append({'scene': scene, 'name': 'to ' + label, 'hash': a['Hash'], 'kind': 'loading',
                                    'polys': place(box({'OffsetTranslation': {'X': 0, 'Y': 0.5, 'Z': 0}}), a)})
                continue
            kind = next((k for pat, k in ZONE_LOGIC if pat in logic), False)
            label = None
            if kind is False and '/Module/' in logic:
                mk = module_kind(logic)
                if mk: kind, label = mk
            if not kind: continue
            for ref in g.get('References', []):
                a = actors.get(ref.get('Reference'))
                if not a or a.get('Gyaml') != 'Area': continue
                st = (a.get('Dynamic') or {}).get('RigidBodyShapeType', 1)
                if st == 3:
                    polys = place(cylinder({'CenterA': {'Y': 1}, 'CenterB': {'Y': 0}, 'Radius': 1.0}), a)
                elif st == 0:
                    polys = place(cylinder({'CenterA': {'Y': 1}, 'CenterB': {'Y': -1}, 'Radius': 1.0}), {**a, 'Translate': a.get('Translate', [0, 0, 0])})
                else:
                    polys = place(box({'OffsetTranslation': {'X': 0, 'Y': 0.5, 'Z': 0}}), a)
                out.append({'scene': scene, 'name': label or re.search(r'Module/([^/]+)/', logic).group(1), 'hash': a['Hash'],
                            'kind': kind, 'polys': polys})
    json.dump(out, open(os.path.join(HERE, 'geom.json'), 'w'))
    json.dump(travel_table(), open(os.path.join(HERE, 'travel.json'), 'w'))
    json.dump({'actors': live_actors, 'boxes': boxes, 'shapes': {k: v for k, v in shapes.items() if v}},
              open(os.path.join(HERE, 'actors.json'), 'w'))
    return out


def travel_table():
    """{map: [[spawn point, arrival event, label]]} from every StartPos in the game plus the exits that lead to it."""
    root = romfs.path('Banc')
    spawns, events = {}, {}
    for area in sorted(os.listdir(root)):
        folder = os.path.join(root, area)
        if not os.path.isdir(folder): continue
        for f in sorted(os.listdir(folder)):
            if not f.endswith('.bcett.byml.zs'): continue
            scene = f.split('.')[0]
            banc = re.sub(r'_(Static|Dynamic)$', '', scene)
            for a in Byml(romfs.read('Banc', area, f)).root.get('Actors', []):
                d = a.get('Dynamic') or {}
                if a.get('Gyaml') == 'StartPos':
                    names = spawns.setdefault(banc, {})
                    if a.get('Name') and a['Name'] != 'EntrancePos': names.setdefault(a['Name'], None)
                pos = d.get('NextStartPosName')
                for target in {d.get('NextBancParamNameChild'), d.get('NextBancParamNameAdult')} - {None}:
                    if pos: events.setdefault((target, pos), (d.get('NextEventName') or 'Dm_SY_BancChangeArrive', scene))
    for target, pos in events: spawns.setdefault(target, {}).setdefault(pos, None)
    out = {}
    for banc, names in sorted(spawns.items()):
        rows = []
        for n in sorted(names) or ['StartPos']:
            ev, src = events.get((banc, n), ('Dm_SY_BancChangeArrive', None))
            rows.append([n, ev, ('from ' + re.sub(r'_(Static|Dynamic)$', '', src)) if src else 'other spawn point' if names else 'default spawn point'])
        out[banc] = rows
    return out


if __name__ == '__main__':
    if not romfs.load_config(): raise SystemExit('Set the romfs path first (run the viewer once).')
    print(len(build(lambda m: None)), 'objects')
