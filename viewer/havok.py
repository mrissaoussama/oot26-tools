"""Phive / Havok tagfile mesh decoding (hknpMeshShape) and zone collision tiles."""
import math, os, re, struct
import numpy as np
import romfs

TILE = 250.0


def tag_sections(b):
    out, i = [], 0
    while True:
        i = b.find(b'TAG0', i)
        if i < 0: return out
        o = i - 4
        sz = struct.unpack_from('>I', b, o)[0] & 0x3FFFFFFF
        out.append((o, sz)); i = o + max(sz, 8)


def parse_tag(b, o, sz):
    end, p = o + sz, o + 8
    data = items = names = None
    while p < end:
        s = struct.unpack_from('>I', b, p)[0] & 0x3FFFFFFF
        if s < 8: break
        tag = b[p + 4:p + 8]
        if tag == b'DATA': data = p + 8
        if tag in (b'TYPE', b'INDX'):
            q = p + 8
            while q < p + s:
                s2 = struct.unpack_from('>I', b, q)[0] & 0x3FFFFFFF
                if s2 < 8: break
                if b[q + 4:q + 8] == b'ITEM': items = [struct.unpack_from('<III', b, r) for r in range(q + 8, q + s2 - 11, 12)]
                if b[q + 4:q + 8] == b'TST1': names = b[q + 8:q + s2].split(b'\0')
                q += s2
        p += s
    return data, items, names


def mesh_tris(b, D, items):
    """Geometry sections: u16 vertices, (sectionOffset + v) / 512 = position; primitives are quads of u8 indices."""
    starts = {off: cnt for _, off, cnt in items if cnt}
    for _, off, cnt in items:
        if not cnt: continue
        secs = []
        for i in range(cnt):
            o = D + off + i * 0x40
            if o + 0x40 > len(b): break
            rel = [struct.unpack_from('<iI', b, o + j) for j in range(0, 32, 8)]
            tg = [o + j * 8 - D + r[0] for j, r in enumerate(rel)]
            if rel[1][1] == 0 or rel[2][1] == 0 or not all(starts.get(t) == r[1] for t, r in zip(tg, rel)): break
            secs.append((o, rel))
        if len(secs) != cnt: continue
        out = []
        for o, rel in secs:
            P = np.frombuffer(b, np.uint8, rel[1][1] * 4, o + 8 + rel[1][0]).reshape(-1, 4).astype(np.int64)
            nv = int(P.max()) + 1
            if o + 16 + rel[2][0] + nv * 6 > len(b): continue
            V = np.frombuffer(b, '<u2', nv * 3, o + 16 + rel[2][0]).reshape(-1, 3).astype(np.float32)
            W = (np.array(struct.unpack_from('<3i', b, o + 32), np.float32) + V) / 512.0
            out.append(W[P[:, [0, 1, 2]]])
            quad = P[:, 2] != P[:, 3]
            if quad.any(): out.append(W[P[quad][:, [0, 2, 3]]])
        return np.concatenate(out) if out else np.zeros((0, 3, 3), np.float32)
    return np.zeros((0, 3, 3), np.float32)


def file_meshes(b):
    res = []
    for o, sz in tag_sections(b):
        D, items, names = parse_tag(b, o, sz)
        if names and names[0] == b'hknpMeshShape' and D and items:
            res.append(mesh_tris(b, D, items))
    return np.concatenate(res) if res else np.zeros((0, 3, 3), np.float32)


def tiles_for(scene):
    """'Graveyard_Child_Static' -> {(x, z): file}. Dungeons use a single untiled file at key (0, 0)."""
    prefix = re.sub(r'_(Static|Dynamic)$', '', scene)
    base = romfs.path('Phive', 'StaticCompoundBody')
    out = {}
    for d in os.listdir(base):
        if not prefix.startswith(d): continue
        for f in os.listdir(os.path.join(base, d)):
            m = re.match(re.escape(prefix) + r'(?:_X(\d+)_Z(\d+))?\.Nin', f)
            if m: out[(int(m.group(1) or 0), int(m.group(2) or 0))] = os.path.join(base, d, f)
    return out


def tile_origin(b, key, tiles):
    """Tile translation stored after a (1, 1, 1, 0) scale vector; falls back to a centred grid."""
    i = b.find(struct.pack('<4f', 1, 1, 1, 0))
    if i >= 0:
        x, y, z = struct.unpack_from('<3f', b, i + 24)
        if all(abs(v) < 1e5 for v in (x, y, z)): return np.array([x, y, z], np.float32)
    nx = max(k[0] for k in tiles) + 1
    nz = max(k[1] for k in tiles) + 1
    return np.array([(key[0] - nx / 2) * TILE, 0, (key[1] - nz / 2) * TILE], np.float32)


NOT_SOLID = re.compile(r'AirWall|Airwall|Area|Anchor|Camera|Sensor')


def load_tile(path, key, tiles, names=None):
    """Terrain mesh (tile-local) plus baked objects (compound shapes, world space).
    names: placement hash -> actor name; baked areas and air walls are left out."""
    b = romfs.dec(open(path, 'rb').read())
    skip = None
    if names:
        skip = {i for i, h in instance_actors(b).items() if NOT_SOLID.search(names.get(h, ''))}
    out = []
    tris = file_meshes(b)
    if len(tris):
        ok = np.all((tris > -2) & (tris < TILE + 2), axis=(1, 2))
        out.append(tris[ok] + tile_origin(b, key, tiles))
    comp = file_compounds(b, skip)
    if len(comp):
        out.append(comp[np.all(np.isfinite(comp) & (np.abs(comp) < 1e5), axis=(1, 2))])
    return np.concatenate(out) if out else np.zeros((0, 3, 3), np.float32)


def _varuint(b, p):
    x = b[p]
    if x & 0x80 == 0: return x, p + 1
    if x & 0xC0 == 0x80: return ((x & 0x3F) << 8) | b[p + 1], p + 2
    if x & 0xE0 == 0xC0: return ((x & 0x1F) << 16) | (b[p + 1] << 8) | b[p + 2], p + 3
    if x & 0xF8 == 0xE0: return ((x & 0x07) << 24) | (b[p + 1] << 16) | (b[p + 2] << 8) | b[p + 3], p + 4
    return ((x & 0x07) << 32) | int.from_bytes(b[p + 1:p + 5], 'big'), p + 5


class Tag:
    """One Havok tagfile: DATA offset, items {offset: (type name, count)}."""
    def __init__(self, b, o, sz):
        self.b = b
        secs, end, p = {}, o + sz, o + 8
        while p < end:
            s = struct.unpack_from('>I', b, p)[0] & 0x3FFFFFFF
            if s < 8: break
            tag = b[p + 4:p + 8]
            secs[tag] = (p + 8, p + s)
            if tag in (b'TYPE', b'INDX'):
                q = p + 8
                while q < p + s:
                    s2 = struct.unpack_from('>I', b, q)[0] & 0x3FFFFFFF
                    if s2 < 8: break
                    secs[b[q + 4:q + 8]] = (q + 8, q + s2); q += s2
            p += s
        self.D = secs[b'DATA'][0]
        names = b[secs[b'TST1'][0]:secs[b'TST1'][1]].split(b'\0')
        p = secs[b'TNA1'][0]
        n, p = _varuint(b, p)
        types = [None]
        for _ in range(1, n):
            ni, p = _varuint(b, p); tc, p = _varuint(b, p)
            for _ in range(tc): p = _varuint(b, _varuint(b, p)[1])[1]
            types.append(names[ni].decode('latin1'))
        it = secs[b'ITEM']
        self.items = {}
        for r in range(it[0], it[1] - 11, 12):
            t, off, cnt = struct.unpack_from('<III', b, r)
            if cnt and 0 < (t & 0xFFFFFF) < len(types): self.items[off] = (types[t & 0xFFFFFF], cnt)
        self.root = types[struct.unpack_from('<I', b, it[0] + 12)[0] & 0xFFFFFF] if it[1] - it[0] >= 24 else None

    def rel(self, off):
        """hkRelArray / hkRelPtr at data offset off -> (target offset, count)."""
        r, n = struct.unpack_from('<iI', self.b, self.D + off)
        return off + r, n

    def f32(self, off, n): return np.frombuffer(self.b, np.float32, n, self.D + off)


def _quat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], np.float32)


def _convex(t, off):
    vo, vn = t.rel(off + 0x38); fo, fn = t.rel(off + 0x48); io, inn = t.rel(off + 0x50)
    if not vn or not fn: return np.zeros((0, 3, 3), np.float32)
    V = t.f32(vo, vn * 3).reshape(-1, 3)
    idx = np.frombuffer(t.b, np.uint8, inn, t.D + io)
    out = []
    for i in range(fn):
        first, cnt = struct.unpack_from('<HB', t.b, t.D + fo + i * 4)
        f = idx[first:first + cnt]
        if len(f) >= 3 and f.max() < vn:
            out.append(np.stack([np.repeat(V[f[:1]], len(f) - 2, 0), V[f[1:-1]], V[f[2:]]], 1))
    return np.concatenate(out) if out else np.zeros((0, 3, 3), np.float32)


def _capsule(t, off, n=12):
    po, pn = t.rel(off + 0x38)
    if pn < 2: return np.zeros((0, 3, 3), np.float32)
    a, c = t.f32(po, 3), t.f32(po + 12, 3)
    r = float(t.f32(off + 0x68, 1)[0])
    axis = c - a; L = np.linalg.norm(axis)
    if L < 1e-6: return np.zeros((0, 3, 3), np.float32)
    axis = axis / L
    u = np.cross(axis, [0, 1, 0] if abs(axis[1]) < 0.9 else [1, 0, 0]); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    ring = [u * math.cos(2 * math.pi * i / n) * r + v * math.sin(2 * math.pi * i / n) * r for i in range(n)]
    out = []
    for i in range(n):
        p0, p1 = ring[i], ring[(i + 1) % n]
        out += [[a + p0, a + p1, c + p1], [a + p0, c + p1, c + p0]]
    return np.array(out, np.float32)


def _mesh_obj(t, off):
    so, sn = t.rel(off + 0x88)
    out = []
    for i in range(sn):
        o = so + i * 0x40
        po, pn = t.rel(o + 8); vo, _ = t.rel(o + 16)
        if not pn: continue
        P = np.frombuffer(t.b, np.uint8, pn * 4, t.D + po).reshape(-1, 4).astype(np.int64)
        nv = int(P.max()) + 1
        V = np.frombuffer(t.b, '<u2', nv * 3, t.D + vo).reshape(-1, 3).astype(np.float32)
        W = (np.array(struct.unpack_from('<3i', t.b, t.D + o + 32), np.float32) + V) / 512.0
        out.append(W[P[:, [0, 1, 2]]])
        quad = P[:, 2] != P[:, 3]
        if quad.any(): out.append(W[P[quad][:, [0, 2, 3]]])
    return np.concatenate(out) if out else np.zeros((0, 3, 3), np.float32)


def _shape(t, off, depth=0):
    kind = t.items.get(off, (None,))[0]
    if kind in ('hknpBoxShape', 'hknpConvexShape', 'hknpConvexPolytopeShape'): return _convex(t, off)
    if kind == 'hknpCapsuleShape': return _capsule(t, off)
    if kind == 'hknpMeshShape': return _mesh_obj(t, off)
    if kind == 'hknpCompoundShape' and depth < 4: return _compound(t, off, depth + 1)
    return np.zeros((0, 3, 3), np.float32)



def _compound(t, off, depth=0, skip=None, base=0):
    """skip: set of top-level instance indices (counted from base) to leave out."""
    inst = None
    for f in range(0, 0x100, 4):
        if t.D + off + f + 8 > len(t.b): break
        tgt, n = t.rel(off + f)
        k = t.items.get(tgt)
        if k and k[0] == 'hkFreeListArrayElement': inst = (tgt, k[1]); break
    if not inst: return np.zeros((0, 3, 3), np.float32), 0
    out, cache = [], {}
    for i in range(inst[1]):
        if skip and base + i in skip: continue
        o = inst[0] + i * 64
        sp, _ = t.rel(o + 48)
        if sp not in t.items: continue
        if sp not in cache:
            r = _shape(t, sp, depth)
            cache[sp] = r[0] if isinstance(r, tuple) else r
        tris = cache[sp]
        if not len(tris): continue
        q = t.f32(o, 4); tr = t.f32(o + 16, 3); sc = t.f32(o + 32, 3)
        if not np.all(np.isfinite(sc)) or not np.any(sc): sc = np.ones(3, np.float32)
        out.append((tris * sc) @ _quat(q).T + tr)
    return np.concatenate(out) if out else np.zeros((0, 3, 3), np.float32), inst[1]


def instance_actors(b):
    """Baked instance index -> placement hash, from the table in the Phive header."""
    hdr = struct.unpack_from('<4I', b, 0x10)
    out = {}
    for o in range(hdr[0], hdr[1] - 0x1F, 0x20):
        first, last, h = struct.unpack_from('<iiQ', b, o)
        if first < 0: continue
        for i in range(first, last + 1): out[i] = h
    return out


def file_compounds(b, skip=None):
    res, base = [], 0
    for o, sz in tag_sections(b):
        try:
            t = Tag(b, o, sz)
        except Exception:
            continue
        if t.root == 'hknpCompoundShape':
            tris, n = _compound(t, 0, skip=skip, base=base)
            res.append(tris); base += n
    return np.concatenate(res) if res else np.zeros((0, 3, 3), np.float32)
