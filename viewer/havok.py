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
    """Terrain mesh plus baked objects, world space.
    Returns (triangles, placement hash, material, submaterial, shape-tag flags) per triangle; hash 0 = plain terrain.
    Baked areas and air walls are left out."""
    b = romfs.dec(open(path, 'rb').read())
    owner = instance_actors(b)
    table, starts = materials(b)
    skip = {i for i, h in owner.items() if NOT_SOLID.search(names.get(h, ''))} if names else None
    tris_out, hash_out, mat_out = [], [], []
    tris, tags = terrain_mesh(b)
    if len(tris):
        ok = np.all((tris > -2) & (tris < TILE + 2), axis=(1, 2))
        mesh = [v for v in starts.values() if v[2] == 2]
        first = max(mesh, key=lambda v: v[1])[0] if mesh else 0
        tris_out.append(tris[ok] + tile_origin(b, key, tiles)); hash_out.append(np.zeros(int(ok.sum()), np.uint64))
        mat_out.append(first + tags[ok])
    comp, idx, tags = file_compounds(b, skip)
    if len(comp):
        ok = np.all(np.isfinite(comp) & (np.abs(comp) < 1e5), axis=(1, 2))
        tris_out.append(comp[ok]); hash_out.append(np.array([owner.get(int(i), 0) for i in idx[ok]], np.uint64))
        first = np.array([starts.get(int(i), (0, 1, 0))[0] for i in idx[ok]], np.int64)
        mat_out.append(first + tags[ok])
    if not tris_out or not len(table):
        z = np.zeros(sum(len(t) for t in tris_out), np.uint64)
        tri = np.concatenate(tris_out) if tris_out else np.zeros((0, 3, 3), np.float32)
        return tri, (np.concatenate(hash_out) if hash_out else z), z.astype(np.uint8), z.astype(np.uint8), z
    mt = table[np.clip(np.concatenate(mat_out), 0, len(table) - 1)]
    return (np.concatenate(tris_out), np.concatenate(hash_out),
            mt['m'].astype(np.uint8), mt['s'].astype(np.uint8), mt['f'].astype(np.uint64))


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
    """Mesh shape at data offset off -> (triangles, shape tag per triangle).
    Tags come from shapeTagTable (+0x60): sorted (key, tag) runs, key = section << 8 | primitive."""
    so, sn = t.rel(off + 0x88)
    to, tn = t.rel(off + 0x60)
    tab = np.frombuffer(t.b, '<u4', tn * 2, t.D + to).reshape(-1, 2) if tn else np.zeros((0, 2), np.uint32)
    out, tags = [], []
    for i in range(sn):
        o = so + i * 0x40
        po, pn = t.rel(o + 8); vo, _ = t.rel(o + 16)
        if not pn: continue
        P = np.frombuffer(t.b, np.uint8, pn * 4, t.D + po).reshape(-1, 4).astype(np.int64)
        nv = int(P.max()) + 1
        V = np.frombuffer(t.b, '<u2', nv * 3, t.D + vo).reshape(-1, 3).astype(np.float32)
        W = (np.array(struct.unpack_from('<3i', t.b, t.D + o + 32), np.float32) + V) / 512.0
        keys = (i << 8) | np.arange(pn)
        if len(tab):
            tg = tab[np.clip(np.searchsorted(tab[:, 0], keys, 'right') - 1, 0, None), 1].astype(np.int64)
        else:
            tg = np.zeros(pn, np.int64)
        out.append(W[P[:, [0, 1, 2]]]); tags.append(tg)
        quad = P[:, 2] != P[:, 3]
        if quad.any(): out.append(W[P[quad][:, [0, 2, 3]]]); tags.append(tg[quad])
    if not out: return np.zeros((0, 3, 3), np.float32), np.zeros(0, np.int64)
    return np.concatenate(out), np.concatenate(tags)


def _shape(t, off, depth=0):
    kind = t.items.get(off, (None,))[0]
    if kind in ('hknpBoxShape', 'hknpConvexShape', 'hknpConvexPolytopeShape'): r = _convex(t, off)
    elif kind == 'hknpCapsuleShape': r = _capsule(t, off)
    elif kind == 'hknpMeshShape': return _mesh_obj(t, off)
    elif kind == 'hknpCompoundShape' and depth < 4:
        tris, _, _, tags = _compound(t, off, depth + 1)
        return tris, tags
    else: r = np.zeros((0, 3, 3), np.float32)
    return r, np.zeros(len(r), np.int64)



def _compound(t, off, depth=0, skip=None, base=0):
    """skip: set of top-level instance indices (counted from base) to leave out."""
    inst = None
    for f in range(0, 0x100, 4):
        if t.D + off + f + 8 > len(t.b): break
        tgt, n = t.rel(off + f)
        k = t.items.get(tgt)
        if k and k[0] == 'hkFreeListArrayElement': inst = (tgt, k[1]); break
    empty = np.zeros(0, np.int64)
    if not inst: return np.zeros((0, 3, 3), np.float32), 0, empty, empty
    out, ids, tags, cache = [], [], [], {}
    for i in range(inst[1]):
        if skip and base + i in skip: continue
        o = inst[0] + i * 64
        sp, _ = t.rel(o + 48)
        if sp not in t.items: continue
        if sp not in cache: cache[sp] = _shape(t, sp, depth)
        tris, tg = cache[sp]
        if not len(tris): continue
        q = t.f32(o, 4); tr = t.f32(o + 16, 3); sc = t.f32(o + 32, 3)
        if not np.all(np.isfinite(sc)) or not np.any(sc): sc = np.ones(3, np.float32)
        out.append((tris * sc) @ _quat(q).T + tr)
        ids.append(np.full(len(tris), base + i, np.int64)); tags.append(tg)
    if not out: return np.zeros((0, 3, 3), np.float32), inst[1], empty, empty
    return np.concatenate(out), inst[1], np.concatenate(ids), np.concatenate(tags)


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
    """Baked objects: (triangles, top-level instance index per triangle, shape tag per triangle)."""
    res, ids, tags, base = [], [], [], 0
    for o, sz in tag_sections(b):
        try:
            t = Tag(b, o, sz)
        except Exception:
            continue
        if t.root == 'hknpCompoundShape':
            tris, n, idx, tg = _compound(t, 0, skip=skip, base=base)
            res.append(tris); ids.append(idx); tags.append(tg); base += n
    z = np.zeros(0, np.int64)
    if not res: return np.zeros((0, 3, 3), np.float32), z, z
    return np.concatenate(res), np.concatenate(ids), np.concatenate(tags)


MAT_DTYPE = np.dtype([('m', '<u4'), ('s', '<u4'), ('f', '<u8')])


def materials(b):
    """Per-file material table (material, submaterial, user shape-tag mask) and, per instance,
    (first table entry, entry count, shape kind: 1 convex, 2 mesh)."""
    hdr = struct.unpack_from('<24I', b, 8)
    table = np.frombuffer(b, MAT_DTYPE, (hdr[7] - hdr[6]) // 16, hdr[6])
    starts = {}
    for o in range(hdr[5], hdr[6] - 19, 20):
        i, flags, _, _, first, count = struct.unpack_from('<IIHHII', b, o)
        starts[i] = (first, count, flags >> 24)
    return table, starts


def terrain_mesh(b):
    """Standalone terrain mesh (tile-local): (triangles, shape tag per triangle)."""
    tris, tags = [], []
    for o, sz in tag_sections(b):
        try:
            t = Tag(b, o, sz)
        except Exception:
            continue
        if t.root == 'hknpMeshShape':
            tr, tg = _mesh_obj(t, 0); tris.append(tr); tags.append(tg)
    if not tris: return np.zeros((0, 3, 3), np.float32), np.zeros(0, np.int64)
    return np.concatenate(tris), np.concatenate(tags)


def instance_records(b):
    """Global instance index -> file offset of its 64-byte hknpShapeInstance record.
    The u32 at +28 is the instance state: flags byte (0x40 = enabled) plus a 24-bit tree index."""
    out, base = {}, 0
    for o, sz in tag_sections(b):
        try:
            t = Tag(b, o, sz)
        except Exception:
            continue
        if t.root != 'hknpCompoundShape': continue
        inst = next(((off, k[1]) for off, k in t.items.items() if k[0] == 'hkFreeListArrayElement'), None)
        if not inst: continue
        for i in range(inst[1]): out[base + i] = t.D + inst[0] + i * 64
        base += inst[1]
    return out


def material_prims(b, instances, material=27):
    """File offsets of the 4-byte mesh primitives (quads) whose material is `material` (27 = AirWall),
    inside the given top-level baked instances. Pointing all four indices at one vertex removes the face
    without growing any bounding box."""
    table, starts = materials(b)
    out, base = [], 0
    for o, sz in tag_sections(b):
        try:
            t = Tag(b, o, sz)
        except Exception:
            continue
        if t.root != 'hknpCompoundShape': continue
        inst = next(((off, k[1]) for off, k in t.items.items() if k[0] == 'hkFreeListArrayElement'), None)
        if not inst: continue
        for i in range(inst[1]):
            g = base + i
            if g not in instances or g not in starts: continue
            sp, _ = t.rel(inst[0] + i * 64 + 48)
            if t.items.get(sp, (None,))[0] != 'hknpMeshShape': continue
            first = starts[g][0]
            so, sn = t.rel(sp + 0x88)
            to, tn = t.rel(sp + 0x60)
            tab = np.frombuffer(t.b, '<u4', tn * 2, t.D + to).reshape(-1, 2) if tn else np.zeros((0, 2), np.uint32)
            for s in range(sn):
                po, pn = t.rel(so + s * 0x40 + 8)
                if not pn: continue
                keys = (s << 8) | np.arange(pn)
                tg = tab[np.clip(np.searchsorted(tab[:, 0], keys, 'right') - 1, 0, None), 1].astype(np.int64) if len(tab) else np.zeros(pn, np.int64)
                idx = np.clip(first + tg, 0, len(table) - 1)
                for k in np.where(table['m'][idx] == material)[0]:
                    out.append(t.D + po + int(k) * 4)
        base += inst[1]
    return out


def terrain_material_prims(b, key, tiles, center, radius, material=27):
    """File offsets of the plain terrain mesh's quads with `material` whose centre lies within radius of center (world)."""
    table, starts = materials(b)
    mesh = [v for v in starts.values() if v[2] == 2]
    first = max(mesh, key=lambda v: v[1])[0] if mesh else 0
    origin = tile_origin(b, key, tiles)
    c = np.asarray(center, np.float32)
    out = []
    for o, sz in tag_sections(b):
        try:
            t = Tag(b, o, sz)
        except Exception:
            continue
        if t.root != 'hknpMeshShape': continue
        so, sn = t.rel(0x88)
        to, tn = t.rel(0x60)
        tab = np.frombuffer(t.b, '<u4', tn * 2, t.D + to).reshape(-1, 2) if tn else np.zeros((0, 2), np.uint32)
        for s in range(sn):
            sec = so + s * 0x40
            po, pn = t.rel(sec + 8); vo, _ = t.rel(sec + 16)
            if not pn: continue
            P = np.frombuffer(t.b, np.uint8, pn * 4, t.D + po).reshape(-1, 4).astype(np.int64)
            V = np.frombuffer(t.b, '<u2', (int(P.max()) + 1) * 3, t.D + vo).reshape(-1, 3).astype(np.float32)
            W = (np.array(struct.unpack_from('<3i', t.b, t.D + sec + 32), np.float32) + V) / 512.0 + origin
            keys = (s << 8) | np.arange(pn)
            tg = tab[np.clip(np.searchsorted(tab[:, 0], keys, 'right') - 1, 0, None), 1].astype(np.int64) if len(tab) else np.zeros(pn, np.int64)
            mat = table['m'][np.clip(first + tg, 0, len(table) - 1)]
            ctr = W[P].mean(1)
            for k in np.where((mat == material) & (np.linalg.norm(ctr - c, axis=1) < radius))[0]:
                out.append(t.D + po + int(k) * 4)
    return out
