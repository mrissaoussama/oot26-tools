"""Phive / Havok tagfile mesh decoding (hknpMeshShape) and zone collision tiles."""
import os, re, struct
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
    """'Graveyard_Child_Static' -> {(x, z): tile path}."""
    prefix = re.sub(r'_(Static|Dynamic)$', '', scene)
    base = romfs.path('Phive', 'StaticCompoundBody')
    out = {}
    for d in os.listdir(base):
        if not prefix.startswith(d): continue
        for f in os.listdir(os.path.join(base, d)):
            m = re.match(re.escape(prefix) + r'_X(\d+)_Z(\d+)\.', f)
            if m: out[(int(m.group(1)), int(m.group(2)))] = os.path.join(base, d, f)
    return out


def tile_origin(tiles, key):
    nx = max(k[0] for k in tiles) + 1
    nz = max(k[1] for k in tiles) + 1
    return np.array([(key[0] - nx / 2) * TILE, 0, (key[1] - nz / 2) * TILE], np.float32)


def load_tile(path, origin):
    tris = file_meshes(romfs.dec(open(path, 'rb').read()))
    if not len(tris): return tris
    ok = np.all((tris > -2) & (tris < TILE + 2), axis=(1, 2))
    return tris[ok] + origin
