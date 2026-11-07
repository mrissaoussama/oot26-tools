import struct


class Byml:
    def __init__(self, b):
        self.b = b
        self.e = '<' if b[:2] == b'YB' else '>'
        _, self.ver, keys, strs, root = self.u('HHIII', 0)
        self.keys = self.strtab(keys) if keys else []
        self.strs = self.strtab(strs) if strs else []
        self.root = self.node(self.b[root], root) if root else None

    def u(self, f, o):
        return struct.unpack_from(self.e + f, self.b, o)

    def u24(self, o):
        x = self.b[o:o + 3]
        return int.from_bytes(x, 'little' if self.e == '<' else 'big')

    def strtab(self, o):
        n = self.u24(o + 1)
        offs = self.u('%dI' % (n + 1), o + 4)
        return [self.b[o + offs[i]:o + offs[i + 1]].split(b'\0')[0].decode('utf8', 'replace') for i in range(n)]

    def node(self, t, v):
        if t == 0xC1:
            n = self.u24(v + 1)
            d = {}
            for i in range(n):
                p = v + 4 + i * 8
                k = self.u24(p)
                tt = self.b[p + 3]
                d[self.keys[k]] = self.value(tt, self.u('I', p + 4)[0])
            return d
        if t == 0xC0:
            n = self.u24(v + 1)
            types = self.b[v + 4:v + 4 + n]
            vo = v + 4 + ((n + 3) & ~3)
            return [self.value(types[i], self.u('I', vo + i * 4)[0]) for i in range(n)]
        if t in (0x20, 0x21):
            n = self.u24(v + 1)
            ks = 4 if t == 0x20 else 8
            es = ks + 4
            types = v + 4 + n * es
            d = {}
            for i in range(n):
                p = v + 4 + i * es
                k = self.u('I' if ks == 4 else 'Q', p)[0]
                d[k] = self.value(self.b[types + i], self.u('I', p + ks)[0])
            return d
        raise ValueError('node type %x' % t)

    def value(self, t, x):
        if t in (0xC0, 0xC1, 0x20, 0x21): return self.node(t, x)
        if t == 0xA0: return self.strs[x]
        if t == 0xD0: return bool(x)
        if t == 0xD1: return struct.unpack('i', struct.pack('I', x))[0]
        if t == 0xD2: return struct.unpack('f', struct.pack('I', x))[0]
        if t == 0xD3: return x
        if t == 0xD4: return self.u('q', x)[0]
        if t == 0xD5: return self.u('Q', x)[0]
        if t == 0xD6: return self.u('d', x)[0]
        if t == 0xA1: return ('bin', x)
        if t == 0xFF: return None
        return ('?%x' % t, x)
