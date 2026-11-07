"""Live game state via Cheat Engine's cepipe (addresses) and ReadProcessMemory (values)."""
import ctypes, ctypes.wintypes as wt, math, struct, time
import numpy as np

k32 = ctypes.WinDLL('kernel32', use_last_error=True)
k32.OpenProcess.restype = wt.HANDLE
k32.ReadProcessMemory.argtypes = [wt.HANDLE, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
k32.VirtualQueryEx.argtypes = [wt.HANDLE, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
k32.WriteProcessMemory.argtypes = [wt.HANDLE, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]

CAM_PTR = 0x4E98068  # [[main + CAM_PTR] + 0x68] = game camera


def ce(lua):
    """Runs Lua in Cheat Engine through the cepipe server and returns its integer result."""
    for attempt in range(20):
        try:
            with open(r'\\.\pipe\cepipe', 'r+b', buffering=0) as p:
                b = lua.encode()
                p.write(struct.pack('<BI', 1, len(b)) + b + struct.pack('<Q', 0))
                return struct.unpack('<Q', p.read(8))[0]
        except OSError:
            if attempt == 19: raise
            time.sleep(0.1)


def murmur3(data, seed=0):
    c1, c2, h, n = 0xcc9e2d51, 0x1b873593, seed, len(data)
    for i in range(0, n // 4 * 4, 4):
        k = (struct.unpack_from('<I', data, i)[0] * c1) & 0xffffffff
        k = (((k << 15) | (k >> 17)) * c2) & 0xffffffff
        h ^= k; h = ((((h << 13) | (h >> 19)) & 0xffffffff) * 5 + 0xe6546b64) & 0xffffffff
    k = 0
    for i, b in enumerate(data[n // 4 * 4:]): k |= b << (8 * i)
    if n % 4:
        k = (k * c1) & 0xffffffff; k = (((k << 15) | (k >> 17)) * c2) & 0xffffffff; h ^= k
    h ^= n; h ^= h >> 16; h = (h * 0x85ebca6b) & 0xffffffff; h ^= h >> 13; h = (h * 0xc2b2ae35) & 0xffffffff
    return h ^ (h >> 16)


class MBI(ctypes.Structure):
    _fields_ = [('BaseAddress', ctypes.c_uint64), ('AllocationBase', ctypes.c_uint64), ('AllocationProtect', wt.DWORD),
                ('PartitionId', wt.WORD), ('RegionSize', ctypes.c_uint64), ('State', wt.DWORD), ('Protect', wt.DWORD), ('Type', wt.DWORD)]


class Game:
    def __init__(self):
        self.h, self.pid, self.addr, self.last = None, None, {}, 0
        self.error = ''

    def refresh(self):
        if time.time() - self.last < 2: return
        self.last = time.time()
        try:
            pid = ce('return getOpenedProcessID()')
            if pid != self.pid:
                self.pid, self.h = pid, k32.OpenProcess(0x0438, False, pid)
            for k in ('linkpos', 'linkobj', 'gb', 'M'):
                self.addr[k] = ce('return LT and LT.%s or 0' % k)
            self.error = '' if self.addr.get('linkpos') else 'CE is connected, but Link tools has no position yet'
        except OSError:
            self.error = 'cepipe not found: enable "Link tools" in Cheat Engine'

    def raw(self, a, n):
        buf = ctypes.create_string_buffer(n)
        got = ctypes.c_size_t()
        if not a or not self.h or not k32.ReadProcessMemory(self.h, a, buf, n, ctypes.byref(got)): return None
        return buf.raw[:got.value]

    def read(self, a, fmt):
        b = self.raw(a, struct.calcsize(fmt))
        return struct.unpack(fmt, b) if b and len(b) == struct.calcsize(fmt) else None

    def camera(self):
        gb, M = self.addr.get('gb'), self.addr.get('M')
        p = gb and self.read(gb + M + CAM_PTR, '<Q')
        c = p and self.read(gb + p[0] + 0x68, '<Q')
        b = c and self.raw(gb + c[0], 0x148)
        if not b or len(b) < 0x148: return None
        return {'view': np.array(struct.unpack_from('<12f', b, 8), np.float32).reshape(3, 4),
                'fovy': struct.unpack_from('<f', b, 0x130)[0]}

    def actor_of(self, rec):
        """Actor object behind a placement record (rec = address of its placement hash), or None."""
        gb, M = self.addr.get('gb'), self.addr.get('M')
        p = self.read(gb + rec - 0x28, '<Q')
        if not p or not 0x2000000000 <= p[0] < 0x2400000000 or p[0] % 8: return None
        v = self.read(gb + p[0], '<Q')
        return p[0] if v and M <= v[0] < M + 0x6000000 else None

    def actor_rot(self, actor):
        """3x3 rotation (row-major at +0x328), or None if it does not look like a rotation."""
        r = self.read(self.addr.get('gb') + actor + 0x328, '<9f')
        if not r: return None
        R = np.array(r, np.float32).reshape(3, 3)
        return R if np.all(np.isfinite(R)) and abs(abs(np.linalg.det(R)) - 1) < 0.05 else None

    def actor_pos(self, actor):
        p = self.read(self.addr.get('gb') + actor + 0x31C, '<3f')
        return p if p and all(math.isfinite(v) and abs(v) < 1e5 for v in p) else None

    def actors(self, lo=0x2140000000, hi=0x2160000000):
        """All live actors: {actor address: (name, placement hash)}. Every actor shares the base vtable at main+4B25E18."""
        gb, M = self.addr.get('gb'), self.addr.get('M')
        vt = np.uint64(M + 0x4B25E18)
        out, a, m = {}, gb + lo, MBI()
        while a < gb + hi and k32.VirtualQueryEx(self.h, a, ctypes.byref(m), ctypes.sizeof(m)):
            if m.State == 0x1000 and m.Protect in (0x04, 0x40):
                for off in range(0, m.RegionSize, 1 << 25):
                    buf = self.raw(m.BaseAddress + off, min(1 << 25, m.RegionSize - off))
                    if not buf: continue
                    arr = np.frombuffer(buf[:len(buf) // 8 * 8], np.uint64)
                    for i in np.where(arr == vt)[0]:
                        act = m.BaseAddress + off + int(i) * 8 - gb
                        np_ = self.read(gb + act + 0x20, '<Q')
                        name = self.raw(gb + np_[0], 64) if np_ and 0x2000000000 <= np_[0] < 0x2400000000 else None
                        name = name.split(b'\0')[0].decode('latin1', 'replace') if name else ''
                        rec = self.read(gb + act + 0x3D8, '<Q')
                        h = self.read(gb + rec[0] + 0x88, '<Q') if rec and 0x2000000000 <= rec[0] < 0x2400000000 else None
                        out[act] = (name, h[0] if h else 0)
            a = m.BaseAddress + m.RegionSize
        return out

    def find_bytes(self, pat, lo=0x1000000000, hi=0x2000000000):
        """Guest address of the first occurrence of pat in [lo, hi)."""
        gb = self.addr.get('gb')
        a, m = gb + lo, MBI()
        while a < gb + hi and k32.VirtualQueryEx(self.h, a, ctypes.byref(m), ctypes.sizeof(m)):
            if m.State == 0x1000 and m.Protect in (0x04, 0x40):
                for off in range(0, m.RegionSize, 1 << 26):
                    buf = self.raw(m.BaseAddress + off, min(1 << 26, m.RegionSize - off))
                    j = buf.find(pat) if buf else -1
                    if j >= 0: return m.BaseAddress + off + j - gb
            a = m.BaseAddress + m.RegionSize
        return None

    def models(self, lo=0x2140000000, hi=0x2160000000):
        """Model transform objects: [(address, (x, y, z))]. Vtable main+4C542F0, self pointer at +0x98,
        3x4 row-major world matrix at +0xA0 (a second copy at +0xD0). Writing it moves the model and its collision."""
        gb, M = self.addr.get('gb'), self.addr.get('M')
        vt = np.uint64(M + 0x4C542F0)
        out, a, m = [], gb + lo, MBI()
        while a < gb + hi and k32.VirtualQueryEx(self.h, a, ctypes.byref(m), ctypes.sizeof(m)):
            if m.State == 0x1000 and m.Protect in (0x04, 0x40):
                for off in range(0, m.RegionSize, 1 << 25):
                    buf = self.raw(m.BaseAddress + off, min(1 << 25, m.RegionSize - off))
                    if not buf: continue
                    arr = np.frombuffer(buf[:len(buf) // 8 * 8], np.uint64)
                    for i in np.where(arr == vt)[0]:
                        o = int(i) * 8
                        addr = m.BaseAddress + off + o - gb
                        if o + 0x100 > len(buf) or struct.unpack_from('<Q', buf, o + 0x98)[0] != addr: continue
                        t = struct.unpack_from('<12f', buf, o + 0xA0)
                        out.append((addr, (t[3], t[7], t[11])))
            a = m.BaseAddress + m.RegionSize
        return out

    def write(self, guest, data):
        got = ctypes.c_size_t()
        return bool(self.h and k32.WriteProcessMemory(self.h, self.addr['gb'] + guest, data, len(data), ctypes.byref(got)))

    def move(self, actor, model, pos, mpos=None):
        """Write the actor position and (both copies of) its model matrix translation."""
        if model and mpos:
            for k, v in zip((0xC, 0x1C, 0x2C), mpos):
                b = struct.pack('<f', v)
                self.write(model + 0xA0 + k, b); self.write(model + 0xD0 + k, b)
        if actor: self.write(actor + 0x31C, struct.pack('<3f', *pos))

    def model_pos(self, model):
        t = self.read(self.addr['gb'] + model + 0xA0, '<12f')
        return (t[3], t[7], t[11]) if t else None

    def teleport(self, x, y, z):
        """Moves Link through Link tools, asking it to re-find his position first if it lost it."""
        if not self.addr.get('linkpos'):
            ce('ltFindPos() return 1')
            self.last = 0; self.refresh()
            if not self.addr.get('linkpos'): return False
        b = self.addr['linkpos']
        cur = self.read(b, '<3d')
        if not cur: return False
        d = (x - cur[0], y + 0.585 - cur[1], z - cur[2])
        # Link tools keeps every copy of the record in LINKPOS_ALL; shift them too, then the driving one.
        ce('for _,a in ipairs(LINKPOS_ALL or {}) do if a ~= %d then writeDouble(a, readDouble(a)+%r) writeDouble(a+8, readDouble(a+8)+%r) '
           'writeDouble(a+16, readDouble(a+16)+%r) end end return 1' % (b, *d))
        k32.WriteProcessMemory(self.h, b, struct.pack('<3d', x, y + 0.585, z), 24, None)
        return True

    def find_bodies(self, positions, lo=0x2160000000, hi=0x2170000000):
        """Physics bodies (hknpBody, stride 0xC0) placed at the given positions: {index: [body addresses]}.
        A body starts with a 3x4 rotation; its translation is at +0x30 and its AABB floats at +0x60..+0x78."""
        gb = self.addr['gb']
        P = np.array(positions, np.float32)
        out, a, m = {}, gb + lo, MBI()
        while a < gb + hi and k32.VirtualQueryEx(self.h, a, ctypes.byref(m), ctypes.sizeof(m)):
            if m.State == 0x1000 and m.Protect in (0x04, 0x40):
                for off in range(0, m.RegionSize, 1 << 26):
                    b = self.raw(m.BaseAddress + off, min(1 << 26, m.RegionSize - off))
                    if not b: continue
                    f = np.frombuffer(b[:len(b) // 4 * 4], np.float32)
                    with np.errstate(invalid='ignore'):
                        cand = np.where(np.isin(np.round(f[12:-2], 2), np.round(P[:, 0], 2)))[0]
                    for j in cand:
                        for i, p in enumerate(P):
                            if abs(f[j + 12] - p[0]) < 0.01 and abs(f[j + 13] - p[1]) < 0.01 and abs(f[j + 14] - p[2]) < 0.01:
                                R = f[j:j + 12].reshape(3, 4)[:, :3]
                                if abs(abs(np.linalg.det(R)) - 1) < 0.02:
                                    out.setdefault(i, []).append(m.BaseAddress + off + int(j) * 4 - gb)
            a = m.BaseAddress + m.RegionSize
        return out

    def shift_body(self, body, dx, dy, dz, raw=None):
        """Moves a body's translation and AABB by (dx, dy, dz)."""
        raw = raw or self.raw(self.addr['gb'] + body, 0x80)
        f = list(struct.unpack('<32f', raw))
        for k, d in ((12, dx), (13, dy), (14, dz), (24, dx), (25, dy), (26, dz), (28, dx), (29, dy)):
            f[k] += d
        f[27] += dz
        self.write(body, struct.pack('<32f', *f))

    SEQ_FIELDS = {'Sequence_CurrentBanc': 'cur', 'Sequence_ReservedBootEventName': 'event',
                  'Sequence_ReservedEventStarterName': 'starter', 'Sequence_ReservedNextBanc': 'banc',
                  'Sequence_ReservedNextStartPosName': 'pos'}

    def seq_slots(self):
        """Addresses of the GameData String64 values that hold the reserved map change (cached).
        Each String64 entry is three sead::FixedSafeString<64> (0x68/0x50/0x50 apart); the second is the live value."""
        gb = self.addr['gb']
        c = getattr(self, '_seq', None)
        if c and (self.raw(gb + c['cur'], 4) or b'\0')[:1] not in (b'\0',): return c
        want = {murmur3(n.encode()): k for n, k in self.SEQ_FIELDS.items()}
        keys = np.array(list(want), np.uint32)
        idx, table, a, m = {}, None, gb + 0x2180000000, MBI()
        while a < gb + 0x2190000000 and len(idx) < len(want) and k32.VirtualQueryEx(self.h, a, ctypes.byref(m), ctypes.sizeof(m)):
            if m.State == 0x1000 and m.Protect in (0x04, 0x40):
                buf = self.raw(m.BaseAddress, m.RegionSize)
                if buf:
                    u = np.frombuffer(buf[:len(buf) // 4 * 4], np.uint32)
                    for j in np.where(np.isin(u[:-1], keys))[0]:
                        if u[j + 1] < 0x400 and (j * 4) % 8 == 0:
                            idx.setdefault(want[int(u[j])], int(u[j + 1])); table = (m.BaseAddress, m.RegionSize)
            a = m.BaseAddress + m.RegionSize
        if len(idx) < len(want) or table is None: return None
        base, size = table
        buf = self.raw(base, size) or b''
        objs = [base + o for o in range(0, len(buf) - 16, 4) if struct.unpack_from('<QI', buf, o) == (base + o + 0xC - gb, 0x40)]
        for start in range(len(objs)):
            if start + 3 * max(idx.values()) + 2 >= len(objs): break
            cur = objs[start + 3 * idx['cur'] + 1]
            if objs[start + 1] - objs[start] == 0x50 and (self.raw(cur + 0xC, 1) or b'\0') != b'\0':
                if all(objs[start + 3 * k + 1] - objs[start + 3 * k] == 0x50 for k in idx.values()):
                    self._seq = {k: objs[start + 3 * i + 1] + 0xC - gb for k, i in idx.items()}
                    return self._seq
        return None

    def seq_get(self, k):
        c = self.seq_slots()
        return c and self.raw(self.addr['gb'] + c[k], 0x40).split(b'\0')[0].decode('latin1')

    def seq_put(self, k, text):
        c = self.seq_slots()
        if c: self.write(c[k], text.encode()[:0x3F].ljust(0x40, b'\0'))

    def drop_slots(self, lo=0x2140000000, hi=0x2160000000):
        """{guest address: (drop table, drop actor)} for every live Drop blackboard (pots, grass, rocks, chests...).
        Each is a 16-byte header, three pointers, then three String128 objects {ptr -> buf, cap 0x80, buf[0x80]}."""
        gb = self.addr.get('gb')
        hdr = b'\x04\0\0\0\x04\0\0\0\x04\0\0\0\0\0\0\0'
        out, a, m = {}, gb + lo, MBI()
        while a < gb + hi and k32.VirtualQueryEx(self.h, a, ctypes.byref(m), ctypes.sizeof(m)):
            if m.State == 0x1000 and m.Protect in (0x04, 0x40):
                for off in range(0, m.RegionSize, 1 << 25):
                    base = m.BaseAddress + off
                    buf = self.raw(base, min((1 << 25) + 0x200, m.RegionSize - off))
                    j = buf.find(hdr) if buf else -1
                    while j != -1 and j < 1 << 25:
                        x = base + j + 0x30 - gb
                        b = buf[j + 0x30:j + 0x30 + 0x1B0]
                        if len(b) == 0x1B0 and all(struct.unpack_from('<QI', b, k * 0x90) == (x + k * 0x90 + 0x10, 0x80) for k in range(3)):
                            table, actor = (b[k * 0x90 + 0x10:k * 0x90 + 0x90].split(b'\0')[0] for k in range(2))
                            if not table or table.startswith((b'Drop', b'None')): out[x] = (table.decode('latin1'), actor.decode('latin1'))
                        j = buf.find(hdr, j + 1)
            a = m.BaseAddress + m.RegionSize
        return out

    def set_drop(self, slot, actor):
        self.write(slot + 0xA0, actor.encode()[:0x7F].ljust(0x80, b'\0'))

    def write_u32(self, guest, value):
        if not self.write(guest, struct.pack('<I', value & 0xFFFFFFFF)):
            ce('writeInteger(%d, %d) return 1' % (self.addr['gb'] + guest, value))

    def player(self):
        """Player actor (guest address), found by name; cached while it still reads as Player."""
        gb = self.addr.get('gb')
        a = getattr(self, '_player', None)
        if a:
            np_ = self.read(gb + a + 0x20, '<Q')
            name = self.raw(gb + np_[0], 7) if np_ and 0x2000000000 <= np_[0] < 0x2400000000 else None
            if name == b'Player\0': return a
        if time.time() - getattr(self, '_player_scan', 0) < 3: return None
        self._player_scan = time.time()
        self._player = next((a for a, (n, h) in self.actors().items() if n == 'Player'), None)
        return self._player

    def state(self):
        self.refresh()
        p = self.read(self.addr.get('linkpos'), '<3d') if self.addr.get('linkpos') else None
        if p:
            p = (p[0], p[1] - 0.585, p[2])
        elif self.addr.get('gb'):
            a = self.player()
            q = a and self.actor_pos(a)
            if not q: return None
            p = q
        else:
            return None
        o = self.addr.get('linkobj')
        f = self.read(o + 0x328, '<3f') if o else None
        return {'x': p[0], 'y': p[1], 'z': p[2], 'face': (f[2], f[0]) if f else None, 'cam': self.camera()}

    def scan_u64(self, keys, lo, hi, progress=None):
        """{key: [guest addresses]} for keys (sorted np.uint64) found in guest memory [lo, hi)."""
        gb = self.addr.get('gb')
        found, a, m = {}, gb + lo, MBI()
        while a < gb + hi and k32.VirtualQueryEx(self.h, a, ctypes.byref(m), ctypes.sizeof(m)):
            if m.State == 0x1000 and m.Protect in (0x04, 0x40):
                for off in range(0, m.RegionSize, 1 << 25):
                    b = self.raw(m.BaseAddress + off, min(1 << 25, m.RegionSize - off))
                    if b:
                        arr = np.frombuffer(b[:len(b) // 8 * 8], np.uint64)
                        for i in np.where(np.isin(arr, keys))[0]:
                            found.setdefault(int(arr[i]), []).append(m.BaseAddress + off + int(i) * 8 - gb)
            a = m.BaseAddress + m.RegionSize
            if progress: progress(min(1, (a - gb - lo) / (hi - lo)))
        return found
