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

    def move(self, actor, model, pos):
        """Write a new world position into the model matrix (both copies) and the actor."""
        for k, v in zip((0xC, 0x1C, 0x2C), pos):
            b = struct.pack('<f', v)
            if model: self.write(model + 0xA0 + k, b); self.write(model + 0xD0 + k, b)
        if actor: self.write(actor + 0x31C, struct.pack('<3f', *pos))

    def model_pos(self, model):
        t = self.read(self.addr['gb'] + model + 0xA0, '<12f')
        return (t[3], t[7], t[11]) if t else None

    def write_u32(self, guest, value):
        if not self.write(guest, struct.pack('<I', value & 0xFFFFFFFF)):
            ce('writeInteger(%d, %d) return 1' % (self.addr['gb'] + guest, value))

    def state(self):
        self.refresh()
        p = self.read(self.addr.get('linkpos'), '<3d')
        if not p: return None
        o = self.addr.get('linkobj')
        f = self.read(o + 0x328, '<3f') if o else None
        return {'x': p[0], 'y': p[1] - 0.585, 'z': p[2], 'face': (f[2], f[0]) if f else None, 'cam': self.camera()}

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
