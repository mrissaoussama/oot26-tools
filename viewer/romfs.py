import json, os, struct
import zstandard

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, 'config.json')
ROOT = None
_DICTS = {}


def load_config():
    global ROOT
    if os.path.exists(CONFIG):
        ROOT = json.load(open(CONFIG)).get('romfs')
    if not (ROOT and os.path.isdir(os.path.join(ROOT, 'Banc'))): ROOT = None
    return ROOT


def save_config(path):
    global ROOT
    ROOT = path
    json.dump({'romfs': path}, open(CONFIG, 'w'))


def path(*parts):
    return os.path.join(ROOT, *parts)


def sarc(buf):
    """SARC archive -> {name: bytes}."""
    data_off = struct.unpack_from('<I', buf, 0xC)[0]
    n = struct.unpack_from('<H', buf, 0x1A)[0]
    names_off = 0x20 + n * 16 + 8
    out = {}
    for i in range(n):
        _, attr, s, e = struct.unpack_from('<IIII', buf, 0x20 + i * 16)
        name = str(i)
        if attr >> 24:
            no = names_off + (attr & 0xFFFF) * 4
            name = buf[no:buf.index(b'\0', no)].decode()
        out[name] = buf[data_off + s:data_off + e]
    return out


def dec(buf):
    if buf[:4] != b'\x28\xb5\x2f\xfd': return buf
    if not _DICTS:
        raw = zstandard.ZstdDecompressor().decompress(open(path('Pack', 'ZsDic.pack.zs'), 'rb').read(), max_output_size=1 << 28)
        for blob in sarc(raw).values():
            d = zstandard.ZstdCompressionDict(blob)
            _DICTS[d.dict_id()] = d
    did = zstandard.get_frame_parameters(buf).dict_id
    z = zstandard.ZstdDecompressor(dict_data=_DICTS[did]) if did else zstandard.ZstdDecompressor()
    return z.decompress(buf, max_output_size=1 << 30)


def read(*parts):
    return dec(open(path(*parts), 'rb').read())
