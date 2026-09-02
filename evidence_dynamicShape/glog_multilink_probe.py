import ctypes
import platform  # noqa

import paddle  # noqa

SYMS = {
    "libpaddle.so": 0x0E825B60,
    "libphi_core.so": 0x14C25940,
    "libphi_gpu.so": 0x0D629500,
    "libcinnapi.so": 0x0639B980,
}

bases = {}
with open("/proc/self/maps") as f:
    for line in f:
        parts = line.split()
        if len(parts) < 6:
            continue
        path = parts[-1]
        for name in SYMS:
            if path.endswith("/" + name) and name not in bases:
                bases[name] = int(parts[0].split("-")[0], 16)

for name, off in SYMS.items():
    if name not in bases:
        print(f"{name}: NOT LOADED")
        continue
    addr = bases[name] + off
    raw = ctypes.string_at(addr, 16)
    ptr = int.from_bytes(raw[:8], "little")
    size = int.from_bytes(raw[8:16], "little")
    content = b""
    if ptr and size < 4096:
        try:
            content = ctypes.string_at(ptr, size)
        except Exception as e:  # noqa
            content = b"<unreadable>"
    print(
        f"{name}: &buf=0x{addr:x} _M_p=0x{ptr:x} size={size} data={content!r}"
    )
print("DONE", flush=True)
