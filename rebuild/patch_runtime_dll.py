"""Patch runtime.py in-place to load nh_*.dll on Windows instead of libnh_*.so."""
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
old = "self.textlib=ct.CDLL(str(HERE/'libnh_hidden.so'));self.vislib=ct.CDLL(str(HERE/'libnh_vision.so'))"
new = """# Windows builds export nh_hidden.dll / nh_vision.dll; Linux keeps libnh_*.so.
        import os as _os
        _dll_dirs = [HERE, HERE/'build'/'bin']
        for _d in _dll_dirs:
            if _d.is_dir():
                try: _os.add_dll_directory(str(_d))
                except (OSError, AttributeError): pass
                _os.environ['PATH'] = str(_d) + _os.pathsep + _os.environ.get('PATH','')
        def _nh_load(stem):
            for name in (f'nh_{stem}.dll', f'libnh_{stem}.so'):
                p = HERE/name
                if p.is_file():
                    return ct.CDLL(str(p))
            raise FileNotFoundError(f'NeoHorse native library for {stem} not found under {HERE}')
        self.textlib=_nh_load('hidden');self.vislib=_nh_load('vision')"""
if old not in text:
    if "nh_hidden.dll" in text or "_nh_load" in text:
        print("already patched")
        raise SystemExit(0)
    raise SystemExit("needle not found in runtime.py")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("patched", path)
