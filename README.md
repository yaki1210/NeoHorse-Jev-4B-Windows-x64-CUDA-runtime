# NeoHorse Windows Runtime (precompiled)

Prebuilt **Windows x64 + CUDA** native libraries and Python loader for [NeoHorse-Jev-4B-GGUF](https://huggingface.co/TokenRhythm/NeoHorse-Jev-4B-GGUF).

This repo ships the **inference runtime only**. Download GGUF model weights separately from Hugging Face (do not commit `.gguf` files here).

Upstream model / GGUF docs: https://huggingface.co/TokenRhythm/NeoHorse-Jev-4B-GGUF  
Upstream project: https://github.com/TokenRhythm/NeoHorse

## What this is

| Piece | Role |
| --- | --- |
| `runtime/nh_hidden.dll`, `runtime/nh_vision.dll` | NeoHorse decision bridges (text / vision) |
| `runtime/llama.dll`, `mtmd.dll`, `ggml*.dll` | Pinned llama.cpp / ggml / mtmd shared libs |
| `runtime/cudart64_12.dll` | CUDA 12 runtime redistributable (small) |
| `runtime/runtime.py` | `NeoHorseGGUF` Python API |
| `runtime/support/` | Decision encoding helpers (+ vendored Kev attribution) |
| `runtime/gguf-py/` | GGUF reader used by the loader |
| `rebuild/` | Optional MSVC+CUDA rebuild scripts (not needed to run) |

**Not included:** GGUF weights (~2.6–9.1 GB), mousemove sandbox / aim scripts, logs, build trees.

## Requirements

- Windows 10/11 x64
- NVIDIA GPU with a recent driver (this build used **CUDA toolkit 12.3**, arch **sm_89**)
- Python 3.12+ recommended
- Visual C++ Redistributable (usually already present)

## Install

```powershell
# 1) Clone this runtime repo
cd E:\project
git clone <THIS_REPO_URL> neohorse-windows-runtime
cd neohorse-windows-runtime

# 2) Python deps
python -m venv .venv
.\.venv\Scripts\activate
pip install -r runtime\requirements.txt

# 3) Download ONE GGUF (example: Q4_K_M ~3.39 GB decimal / ~3.16 GiB on disk)
pip install huggingface_hub
hf download TokenRhythm/NeoHorse-Jev-4B-GGUF NeoHorse-Jev-4B-Q4_K_M.gguf --local-dir .
# Or copy from a local cache, e.g. E:\models\NeoHorse-Jev-4B-Q4_K_M.gguf
```

### Git LFS (important)

`runtime/ggml-cuda.dll` is **~183 MB** and is tracked with **Git LFS** (only that file; other DLLs stay as normal git blobs).

- **Clone with LFS:** install [Git LFS](https://git-lfs.com), then `git lfs install` once, then clone as usual. After clone, `git lfs ls-files` should list `runtime/ggml-cuda.dll`.
- **If you clone without LFS:** that path may be a tiny pointer file instead of the real DLL. Either install LFS and run `git lfs pull`, or download the DLL from a GitHub **Release** asset / zip (recommended fallback).

> Prefer Git LFS for the repo, or attach a Release zip containing `ggml-cuda.dll`, rather than committing it as a plain git blob.

## Quick run

From the repo root (directory that contains `runtime/`):

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
python runtime\example.py --model NeoHorse-Jev-4B-Q4_K_M.gguf
# optional vision:
# python runtime\example.py --model NeoHorse-Jev-4B-Q4_K_M.gguf --image example.png
```

Python API:

```python
import sys
sys.path.insert(0, "runtime")
from runtime import NeoHorseGGUF

model = NeoHorseGGUF("NeoHorse-Jev-4B-Q4_K_M.gguf")
try:
    result = model.predict({
        "state": "Refund policy: a receipt is required. The customer has no receipt.",
        "questions": {
            "refund": {
                "type": "noul",
                "instructions": "Is a refund allowed under the policy?"
            }
        }
    })
    print(result["answers"])
finally:
    model.close()
```

## Build info (this prebuild)

- Built on: 2026-09-27 ~02:20 (Asia/Shanghai)
- Script: `neohorse-jev-gguf/run_build_windows.bat` → `runtime/build_windows.py --cuda-arch 89 --jobs 8`
- llama.cpp pin: `94256114c229674ef96e76eb2dea596e65b43818`
- CUDA toolkit: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3`
- Result: `EXIT=0` (DLLs copied next to `nh_*.dll`)

To rebuild yourself (MSVC + CUDA + Ninja required), see `rebuild/` and adapt paths in `run_build_windows.bat`.

## License / attribution

- NeoHorse package: Apache-2.0 (`LICENSE`)
- llama.cpp / ggml: MIT (`LICENSE.llama.cpp`)
- Vendored decision helpers: see `runtime/support/neohorse_decision/_vendor/NOTICE.md` (Kev, Apache-2.0)

## Layout

```
runtime/           # drop-in GGUF runtime (binaries + Python)
rebuild/           # optional Windows rebuild sources/scripts
LICENSE*
README.md
RELEASE_NOTES.md
```

Place any downloaded `*.gguf` beside this repo (or pass an absolute path to `--model`).