# Release notes — NeoHorse Windows Runtime

## v0.1.0 (suggested first tag)

**Initial public packaging** of the precompiled Windows x64 + CUDA inference runtime for NeoHorse-Jev-4B-GGUF.

### Contents
- Native DLLs: `nh_hidden.dll`, `nh_vision.dll`, `llama.dll`, `mtmd.dll`, `ggml*.dll`, `cudart64_12.dll`
- Python loader: `runtime/runtime.py` + `example.py`, `gguf-py/`, decision helpers under `support/`
- Optional rebuild scripts under `rebuild/`

### Build fingerprint
- Date: 2026-09-27 ~02:20 Asia/Shanghai
- CUDA toolkit: 12.3
- GPU arch: sm_89
- llama.cpp commit: `94256114c229674ef96e76eb2dea596e65b43818`

### Git LFS
- Tracked: `runtime/ggml-cuda.dll` (~183 MB) only
- Cloners need Git LFS, or grab that DLL from a Release asset

### Not in this repo
- GGUF weights (download from Hugging Face)
- Mousemove / aim sandbox scripts

### Suggested GitHub Release assets
- Full tree zip of this repo (with LFS-resolved `ggml-cuda.dll`), or
- Standalone `ggml-cuda.dll` for users who clone without LFS