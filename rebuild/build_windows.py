"""Windows MSVC+CUDA build for NeoHorse GGUF runtime. Keep artifacts under this runtime dir (E:)."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIN = "94256114c229674ef96e76eb2dea596e65b43818"
CUDA_ROOT = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3")
NVCC = CUDA_ROOT / "bin" / "nvcc.exe"
CUDA_ROOT_POSIX = CUDA_ROOT.as_posix()


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda-arch", default="89")
    ap.add_argument("--jobs", default="8")
    ap.add_argument("--source", type=Path, default=None)
    args = ap.parse_args()

    src = HERE / "llama.cpp"
    build = HERE / "build"

    if not src.exists():
        if args.source:
            shutil.copytree(args.source, src)
        else:
            run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "https://github.com/ggml-org/llama.cpp.git",
                    str(src),
                ]
            )
        run(["git", "-C", str(src), "checkout", "--detach", PIN])

    head = subprocess.check_output(
        ["git", "-C", str(src), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != PIN:
        raise SystemExit(f"unexpected llama.cpp HEAD {head}, want {PIN}")

    target = src / "src" / "llama-model-loader.cpp"
    original = target.read_text(encoding="utf-8")
    marker = "// NeoHorse unified: auxiliary tensors stay in the same GGUF."
    if marker not in original:
        needle = "            // make sure there is no duplicated tensor names"
        patch = (
            "            // NeoHorse unified: auxiliary tensors stay in the same GGUF.\n"
            '            const int nh_key = gguf_find_key(metadata, "neohorse.unified_version");\n'
            "            if (nh_key >= 0 && gguf_get_val_u32(metadata, nh_key) == 1 &&\n"
            '                (tensor_name.rfind("v.", 0) == 0 || tensor_name.rfind("mm.", 0) == 0 ||\n'
            '                 tensor_name.rfind("neohorse.pointer.", 0) == 0)) {\n'
            "                continue;\n"
            "            }\n"
        )
        count = original.count("\n" + needle + "\n")
        if count != 2:
            raise SystemExit(f"patch needle count={count}, expected 2")
        target.write_text(
            original.replace("\n" + needle + "\n", "\n" + patch + needle + "\n"),
            encoding="utf-8",
        )
        print("patched llama-model-loader.cpp", flush=True)
    else:
        print("loader already patched", flush=True)

    if not NVCC.is_file():
        raise SystemExit(f"nvcc not found: {NVCC}")

    # Force toolkit 12.3 libs (avoid Anaconda cudart64_110 / LNK2005).
    cudart = (CUDA_ROOT / "lib" / "x64" / "cudart.lib").as_posix()
    cudart_static = (CUDA_ROOT / "lib" / "x64" / "cudart_static.lib").as_posix()
    cmake = [
        "cmake",
        "-S",
        str(src),
        "-B",
        str(build),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_CUDA=ON",
        f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}",
        f"-DCMAKE_CUDA_COMPILER={NVCC.as_posix()}",
        f"-DCUDAToolkit_ROOT={CUDA_ROOT_POSIX}",
        f"-DCUDA_TOOLKIT_ROOT_DIR={CUDA_ROOT_POSIX}",
        f"-DCMAKE_PREFIX_PATH={CUDA_ROOT_POSIX}",
        f"-DCUDA_CUDART={cudart}",
        f"-DCUDA_cudart_LIBRARY={cudart}",
        f"-DCUDA_cudart_static_LIBRARY={cudart_static}",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DLLAMA_CURL=OFF",
        "-DBUILD_SHARED_LIBS=ON",
        "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
    ]
    run(cmake)
    run(["cmake", "--build", str(build), "--target", "mtmd", "-j", str(args.jobs)])

    lib_dirs: list[Path] = []
    for p in build.rglob("mtmd.lib"):
        lib_dirs.append(p.parent)
    for p in build.rglob("llama.lib"):
        lib_dirs.append(p.parent)
    for p in build.rglob("ggml.lib"):
        lib_dirs.append(p.parent)
    bin_dirs = []
    for p in build.rglob("mtmd.dll"):
        bin_dirs.append(p.parent)
    lib_dirs = list(dict.fromkeys(lib_dirs))
    bin_dirs = list(dict.fromkeys(bin_dirs))
    print("lib_dirs:", *[str(x) for x in lib_dirs], flush=True)
    print("bin_dirs:", *[str(x) for x in bin_dirs], flush=True)
    if not lib_dirs:
        raise SystemExit(f"no *.lib found under {build}")

    includes = [
        str(src / "include"),
        str(src / "ggml" / "include"),
        str(src / "tools" / "mtmd"),
    ]
    # Prefer import libs that exist
    wanted = ["mtmd", "llama", "ggml", "ggml-base"]
    link_libs = []
    for name in wanted:
        found = None
        for d in lib_dirs:
            cand = d / f"{name}.lib"
            if cand.is_file():
                found = cand
                break
        if not found:
            raise SystemExit(f"missing {name}.lib under {lib_dirs}")
        link_libs.append(str(found))

    for stem, exports in (
        ("vision", ["nv_open", "nv_hidden", "nv_close"]),
        ("hidden", ["nh_open", "nh_hidden", "nh_close"]),
    ):
        cpp = HERE / f"{stem}_bridge.cpp"
        out_dll = HERE / f"nh_{stem}.dll"
        obj = HERE / f"{stem}_bridge.obj"
        cmd = [
            "cl",
            "/nologo",
            "/O2",
            "/std:c++17",
            "/EHsc",
            "/MD",
            "/LD",
            *[f"/I{i}" for i in includes],
            str(cpp),
            f"/Fe:{out_dll}",
            f"/Fo:{obj}",
            "/link",
            *[f"/LIBPATH:{d}" for d in lib_dirs],
            *link_libs,
            *[f"/EXPORT:{e}" for e in exports],
        ]
        run(cmd)

    # Convenience: copy dependency DLLs next to nh_*.dll for easy loading
    for d in bin_dirs:
        for dll in d.glob("*.dll"):
            dest = HERE / dll.name
            if not dest.exists() or dest.stat().st_mtime < dll.stat().st_mtime:
                shutil.copy2(dll, dest)
                print("copied", dll.name, flush=True)

    print("DONE", flush=True)
    for p in sorted(HERE.glob("nh_*.dll")):
        print(f"  {p.name} {p.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
