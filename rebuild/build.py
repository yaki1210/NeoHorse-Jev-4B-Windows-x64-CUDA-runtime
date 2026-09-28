"""Build a separate pinned llama.cpp with opt-in unified-tensor filtering."""
import argparse, subprocess, shutil, os
from pathlib import Path
HERE=Path(__file__).resolve().parent
PIN='94256114c229674ef96e76eb2dea596e65b43818'

def run(cmd):subprocess.run(list(map(str,cmd)),check=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path);ap.add_argument('--cuda-arch',default='native');ap.add_argument('--jobs',default='8');args=ap.parse_args()
    src=HERE/'llama.cpp';build=HERE/'build'
    if not src.exists():
        if args.source:
            source=args.source.resolve()
            shutil.copytree(source,src)
        else:run(['git','clone','https://github.com/ggml-org/llama.cpp.git',src])
        run(['git','-C',src,'checkout','--detach',PIN])
    assert subprocess.check_output(['git','-C',src,'rev-parse','HEAD'],text=True).strip()==PIN
    target=src/'src/llama-model-loader.cpp'
    original=target.read_text(); marker='// NeoHorse unified: auxiliary tensors stay in the same GGUF.'
    if marker not in original:
        needle='            // make sure there is no duplicated tensor names'
        patch='''            // NeoHorse unified: auxiliary tensors stay in the same GGUF.
            const int nh_key = gguf_find_key(metadata, "neohorse.unified_version");
            if (nh_key >= 0 && gguf_get_val_u32(metadata, nh_key) == 1 &&
                (tensor_name.rfind("v.", 0) == 0 || tensor_name.rfind("mm.", 0) == 0 ||
                 tensor_name.rfind("neohorse.pointer.", 0) == 0)) {
                continue;
            }
'''
        # Main-file and FILE* paths only. Sharded files are not our container format.
        assert original.count('\n'+needle+'\n')==2
        target.write_text(original.replace('\n'+needle+'\n','\n'+patch+needle+'\n'))
    cuda=[]
    if Path('/usr/local/cuda/bin/nvcc').is_file():cuda=['-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc']
    run(['cmake','-S',src,'-B',build,'-DGGML_CUDA=ON','-DCMAKE_CUDA_ARCHITECTURES='+args.cuda_arch,*cuda,
         '-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_CURL=OFF','-DBUILD_SHARED_LIBS=ON'])
    run(['cmake','--build',build,'--target','mtmd','-j',args.jobs])
    for stem in ('vision','hidden'):
        run(['g++','-shared','-fPIC','-O2','-std=c++17',HERE/(stem+'_bridge.cpp'),'-I'+str(src/'include'),
             '-I'+str(src/'ggml/include'),'-I'+str(src/'tools/mtmd'),'-L'+str(build/'bin'),
             '-lmtmd','-lllama','-lggml-base','-lggml','-Wl,-rpath,$ORIGIN/build/bin',
             '-o',HERE/('libnh_'+stem+'.so')])

if __name__=='__main__':main()
