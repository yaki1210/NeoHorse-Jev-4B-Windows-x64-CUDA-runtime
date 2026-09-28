"""One GGUF contains all weights AND tokenizer; no model-weight extraction."""
import base64, ctypes as ct, gzip, hashlib, json, sys, tempfile, threading
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'gguf-py'),str(HERE/'llama.cpp/gguf-py'),str(HERE/'support')]
import gguf
from neohorse_decision._vendor.schema import SystemOneRequest,to_record
from neohorse_decision._vendor.model import encode
from neohorse_decision.systemone import MODEL_ID,MODEL_NAMES,with_confidence

class NeoHorseGGUF:
    def __init__(self,model_path):
        self.path=Path(model_path).resolve();self.lock=threading.Lock();self.handle=None;self.mode=None
        r=gguf.GGUFReader(self.path)
        if r.fields['neohorse.unified_version'].contents()!=1:raise ValueError('Expected unified GGUF v1')
        self.precision=r.fields['neohorse.language_precision'].contents()
        self.head={t.name.removeprefix('neohorse.pointer.'):np.array(t.data,copy=True) for t in r.tensors if t.name.startswith('neohorse.pointer.')}
        if set(self.head)!={'q.weight','q.bias','k.weight','k.bias'}:raise ValueError('Incomplete decision head')
        packed=base64.b64decode(r.fields['neohorse.tokenizer_gzip_base64'].contents(),validate=True)
        if hashlib.sha256(packed).hexdigest()!=r.fields['neohorse.tokenizer_sha256'].contents():raise ValueError('Tokenizer identity mismatch')
        files=json.loads(gzip.decompress(packed))
        if set(files)!={'tokenizer.json','tokenizer_config.json','chat_template.jinja'}:raise ValueError('Unexpected tokenizer files')
        # Only small tokenizer configuration is materialized. No vision/head/LLM weight file is extracted.
        with tempfile.TemporaryDirectory(prefix='neohorse-tokenizer-') as directory:
            for name,text in files.items():(Path(directory)/name).write_text(text, encoding='utf-8')
            self.tok=AutoTokenizer.from_pretrained(directory,local_files_only=True)
        self.dim=self.head['q.weight'].shape[1]
        # Windows builds export nh_hidden.dll / nh_vision.dll; Linux keeps libnh_*.so.
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
        self.textlib=_nh_load('hidden');self.vislib=_nh_load('vision')
        self.textlib.nh_open.argtypes=[ct.c_char_p,ct.c_int];self.textlib.nh_open.restype=ct.c_void_p
        self.textlib.nh_close.argtypes=[ct.c_void_p]
        self.textlib.nh_hidden.argtypes=[ct.c_void_p,ct.POINTER(ct.c_int32),ct.c_int,ct.POINTER(ct.c_int32),ct.c_int,ct.POINTER(ct.c_float)]
        self.textlib.nh_hidden.restype=ct.c_int
        self.vislib.nv_open.argtypes=[ct.c_char_p,ct.c_char_p];self.vislib.nv_open.restype=ct.c_void_p
        self.vislib.nv_close.argtypes=[ct.c_void_p]
        self.vislib.nv_hidden.argtypes=[ct.c_void_p,ct.POINTER(ct.c_int32),ct.c_int,ct.POINTER(ct.c_int32),ct.c_int,ct.POINTER(ct.c_ubyte),ct.c_int,ct.c_int,ct.POINTER(ct.c_float),ct.POINTER(ct.c_int)]
        self.vislib.nv_hidden.restype=ct.c_int

    def _switch(self,vision):
        mode='vision' if vision else 'text'
        if self.mode==mode and self.handle:return
        self._close()
        path=str(self.path).encode()
        self.handle=self.vislib.nv_open(path,path) if vision else self.textlib.nh_open(path,99)
        if not self.handle:raise RuntimeError('Model initialization failed')
        self.mode=mode

    def predict(self,request,image=None):
        from PIL import Image
        if image is not None and (not isinstance(image,Image.Image) or image.width*image.height>4194304):raise ValueError('Expected a local PIL image <=4 megapixels')
        if len(json.dumps(request).encode())>1024*1024:raise ValueError('Request exceeds 1 MiB')
        req=SystemOneRequest(**request)
        if req.model not in MODEL_NAMES:raise ValueError('Unknown model')
        if image is not None and len(req.questions)!=1:raise ValueError('Image mode supports one question')
        rec,metadata=to_record(req)
        if not self.lock.acquire(blocking=False):raise RuntimeError('Model busy')
        try:
            self._switch(image is not None);answers={};tokens=0;image_tokens=0
            for q,m in zip(rec['questions'],metadata):
                single={'state':rec['state'],'questions':[q]}
                enc=encode(self.tok,single,strict=True,max_state=2048 if image is not None else 384,
                           max_branch=8192 if image is not None else 1024,option_isolation=False)
                ids=np.asarray(enc['ids'],np.int32)
                pos=np.asarray([enc['decide_idx'][0]]+list(enc['opt_idx'][0]),np.int32)
                hidden=np.empty((len(pos),self.dim),np.float32)
                args=[self.handle,ids.ctypes.data_as(ct.POINTER(ct.c_int32)),len(ids),pos.ctypes.data_as(ct.POINTER(ct.c_int32)),len(pos)]
                if image is None:
                    if len(ids)>2048:raise ValueError('Text exceeds 2048 tokens')
                    rc=self.textlib.nh_hidden(*args,hidden.ctypes.data_as(ct.POINTER(ct.c_float)))
                else:
                    pixels=np.asarray(image.convert('RGB'),np.uint8).copy();nt=ct.c_int()
                    rc=self.vislib.nv_hidden(*args,pixels.ctypes.data_as(ct.POINTER(ct.c_ubyte)),image.width,image.height,hidden.ctypes.data_as(ct.POINTER(ct.c_float)),ct.byref(nt))
                    image_tokens=nt.value
                if rc or not np.isfinite(hidden).all():raise RuntimeError(f'Decode failed: {rc}')
                query=self.head['q.weight']@hidden[0]+self.head['q.bias']
                keys=hidden[1:]@self.head['k.weight'].T+self.head['k.bias']
                z=keys@query/(len(query)**.5);p=np.exp(z-z.max());p/=p.sum();p=p.tolist()
                if m['type']=='choice':a=dict(type='choice',choice=m['keys'][int(np.argmax(p))],probabilities=dict(zip(m['keys'],p)))
                elif m['type']=='noul':a=dict(type='noul',noul=p[1],probabilities=dict(zip(['false','true'],p)))
                else:a=dict(type='score',score=sum(i*v for i,v in enumerate(p)),legend=m['legend'],probabilities=dict(zip(map(str,range(len(p))),p)))
                answers[m['id']]=a;tokens+=len(ids)+(image_tokens+2 if image is not None else 0)
            result=dict(model=MODEL_ID,answers=answers,input_tokens=tokens)
            if image is not None:result['image_tokens']=image_tokens
            return with_confidence(result)
        finally:self.lock.release()

    def _close(self):
        if self.handle:
            (self.vislib.nv_close if self.mode=='vision' else self.textlib.nh_close)(self.handle)
            self.handle=None;self.mode=None

    def close(self):
        with self.lock:self._close()
