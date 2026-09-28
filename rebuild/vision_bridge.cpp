#include "llama.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <cstring>
#include <vector>
struct Engine {llama_model* model;llama_context* ctx;mtmd_context* vision;int dim;};
extern "C" void* nv_open(const char* model,const char* projector) {
    ggml_backend_load_all();llama_backend_init();
    auto mp=llama_model_default_params();mp.n_gpu_layers=99;
    auto m=llama_model_load_from_file(model,mp);if(!m)return nullptr;
    auto cp=llama_context_default_params();cp.n_ctx=12288;cp.n_batch=2048;cp.n_ubatch=2048;
    cp.n_seq_max=1;cp.n_threads=4;cp.n_threads_batch=4;cp.embeddings=true;
    cp.pooling_type=LLAMA_POOLING_TYPE_NONE;cp.attention_type=LLAMA_ATTENTION_TYPE_CAUSAL;
    auto c=llama_init_from_model(m,cp);if(!c){llama_model_free(m);return nullptr;}
    auto vp=mtmd_context_params_default();vp.use_gpu=true;vp.n_threads=4;vp.warmup=false;
    vp.image_min_tokens=64;vp.image_max_tokens=1024;
    auto v=mtmd_init_from_file(projector,m,vp);
    if(!v){llama_free(c);llama_model_free(m);return nullptr;}
    return new Engine{m,c,v,llama_model_n_embd(m)};
}
static int text(Engine* e,const int32_t* ids,int n,int start) {
    std::vector<llama_pos> pos(n);std::vector<int8_t> flags(n,1);
    for(int i=0;i<n;++i)pos[i]=start+i;
    llama_batch b{n,const_cast<int32_t*>(ids),nullptr,pos.data(),nullptr,nullptr,flags.data()};
    return llama_decode(e->ctx,b);
}
extern "C" int nv_hidden(void* handle,const int32_t* ids,int n,const int32_t* wanted,int k,
                         const unsigned char* rgb,int w,int h,float* out,int* image_tokens) {
    auto e=static_cast<Engine*>(handle);
    if(!e||n<2||n>10240||k<1||w<1||h<1||(int64_t)w*h>4194304)return -1;
    for(int j=0;j<k;++j)if(wanted[j]<1||wanted[j]>=n)return -2;
    auto bitmap=mtmd_bitmap_init(w,h,rgb);auto chunks=mtmd_input_chunks_init();
    const mtmd_bitmap* bitmaps[]={bitmap};const char* marker=mtmd_default_marker();
    mtmd_input_text input{marker,strlen(marker),false,true};
    int rc=mtmd_tokenize(e->vision,chunks,&input,bitmaps,1);
    if(rc){mtmd_bitmap_free(bitmap);mtmd_input_chunks_free(chunks);return -3;}
    llama_memory_clear(llama_get_memory(e->ctx),true);
    rc=text(e,ids,1,0);llama_pos past=1;*image_tokens=0;
    for(size_t i=0;!rc&&i<mtmd_input_chunks_size(chunks);++i){
        auto chunk=mtmd_input_chunks_get(chunks,i);
        if(mtmd_input_chunk_get_type(chunk)==MTMD_INPUT_CHUNK_TYPE_IMAGE)*image_tokens+=mtmd_input_chunk_get_n_tokens(chunk);
        llama_pos next=past;
        rc=mtmd_helper_eval_chunk_single(e->vision,e->ctx,chunk,past,0,2048,false,&next);
        past=next;
    }
    if(!rc&&(*image_tokens<1||*image_tokens>1024||n+*image_tokens+2>12288))rc=-4;
    // Capture selected final hidden vectors as each text batch is evaluated.
    for(int offset=1;!rc&&offset<n;offset+=2048){
        int count=std::min(2048,n-offset);rc=text(e,ids+offset,count,past);past+=count;
        for(int j=0;!rc&&j<k;++j)if(wanted[j]>=offset&&wanted[j]<offset+count){
            auto emb=llama_get_embeddings_ith(e->ctx,wanted[j]-offset);
            if(!emb){rc=-5;break;}memcpy(out+j*e->dim,emb,sizeof(float)*e->dim);
        }
    }
    mtmd_bitmap_free(bitmap);mtmd_input_chunks_free(chunks);return rc;
}
extern "C" void nv_close(void* handle){auto e=static_cast<Engine*>(handle);if(!e)return;mtmd_free(e->vision);llama_free(e->ctx);llama_model_free(e->model);delete e;}
