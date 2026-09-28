#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstring>
#include <vector>

struct engine { llama_model * model; llama_context * ctx; int dim; };

extern "C" void * nh_open(const char * path, int gpu_layers) {
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = gpu_layers;
    auto model = llama_model_load_from_file(path, mp);
    if (!model) return nullptr;
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 2048;
    cp.n_ubatch = 2048;
    cp.n_seq_max = 1;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cp.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
    auto ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); return nullptr; }
    return new engine{model, ctx, llama_model_n_embd(model)};
}

extern "C" int nh_hidden(void * handle, const int32_t * ids, int count,
                          const int32_t * positions, int n_positions, float * out) {
    auto e = static_cast<engine *>(handle);
    if (!e || count < 1 || count > 2048 || n_positions < 1) return -1;
    std::vector<int8_t> outputs(count, 0);
    for (int j = 0; j < n_positions; ++j) {
        if (positions[j] < 0 || positions[j] >= count) return -2;
        outputs[positions[j]] = 1;
    }
    llama_memory_clear(llama_get_memory(e->ctx), true);
    llama_batch batch{count, const_cast<int32_t *>(ids), nullptr, nullptr, nullptr, nullptr, outputs.data()};
    int rc = llama_decode(e->ctx, batch);
    if (rc) return rc;
    for (int j = 0; j < n_positions; ++j) {
        auto h = llama_get_embeddings_ith(e->ctx, positions[j]);
        if (!h) return -3;
        std::memcpy(out + j * e->dim, h, sizeof(float) * e->dim);
    }
    return 0;
}

extern "C" void nh_close(void * handle) {
    auto e = static_cast<engine *>(handle);
    if (!e) return;
    llama_free(e->ctx);
    llama_model_free(e->model);
    delete e;
}
