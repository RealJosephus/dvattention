#include <ATen/Tensor.h>
#include <ATen/ops/empty.h>
#include <Python.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <torch/library.h>
#include <vector>

#include "kernels/kernel_v21.cuh"
#include "kernels/paged_kernel_v21.cuh"
#include "kernels/varlen_paged_kernel_v21.cuh"
#include "rope.cuh"

template<class scalar_t>
const scalar_t* torch_get_pointer(const at::Tensor& tensor) {
    if constexpr (std::is_same_v<scalar_t, float>) {
        return tensor.const_data_ptr<float>();
    } else if constexpr (std::is_same_v<scalar_t, half>) {
        return reinterpret_cast<const half*>(tensor.const_data_ptr<at::Half>());
    } else if constexpr (std::is_same_v<scalar_t, nv_bfloat16>) {
        return reinterpret_cast<const nv_bfloat16*>(tensor.const_data_ptr<at::BFloat16>());
    } else {
        return nullptr;
    }
}

template<class scalar_t>
scalar_t* torch_get_pointer(at::Tensor& tensor) {
    if constexpr (std::is_same_v<scalar_t, float>) {
        return tensor.data_ptr<float>();
    } else if constexpr (std::is_same_v<scalar_t, half>) {
        return reinterpret_cast<half*>(tensor.data_ptr<at::Half>());
    } else if constexpr (std::is_same_v<scalar_t, nv_bfloat16>) {
        return reinterpret_cast<nv_bfloat16*>(tensor.data_ptr<at::BFloat16>());
    } else {
        return nullptr;
    }
}


template<class scalar_t>
void dvattn_rope_tpl(
        at::Tensor& out, const at::Tensor& queries, const at::Tensor& cosines,
        const at::Tensor& sines)
{
    // extract pointers and sizes
    int F = out.size(0);
    int W = out.size(1);
    int Hq = out.size(2);
    int S = out.size(3);
    int E = out.size(4);
    int RotaryE = cosines.size(3);
    TORCH_CHECK(out.is_contiguous());
    TORCH_CHECK(queries.is_contiguous());
    TORCH_CHECK(cosines.is_contiguous());
    TORCH_CHECK(sines.is_contiguous());

    TORCH_CHECK_EQ(queries.size(0), W);
    TORCH_CHECK_EQ(queries.size(1), Hq);
    TORCH_CHECK_EQ(queries.size(2), S);
    TORCH_CHECK_EQ(queries.size(3), E);

    TORCH_CHECK_EQ(cosines.size(0), F);
    TORCH_CHECK_EQ(cosines.size(1), W);
    TORCH_CHECK_EQ(cosines.size(2), S);
    TORCH_CHECK_EQ(cosines.size(3), RotaryE);

    TORCH_CHECK_EQ(sines.size(0), F);
    TORCH_CHECK_EQ(sines.size(1), W);
    TORCH_CHECK_EQ(sines.size(2), S);
    TORCH_CHECK_EQ(sines.size(3), RotaryE);

    rope_gpu(torch_get_pointer<scalar_t>(out), torch_get_pointer<scalar_t>(queries),
             torch_get_pointer<float>(cosines), torch_get_pointer<float>(sines),
                     F, W, Hq, S, E, RotaryE);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<class scalar_t>
void dvattn_attention_tpl(
        at::Tensor& out, double scale, const at::Tensor& locations, const at::Tensor& queries,
        const at::Tensor& fragment_lengths, const std::vector<at::Tensor>& key_fragments,
        const std::vector<at::Tensor>& value_fragments)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // extract pointers and sizes
    int W = out.size(0);
    int Hq = out.size(1);
    int S = out.size(2);
    int Ev = out.size(3);
    TORCH_CHECK(out.is_contiguous());
    scalar_t* out_ptr = torch_get_pointer<scalar_t>(out);
    // Input:   keys: [Hkv, fragment_lengths[i], E] for i in [F]
    //          values: [Hkv, fragment_lengths[i], Ev] for i in [F]

    int F = locations.size(0);
    TORCH_CHECK_EQ(locations.size(1), W);
    TORCH_CHECK_EQ(locations.size(2), S);
    TORCH_CHECK(locations.is_contiguous());
    const int* loc_ptr = locations.const_data_ptr<int>();

    int E = queries.size(4);
    TORCH_CHECK_EQ(queries.size(0), F);
    TORCH_CHECK_EQ(queries.size(1), W);
    TORCH_CHECK_EQ(queries.size(2), Hq);
    TORCH_CHECK_EQ(queries.size(3), S);
    TORCH_CHECK(queries.is_contiguous());
    const scalar_t* query_ptr = torch_get_pointer<scalar_t>(queries);

    TORCH_CHECK_EQ(fragment_lengths.size(0), F);
    TORCH_CHECK(fragment_lengths.is_contiguous());
    const int* fl_ptr = fragment_lengths.const_data_ptr<int>();

    // check key and value fragments
    TORCH_CHECK_EQ(key_fragments.size(), F);
    TORCH_CHECK_EQ(value_fragments.size(), F);
    // Make exactly one cached memory allocation to store the pointers in
    // NOTE: This is neither thread safe, nor will this memory ever be released again.
    static const scalar_t** frag_ptrs = nullptr;
    if(frag_ptrs == nullptr) {
        C10_CUDA_CHECK(cudaMalloc(&frag_ptrs, sizeof(void *) * 1024));
    }

    std::vector<const scalar_t*> frag_ptrs_host(2*F);
    bool has_batch_dim = key_fragments[0].dim() == 4;
    int fo = has_batch_dim ? 1 : 0;
    int Hkv = key_fragments[0].size(fo);
    for(int f = 0; f < F; ++f) {
        TORCH_CHECK_EQ(key_fragments[f].size(fo + 0), Hkv);
        TORCH_CHECK_EQ(value_fragments[f].size(fo + 0), Hkv);
        int fl = key_fragments[f].size(fo + 1);
        TORCH_CHECK_EQ(value_fragments[f].size(fo + 1), fl);
        TORCH_CHECK_EQ(key_fragments[f].size(fo + 2), E);
        TORCH_CHECK_EQ(value_fragments[f].size(fo + 2), Ev);

        TORCH_CHECK(key_fragments[f].is_contiguous());
        TORCH_CHECK(value_fragments[f].is_contiguous());

        frag_ptrs_host[f] = torch_get_pointer<scalar_t>(key_fragments[f]);
        frag_ptrs_host[F + f] = torch_get_pointer<scalar_t>(value_fragments[f]);
    }

    C10_CUDA_CHECK(cudaMemcpyAsync(frag_ptrs, frag_ptrs_host.data(), 2*sizeof(void*)*F, cudaMemcpyHostToDevice));

    // finally, launch
    Shape shape = {F, W, Hq, Hkv, E, Ev, S};
    C10_CUDA_CHECK(v21::dvattn_attention_gpu(out_ptr, (float)scale, loc_ptr, query_ptr, fl_ptr,
                          frag_ptrs, frag_ptrs + F, shape));
}

template<class scalar_t>
void dvattn_paged_attention_tpl(
        at::Tensor& out, double scale, const at::Tensor& locations, const at::Tensor& queries,
        const at::Tensor& cosines, const at::Tensor& sines,
        const at::Tensor& fragment_lengths,
        const at::Tensor& k_cache, const at::Tensor& v_cache,
        const at::Tensor& block_tables)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int W = out.size(0);
    int Hq = out.size(1);
    int S = out.size(2); // S=1 for decode
    int Ev = out.size(3);
    TORCH_CHECK(out.is_contiguous());
    scalar_t* out_ptr = torch_get_pointer<scalar_t>(out);

    int F = locations.size(0);
    TORCH_CHECK_EQ(locations.size(1), W);
    TORCH_CHECK_EQ(locations.size(2), S);
    TORCH_CHECK(locations.is_contiguous());
    const int* loc_ptr = locations.const_data_ptr<int>();

    // The query from python is [W, Hq, S, E].
    // The dvattn kernel expects a rotated query of shape [F, W, Hq, S, E].
    // Let's create and rotate it.
    auto rotated_queries = at::empty({F, W, Hq, S, queries.size(3)}, queries.options());
    dvattn_rope_tpl<scalar_t>(rotated_queries, queries, cosines, sines);

    const int E = rotated_queries.size(4);
    const scalar_t* query_ptr = torch_get_pointer<scalar_t>(rotated_queries);

    TORCH_CHECK_EQ(fragment_lengths.size(0), F);
    TORCH_CHECK(fragment_lengths.is_contiguous());
    const int* fl_ptr = fragment_lengths.const_data_ptr<int>();

    TORCH_CHECK(k_cache.is_contiguous());
    TORCH_CHECK(v_cache.is_contiguous());
    TORCH_CHECK(block_tables.is_contiguous());
    const scalar_t* k_cache_ptr = torch_get_pointer<scalar_t>(k_cache);
    const scalar_t* v_cache_ptr = torch_get_pointer<scalar_t>(v_cache);
    const int* block_tables_ptr = block_tables.const_data_ptr<int>();

    int block_size = k_cache.size(1);
    int Hkv = k_cache.size(2);
    int max_blocks_per_seq = block_tables.size(1);

    Shape shape = {F, W, Hq, Hkv, E, Ev, S, block_size, max_blocks_per_seq};

    C10_CUDA_CHECK(paged_v21::dvattn_paged_attention_gpu(
        out_ptr, (float)scale, loc_ptr, query_ptr, fl_ptr,
        k_cache_ptr, v_cache_ptr, block_tables_ptr, shape));
}

template<class scalar_t>
void dvattn_varlen_rope_tpl(
        at::Tensor& rotated_queries, // [F, T, H, E]
        const at::Tensor& queries,   // [T, H, E]
        const at::Tensor& cosines,   // [F, T, R/2]
        const at::Tensor& sines)     // [F, T, R/2]
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int F = rotated_queries.size(0);
    const int T = rotated_queries.size(1);
    const int H = rotated_queries.size(2);
    const int E = rotated_queries.size(3);
    const int R = cosines.size(2) * 2;

    TORCH_CHECK(rotated_queries.is_contiguous());
    TORCH_CHECK(queries.is_contiguous());
    TORCH_CHECK(cosines.is_contiguous());
    TORCH_CHECK(sines.is_contiguous());
    TORCH_CHECK_EQ(cosines.scalar_type(), at::kFloat);
    TORCH_CHECK_EQ(sines.scalar_type(), at::kFloat);

    TORCH_CHECK_EQ(queries.dim(), 3);
    TORCH_CHECK_EQ(queries.size(0), T);
    TORCH_CHECK_EQ(queries.size(1), H);
    TORCH_CHECK_EQ(queries.size(2), E);

    TORCH_CHECK_EQ(cosines.dim(), 3);
    TORCH_CHECK_EQ(cosines.size(0), F);
    TORCH_CHECK_EQ(cosines.size(1), T);

    TORCH_CHECK_EQ(sines.dim(), 3);
    TORCH_CHECK_EQ(sines.size(0), F);
    TORCH_CHECK_EQ(sines.size(1), T);
    TORCH_CHECK_EQ(sines.size(2), R / 2);

    rope_varlen_gpu(
        torch_get_pointer<scalar_t>(rotated_queries),
        torch_get_pointer<scalar_t>(queries),
        torch_get_pointer<float>(cosines),
        torch_get_pointer<float>(sines),
        F, T, H, E, R);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<class scalar_t>
void dvattn_varlen_paged_attention_tpl(
        at::Tensor& out, double scale, const at::Tensor& locations, const at::Tensor& queries,
        const at::Tensor& cosines, const at::Tensor& sines,
        const at::Tensor& fragment_lengths,
        const at::Tensor& k_cache, const at::Tensor& v_cache,
        const at::Tensor& block_tables,
        const at::Tensor& cu_seqlens_q, const at::Tensor& cu_seqlens_k)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int T = queries.size(0); // Total tokens
    const int Hq = queries.size(1);
    const int E = queries.size(2);
    const int F = locations.size(0);

    auto rotated_queries = at::empty({F, T, Hq, E}, queries.options());
    dvattn_varlen_rope_tpl<scalar_t>(rotated_queries, queries, cosines, sines);

    TORCH_CHECK(out.is_contiguous());
    scalar_t* out_ptr = torch_get_pointer<scalar_t>(out);

    const int* loc_ptr = locations.const_data_ptr<int>();
    const scalar_t* query_ptr = torch_get_pointer<scalar_t>(rotated_queries);
    const int* fl_ptr = fragment_lengths.const_data_ptr<int>();

    const scalar_t* k_cache_ptr = torch_get_pointer<scalar_t>(k_cache);
    const scalar_t* v_cache_ptr = torch_get_pointer<scalar_t>(v_cache);
    const int* block_tables_ptr = block_tables.const_data_ptr<int>();
    const int* cu_seqlens_q_ptr = cu_seqlens_q.const_data_ptr<int>();
    const int* cu_seqlens_k_ptr = cu_seqlens_k.const_data_ptr<int>();

    const int W = cu_seqlens_q.size(0) - 1;
    const int block_size = k_cache.size(1);
    const int Hkv = k_cache.size(2);
    const int Ev = v_cache.size(3);
    const int max_blocks_per_seq = block_tables.size(1);

    Shape shape = {
        F, W, Hq, Hkv, E, Ev, /*S=*/-1,
        block_size, max_blocks_per_seq,
        T, cu_seqlens_q_ptr, cu_seqlens_k_ptr
    };

    C10_CUDA_CHECK(varlen_paged_v21::dvattn_varlen_paged_attention_gpu(
        out_ptr, (float)scale, loc_ptr, query_ptr, fl_ptr,
        k_cache_ptr, v_cache_ptr, block_tables_ptr, shape));
}

void dvattn_attention(
        at::Tensor& out, double scale, const at::Tensor& locations, const at::Tensor& queries,
        const at::Tensor& fragment_lengths, const std::vector<at::Tensor>& key_fragments,
        const std::vector<at::Tensor>& value_fragments)
{
    if(out.dtype() == at::kHalf) {
        dvattn_attention_tpl<half>(out, scale, locations, queries, fragment_lengths, key_fragments, value_fragments);
    } else if (out.dtype() == at::kFloat) {
        dvattn_attention_tpl<float>(out, scale, locations, queries, fragment_lengths, key_fragments, value_fragments);
    } else if (out.dtype() == at::kBFloat16) {
        dvattn_attention_tpl<nv_bfloat16>(out, scale, locations, queries, fragment_lengths, key_fragments, value_fragments);
    }
}

void dvattn_paged_attention(
        at::Tensor& out, double scale, const at::Tensor& locations, const at::Tensor& queries,
        const at::Tensor& cosines, const at::Tensor& sines,
        const at::Tensor& fragment_lengths,
        const at::Tensor& k_cache, const at::Tensor& v_cache, const at::Tensor& block_tables)
{
    if(out.dtype() == at::kHalf) {
        dvattn_paged_attention_tpl<half>(out, scale, locations, queries, cosines, sines, fragment_lengths, k_cache, v_cache, block_tables);
    } else if (out.dtype() == at::kFloat) {
        dvattn_paged_attention_tpl<float>(out, scale, locations, queries, cosines, sines, fragment_lengths, k_cache, v_cache, block_tables);
    } else if (out.dtype() == at::kBFloat16) {
        dvattn_paged_attention_tpl<nv_bfloat16>(out, scale, locations, queries, cosines, sines, fragment_lengths, k_cache, v_cache, block_tables);
    }
}

void dvattn_varlen_paged_attention(
        at::Tensor& out, double scale, const at::Tensor& locations, const at::Tensor& queries,
        const at::Tensor& cosines, const at::Tensor& sines,
        const at::Tensor& fragment_lengths,
        const at::Tensor& k_cache, const at::Tensor& v_cache,
        const at::Tensor& block_tables,
        const at::Tensor& cu_seqlens_q, const at::Tensor& cu_seqlens_k)
{
    if(out.dtype() == at::kHalf) {
        dvattn_varlen_paged_attention_tpl<half>(out, scale, locations, queries, cosines, sines, fragment_lengths, k_cache, v_cache, block_tables, cu_seqlens_q, cu_seqlens_k);
    } else if (out.dtype() == at::kFloat) {
        dvattn_varlen_paged_attention_tpl<float>(out, scale, locations, queries, cosines, sines, fragment_lengths, k_cache, v_cache, block_tables, cu_seqlens_q, cu_seqlens_k);
    } else if (out.dtype() == at::kBFloat16) {
        dvattn_varlen_paged_attention_tpl<nv_bfloat16>(out, scale, locations, queries, cosines, sines, fragment_lengths, k_cache, v_cache, block_tables, cu_seqlens_q, cu_seqlens_k);
    }
}

void dvattn_rope(
        at::Tensor& out, const at::Tensor& queries, const at::Tensor& cosines, const at::Tensor& sines)
{
    if(out.dtype() == at::kHalf) {
        dvattn_rope_tpl<half>(out, queries, cosines, sines);
    } else if (out.dtype() == at::kFloat) {
        dvattn_rope_tpl<float>(out, queries, cosines, sines);
    } else if (out.dtype() == at::kBFloat16) {
        dvattn_rope_tpl<nv_bfloat16>(out, queries, cosines, sines);
    }
}

void dvattn_fused(
        at::Tensor& out, at::Tensor& rotated_queries, double scale, const at::Tensor& locations, const at::Tensor& queries,
        const at::Tensor& fragment_lengths, const std::vector<at::Tensor>& key_fragments,
        const std::vector<at::Tensor>& value_fragments,
        const at::Tensor& cosines, const at::Tensor& sines)
{
    std::vector<at::Tensor> key_fragments_contiguous;
    std::vector<at::Tensor> val_fragments_contiguous;
    key_fragments_contiguous.reserve(key_fragments.size());
    val_fragments_contiguous.reserve(key_fragments.size());
    for(int i = 0; i < key_fragments.size(); ++i) {
        key_fragments_contiguous.push_back(key_fragments[i].contiguous());
        val_fragments_contiguous.push_back(value_fragments[i].contiguous());
    }
    dvattn_rope(rotated_queries, queries, cosines, sines);
    dvattn_attention(out, scale, locations, rotated_queries, fragment_lengths, key_fragments_contiguous, val_fragments_contiguous);
}

void dvattn_varlen_rope(
        at::Tensor& rotated_queries,
        const at::Tensor& queries,
        const at::Tensor& cosines,
        const at::Tensor& sines)
{
    if(rotated_queries.dtype() == at::kHalf) {
        dvattn_varlen_rope_tpl<half>(rotated_queries, queries, cosines, sines);
    } else if (rotated_queries.dtype() == at::kFloat) {
        dvattn_varlen_rope_tpl<float>(rotated_queries, queries, cosines, sines);
    } else if (rotated_queries.dtype() == at::kBFloat16) {
        dvattn_varlen_rope_tpl<nv_bfloat16>(rotated_queries, queries, cosines, sines);
    }
}

extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
   The import from Python will load the .so consisting of this file
   in this extension, so that the TORCH_LIBRARY static initializers
   below are run. */
PyObject* PyInit_libdvatt(void) {
    static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "libdvatt", /* name of module */
            NULL,            /* module documentation, may be NULL */
            -1,              /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
            NULL,            /* methods */
    };
    return PyModule_Create(&module_def);
}
}

TORCH_LIBRARY(libdvatt, m) {
    std::vector<at::Tag> tags;
    tags.push_back(at::Tag::needs_fixed_stride_order);
    m.def("dvattn_sdpa(Tensor(a!) output, float scale, Tensor locations, Tensor queries, "
          "Tensor fragment_lengths, Tensor[] key_fragments, Tensor[] value_fragments) -> ()", tags, torch::_RegisterOrVerify::REGISTER);
    m.def("dvattn_paged_sdpa(Tensor(a!) output, float scale, Tensor locations, Tensor queries, "
          "Tensor cosines, Tensor sines, Tensor fragment_lengths, Tensor k_cache, Tensor v_cache, Tensor block_tables) -> ()", tags, torch::_RegisterOrVerify::REGISTER);
    m.def("dvattn_rope(Tensor(a!) output, Tensor queries, Tensor cosines, Tensor sines) -> ()", tags, torch::_RegisterOrVerify::REGISTER);
    m.def("dvattn_fused(Tensor(a!) output, Tensor(b!) rq, float scale, Tensor locations, Tensor queries, "
          "Tensor fragment_lengths, Tensor[] key_fragments, Tensor[] value_fragments, Tensor cosines, Tensor sines) -> ()", tags, torch::_RegisterOrVerify::REGISTER);
    m.def("dvattn_varlen_rope(Tensor(a!) output, Tensor queries, Tensor cosines, Tensor sines) -> ()", tags, torch::_RegisterOrVerify::REGISTER);
    m.def("dvattn_varlen_paged_sdpa(Tensor(a!) output, float scale, Tensor locations, Tensor queries, "
          "Tensor cosines, Tensor sines, Tensor fragment_lengths, Tensor k_cache, Tensor v_cache, "
          "Tensor block_tables, Tensor cu_seqlens_q, Tensor cu_seqlens_k) -> ()", tags, torch::_RegisterOrVerify::REGISTER);
}

TORCH_LIBRARY_IMPL(libdvatt, CUDA, m) {
    m.impl("dvattn_sdpa", dvattn_attention);
    m.impl("dvattn_paged_sdpa", dvattn_paged_attention);
    m.impl("dvattn_rope", dvattn_rope);
    m.impl("dvattn_fused", dvattn_fused);
    m.impl("dvattn_varlen_rope", dvattn_varlen_rope);
    m.impl("dvattn_varlen_paged_sdpa", dvattn_varlen_paged_attention);
}
