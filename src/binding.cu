#include <ATen/Tensor.h>
#include <ATen/ops/empty.h>
#include <Python.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <torch/library.h>
#include <vector>

#include "cache.cuh"
#include "kernels/kernel_v21.cuh"
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

    C10_CUDA_CHECK(cudaMemcpyAsync(frag_ptrs, frag_ptrs_host.data(), 2*sizeof(void*)*F, cudaMemcpyHostToDevice, stream));

    Shape shape = {F, W, Hq, Hkv, E, Ev, S, 0, 0, 0, nullptr, nullptr}; // block_size, max_blocks not used
    C10_CUDA_CHECK(v21::dvattn_attention_gpu(out_ptr, (float)scale, loc_ptr, query_ptr, fl_ptr,
                          frag_ptrs, frag_ptrs + F, shape));
}

template<class scalar_t>
void dvattn_paged_attention_tpl(
        at::Tensor& out, double scale, const at::Tensor& locations, const at::Tensor& queries,
        const at::Tensor& fragment_lengths, const at::Tensor& key_cache,
        const at::Tensor& value_cache, const at::Tensor& block_table)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    scalar_t* out_ptr = torch_get_pointer<scalar_t>(out);
    const scalar_t* query_ptr = torch_get_pointer<scalar_t>(queries);
    const scalar_t* key_cache_ptr = torch_get_pointer<scalar_t>(key_cache);
    const scalar_t* value_cache_ptr = torch_get_pointer<scalar_t>(value_cache);
    const int* loc_ptr = locations.const_data_ptr<int>();
    const int* fl_ptr = fragment_lengths.const_data_ptr<int>();
    const int* block_table_ptr = block_table.const_data_ptr<int>();

    Shape shape = {
        (int)locations.size(0),       // F
        (int)out.size(0),             // W (batch_size)
        (int)out.size(1),             // Hq
        (int)key_cache.size(1),       // Hkv
        (int)queries.size(4),         // E
        (int)out.size(3),             // Ev
        (int)out.size(2),             // S
        (int)key_cache.size(2),       // block_size
        (int)block_table.size(1),     // max_blocks_per_seq
        0, nullptr, nullptr
    };

    C10_CUDA_CHECK(v21::dvattn_paged_attention_gpu<scalar_t>(
        out_ptr, (float)scale, loc_ptr, query_ptr, fl_ptr,
        key_cache_ptr, value_cache_ptr, block_table_ptr, shape));
}

template<class scalar_t>
void dvattn_varlen_paged_attention_tpl(
        at::Tensor& out, double scale, const at::Tensor& locations, const at::Tensor& queries,
        const at::Tensor& fragment_lengths, const at::Tensor& key_cache,
        const at::Tensor& value_cache, const at::Tensor& block_table,
        const at::Tensor& cu_seqlens_q, const at::Tensor& cu_seqlens_k)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    scalar_t* out_ptr = torch_get_pointer<scalar_t>(out);
    const scalar_t* query_ptr = torch_get_pointer<scalar_t>(queries);
    const scalar_t* key_cache_ptr = torch_get_pointer<scalar_t>(key_cache);
    const scalar_t* value_cache_ptr = torch_get_pointer<scalar_t>(value_cache);
    const int* loc_ptr = locations.const_data_ptr<int>();
    const int* fl_ptr = fragment_lengths.const_data_ptr<int>();
    const int* block_table_ptr = block_table.const_data_ptr<int>();
    const int* cu_seqlens_q_ptr = cu_seqlens_q.const_data_ptr<int>();
    const int* cu_seqlens_k_ptr = cu_seqlens_k.const_data_ptr<int>();

    Shape shape = {
        (int)locations.size(0),       // F
        (int)cu_seqlens_q.size(0) - 1,  // W (batch_size)
        (int)out.size(1),             // Hq
        (int)key_cache.size(1),       // Hkv
        (int)queries.size(3),         // E
        (int)out.size(2),             // Ev
        0,                            // S (not used in varlen)
        (int)key_cache.size(2),       // block_size
        (int)block_table.size(1),     // max_blocks_per_seq
        (int)out.size(0),             // total_q_tokens
        cu_seqlens_q_ptr,
        cu_seqlens_k_ptr
    };

    C10_CUDA_CHECK(v21::dvattn_varlen_paged_attention_gpu<scalar_t>(
        out_ptr, (float)scale, loc_ptr, query_ptr, fl_ptr,
        key_cache_ptr, value_cache_ptr, block_table_ptr, shape));
}

template<class scalar_t>
void copy_to_blocks_tpl(
    const at::Tensor& key_states, const at::Tensor& value_states,
    at::Tensor& key_cache, at::Tensor& value_cache,
    const at::Tensor& block_table, const at::Tensor& seq_lengths)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int B = key_states.size(0);
    const int Hkv = key_states.size(1);
    const int S_new = key_states.size(2);
    const int E = key_states.size(3);
    const int Ev = value_states.size(3);
    const int block_size = key_cache.size(2);
    const int max_blocks = block_table.size(1);

    Shape shape = {0, B, 0, Hkv, E, Ev, S_new, block_size, max_blocks, 0, nullptr, nullptr};

    dim3 grid_dim(B, Hkv);
    // Each thread handles one element in the embedding dimension
    dim3 block_dim(S_new, std::max(E, Ev));

    copy_to_blocks_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
        torch_get_pointer<scalar_t>(key_states),
        torch_get_pointer<scalar_t>(value_states),
        torch_get_pointer<scalar_t>(key_cache),
        torch_get_pointer<scalar_t>(value_cache),
        block_table.const_data_ptr<int>(),
        seq_lengths.const_data_ptr<int>(),
        shape
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
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
        const at::Tensor& fragment_lengths, const at::Tensor& key_cache,
        const at::Tensor& value_cache, const at::Tensor& block_table)
{
    if(out.dtype() == at::kHalf) {
        dvattn_paged_attention_tpl<half>(out, scale, locations, queries, fragment_lengths, key_cache, value_cache, block_table);
    } else if (out.dtype() == at::kFloat) {
        dvattn_paged_attention_tpl<float>(out, scale, locations, queries, fragment_lengths, key_cache, value_cache, block_table);
    } else if (out.dtype() == at::kBFloat16) {
        dvattn_paged_attention_tpl<nv_bfloat16>(out, scale, locations, queries, fragment_lengths, key_cache, value_cache, block_table);
    }
}

void dvattn_varlen_paged_attention(
        at::Tensor& out, double scale, const at::Tensor& locations, const at::Tensor& queries,
        const at::Tensor& fragment_lengths, const at::Tensor& key_cache,
        const at::Tensor& value_cache, const at::Tensor& block_table,
        const at::Tensor& cu_seqlens_q, const at::Tensor& cu_seqlens_k)
{
    if(out.dtype() == at::kHalf) {
        dvattn_varlen_paged_attention_tpl<half>(out, scale, locations, queries, fragment_lengths, key_cache, value_cache, block_table, cu_seqlens_q, cu_seqlens_k);
    } else if (out.dtype() == at::kFloat) {
        dvattn_varlen_paged_attention_tpl<float>(out, scale, locations, queries, fragment_lengths, key_cache, value_cache, block_table, cu_seqlens_q, cu_seqlens_k);
    } else if (out.dtype() == at::kBFloat16) {
        dvattn_varlen_paged_attention_tpl<nv_bfloat16>(out, scale, locations, queries, fragment_lengths, key_cache, value_cache, block_table, cu_seqlens_q, cu_seqlens_k);
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

void copy_to_blocks(
    const at::Tensor& key_states, const at::Tensor& value_states,
    at::Tensor& key_cache, at::Tensor& value_cache,
    const at::Tensor& block_table, const at::Tensor& seq_lengths)
{
    if(key_states.dtype() == at::kHalf) {
        copy_to_blocks_tpl<half>(key_states, value_states, key_cache, value_cache, block_table, seq_lengths);
    } else if (key_states.dtype() == at::kFloat) {
        copy_to_blocks_tpl<float>(key_states, value_states, key_cache, value_cache, block_table, seq_lengths);
    } else if (key_states.dtype() == at::kBFloat16) {
        copy_to_blocks_tpl<nv_bfloat16>(key_states, value_states, key_cache, value_cache, block_table, seq_lengths);
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
    m.def("dvattn_rope(Tensor(a!) output, Tensor queries, Tensor cosines, Tensor sines) -> ()", tags, torch::_RegisterOrVerify::REGISTER);
    m.def("dvattn_fused(Tensor(a!) output, Tensor(b!) rq, float scale, Tensor locations, Tensor queries, "
          "Tensor fragment_lengths, Tensor[] key_fragments, Tensor[] value_fragments, Tensor cosines, Tensor sines) -> ()", tags, torch::_RegisterOrVerify::REGISTER);

    m.def("dvattn_paged_sdpa(Tensor(a!) output, float scale, Tensor locations, Tensor queries, "
          "Tensor fragment_lengths, Tensor key_cache, Tensor value_cache, Tensor block_table) -> ()", tags, torch::_RegisterOrVerify::REGISTER);
    m.def("copy_to_blocks(Tensor key_states, Tensor value_states, Tensor(a!) key_cache, Tensor(b!) value_cache, "
          "Tensor block_table, Tensor seq_lengths) -> ()", tags, torch::_RegisterOrVerify::REGISTER);
    m.def("dvattn_varlen_paged_sdpa(Tensor(a!) output, float scale, Tensor locations, Tensor queries, "
          "Tensor fragment_lengths, Tensor key_cache, Tensor value_cache, Tensor block_table, Tensor cu_seqlens_q, Tensor cu_seqlens_k) -> ()", tags, torch::_RegisterOrVerify::REGISTER);

}

TORCH_LIBRARY_IMPL(libdvatt, CUDA, m) {
    m.impl("dvattn_sdpa", dvattn_attention);
    m.impl("dvattn_rope", dvattn_rope);
    m.impl("dvattn_fused", dvattn_fused);

    m.impl("dvattn_paged_sdpa", dvattn_paged_attention);
    m.impl("copy_to_blocks", copy_to_blocks);
    m.impl("dvattn_varlen_paged_sdpa", dvattn_varlen_paged_attention);
}
