#pragma once

#include "common.h"
#include "vec.cuh"
#include "cuda_check.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_pipeline_primitives.h>

namespace cg = cooperative_groups;

namespace paged_v21
{
constexpr const int SubWarpSize = 8;
constexpr const int WarpSize = 32;

template<int E, int Ev, int GQA, class scalar_t>
__global__ __launch_bounds__(256) void dvattn_paged_attention_gpu_kernel21(
        scalar_t* out, char* workspace, float scale,
        const int* locations, const scalar_t* queries,
        const int* fragment_lengths,
        const scalar_t* k_cache,
        const scalar_t* v_cache, // Kernel parameter
        const int* block_tables,
        Shape shape) {
    // Paged Attention version of the kernel.
    // K/V caches are non-contiguous and accessed via block_tables.
    // k_cache shape: [num_total_blocks, Hkv, block_size, E]
    // v_cache shape: [num_total_blocks, Hkv, block_size, Ev]
    // block_tables shape: [W, max_blocks_per_seq]

    int W = shape.W;
    int Hq = shape.Hq;
    int S = shape.S; // S=1 for decode
    assert(E == shape.E);
    assert(Ev == shape.Ev);

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);
    auto sub_warp = cg::tiled_partition<SubWarpSize>(block);
    constexpr const int SubWarpMetaSize = 256 / SubWarpSize;

    ptrdiff_t q_stride = E * S * Hq * W;
    extern __shared__ float scratch[];

    // adjust scale so we can use base-2 exponent later on
    float l2scale = scale / std::log(2.f);

    int hkv = blockIdx.x;
    int w = blockIdx.y % W;
    int s = blockIdx.y / W;
    int split = blockIdx.z;
    int splits = gridDim.z;

    int hq = hkv * GQA;
    ptrdiff_t q_offset = ((w * Hq + hq) * S + s) * E;

    constexpr const int VecSize = 16 / sizeof(scalar_t);
    constexpr int VPH_k = E / (SubWarpSize * VecSize);   // vectors per head per thread
    constexpr int VPH_v = Ev / (SubWarpSize * VecSize);  // vectors per head per thread

    using full_vec_t = GenericVector<scalar_t, VecSize>;
    using full_fvec_t = GenericVector<float, VecSize>;
    using qk_cache_t = GenericVector<float, E / SubWarpSize>;
    qk_cache_t q_cache[GQA];

    using v_acc_t = GenericVector<float, Ev / SubWarpSize>;
    v_acc_t v_accumulator[GQA];
    float maximum[GQA];
    for (int gqa = 0; gqa < GQA; ++gqa) {
        v_accumulator[gqa] = v_acc_t::zeros();
        maximum[gqa] = std::numeric_limits<float>::lowest();
    }

    float lse[GQA] = {};

    full_vec_t* keys_lookahead = reinterpret_cast<full_vec_t*>(scratch);
    full_vec_t* vals_lookahead = keys_lookahead + 2 * VPH_k * 256;

    int turn_token_offset = 0;

    for (int f = 0; f < shape.F; ++f) {
        int q_loc = locations[(f * W + w) * S + s];
        int L = fragment_lengths[f];
        int maxL = std::min(L, q_loc + 1);

        for (int gqa = 0; gqa < GQA; ++gqa) {
            for (int ee = 0; ee < VPH_k; ++ee) {
                int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize;
                full_vec_t qv = full_vec_t::load(queries + f * q_stride + q_offset + gqa * S * E + e);
                for (int j = 0; j < VecSize; ++j) {
                    q_cache[gqa][ee * VecSize + j] = qv[j];
                }
            }
        }

        const int StepSize = SubWarpMetaSize * splits;

        auto ldg_sts = [&](int stage, int l) {
            if (l >= maxL) return;

            int absolute_key_pos = turn_token_offset + l;
            int block_in_seq_idx = absolute_key_pos / shape.block_size;
            int offset_in_block = absolute_key_pos % shape.block_size;
            int physical_block_id = block_tables[w * shape.max_blocks_per_seq + block_in_seq_idx];

            size_t k_block_stride = (size_t)shape.block_size * shape.Hkv * E;
            size_t k_token_stride = (size_t)shape.Hkv * E;
            const scalar_t* key_block_ptr = k_cache + (size_t)physical_block_id * k_block_stride;
            const scalar_t* key_token_ptr = key_block_ptr + (size_t)offset_in_block * k_token_stride;

            size_t v_block_stride = (size_t)shape.block_size * shape.Hkv * Ev;
            size_t v_token_stride = (size_t)shape.Hkv * Ev;
            const scalar_t* val_block_ptr = v_cache + (size_t)physical_block_id * v_block_stride;
            const scalar_t* val_token_ptr = val_block_ptr + (size_t)offset_in_block * v_token_stride;

            for (int ee = 0; ee < VPH_k; ++ee) {
                int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize;
                __pipeline_memcpy_async(keys_lookahead + (stage * VPH_k + ee) * 256 + threadIdx.x,
                                        key_token_ptr + hkv * E + e, sizeof(full_vec_t));
            }
            for (int ee = 0; ee < VPH_v; ++ee) {
                int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize;
                 __pipeline_memcpy_async(vals_lookahead + (stage * VPH_v + ee) * 256 + threadIdx.x,
                                        val_token_ptr + hkv * Ev + e, sizeof(full_vec_t));
            }
        };

        int stage = 0;
        ldg_sts(0, sub_warp.meta_group_rank() * splits + split);
        __pipeline_commit();
        ldg_sts(1, sub_warp.meta_group_rank() * splits + split + StepSize);
        __pipeline_commit();

        for (int ll = split; ll < maxL; ll += StepSize) {
            int l = ll + sub_warp.meta_group_rank() * splits;
            qk_cache_t keys;
            v_acc_t vals;
            __pipeline_wait_prior(1);
            if (l >= maxL) continue;
            unsigned mask = __activemask();

            for (int ee = 0; ee < VPH_k; ++ee) {
                full_vec_t tmp = keys_lookahead[(stage * VPH_k + ee) * 256 + threadIdx.x];
                for (int j = 0; j < VecSize; ++j) {
                    keys[ee * VecSize + j] = (float)tmp[j];
                }
            }
            for (int ee = 0; ee < VPH_v; ++ee) {
                full_vec_t tmp = vals_lookahead[(stage * VPH_v + ee) * 256 + threadIdx.x];
                for (int j = 0; j < VecSize; ++j) {
                    vals[ee * VecSize + j] = (float)tmp[j];
                }
            }

            ldg_sts((stage + 2) % 2, l + 2 * StepSize);
            stage = (stage + 1) % 2;
            __pipeline_commit();

            float qk[GQA] = {};
            #pragma unroll
            for (int gqa = 0; gqa < GQA; ++gqa) {
                for (int ee = 0; ee < VPH_k; ++ee) {
                    for (int j = 0; j < VecSize; ++j) {
                        qk[gqa] += q_cache[gqa][ee * VecSize + j] * keys[ee * VecSize + j];
                    }
                }
            }

            #pragma unroll
            for (int gqa = 0; gqa < GQA; ++gqa) {
                qk[gqa] += __shfl_xor_sync(mask, qk[gqa], 0b0100, 8);
                qk[gqa] += __shfl_xor_sync(mask, qk[gqa], 0b0010, 8);
                qk[gqa] += __shfl_xor_sync(mask, qk[gqa], 0b0001, 8);
            }

            #pragma unroll
            for (int gqa = 0; gqa < GQA; ++gqa) {
                if (qk[gqa] > maximum[gqa]) {
                    float rescale = std::exp2f(l2scale * (maximum[gqa] - qk[gqa]));
                    for (int j = 0; j < v_acc_t::size; ++j) {
                        v_accumulator[gqa][j] *= rescale;
                    }
                    lse[gqa] *= rescale;
                    maximum[gqa] = qk[gqa];
                }
                float att = std::exp2f(l2scale * (qk[gqa] - maximum[gqa]));
                lse[gqa] += att;

                for (int ee = 0; ee < VPH_v; ++ee) {
                    for (int j = 0; j < VecSize; ++j) {
                        v_accumulator[gqa][ee * VecSize + j] += att * vals[ee * VecSize + j];
                    }
                }
            }
        }
        __pipeline_wait_prior(0);
        turn_token_offset += L;
    }

    using vec_t = GenericVector<scalar_t, 4>;
    using fvec_t = GenericVector<float, 4>;
    using stats_t = GenericVector<float, 2>;

    __syncthreads();
    #pragma unroll
    for (int gqa = 0; gqa < GQA; ++gqa) {
        if (sub_warp.thread_rank() == 0) {
            stats_t data;
            data[0] = maximum[gqa];
            data[1] = lse[gqa];
            data.store(scratch + 2 * sub_warp.meta_group_rank() + 2 * WarpSize * gqa);
        }
    }

    __syncthreads();
    #pragma unroll
    for (int gqa = 0; gqa < GQA; ++gqa) {
        float r_max = maximum[gqa];
        float l_max = maximum[gqa];
        float r_lse = 0;
        if (warp.thread_rank() < SubWarpMetaSize) {
            stats_t data = stats_t::load(scratch + 2 * warp.thread_rank() + 2 * WarpSize * gqa);
            r_max = data[0];
            r_lse = data[1];
        }

        maximum[gqa] = cg::reduce(warp, r_max, cg::greater<float>{});
        r_lse *= std::exp2f(l2scale * (r_max - maximum[gqa]));
        lse[gqa] = cg::reduce(warp, r_lse, cg::plus<float>{});

        if (lse[gqa] != 0) {
            float rescale = std::exp2f(l2scale * (l_max - maximum[gqa])) / lse[gqa];
            for (int j = 0; j < v_acc_t::size; ++j) {
                v_accumulator[gqa][j] *= rescale;
            }
        }

        if (threadIdx.x == 0) {
            stats_t data;
            data[0] = maximum[gqa];
            data[1] = lse[gqa];
            data.store(scratch + GQA * 256 / WarpSize * Ev + gqa * 2);
        }

        for (int ee = 0; ee < VPH_v; ++ee) {
            for (int j = 0; j < VecSize; ++j) {
                float v = v_accumulator[gqa][ee * VecSize + j];
                static_assert(SubWarpSize == 8);
                v += __shfl_xor_sync(0xffffffff, v, 0b10000, WarpSize);
                v += __shfl_xor_sync(0xffffffff, v, 0b01000, WarpSize);
                v_accumulator[gqa][ee * VecSize + j] = v;
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for (int gqa = 0; gqa < GQA; ++gqa) {
        if (sub_warp.meta_group_rank() % (WarpSize / SubWarpSize) == 0) {
            for (int ee = 0; ee < VPH_v; ++ee) {
                int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize;
                full_fvec_t store;
                for (int j = 0; j < VecSize; ++j) {
                    store[j] = v_accumulator[gqa][ee * VecSize + j];
                }
                store.store(scratch + e + Ev * sub_warp.meta_group_rank() / (WarpSize / SubWarpSize) + gqa * 256 / WarpSize * Ev);
            }
        }
    }
    __syncthreads();

    for (int gqa_offset = 0; gqa_offset < GQA; gqa_offset += (blockDim.x / WarpSize)) {
        int gqa = warp.meta_group_rank() + gqa_offset;
        if (gqa >= GQA) {
            continue;
        }

        int h = hkv * GQA + gqa;
        int res_base = ((w * Hq + h) * S + s);
        int res_inc = W * Hq * S;
        int res_idx = res_base + split * res_inc;
        float* global_accumulator = reinterpret_cast<float*>(workspace);
        float* lse_target = global_accumulator + W * Hq * S * Ev * splits;

        stats_t data = stats_t::load(scratch + GQA * 256 / WarpSize * Ev + gqa * 2);
        float own_lse = data[1];
        float own_max = data[0];
        own_lse = std::log2(own_lse) + l2scale * own_max;

        for (int e = vec_t::size * warp.thread_rank(); e < Ev; e += vec_t::size * warp.size()) {
            // merge the local results
            fvec_t res = fvec_t::zeros();
            for (int j = 0; j < SubWarpMetaSize / (WarpSize / SubWarpSize); ++j) {
                fvec_t sv = fvec_t::load(scratch + e + Ev * j + gqa * 256 / WarpSize * Ev);
                for (int jj = 0; jj < vec_t::size; ++jj) {
                    res[jj] += sv[jj];
                }
            }
            res.store(global_accumulator + res_idx * Ev + e);
        }

        lse_target[res_idx] = own_lse;
    }
}

template<int Ev, class scalar_t>
__global__ __launch_bounds__(32) void dvattn_paged_attention_reduce_kernel(
        scalar_t* out, const float* v_buffer, const float* lse_buffer, int splits, Shape shape) {
    int h = blockIdx.x;
    int w = blockIdx.y % shape.W;
    int s = blockIdx.y / shape.W;

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    using v_acc_t = GenericVector<float, Ev / warp.size()>;
    v_acc_t v_accumulator = v_acc_t::zeros();

    using vec_t = GenericVector<scalar_t, 4>;
    using fvec_t = GenericVector<float, 4>;

    float own_lse = std::numeric_limits<float>::lowest();

    for (int split = 0; split < splits; ++split) {
        int res_idx = ((w * shape.Hq + h) * shape.S + s) + split * (shape.W * shape.Hq * shape.S);
        const float* split_res = v_buffer + (size_t)res_idx * Ev;
        float res_lse = lse_buffer[res_idx];

        if (res_lse <= std::numeric_limits<float>::lowest()) {
            continue;
        }

        if (own_lse <= std::numeric_limits<float>::lowest()) {
            #pragma unroll
            for (int ee = 0; ee < Ev / warp.size(); ee += fvec_t::size) {
                 int e = ee * warp.size() + warp.thread_rank() * fvec_t::size;
                fvec_t sv = fvec_t::load_lu(split_res + e);
                for (int jj = 0; jj < fvec_t::size; ++jj) {
                    v_accumulator[ee + jj] = sv[jj];
                }
            }
            own_lse = res_lse;
            continue;
        }

        float max_lse = std::max(own_lse, res_lse);
        float sa = std::exp2f(own_lse - max_lse);
        float sb = std::exp2f(res_lse - max_lse);
        float sum_ab = sa + sb;
        float rescaler_a = sa / sum_ab;
        float rescaler_b = sb / sum_ab;

        #pragma unroll
        for (int ee = 0; ee < Ev / warp.size(); ee += fvec_t::size) {
            int e = ee * warp.size() + warp.thread_rank() * fvec_t::size;
            fvec_t sv = fvec_t::load_lu(split_res + e);
            for (int jj = 0; jj < fvec_t::size; ++jj) {
                float old = v_accumulator[ee + jj];
                v_accumulator[ee + jj] = old * rescaler_a + sv[jj] * rescaler_b;
            }
        }
        own_lse = std::log2(sum_ab) + max_lse;
    }

    for (int ee = 0; ee < Ev / warp.size(); ee += fvec_t::size) {
        int e = ee * warp.size() + warp.thread_rank() * fvec_t::size;
        vec_t st = vec_t::zeros();
        for (int jj = 0; jj < fvec_t::size; ++jj) {
            st[jj] = (scalar_t)v_accumulator[ee + jj];
        }
        st.store(out + ((w * shape.Hq + h) * shape.S + s) * Ev + e);
    }
}


template<class scalar_t>
cudaError_t dvattn_paged_attention_gpu(scalar_t* out, float scale,
                           const int* locations, const scalar_t* queries,
                           const int* fragment_lengths,
                           const scalar_t* k_cache,
                           const scalar_t* v_cache,
                           const int* block_tables,
                           Shape shape) {
    int problem_size = shape.Hkv * shape.W * shape.S;
    if (problem_size == 0) return cudaSuccess;
    int sms = -1;
    CUDA_RETURN_ON_ERROR(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    int splits = max(2, sms / problem_size);

    dim3 grid_dim{(unsigned)shape.Hkv, (unsigned)shape.W * (unsigned)shape.S, (unsigned)splits};
    dim3 block_dim{256, 1, 1};
    size_t smem = shape.Ev * sizeof(float) * block_dim.x / 32 * (shape.Hq / shape.Hkv);
    smem += 2 * sizeof(float) * (shape.Hq / shape.Hkv);
    smem = std::max(smem, 2 * (shape.E + shape.Ev) * (block_dim.x / SubWarpSize) * sizeof(scalar_t));

    static char* workspace = nullptr;
    static std::size_t workspace_size = 0;

    size_t required_workspace_elems = (size_t)shape.W * shape.Hq * shape.S * splits;
    size_t required_bytes = required_workspace_elems * (shape.Ev + 1) * sizeof(float);
    if (workspace_size < required_bytes) {
        if (workspace)
            CUDA_RETURN_ON_ERROR(cudaFree(workspace));
        CUDA_RETURN_ON_ERROR(cudaMalloc(&workspace, required_bytes));
        workspace_size = required_bytes;
    }
    CUDA_RETURN_ON_ERROR(cudaMemset(workspace, 0, required_bytes));

    float* v_buffer = reinterpret_cast<float*>(workspace);
    float* lse_buffer = v_buffer + required_workspace_elems * shape.Ev;

    if (shape.E == 128 && shape.Ev == 128 && (shape.Hq / shape.Hkv) == 16) {
        CUDA_RETURN_ON_ERROR(cudaFuncSetAttribute(dvattn_paged_attention_gpu_kernel21<128, 128, 16, scalar_t>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        dvattn_paged_attention_gpu_kernel21<128, 128, 16><<<grid_dim, block_dim, smem>>>(
                out, workspace, scale, locations, queries, fragment_lengths,
                k_cache, v_cache, block_tables, shape);
        CUDA_CHECK_THROW(cudaGetLastError());

        dim3 r_grid_dim{(unsigned)shape.Hq, (unsigned)shape.W * (unsigned)shape.S, 1};
        dim3 r_block_dim{32};
        dvattn_paged_attention_reduce_kernel<128><<<r_grid_dim, r_block_dim>>>(
                out, v_buffer, lse_buffer, splits, shape);
        CUDA_CHECK_THROW(cudaGetLastError());
    } else {
        printf("Unsupported head dimension for paged attention kernel");
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

}  // namespace paged_v21
