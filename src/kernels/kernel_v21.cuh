// move reduction back into a separate kernel.
// in terms of pure kernel times, this is faster
#include "common.h"
#include "vec.cuh"
#include "cuda_check.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_pipeline_primitives.h>

namespace cg = cooperative_groups;

namespace v21
{

__device__ inline void find_seq_idx(int& w, int& s, int global_q_idx, const int* cu_seqlens_q, int num_seqs) {
    // A simple linear scan should be fast enough given typical batch sizes.
    // For very large batches, a binary search could be used.
    w = 0;
    while (w < num_seqs && global_q_idx >= cu_seqlens_q[w+1]) {
        w++;
    }
    s = global_q_idx - cu_seqlens_q[w];
}

constexpr const int SubWarpSize = 8;
constexpr const int WarpSize = 32;

template<int E, int Ev, int GQA, class scalar_t>
__global__ __launch_bounds__(256) void dvattn_attention_gpu_kernel21(
        scalar_t* out, char* workspace, float scale,
        const int* locations, const scalar_t* queries,
        const float* cosines, const float* sines, int rotary_dim,
        const int* fragment_lengths,
        const scalar_t* const* key_fragments,
        const scalar_t* const* value_fragments,
        Shape shape) {
    // Input:   keys: [Hkv, fragment_lengths[i], E] for i in [F]
    //          values: [Hkv, fragment_lengths[i], Ev] for i in [F]
    //          fragment_lengths: [F]
    //          queries: [F, W, Hq, S, E]
    //          locations [F, W, S]
    // Scratch: workspace [W, Hq, S, Ev] (in float32, iff scalar_t != float32) + [W, Hq, S] BlockResult
    // Output:  [W, Hq, S, Ev]
    // attention mask: s attends to l iff locations[b, s] >= l (i.e., shifted causal masking)

    int W = shape.W;
    int Hq = shape.Hq;
    int S = shape.S;
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

    // combine values
    using v_cache_t = GenericVector<float, Ev / SubWarpSize>;
    v_cache_t v_cache[GQA];
    float maximum[GQA];
    for (int gqa = 0; gqa < GQA; ++gqa) {
        v_cache[gqa] = v_cache_t::zeros();
        maximum[gqa] = std::numeric_limits<float>::lowest();
    }

    // determine maximum and online logsumexp
    float lse[GQA] = {};
    {
        full_vec_t* keys_lookahead = reinterpret_cast<full_vec_t*>(scratch);
        full_vec_t* vals_lookahead = keys_lookahead + 2 * VPH_k * 256;

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

            const scalar_t* value_fragment = value_fragments[f];
            const scalar_t* key_fragment = key_fragments[f];

            const int StepSize = SubWarpMetaSize * splits;
            auto ldg_sts = [&](int stage, int l) {
                if (l >= maxL) return;
                ptrdiff_t k_offset = (hkv * L + l) * E;
                ptrdiff_t v_offset = (hkv * L + l) * Ev;
                for (int ee = 0; ee < VPH_k; ++ee) {
                    int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize;
                    __pipeline_memcpy_async(keys_lookahead + (stage * VPH_k + ee) * 256 + threadIdx.x,
                                            key_fragment + k_offset + e, sizeof(full_vec_t));
                }
                for (int ee = 0; ee < VPH_v; ++ee) {
                    int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize;
                    __pipeline_memcpy_async(vals_lookahead + (stage * VPH_v + ee) * 256 + threadIdx.x,
                                            value_fragment + v_offset + e, sizeof(full_vec_t));
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
                v_cache_t vals;
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

                // important: By having the warp shuffles together like this in a separate loop,
                // the compiler ends up generating better sequenced assembly, where we first initiate a
                // bunch of shuffles and only then do the addition, hiding the latency much better
                // than in the single-loop version.
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
                        for (int j = 0; j < v_cache_t::size; ++j) {
                            v_cache[gqa][j] *= rescale;
                        }
                        lse[gqa] *= rescale;
                        maximum[gqa] = qk[gqa];
                    }
                    float att = std::exp2f(l2scale * (qk[gqa] - maximum[gqa]));
                    lse[gqa] += std::exp2f(l2scale * (qk[gqa] - maximum[gqa]));

                    for (int ee = 0; ee < VPH_v; ++ee) {
                        for (int j = 0; j < VecSize; ++j) {
                            v_cache[gqa][ee * VecSize + j] += att * vals[ee * VecSize + j];
                        }
                    }
                }
            }
            __pipeline_wait_prior(0);
        }
    }

    using vec_t = GenericVector<scalar_t, 4>;
    using fvec_t = GenericVector<float, 4>;
    using stats_t = GenericVector<float, 2>;

    __syncthreads();
    #pragma unroll
    for (int gqa = 0; gqa < GQA; ++gqa) {
        // combine split-k results
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

        // Note: It *is* possible that no thread in this warp had any valid position (due to causal masking),
        // which would lead to division by zero -> 0 * inf = NaN here.
        if (lse[gqa] != 0) {
            float rescale = std::exp2f(l2scale * (l_max - maximum[gqa])) / lse[gqa];
            for (int j = 0; j < v_cache_t::size; ++j) {
                v_cache[gqa][j] *= rescale;
            }
        }

        if (threadIdx.x == 0) {
            stats_t data;
            data[0] = maximum[gqa];
            data[1] = lse[gqa];
            data.store(scratch + GQA * 256 / WarpSize * Ev + gqa * 2);
        }

        // now reduce value across subwarp within a warp
        for (int ee = 0; ee < VPH_v; ++ee) {
            for (int j = 0; j < VecSize; ++j) {
                float v = v_cache[gqa][ee * VecSize + j];
                static_assert(SubWarpSize == 8);
                v += __shfl_xor_sync(0xffffffff, v, 0b10000, WarpSize);
                v += __shfl_xor_sync(0xffffffff, v, 0b01000, WarpSize);
                v_cache[gqa][ee * VecSize + j] = v;
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
                    store[j] = v_cache[gqa][ee * VecSize + j];
                }
                store.store(scratch + e + Ev * sub_warp.meta_group_rank() / (WarpSize / SubWarpSize) + gqa * 256 / WarpSize * Ev);
            }
        }
    }
    __syncthreads();

    // Let all warps in the block collaborate to reduce all GQA groups.
    // Each warp (ID 0-7 for a 256-thread block) will handle a different GQA group in each iteration of the loop.
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

template<int E, int Ev, int GQA, class scalar_t>
__global__ __launch_bounds__(256) void dvattn_paged_attention_gpu_kernel21(
        scalar_t* out, char* workspace, float scale,
        const int* locations, const scalar_t* queries,
        const float* cosines, const float* sines, int rotary_dim,
        const int* fragment_lengths, // Corresponds to sequence lengths
        const scalar_t* key_cache,
        const scalar_t* value_cache,
        const int* block_table,
        Shape shape) {
    // Input:   key_cache/value_cache: [NumBlocks, Hkv, BlockSize, E/Ev]
    //          block_table: [W, max_blocks_per_seq]  (W is batch size)
    //          queries: [W, Hq, S, E]
    //          locations/cosines/sines: [F, W, S, ...]
    // Output:  [W, Hq, S, Ev]

    int W = shape.W;
    int Hq = shape.Hq;
    int S = shape.S;
    int block_size = shape.block_size;
    int max_blocks = shape.max_blocks_per_seq;
    assert(E == shape.E);
    assert(Ev == shape.Ev);

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);
    auto sub_warp = cg::tiled_partition<SubWarpSize>(block);
    constexpr const int SubWarpMetaSize = 256 / SubWarpSize;

    extern __shared__ float scratch[];

    // adjust scale so we can use base-2 exponent later on
    float l2scale = scale / std::log(2.f);

    int hkv = blockIdx.x;
    int w = blockIdx.y % W;
    int s = blockIdx.y / W;
    int split = blockIdx.z;
    int splits = gridDim.z;

    int hq = hkv * GQA;

    constexpr const int VecSize = 16 / sizeof(scalar_t);
    constexpr int VPH_k = E / (SubWarpSize * VecSize);   // vectors per head per thread
    constexpr int VPH_v = Ev / (SubWarpSize * VecSize);  // vectors per head per thread

    using full_vec_t = GenericVector<scalar_t, VecSize>;
    using full_fvec_t = GenericVector<float, VecSize>;
    using qk_cache_t = GenericVector<float, E / SubWarpSize>; // E / 8

    // Load unrotated query once
    qk_cache_t unrotated_q_cache[GQA];
    ptrdiff_t q_base_offset = ((w * Hq + hq) * S + s) * E;

    #pragma unroll
    for (int gqa = 0; gqa < GQA; ++gqa) {
        #pragma unroll
        for (int ee = 0; ee < VPH_k; ++ee) {
            full_vec_t qv = full_vec_t::load(queries + q_base_offset + gqa * S * E + (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize);
            #pragma unroll
            for(int v=0; v < VecSize; ++v) {
                unrotated_q_cache[gqa][ee * VecSize + v] = (float)qv[v];
            }
        }
    }

    // combine values
    using v_cache_t = GenericVector<float, Ev / SubWarpSize>;
    v_cache_t v_cache[GQA];
    float maximum[GQA];
    for (int gqa = 0; gqa < GQA; ++gqa) {
        v_cache[gqa] = v_cache_t::zeros();
        maximum[gqa] = std::numeric_limits<float>::lowest();
    }

    // determine maximum and online logsumexp
    float lse[GQA] = {};
    {
        full_vec_t* keys_lookahead = reinterpret_cast<full_vec_t*>(scratch);
        full_vec_t* vals_lookahead = keys_lookahead + 2 * VPH_k * 256;

        for (int f = 0; f < shape.F; ++f) {
            int q_loc = locations[(f * W + w) * S + s];
            int L = fragment_lengths[f];
            int maxL = std::min(L, q_loc + 1);

            qk_cache_t q_cache[GQA];
            const float* cos_ptr = cosines + ((f * W + w) * S + s) * rotary_dim;
            const float* sin_ptr = sines + ((f * W + w) * S + s) * rotary_dim;

            #pragma unroll
            for (int gqa = 0; gqa < GQA; ++gqa) {
                #pragma unroll
                for (int ee = 0; ee < VPH_k; ++ee) {
                    #pragma unroll
                    for (int v = 0; v < VecSize; ++v) {
                        int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize + v;
                        float val = unrotated_q_cache[gqa][ee * VecSize + v];

                        if (e >= rotary_dim) {
                            q_cache[gqa][ee * VecSize + v] = val;
                            continue;
                        }

                        int e_pair = (e < rotary_dim / 2) ? (e + rotary_dim / 2) : (e - rotary_dim / 2);
                        int pair_sub_warp_rank = (e_pair / VecSize) % SubWarpSize;
                        int pair_ee = e_pair / (SubWarpSize * VecSize);
                        int pair_v = e_pair % VecSize;

                        float pair_val = __shfl_sync(0xffffffff, unrotated_q_cache[gqa][pair_ee * VecSize + pair_v], pair_sub_warp_rank, SubWarpSize);

                        if (e < rotary_dim / 2) {
                            float cos_val = __ldg(cos_ptr + e);
                            float sin_val = __ldg(sin_ptr + e);
                            q_cache[gqa][ee * VecSize + v] = val * cos_val - pair_val * sin_val;
                        } else {
                            float cos_val = __ldg(cos_ptr + e_pair);
                            float sin_val = __ldg(sin_ptr + e_pair);
                            q_cache[gqa][ee * VecSize + v] = val * cos_val + pair_val * sin_val;
                        }
                    }
                }
            }

            // MODIFICATION: Use block_table to calculate addresses instead of fragment pointers
            const int StepSize = SubWarpMetaSize * splits;
            auto ldg_sts = [&](int stage, int l) {
                if (l >= maxL) return;

                // Paged attention address calculation
                int block_idx_in_seq = l / block_size;
                int offset_in_block = l % block_size;
                int physical_block_id = block_table[w * max_blocks + block_idx_in_seq];

                ptrdiff_t k_offset = (((ptrdiff_t)physical_block_id * shape.Hkv + hkv) * block_size + offset_in_block) * E;
                ptrdiff_t v_offset = (((ptrdiff_t)physical_block_id * shape.Hkv + hkv) * block_size + offset_in_block) * Ev;

                for (int ee = 0; ee < VPH_k; ++ee) {
                    int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize;
                    __pipeline_memcpy_async(keys_lookahead + (stage * VPH_k + ee) * 256 + threadIdx.x,
                                            key_cache + k_offset + e, sizeof(full_vec_t));
                }
                for (int ee = 0; ee < VPH_v; ++ee) {
                    int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize;
                    __pipeline_memcpy_async(vals_lookahead + (stage * VPH_v + ee) * 256 + threadIdx.x,
                                            value_cache + v_offset + e, sizeof(full_vec_t));
                }
            };

            int stage = 0;
            ldg_sts(0, sub_warp.meta_group_rank() * splits + split);
            __pipeline_commit();
            ldg_sts(1, sub_warp.meta_group_rank() * splits + split + StepSize);
            __pipeline_commit();

            for (int ll = split; ll < maxL; ll += StepSize) {
                // This part of the loop is identical to the original kernel
                int l = ll + sub_warp.meta_group_rank() * splits;
                qk_cache_t keys;
                v_cache_t vals;
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
                        for (int j = 0; j < v_cache_t::size; ++j) {
                            v_cache[gqa][j] *= rescale;
                        }
                        lse[gqa] *= rescale;
                        maximum[gqa] = qk[gqa];
                    }
                    float att = std::exp2f(l2scale * (qk[gqa] - maximum[gqa]));
                    lse[gqa] += std::exp2f(l2scale * (qk[gqa] - maximum[gqa]));

                    for (int ee = 0; ee < VPH_v; ++ee) {
                        for (int j = 0; j < VecSize; ++j) {
                            v_cache[gqa][ee * VecSize + j] += att * vals[ee * VecSize + j];
                        }
                    }
                }
            }
            __pipeline_wait_prior(0);
        }
    }

    // The reduction part is identical to the original kernel
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
            for (int j = 0; j < v_cache_t::size; ++j) {
                v_cache[gqa][j] *= rescale;
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
                float v = v_cache[gqa][ee * VecSize + j];
                static_assert(SubWarpSize == 8);
                v += __shfl_xor_sync(0xffffffff, v, 0b10000, WarpSize);
                v += __shfl_xor_sync(0xffffffff, v, 0b01000, WarpSize);
                v_cache[gqa][ee * VecSize + j] = v;
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
                    store[j] = v_cache[gqa][ee * VecSize + j];
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
__global__ __launch_bounds__(32) void dvattn_attention_reduce_kernel(
        scalar_t* out, const float* v_buffer, const float* lse_buffer, int splits, Shape shape) {
    int h = blockIdx.x;
    int w = blockIdx.y % shape.W;
    int s = blockIdx.y / shape.W;

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    using v_cache_t = GenericVector<float, Ev / warp.size()>;
    v_cache_t v_cache = v_cache_t::zeros();

    using vec_t = GenericVector<scalar_t, 4>;
    using fvec_t = GenericVector<float, 4>;

    float own_lse = std::numeric_limits<float>::lowest();

    for (int split = 0; split < splits; ++split) {
        int res_idx = ((split * shape.W + w) * shape.Hq + h) * shape.S + s;
        const float* split_res = v_buffer + res_idx * Ev;
        float res_lse = lse_buffer[res_idx];
        if (res_lse == std::numeric_limits<float>::lowest()) {
            continue;
        }
        float max = std::max(own_lse, res_lse);
        float sa = std::exp2f(own_lse - max);
        float sb = std::exp2f(res_lse - max);
        float rescaler_a = sa / (sa + sb);
        float rescaler_b = sb / (sa + sb);
        #pragma unroll
        for (int ee = 0; ee < Ev / warp.size(); ee += fvec_t::size) {
            int e = ee * warp.size() + warp.thread_rank() * fvec_t::size;
            fvec_t sv = fvec_t::load_lu(split_res + e);
            for (int jj = 0; jj < fvec_t::size; ++jj) {
                float old = v_cache[ee + jj];
                float upd = old * rescaler_a + sv[jj] * rescaler_b;
                v_cache[ee + jj] = upd;
            }
        }
        own_lse = std::log2(sa + sb) + max;
    }

    for (int ee = 0; ee < Ev / warp.size(); ee += fvec_t::size) {
        int e = ee * warp.size() + warp.thread_rank() * fvec_t::size;
        vec_t st = vec_t::zeros();
        for (int jj = 0; jj < fvec_t::size; ++jj) {
            st[jj] = (scalar_t)v_cache[ee + jj];
        }
        st.store(out + ((w * shape.Hq + h) * shape.S + s) * Ev + e);
    }
}

template<int E, int Ev, int GQA, class scalar_t>
__global__ __launch_bounds__(256) void dvattn_varlen_paged_attention_gpu_kernel21(
        scalar_t* out, char* workspace, float scale,
        const int* locations, const scalar_t* queries,
        const float* cosines, const float* sines, int rotary_dim,
        const int* fragment_lengths, // Corresponds to sequence lengths
        const scalar_t* key_cache,
        const scalar_t* value_cache,
        const int* block_table,
        Shape shape) {
    // Varlen version of paged attention kernel.
    // Grid.y is total_q_tokens.
    // queries: [total_q_tokens, Hq, E]
    // locations/cosines/sines: [F, total_q_tokens, ...]

    int W = shape.W;
    int Hq = shape.Hq;
    int total_q_tokens = shape.total_q_tokens;
    int block_size = shape.block_size;
    int max_blocks = shape.max_blocks_per_seq;
    assert(E == shape.E);
    assert(Ev == shape.Ev);

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);
    auto sub_warp = cg::tiled_partition<SubWarpSize>(block);
    constexpr const int SubWarpMetaSize = 256 / SubWarpSize;

    extern __shared__ float scratch[];

    float l2scale = scale / std::log(2.f);

    int hkv = blockIdx.x;
    int global_q_idx = blockIdx.y;
    int split = blockIdx.z;
    int splits = gridDim.z;

    int w, s; // batch_idx, seq_idx within batch
    find_seq_idx(w, s, global_q_idx, shape.cu_seqlens_q, W);

    int hq = hkv * GQA;
    // queries are packed: [total_q_tokens, Hq, E]
    ptrdiff_t q_base_offset = ((ptrdiff_t)global_q_idx * Hq + hq) * E;

    constexpr const int VecSize = 16 / sizeof(scalar_t);
    constexpr int VPH_k = E / (SubWarpSize * VecSize);
    constexpr int VPH_v = Ev / (SubWarpSize * VecSize);

    using full_vec_t = GenericVector<scalar_t, VecSize>;
    using full_fvec_t = GenericVector<float, VecSize>;
    using qk_cache_t = GenericVector<float, E / SubWarpSize>;

    using v_cache_t = GenericVector<float, Ev / SubWarpSize>;
    v_cache_t v_cache[GQA];
    float maximum[GQA];

    // Load unrotated query once before the fragment loop
    qk_cache_t unrotated_q_cache[GQA];
    #pragma unroll
    for (int gqa = 0; gqa < GQA; ++gqa) {
        #pragma unroll
        for (int ee = 0; ee < VPH_k; ++ee) {
            full_vec_t qv = full_vec_t::load(queries + q_base_offset + gqa * E + (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize);
            #pragma unroll
            for(int v=0; v < VecSize; ++v) {
                unrotated_q_cache[gqa][ee * VecSize + v] = (float)qv[v];
            }
        }
    }

    for (int gqa = 0; gqa < GQA; ++gqa) {
        v_cache[gqa] = v_cache_t::zeros();
        maximum[gqa] = std::numeric_limits<float>::lowest();
    }

    float lse[GQA] = {};
    {
        full_vec_t* keys_lookahead = reinterpret_cast<full_vec_t*>(scratch);
        full_vec_t* vals_lookahead = keys_lookahead + 2 * VPH_k * 256;

        for (int f = 0; f < shape.F; ++f) {
            // locations are packed: [F, total_q_tokens]
            int q_loc = locations[f * total_q_tokens + global_q_idx];
            int L = fragment_lengths[f * W + w]; // fragment_lengths is [F, W]
            int maxL = std::min(L, q_loc + 1);

            qk_cache_t q_cache[GQA];
            const float* cos_ptr = cosines + ((ptrdiff_t)f * total_q_tokens + global_q_idx) * rotary_dim;
            const float* sin_ptr = sines + ((ptrdiff_t)f * total_q_tokens + global_q_idx) * rotary_dim;

            #pragma unroll
            for (int gqa = 0; gqa < GQA; ++gqa) {
                #pragma unroll
                for (int ee = 0; ee < VPH_k; ++ee) {
                    #pragma unroll
                    for (int v = 0; v < VecSize; ++v) {
                        int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize + v;
                        float val = unrotated_q_cache[gqa][ee * VecSize + v];

                        if (e >= rotary_dim) {
                            q_cache[gqa][ee * VecSize + v] = val;
                            continue;
                        }

                        int e_pair = (e < rotary_dim / 2) ? (e + rotary_dim / 2) : (e - rotary_dim / 2);
                        int pair_sub_warp_rank = (e_pair / VecSize) % SubWarpSize;
                        int pair_ee = e_pair / (SubWarpSize * VecSize);
                        int pair_v = e_pair % VecSize;

                        float pair_val = __shfl_sync(0xffffffff, unrotated_q_cache[gqa][pair_ee * VecSize + pair_v], pair_sub_warp_rank, SubWarpSize);

                        if (e < rotary_dim / 2) {
                            q_cache[gqa][ee * VecSize + v] = val * __ldg(cos_ptr + e) - pair_val * __ldg(sin_ptr + e);
                        } else {
                            q_cache[gqa][ee * VecSize + v] = val * __ldg(cos_ptr + e_pair) + pair_val * __ldg(sin_ptr + e_pair);
                        }
                    }
                }
            }

            const int StepSize = SubWarpMetaSize * splits;
            auto ldg_sts = [&](int stage, int l) {
                if (l >= maxL) return;
                int block_idx_in_seq = l / block_size;
                int offset_in_block = l % block_size;
                int physical_block_id = block_table[w * max_blocks + block_idx_in_seq];
                ptrdiff_t k_offset = (((ptrdiff_t)physical_block_id * shape.Hkv + hkv) * block_size + offset_in_block) * E;
                ptrdiff_t v_offset = (((ptrdiff_t)physical_block_id * shape.Hkv + hkv) * block_size + offset_in_block) * Ev;

                for (int ee = 0; ee < VPH_k; ++ee) {
                    int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize;
                    __pipeline_memcpy_async(keys_lookahead + (stage * VPH_k + ee) * 256 + threadIdx.x, key_cache + k_offset + e, sizeof(full_vec_t));
                }
                for (int ee = 0; ee < VPH_v; ++ee) {
                    int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize;
                    __pipeline_memcpy_async(vals_lookahead + (stage * VPH_v + ee) * 256 + threadIdx.x, value_cache + v_offset + e, sizeof(full_vec_t));
                }
            };

            int stage = 0;
            ldg_sts(0, sub_warp.meta_group_rank() * splits + split);
            __pipeline_commit();
            ldg_sts(1, sub_warp.meta_group_rank() * splits + split + StepSize);
            __pipeline_commit();

            for (int ll = split; ll < maxL; ll += StepSize) {
                int l = ll + sub_warp.meta_group_rank() * splits;
                qk_cache_t keys; v_cache_t vals;
                __pipeline_wait_prior(1);
                if (l >= maxL) continue;
                unsigned mask = __activemask();

                for (int ee = 0; ee < VPH_k; ++ee) {
                    full_vec_t tmp = keys_lookahead[(stage * VPH_k + ee) * 256 + threadIdx.x];
                    for (int j = 0; j < VecSize; ++j) { keys[ee * VecSize + j] = (float)tmp[j]; }
                }
                for (int ee = 0; ee < VPH_v; ++ee) {
                    full_vec_t tmp = vals_lookahead[(stage * VPH_v + ee) * 256 + threadIdx.x];
                    for (int j = 0; j < VecSize; ++j) { vals[ee * VecSize + j] = (float)tmp[j]; }
                }

                ldg_sts((stage + 2) % 2, l + 2 * StepSize);
                stage = (stage + 1) % 2;
                __pipeline_commit();

                float qk[GQA] = {};
                #pragma unroll
                for (int gqa = 0; gqa < GQA; ++gqa) {
                    for (int ee = 0; ee < VPH_k; ++ee) {
                        for (int j = 0; j < VecSize; ++j) { qk[gqa] += q_cache[gqa][ee * VecSize + j] * keys[ee * VecSize + j]; }
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
                        for (int j = 0; j < v_cache_t::size; ++j) { v_cache[gqa][j] *= rescale; }
                        lse[gqa] *= rescale;
                        maximum[gqa] = qk[gqa];
                    }
                    float att = std::exp2f(l2scale * (qk[gqa] - maximum[gqa]));
                    lse[gqa] += att;
                    for (int ee = 0; ee < VPH_v; ++ee) {
                        for (int j = 0; j < VecSize; ++j) { v_cache[gqa][ee * VecSize + j] += att * vals[ee * VecSize + j]; }
                    }
                }
            }
            __pipeline_wait_prior(0);
        }
    }

    // Reduction logic is identical to paged attention, just writing to a different global memory layout
    using vec_t = GenericVector<scalar_t, 4>;
    using fvec_t = GenericVector<float, 4>;
    using stats_t = GenericVector<float, 2>;

    __syncthreads();
    #pragma unroll
    for (int gqa = 0; gqa < GQA; ++gqa) {
        if (sub_warp.thread_rank() == 0) {
            stats_t data; data[0] = maximum[gqa]; data[1] = lse[gqa];
            data.store(scratch + 2 * sub_warp.meta_group_rank() + 2 * WarpSize * gqa);
        }
    }
    __syncthreads();
    // Identical reduction logic...
    #pragma unroll
    for (int gqa = 0; gqa < GQA; ++gqa) {
        float r_max = maximum[gqa]; float l_max = maximum[gqa]; float r_lse = 0;
        if (warp.thread_rank() < SubWarpMetaSize) {
            stats_t data = stats_t::load(scratch + 2 * warp.thread_rank() + 2 * WarpSize * gqa);
            r_max = data[0]; r_lse = data[1];
        }
        maximum[gqa] = cg::reduce(warp, r_max, cg::greater<float>{});
        r_lse *= std::exp2f(l2scale * (r_max - maximum[gqa]));
        lse[gqa] = cg::reduce(warp, r_lse, cg::plus<float>{});
        if (lse[gqa] != 0) {
            float rescale = std::exp2f(l2scale * (l_max - maximum[gqa])) / lse[gqa];
            for (int j = 0; j < v_cache_t::size; ++j) { v_cache[gqa][j] *= rescale; }
        }
        if (threadIdx.x == 0) {
            stats_t data; data[0] = maximum[gqa]; data[1] = lse[gqa];
            data.store(scratch + GQA * 256 / WarpSize * Ev + gqa * 2);
        }
        for (int ee = 0; ee < VPH_v; ++ee) {
            for (int j = 0; j < VecSize; ++j) {
                float v = v_cache[gqa][ee * VecSize + j];
                v += __shfl_xor_sync(0xffffffff, v, 0b10000, WarpSize);
                v += __shfl_xor_sync(0xffffffff, v, 0b01000, WarpSize);
                v_cache[gqa][ee * VecSize + j] = v;
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
                for (int j = 0; j < VecSize; ++j) { store[j] = v_cache[gqa][ee * VecSize + j]; }
                store.store(scratch + e + Ev * sub_warp.meta_group_rank() / (WarpSize / SubWarpSize) + gqa * 256 / WarpSize * Ev);
            }
        }
    }
    __syncthreads();

    for (int gqa_offset = 0; gqa_offset < GQA; gqa_offset += (blockDim.x / WarpSize)) {
        int gqa = warp.meta_group_rank() + gqa_offset;
        if (gqa >= GQA) continue;

        int h = hkv * GQA + gqa;
        // The workspace is laid out as [splits, total_q, Hq, Ev]
        int res_base = (global_q_idx * Hq + h);
        int res_inc = total_q_tokens * Hq;
        int res_idx = res_base + split * res_inc;
        float* global_accumulator = reinterpret_cast<float*>(workspace);
        float* lse_target = global_accumulator + (size_t)splits * total_q_tokens * Hq * Ev;

        stats_t data = stats_t::load(scratch + GQA * 256 / WarpSize * Ev + gqa * 2);
        float own_lse = data[1]; float own_max = data[0];
        own_lse = std::log2(own_lse) + l2scale * own_max;

        for (int e = vec_t::size * warp.thread_rank(); e < Ev; e += vec_t::size * warp.size()) {
            fvec_t res = fvec_t::zeros();
            for (int j = 0; j < SubWarpMetaSize / (WarpSize / SubWarpSize); ++j) {
                fvec_t sv = fvec_t::load(scratch + e + Ev * j + gqa * 256 / WarpSize * Ev);
                for (int jj = 0; jj < vec_t::size; ++jj) { res[jj] += sv[jj]; }
            }
            res.store(global_accumulator + (size_t)res_idx * Ev + e);
        }
        lse_target[res_idx] = own_lse;
    }
}


template<int Ev, class scalar_t>
__global__ __launch_bounds__(32) void dvattn_varlen_attention_reduce_kernel(
        scalar_t* out, const float* v_buffer, const float* lse_buffer, int splits, Shape shape) {
    int h = blockIdx.x;
    int global_q_idx = blockIdx.y;

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    using v_cache_t = GenericVector<float, Ev / warp.size()>;
    v_cache_t v_cache = v_cache_t::zeros();

    using vec_t = GenericVector<scalar_t, 4>;
    using fvec_t = GenericVector<float, 4>;

    float own_lse = std::numeric_limits<float>::lowest();

    for (int split = 0; split < splits; ++split) {
        int res_idx = (split * shape.total_q_tokens * shape.Hq) + (global_q_idx * shape.Hq) + h;
        const float* split_res = v_buffer + (size_t)res_idx * Ev;
        float res_lse = lse_buffer[res_idx];
        if (res_lse == std::numeric_limits<float>::lowest()) continue;

        float max_lse = std::max(own_lse, res_lse);
        float scale_a = std::exp2f(own_lse - max_lse);
        float scale_b = std::exp2f(res_lse - max_lse);
        float inv_sum = 1.0f / (scale_a + scale_b);

        #pragma unroll
        for (int ee = 0; ee < Ev / warp.size(); ee += fvec_t::size) {
            int e = ee * warp.size() + warp.thread_rank() * fvec_t::size;
            fvec_t sv = fvec_t::load_lu(split_res + e);
            for (int jj = 0; jj < fvec_t::size; ++jj) {
                v_cache[ee + jj] = (v_cache[ee + jj] * scale_a + sv[jj] * scale_b) * inv_sum;
            }
        }
        own_lse = std::log2(scale_a + scale_b) + max_lse;
    }

    for (int ee = 0; ee < Ev / warp.size(); ee += fvec_t::size) {
        int e = ee * warp.size() + warp.thread_rank() * fvec_t::size;
        vec_t st;
        for (int jj = 0; jj < fvec_t::size; ++jj) {
            st[jj] = (scalar_t)v_cache[ee + jj];
        }
        st.store(out + ((size_t)global_q_idx * shape.Hq + h) * Ev + e);
    }
}

template<class scalar_t>
cudaError_t dvattn_attention_gpu(scalar_t* out, float scale,
                           const int* locations, const scalar_t* queries,
                           const float* cosines, const float* sines, int rotary_dim,
                           const int* fragment_lengths,
                           const scalar_t** key_fragments,
                           const scalar_t** value_fragments,
                           Shape shape) {
    int problem_size = shape.Hkv * shape.W * shape.S;
    int sms = -1;
    CUDA_RETURN_ON_ERROR(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    // Note: The current kernel will **not** work if there is only one split!
    int splits = max(2, sms / problem_size);

    dim3 grid_dim{(unsigned)shape.Hkv, (unsigned)shape.W * (unsigned)shape.S, (unsigned)splits};
    dim3 block_dim{256, 1, 1};
    size_t smem = shape.Ev * sizeof(float) * block_dim.x / 32 * (shape.Hq / shape.Hkv);
    smem += 2 * sizeof(float) * (shape.Hq / shape.Hkv);
    smem = std::max(smem, 2 * (shape.E + shape.Ev) * (block_dim.x / SubWarpSize) * sizeof(scalar_t));
    static char* workspace = nullptr;
    static std::size_t workspace_size = 0;

    std::size_t required_workspace = shape.W * shape.Hq * shape.S * splits;  // [W, Hq, S, K]
    size_t alloc = required_workspace * (shape.Ev + 1) * sizeof(float);
    if (workspace_size < required_workspace) {
        if (workspace)
            CUDA_RETURN_ON_ERROR(cudaFree(workspace));
        CUDA_RETURN_ON_ERROR(cudaMalloc(&workspace, alloc));
        CUDA_RETURN_ON_ERROR(cudaMemset(workspace, 0, alloc));
        workspace_size = required_workspace;
    }

    if (shape.E == 128 && shape.Ev == 128 && shape.Hq == shape.Hkv * 16) {
        CUDA_RETURN_ON_ERROR(cudaFuncSetAttribute(dvattn_attention_gpu_kernel21<128, 128, 16, scalar_t>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        dvattn_attention_gpu_kernel21<128, 128, 16><<<grid_dim, block_dim, smem>>>(
                out, workspace, scale, locations, queries, cosines, sines, rotary_dim, fragment_lengths, key_fragments, value_fragments, shape);

        dim3 r_grid_dim{(unsigned)shape.Hq, (unsigned)shape.W * (unsigned)shape.S, 1};
        dvattn_attention_reduce_kernel<128><<<r_grid_dim, 32>>>(
                out, (float*)workspace, (float*)workspace + splits * shape.W * shape.Hq * shape.S * shape.Ev,
                splits, shape);
    } else {
        printf("Unsupported head dimension");
    }
    return cudaGetLastError();
}

template<class scalar_t>
cudaError_t dvattn_paged_attention_gpu(scalar_t* out, float scale,
                           const int* locations, const scalar_t* queries,
                           const float* cosines, const float* sines, int rotary_dim,
                           const int* fragment_lengths,
                           const scalar_t* key_cache,
                           const scalar_t* value_cache,
                           const int* block_table,
                           Shape shape) {
    int problem_size = shape.Hkv * shape.W * shape.S;
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

    std::size_t required_workspace = shape.W * shape.Hq * shape.S * splits;  // [W, Hq, S, K]
    size_t alloc = required_workspace * (shape.Ev + 1) * sizeof(float);
    if (workspace_size < required_workspace) {
        if (workspace)
            CUDA_RETURN_ON_ERROR(cudaFree(workspace));
        CUDA_RETURN_ON_ERROR(cudaMalloc(&workspace, alloc));
        CUDA_RETURN_ON_ERROR(cudaMemset(workspace, 0, alloc));
        workspace_size = required_workspace;
    }

    if (shape.E == 128 && shape.Ev == 128 && shape.Hq == shape.Hkv * 16) {
        CUDA_RETURN_ON_ERROR(cudaFuncSetAttribute(dvattn_paged_attention_gpu_kernel21<128, 128, 16, scalar_t>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        dvattn_paged_attention_gpu_kernel21<128, 128, 16><<<grid_dim, block_dim, smem>>>(
                out, workspace, scale, locations, queries, cosines, sines, rotary_dim, fragment_lengths, key_cache, value_cache, block_table, shape);

        dim3 r_grid_dim{(unsigned)shape.Hq, (unsigned)shape.W * (unsigned)shape.S, 1};
        dvattn_attention_reduce_kernel<128><<<r_grid_dim, 32>>>(
                out, (float*)workspace, (float*)workspace + splits * shape.W * shape.Hq * shape.S * shape.Ev,
                splits, shape);
    } else {
        printf("Unsupported head dimension");
    }
    return cudaGetLastError();
}

template<class scalar_t>
cudaError_t dvattn_varlen_paged_attention_gpu(scalar_t* out, float scale,
                           const int* locations, const scalar_t* queries,
                           const float* cosines, const float* sines, int rotary_dim,
                           const int* fragment_lengths,
                           const scalar_t* key_cache,
                           const scalar_t* value_cache,
                           const int* block_table,
                           Shape shape) {
    int problem_size = shape.total_q_tokens;
    int sms = -1;
    CUDA_RETURN_ON_ERROR(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    int splits = max(2, sms * 4 / problem_size); // Heuristic for split-k

    dim3 grid_dim{(unsigned)shape.Hkv, (unsigned)shape.total_q_tokens, (unsigned)splits};
    dim3 block_dim{256, 1, 1};
    size_t smem = shape.Ev * sizeof(float) * block_dim.x / 32 * (shape.Hq / shape.Hkv);
    smem += 2 * sizeof(float) * (shape.Hq / shape.Hkv);
    smem = std::max(smem, 2 * (shape.E + shape.Ev) * (block_dim.x / SubWarpSize) * sizeof(scalar_t));
    static char* workspace = nullptr;
    static std::size_t workspace_size = 0;

    std::size_t required_workspace = (size_t)splits * shape.total_q_tokens * shape.Hq * (shape.Ev + 1) * sizeof(float);
    if (workspace_size < required_workspace) {
        if (workspace) CUDA_RETURN_ON_ERROR(cudaFree(workspace));
        CUDA_RETURN_ON_ERROR(cudaMalloc(&workspace, required_workspace));
        workspace_size = required_workspace;
    }

    if (shape.E == 128 && shape.Ev == 128 && shape.Hq == shape.Hkv * 16) {
        CUDA_RETURN_ON_ERROR(cudaFuncSetAttribute(dvattn_varlen_paged_attention_gpu_kernel21<128, 128, 16, scalar_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        dvattn_varlen_paged_attention_gpu_kernel21<128, 128, 16><<<grid_dim, block_dim, smem>>>(
                out, workspace, scale, locations, queries, cosines, sines, rotary_dim, fragment_lengths, key_cache, value_cache, block_table, shape);

        dim3 r_grid_dim{(unsigned)shape.Hq, (unsigned)shape.total_q_tokens, 1};
        dvattn_varlen_attention_reduce_kernel<128><<<r_grid_dim, 32>>>(
                out, (float*)workspace, (float*)workspace + (size_t)splits * shape.total_q_tokens * shape.Hq * shape.Ev,
                splits, shape);
    } else {
        printf("Unsupported head dimension for varlen paged attention");
    }
    return cudaGetLastError();
}

}  // namespace v21
