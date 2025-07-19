#pragma once

#include "common.h"
#include <cuda_runtime.h>

template<class scalar_t>
__global__ void copy_to_blocks_kernel(
    const scalar_t* key_states,
    const scalar_t* value_states,
    scalar_t* key_cache,
    scalar_t* value_cache,
    const int* block_table,
    const int* seq_lengths,
    Shape shape)
{
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int new_token_idx = threadIdx.x;

    int S_new = shape.S;
    if (new_token_idx >= S_new) return;

    const int Hkv = shape.Hkv;
    const int E = shape.E;
    const int Ev = shape.Ev;
    const int block_size = shape.block_size;
    const int max_blocks = shape.max_blocks_per_seq;

    const int seq_len_before = seq_lengths[batch_idx];
    const int total_pos = seq_len_before + new_token_idx;

    const int block_idx_in_seq = total_pos / block_size;
    const int offset_in_block = total_pos % block_size;
    const int physical_block_id = block_table[batch_idx * max_blocks + block_idx_in_seq];

    const scalar_t* src_k_base = key_states + (batch_idx * Hkv + head_idx) * S_new * E + new_token_idx * E;
    const scalar_t* src_v_base = value_states + (batch_idx * Hkv + head_idx) * S_new * Ev + new_token_idx * Ev;

    scalar_t* dst_k_base = key_cache + ((ptrdiff_t)physical_block_id * Hkv + head_idx) * block_size * E + offset_in_block * E;
    scalar_t* dst_v_base = value_cache + ((ptrdiff_t)physical_block_id * Hkv + head_idx) * block_size * Ev + offset_in_block * Ev;

    for (int embed_idx = 0; embed_idx < E; ++embed_idx) {
        dst_k_base[embed_idx] = src_k_base[embed_idx];
    }

    for (int embed_idx = 0; embed_idx < Ev; ++embed_idx) {
        dst_v_base[embed_idx] = src_v_base[embed_idx];
    }
}
