#pragma once

#include <cstdio>

struct Shape {
    int F;          // fragments
    int W;          // workers
    int Hq;         // q heads
    int Hkv;        // kv heads
    int E;          // qk dim
    int Ev;         // v dim
    int S;          // q seq len
    // --- paged attention ---
    int block_size;
    int max_blocks_per_seq;
    // --- varlen prefill ---
    int total_q_tokens;
    const int* cu_seqlens_q;
    const int* cu_seqlens_k;
};
