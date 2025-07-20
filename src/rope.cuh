
template<class scalar_t>
__global__ void rope_kernel(
        scalar_t* rotated_queries, const scalar_t* queries, const float* cosines, const float* sines,
        int F, int W, int Hq, int S, int E, int RotaryE)
{
    int f = blockIdx.x / S;
    int s = blockIdx.x % S;
    int h = blockIdx.y;
    int w = blockIdx.z;

    const scalar_t* query = queries + ((w * Hq + h) * S + s) * E;
    scalar_t* result = rotated_queries + (((f * W + w) * Hq + h) * S + s) * E;

    // Each thread handles one element in the E dimension.
    int e = threadIdx.x;

    if (e < RotaryE) {
        // This thread is in the part that needs to be rotated.
        int e_pair;
        float x1, x2;
        int cos_sin_idx;

        if (e < RotaryE / 2) {
            // First element of a pair.
            e_pair = e + RotaryE / 2;
            x1 = query[e];
            x2 = query[e_pair];
            cos_sin_idx = e;
        } else {
            // Second element of a pair.
            e_pair = e - RotaryE / 2;
            x1 = query[e_pair];
            x2 = query[e];
            cos_sin_idx = e_pair;
        }

        int offset = (((f*W + w) * S + s) * RotaryE);
        const float* cos_vec = cosines + offset;
        const float* sin_vec = sines + offset;

        if (e < RotaryE / 2) {
            result[e] = x1 * cos_vec[cos_sin_idx] - x2 * sin_vec[cos_sin_idx];
        } else {
            result[e] = x2 * cos_vec[cos_sin_idx] + x1 * sin_vec[cos_sin_idx];
        }

    } else {
        // This thread is in the pass-through part. Simple copy.
        result[e] = query[e];
    }
}

template<class scalar_t>
void rope_gpu(
        scalar_t* rotated_queries, const scalar_t* queries, const float* cosines, const float* sines,
        int F, int W, int Hq, int S, int E, int RotaryE) {
    if (S == 0) {
        return;
    }
    dim3 grid_dim(F*S , Hq, W);
    // Launch E threads per query vector.
    dim3 block_dim(E, 1, 1);
    rope_kernel<<<grid_dim, block_dim>>>(rotated_queries, queries, cosines, sines, F, W, Hq, S, E, RotaryE);
}

template<class scalar_t>
__global__ void rope_varlen_kernel(
        scalar_t* rotated_queries,
        const scalar_t* queries,
        const float* cosines,
        const float* sines,
        int F, int T, int H, int E, int RotaryE)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int f = blockIdx.z;
    const int e = threadIdx.x;

    const scalar_t* query_vec = queries + (t * H + h) * E;
    scalar_t* result_vec = rotated_queries + ((f * T + t) * H + h) * E;

    if (e < RotaryE) {
        const int e_rot = e % (RotaryE / 2);
        const int cos_sin_offset = (f * T + t) * RotaryE/2 + e_rot;
        const float cos_val = cosines[cos_sin_offset];
        const float sin_val = sines[cos_sin_offset];

        const float q_val = query_vec[e];
        const float q_pair_val = (e < RotaryE/2) ? query_vec[e + RotaryE/2] : query_vec[e - RotaryE/2];

        if (e < RotaryE/2) {
            result_vec[e] = q_val * cos_val - q_pair_val * sin_val;
        } else {
            result_vec[e] = q_val * cos_val + q_pair_val * sin_val;
        }
    } else {
        result_vec[e] = query_vec[e];
    }
}

template<class scalar_t>
void rope_varlen_gpu(
        scalar_t* rotated_queries, const scalar_t* queries, const float* cosines, const float* sines,
        int F, int T, int H, int E, int R) {
    if (T == 0) {
        return;
    }
    // Grid maps to (total_tokens, num_heads, num_fragments)
    dim3 grid_dim(T, H, F);
    // Each block handles one query vector, with one thread per element in head_dim
    dim3 block_dim(E, 1, 1);
    rope_varlen_kernel<<<grid_dim, block_dim>>>(rotated_queries, queries, cosines, sines, F, T, H, E, R);
}
