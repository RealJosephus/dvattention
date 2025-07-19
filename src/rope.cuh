
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
    dim3 grid_dim(F*S , Hq, W);
    // Launch E threads per query vector.
    dim3 block_dim(E, 1, 1);
    rope_kernel<<<grid_dim, block_dim>>>(rotated_queries, queries, cosines, sines, F, W, Hq, S, E, RotaryE);
}
