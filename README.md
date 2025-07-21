## Dynamic View Attention

<img width="384" height="256" alt="Dynamic View Attention" src="https://github.com/user-attachments/assets/b2ce2bfb-d62c-4839-9f8e-b4c0420eb452" />

This repository provides the implementation of DVAttn, a custom attention mechanism designed for efficient LLM inference.

Instead of rotating KV Cache, the library applies an equivalent relative rotation solely to the query for each cache turn.

The implementation features custom CUDA kernels that support paged attention, varlen sequences, and fused operations. It is hardcoded for a GQA factor of 16, supports partial RoPE application on the head dimension, and operates on fp32, fp16, and bf16 dtypes.
