# Triton Kernel(s) for Implementing Telescoping Caches

This folder contains Triton kernels for indexed matrix multiplication: one argument is a standard matrix, and the other is a bank of values plus an index map describing how to form the second multiplicand. This indirection is crucial in cases where the second multiplicand is a large matrix containing many repeated values - in this case, we can avoid materializing the large matrix and allow the kernel(s) to handle it internally. This matmul-with-indirection is crucial for implementing efficient [telescoping caches](https://github.com/foundation-model-stack/avengers/blob/telescoping-kernel/docs/telescoping-cache.md) over multiple time steps in parallel.

When describing tensor shapes, we use the following variables:

 - b: batch size
 - l: sequence length
 - n: number of unique values in the value buffer
 - h: number of k/v heads
 - e: expansion factor for GQA (h*e = number of q heads)
 - d: head dimensionality, typically 64 or 128
 - c: number of entries in the telescoping cache

Consider the case of QK-multiplication (Attn/V-multiplication is analogous). We are given a fully-materialized Q tensor of size `b l h e d`, a key buffer of size `b n h d`, and an index map of size `l c` that describes, for each position `l` and `c`, which slice along `n` to take from the key buffer. Using the index map, we can form the full K tensor of size `b l h d c`: `c` buffer entries for each head, time step, and batch entry. We then matrix multiply the Q and K tensors to get our attention scores `b l h e c`. Unfortunately, `d >> e`, so while the Q and output tensors are feasible to hold in memory, the K tensor is too large to materialize. 

The kernels here implement the above procedure as a single fused operation, plus the two corresponding backward operations. Attn/V-multiplication follows a transposed procedure: the Attn tensor has size `b l h e c`, and the value buffer and index map match the key buffer/map from above (`b n h d` and `l c` respectively). The materialized V tensor has shape `b l h c d`, and we output `b l h e d`. This operation uses the same kernels, and simply swaps the forward/backward kernels and some of the arguments. 

The correctness checks for these kernels (forward and backward passes) can be run via `pytest -vs test_telescoping_kernel.py`. 