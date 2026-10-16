// Row-wise layer normalization of a float32 matrix, without affine parameters: a CUDA
// port of the known Triton kernel kernels/triton/ladder.py::_layernorm_kernel.
//
// Same arithmetic as the Triton kernel -- mean, then the population variance of the
// centered row, then centered / sqrt(var + eps) -- through a different reduction tree:
// a sequential fold per thread over the columns tid, tid + width, ..., then a balanced
// tree over the block's `width` partials in shared memory. See softmax.cu.
//
// Launch contract (kernels/cuda/ladder.py enforces it):
//   grid  = (n_rows, 1, 1), one block per row;
//   block = (width, 1, 1), width a power of two -- the tree halves it;
//   dynamic shared memory = width * sizeof(float).
//
// Compiled by NVRTC with no host headers, so only device built-ins are available.

__device__ __forceinline__ float block_sum(float value, float* partial) {
    const int tid = threadIdx.x;
    partial[tid] = value;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
        }
        __syncthreads();
    }
    const float total = partial[0];
    // Every thread reads the total before any thread reuses partial[].
    __syncthreads();
    return total;
}

extern "C" __global__ void layernorm_rows(const float* __restrict__ x,
                                          float* __restrict__ y,
                                          int n_cols,
                                          float eps) {
    extern __shared__ float partial[];
    const int tid = threadIdx.x;
    const int width = blockDim.x;
    const size_t base = (size_t)blockIdx.x * (size_t)n_cols;

    // A thread with no column contributes 0, the identity of +.
    float acc = 0.0f;
    for (int c = tid; c < n_cols; c += width) {
        acc += x[base + c];
    }
    const float mean = block_sum(acc, partial) / (float)n_cols;

    float sq = 0.0f;
    for (int c = tid; c < n_cols; c += width) {
        const float d = x[base + c] - mean;
        sq += d * d;
    }
    const float var = block_sum(sq, partial) / (float)n_cols;
    const float denom = sqrtf(var + eps);

    for (int c = tid; c < n_cols; c += width) {
        y[base + c] = (x[base + c] - mean) / denom;
    }
}
