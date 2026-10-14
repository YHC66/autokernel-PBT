// Row-wise softmax of a float32 matrix: a CUDA port of the known Triton kernel
// kernels/triton/ladder.py::_softmax_kernel.
//
// The two compute the same function through different reduction trees, which is the
// point of porting it. Triton's tl.max / tl.sum reduce one tile, as wide as the row, in
// a single balanced tree. Here each thread first folds the columns tid, tid + width,
// tid + 2*width, ... sequentially, and the block then combines its `width` partials in
// a balanced tree in shared memory. While n_cols <= width every thread holds at most one
// column and the trees have the same depth; above it they differ, and so do the last
// bits of the output.
//
// Launch contract (kernels/cuda/ladder.py enforces it):
//   grid  = (n_rows, 1, 1), one block per row;
//   block = (width, 1, 1), width a power of two -- the tree halves it;
//   dynamic shared memory = width * sizeof(float).
//
// Compiled by NVRTC with no host headers, so only device built-ins are available.

__device__ __forceinline__ float negative_infinity() {
    return -__int_as_float(0x7f800000);
}

extern "C" __global__ void softmax_rows(const float* __restrict__ x,
                                        float* __restrict__ y,
                                        int n_cols) {
    extern __shared__ float partial[];
    const int tid = threadIdx.x;
    const int width = blockDim.x;
    const size_t base = (size_t)blockIdx.x * (size_t)n_cols;

    // Row max. A thread with no column keeps -inf, the identity of max, so a row
    // narrower than the block cannot be raised by an idle thread.
    float m = negative_infinity();
    for (int c = tid; c < n_cols; c += width) {
        m = fmaxf(m, x[base + c]);
    }
    partial[tid] = m;
    __syncthreads();
    for (int s = width >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] = fmaxf(partial[tid], partial[tid + s]);
        }
        __syncthreads();
    }
    const float row_max = partial[0];
    // Every thread reads the max before any thread overwrites partial[] below.
    __syncthreads();

    // Denominator. A thread with no column contributes 0, the identity of +.
    float acc = 0.0f;
    for (int c = tid; c < n_cols; c += width) {
        acc += expf(x[base + c] - row_max);
    }
    partial[tid] = acc;
    __syncthreads();
    for (int s = width >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
        }
        __syncthreads();
    }
    const float denom = partial[0];

    for (int c = tid; c < n_cols; c += width) {
        y[base + c] = expf(x[base + c] - row_max) / denom;
    }
}
