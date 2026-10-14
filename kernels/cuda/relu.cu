// ReLU over the rows of a float32 matrix: a CUDA port of the known Triton kernel
// kernels/triton/ladder.py::_relu_kernel.
//
// One block per row; the block's threads stride across the row. Unlike the Triton
// kernel, whose single tile must be at least as wide as the row, a strided loop covers
// any n_cols with any block width, so this port has no tail to drop.
//
// Compiled by NVRTC with no host headers, so only device built-ins are available.

extern "C" __global__ void relu_rows(const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int n_cols) {
    const size_t base = (size_t)blockIdx.x * (size_t)n_cols;
    for (int c = threadIdx.x; c < n_cols; c += blockDim.x) {
        y[base + c] = fmaxf(x[base + c], 0.0f);
    }
}
