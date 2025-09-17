// Copyright (c) OpenMMLab. All rights reserved.
#ifndef TRT_GRID_SAMPLER_KERNEL_HPP
#define TRT_GRID_SAMPLER_KERNEL_HPP
#include <cuda_runtime.h>

enum class GridSamplerInterpolation { Bilinear, Nearest };
enum class GridSamplerPadding { Zeros, Border, Reflection };

#if NV_TENSORRT_MAJOR < 10
template <typename T>
void grid_sample(T *output, const T *input, const T *grid, int *output_dims, int *input_dims,
                 int *grid_dims, int nb_dims, GridSamplerInterpolation interp,
                 GridSamplerPadding padding, bool align_corners, cudaStream_t stream);
#else
template <typename T>
void grid_sample(T *output, const T *input, const T *grid, long int *output_dims,
                 long int *input_dims, long int *grid_dims, int nb_dims,
                 GridSamplerInterpolation interp, GridSamplerPadding padding, bool align_corners,
                 cudaStream_t stream);
#endif  // NV_TENSORRT_MAJOR
#endif  // TRT_GRID_SAMPLER_KERNEL_HPP
