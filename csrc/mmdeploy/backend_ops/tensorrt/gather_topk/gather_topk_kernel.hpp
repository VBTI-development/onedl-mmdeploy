// Copyright (c) OpenMMLab. All rights reserved.
#ifndef TRT_GRID_SAMPLER_KERNEL_HPP
#define TRT_GRID_SAMPLER_KERNEL_HPP
#include <cuda_runtime.h>

#if NV_TENSORRT_MAJOR < 10
template <typename scalar_t>
void gather_topk_impl(const scalar_t* input, const int32_t* indices, const int* dims, int nbDims,
                      const int* indices_dims, int indice_nbDims, scalar_t* output,
                      cudaStream_t stream);
#else
template <typename scalar_t>
void gather_topk_impl(const scalar_t* input, const int64_t* indices, const long int* dims,
                      int nbDims, const long int* indices_dims, int indice_nbDims, scalar_t* output,
                      cudaStream_t stream);
#endif  // NV_TENSORRT_MAJOR < 10
#endif  // TRT_GRID_SAMPLER_KERNEL_HPP
