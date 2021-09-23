/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifdef WITH_CUDA

#include "oneflow/core/primitive/copy_nd.h"
#include "oneflow/core/common/memory_copy_nd_desc.h"
#include "oneflow/core/stream/cuda_stream_context.h"
#include "oneflow/core/primitive/cuda/cuda_graph_support.h"
#include <cuda_runtime.h>
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace primitive {

namespace {

template<int32_t NDIMS, typename I>
struct SOA {
  I val[NDIMS];
};

template<int32_t NDIMS, typename T, typename I>
__global__ void CopyNDGpu(const int n, T* dst, const T* src,
                          NdIndexOffsetHelper<I, NDIMS> dst_helper,
                          NdIndexOffsetHelper<I, NDIMS> src_helper,
                          NdIndexOffsetHelper<I, NDIMS> copy_helper, SOA<NDIMS, I> dst_pos,
                          SOA<NDIMS, I> src_pos) {
  CUDA_1D_KERNEL_LOOP_T(I, i, n) {
    I copy_idx[NDIMS];
    I src_idx[NDIMS];
    I dst_idx[NDIMS];
    copy_helper.OffsetToNdIndex(i, copy_idx);
#pragma unroll
    for (I j = 0; j < NDIMS; j++) {
      src_idx[j] = src_pos.val[j] + copy_idx[j];
      dst_idx[j] = dst_pos.val[j] + copy_idx[j];
    }
    const I src_offset = src_helper.NdIndexToOffset(src_idx);
    const I dst_offset = dst_helper.NdIndexToOffset(dst_idx);
    dst[dst_offset] = src[src_offset];
  }
}

size_t GetPackSize(const MemoryCopyNdDesc& desc, const void* dst, const void* src) {
  const int64_t mask = desc.src_shape.dim_vec().back() | desc.dst_shape.dim_vec().back()
                       | desc.extent.dim_vec().back() | desc.src_pos.dim_vec().back()
                       | desc.dst_pos.dim_vec().back()
                       | static_cast<int64_t>(reinterpret_cast<uintptr_t>(dst))
                       | static_cast<int64_t>(reinterpret_cast<uintptr_t>(src));
  if ((mask & 0xF) == 0) {
    return 16;
  } else if ((mask & 0x7) == 0) {
    return 8;
  } else if ((mask & 0x3) == 0) {
    return 4;
  } else if ((mask & 0x1) == 0) {
    return 2;
  } else {
    return 1;
  }
}

template<int32_t NDIMS, typename P, typename I>
void CopyNDByPackByIndexTypeGpu(cudaStream_t stream, void* dst, const void* src,
                                const MemoryCopyNdDesc& desc) {
  CHECK_EQ(desc.dst_pos.NumAxes(), NDIMS);
  CHECK_EQ(desc.src_pos.NumAxes(), NDIMS);
  CHECK_EQ(desc.dst_shape.NumAxes(), NDIMS);
  CHECK_EQ(desc.src_shape.NumAxes(), NDIMS);
  CHECK_EQ(desc.extent.NumAxes(), NDIMS);
  constexpr size_t pack_size = sizeof(P);
  I dst_shape_dim_arr[NDIMS];
  I src_shape_dim_arr[NDIMS];
  I extent_dim_arr[NDIMS];
  SOA<NDIMS, I> src_pos;
  SOA<NDIMS, I> dst_pos;
  FOR_RANGE(int64_t, i, 0, NDIMS) {
    if (i == NDIMS - 1) {
      dst_pos.val[i] = desc.dst_pos.dim_vec().at(i) / pack_size;
      src_pos.val[i] = desc.src_pos.dim_vec().at(i) / pack_size;
      dst_shape_dim_arr[i] = desc.dst_shape.dim_vec().at(i) / pack_size;
      src_shape_dim_arr[i] = desc.src_shape.dim_vec().at(i) / pack_size;
      extent_dim_arr[i] = desc.extent.dim_vec().at(i) / pack_size;
    } else {
      dst_pos.val[i] = desc.dst_pos.dim_vec().at(i);
      src_pos.val[i] = desc.src_pos.dim_vec().at(i);
      dst_shape_dim_arr[i] = desc.dst_shape.dim_vec().at(i);
      src_shape_dim_arr[i] = desc.src_shape.dim_vec().at(i);
      extent_dim_arr[i] = desc.extent.dim_vec().at(i);
    }
  }
  NdIndexOffsetHelper<I, NDIMS> dst_helper(dst_shape_dim_arr);
  NdIndexOffsetHelper<I, NDIMS> src_helper(src_shape_dim_arr);
  NdIndexOffsetHelper<I, NDIMS> copy_helper(extent_dim_arr);
  const int64_t elem_cnt = desc.extent.elem_cnt() / pack_size;
  CopyNDGpu<NDIMS, P, I><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, stream>>>(
      elem_cnt, reinterpret_cast<P*>(dst), reinterpret_cast<const P*>(src), dst_helper, src_helper,
      copy_helper, dst_pos, src_pos);
}

template<int32_t NDIMS, typename P>
void CopyNDByPackGpu(cudaStream_t stream, void* dst, const void* src,
                     const MemoryCopyNdDesc& desc) {
  if (std::max(desc.dst_shape.elem_cnt(), desc.src_shape.elem_cnt())
      > static_cast<int64_t>(GetMaxVal<int32_t>() / 2)) {
    CopyNDByPackByIndexTypeGpu<NDIMS, P, int64_t>(stream, dst, src, desc);
  } else {
    CopyNDByPackByIndexTypeGpu<NDIMS, P, int32_t>(stream, dst, src, desc);
  }
}

template<int32_t NDIMS>
void CopyNDGpuImpl(cudaStream_t stream, void* dst, const void* src, const MemoryCopyNdDesc& desc) {
  const size_t pack_size = GetPackSize(desc, dst, src);
  if (pack_size == 1) {
    CopyNDByPackGpu<NDIMS, uint8_t>(stream, dst, src, desc);
  } else if (pack_size == 2) {
    CopyNDByPackGpu<NDIMS, uint16_t>(stream, dst, src, desc);
  } else if (pack_size == 4) {
    CopyNDByPackGpu<NDIMS, uint32_t>(stream, dst, src, desc);
  } else if (pack_size == 8) {
    CopyNDByPackGpu<NDIMS, uint64_t>(stream, dst, src, desc);
  } else if (pack_size == 16) {
    static_assert(sizeof(uint4) == 16, "");
    CopyNDByPackGpu<NDIMS, uint4>(stream, dst, src, desc);
  } else {
    UNIMPLEMENTED();
  }
}

void Copy1D(cudaStream_t stream, void* dst, const void* src, size_t count) {
  OF_CUDA_CHECK(cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream));
}

void CopyND(cudaStream_t stream, void* dst, const void* src, const MemoryCopyNdDesc& desc) {
  const int32_t num_axes = desc.src_shape.NumAxes();
  if (num_axes == 1) {
    Copy1D(stream, (unsigned char*)dst + desc.dst_pos.At(0),
           (unsigned char*)src + desc.src_pos.At(0), desc.extent.At(0));
  } else if (num_axes == 2) {
    CopyNDGpuImpl<2>(stream, dst, src, desc);
  } else if (num_axes == 3) {
    CopyNDGpuImpl<3>(stream, dst, src, desc);
  } else if (num_axes == 4) {
    CopyNDGpuImpl<4>(stream, dst, src, desc);
  } else if (num_axes == 5) {
    CopyNDGpuImpl<5>(stream, dst, src, desc);
  } else if (num_axes == 6) {
    CopyNDGpuImpl<6>(stream, dst, src, desc);
  } else {
    UNIMPLEMENTED();
  }
}

class CopyNdImpl : public CopyNd, public CudaGraphSupport {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyNdImpl);
  CopyNdImpl() = default;
  ~CopyNdImpl() override = default;

  void Launch(StreamContext* stream_ctx, void* dst, const void* src,
              const MemoryCopyNdDesc& desc) const override {
    cudaStream_t cuda_stream =
        CHECK_NOTNULL(dynamic_cast<CudaStreamContext*>(stream_ctx))->cuda_stream();
    CopyND(cuda_stream, dst, src, desc);
  }
};

class CopyNdFactoryImpl : public CopyNdFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyNdFactoryImpl);
  CopyNdFactoryImpl() = default;
  ~CopyNdFactoryImpl() override = default;

  std::unique_ptr<CopyNd> New() override { return std::unique_ptr<CopyNd>(new CopyNdImpl()); }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kGPU, CopyNdFactory, CopyNdFactoryImpl);

}  // namespace

}  // namespace primitive

}  // namespace oneflow

#endif
