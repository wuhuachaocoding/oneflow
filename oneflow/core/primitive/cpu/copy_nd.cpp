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
#include "oneflow/core/primitive/copy_nd.h"
#include "oneflow/core/common/memory_copy_nd_desc.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace primitive {

namespace {

void Copy1D(void* dst, const void* src, size_t count) { std::memcpy(dst, src, count); }

void Copy2D(void* dst, size_t dst_pitch, const void* src, size_t src_pitch, size_t width,
            size_t height) {
  unsigned char* dst_ptr = (unsigned char*)dst;
  const unsigned char* src_ptr = (unsigned char*)src;
  FOR_RANGE(size_t, i, 0, height) {
    Copy1D(dst_ptr, src_ptr, width);
    dst_ptr += dst_pitch;
    src_ptr += src_pitch;
  }
}

void Copy3D(void* dst, const void* src, const MemoryCopyNdDesc& desc) {
  const size_t dst_pitch = desc.dst_shape.Count(2);
  const size_t src_pitch = desc.src_shape.Count(2);
  const size_t dst_inner_area = desc.dst_shape.Count(1);
  const size_t src_inner_area = desc.src_shape.Count(1);
  const size_t width = desc.extent.At(2);
  const size_t height = desc.extent.At(1);
  const size_t depth = desc.extent.At(0);
  FOR_RANGE(size_t, i, 0, depth) {
    void* dst_2d = (unsigned char*)dst + (desc.dst_pos.At(0) + i) * dst_inner_area
                   + desc.dst_pos.At(1) * dst_pitch + desc.dst_pos.At(2);
    const void* src_2d = (unsigned char*)src + (desc.src_pos.At(0) + i) * src_inner_area
                         + desc.src_pos.At(1) * src_pitch + desc.src_pos.At(2);
    Copy2D(dst_2d, dst_pitch, src_2d, src_pitch, width, height);
  }
}

template<int32_t NDIMS>
void CopyNDCpuImpl(void* dst, const void* src, const MemoryCopyNdDesc& desc) {
  NdIndexOffsetHelper<int64_t, NDIMS> src_helper(desc.src_shape.dim_vec().data());
  NdIndexOffsetHelper<int64_t, NDIMS> dst_helper(desc.dst_shape.dim_vec().data());
  NdIndexOffsetHelper<int64_t, NDIMS> copy_helper(desc.extent.dim_vec().data());
  FOR_RANGE(int64_t, i, 0, desc.extent.elem_cnt()) {
    int64_t copy_idx[NDIMS];
    int64_t src_idx[NDIMS];
    int64_t dst_idx[NDIMS];
    copy_helper.OffsetToNdIndex(i, copy_idx);
    FOR_RANGE(int64_t, j, 0, NDIMS) {
      src_idx[j] = desc.src_pos.At(j) + copy_idx[j];
      dst_idx[j] = desc.dst_pos.At(j) + copy_idx[j];
    }
    const int64_t src_offset = src_helper.NdIndexToOffset(src_idx);
    const int64_t dst_offset = dst_helper.NdIndexToOffset(dst_idx);
    unsigned char* dst_ptr = reinterpret_cast<unsigned char*>(dst) + dst_offset;
    const unsigned char* src_ptr = reinterpret_cast<const unsigned char*>(src) + src_offset;
    *dst_ptr = *src_ptr;
  }
}

void CopyND(void* dst, const void* src, const MemoryCopyNdDesc& desc) {
  const int32_t num_axes = desc.src_shape.NumAxes();
  if (num_axes == 4) {
    CopyNDCpuImpl<4>(dst, src, desc);
  } else if (num_axes == 5) {
    CopyNDCpuImpl<5>(dst, src, desc);
  } else if (num_axes == 6) {
    CopyNDCpuImpl<6>(dst, src, desc);
  } else {
    UNIMPLEMENTED();
  }
}

class CopyNdImpl : public CopyNd {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyNdImpl);
  CopyNdImpl() = default;
  ~CopyNdImpl() = default;

  void Launch(StreamContext* stream_ctx, void* dst, const void* src,
              const MemoryCopyNdDesc& desc) const override {
    const int32_t num_axes = desc.src_shape.NumAxes();
    if (num_axes == 1) {
      Copy1D((unsigned char*)dst + desc.dst_pos.At(0), (unsigned char*)src + desc.src_pos.At(0),
             desc.extent.At(0));
    } else if (num_axes == 2) {
      const size_t dst_pitch = desc.dst_shape.At(1);
      const size_t src_pitch = desc.src_shape.At(1);
      const size_t width = desc.extent.At(1);
      const size_t height = desc.extent.At(0);
      void* dst_2d = (unsigned char*)dst + desc.dst_pos.At(0) * dst_pitch + desc.dst_pos.At(1);
      const void* src_2d =
          (const unsigned char*)src + desc.src_pos.At(0) * src_pitch + desc.src_pos.At(1);
      Copy2D(dst_2d, dst_pitch, src_2d, src_pitch, width, height);
    } else if (num_axes == 3) {
      Copy3D(dst, src, desc);
    } else {
      CopyND(dst, src, desc);
    }
  }
};

class CopyNdFactoryImpl : public CopyNdFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyNdFactoryImpl);
  CopyNdFactoryImpl() = default;
  ~CopyNdFactoryImpl() override = default;

  std::unique_ptr<CopyNd> New() override { return std::make_unique<CopyNdImpl>(); }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, CopyNdFactory, CopyNdFactoryImpl);

}  // namespace

}  // namespace primitive

}  // namespace oneflow
