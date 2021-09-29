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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/primitive/include/copy_nd.h"

namespace oneflow {

namespace {

template<typename T>
T GetDtypeMatchedValue(double floating, int64_t integral);

template<>
float16 GetDtypeMatchedValue(double floating, int64_t integral) {
  return static_cast<float16>(floating);
}

template<>
float GetDtypeMatchedValue(double floating, int64_t integral) {
  return static_cast<float>(floating);
}

template<>
double GetDtypeMatchedValue(double floating, int64_t integral) {
  return floating;
}

template<>
int8_t GetDtypeMatchedValue(double floating, int64_t integral) {
  return static_cast<int8_t>(integral);
}

template<>
int32_t GetDtypeMatchedValue(double floating, int64_t integral) {
  return static_cast<int32_t>(integral);
}

template<>
int64_t GetDtypeMatchedValue(double floating, int64_t integral) {
  return integral;
}

}  // namespace

template<DeviceType device_type, typename T>
class PadKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  PadKernel() = default;
  ~PadKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    if (y->shape().NumAxes() > 0 && y->shape().elem_cnt() == 0) {
      // if output is 0-shape tensor, than do nothing and return
      return;
    }
    const T constant_value = GetDtypeMatchedValue<T>(ctx->Attr<double>("floating_constant_value"),
                                                     ctx->Attr<int64_t>("integral_constant_value"));
    const auto& padding_before = ctx->Attr<std::vector<int64_t>>("padding_before");
    const auto& padding_after = ctx->Attr<std::vector<int64_t>>("padding_after");
    const int64_t ndims = x->shape().NumAxes();
    CHECK_EQ(padding_before.size(), ndims);

    NewKernelUtil<device_type>::Fill(ctx->device_ctx(), y->shape().elem_cnt(),
                                     static_cast<T>(constant_value), y->mut_dptr<T>());

    DimVector src_pos_vec(ndims, 0);
    DimVector dst_pos_vec(padding_before.cbegin(), padding_before.cend());
    DimVector pad_before_vec(padding_before.cbegin(), padding_before.cend());
    DimVector pad_after_vec(padding_after.cbegin(), padding_after.cend());

    for (int i = 0; i < ndims; ++i) {
      if (dst_pos_vec[i] < 0) {
        // When padding[i] < 0 , dst_pos_vec[i] will < 0 too , src_pos_vec[i] should adjust coords
        // relative and dst_pos_vec[i] will == 0
        src_pos_vec[i] -= dst_pos_vec[i];
        dst_pos_vec[i] = 0;
      }
    }

    DimVector extent_vec(ndims, 0);
    for (int i = 0; i < extent_vec.size(); ++i) {
      if (y->shape().At(i) < x->shape().At(i)) {
        extent_vec[i] = y->shape().At(i);
      } else {
        extent_vec[i] = x->shape().At(i);
        if (pad_before_vec[i] < 0) { extent_vec[i] = extent_vec[i] + pad_before_vec[i]; }
        if (pad_after_vec[i] < 0) { extent_vec[i] = extent_vec[i] + pad_after_vec[i]; }
      }
    }
    std::unique_ptr<primitive::CopyNd> primitive =
        primitive::NewPrimitive<primitive::CopyNdFactory>(device_type, ndims);
    CHECK(primitive);
    primitive->Launch(ctx->stream_ctx(), x->data_type(), x->shape().NumAxes(), y->mut_dptr(),
                      y->shape().ptr(), dst_pos_vec.data(), x->dptr(), x->shape().ptr(),
                      src_pos_vec.data(), extent_vec.data());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_PAD_KERNEL(dev, dtype)                                             \
  REGISTER_USER_KERNEL("pad").SetCreateFn<PadKernel<dev, dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == dev)                                              \
      & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

#ifdef WITH_CUDA
REGISTER_PAD_KERNEL(DeviceType::kGPU, double)
REGISTER_PAD_KERNEL(DeviceType::kGPU, float)
REGISTER_PAD_KERNEL(DeviceType::kGPU, float16)
REGISTER_PAD_KERNEL(DeviceType::kGPU, int32_t)
REGISTER_PAD_KERNEL(DeviceType::kGPU, int64_t)
REGISTER_PAD_KERNEL(DeviceType::kGPU, int8_t)
#endif
REGISTER_PAD_KERNEL(DeviceType::kCPU, double)
REGISTER_PAD_KERNEL(DeviceType::kCPU, float)
REGISTER_PAD_KERNEL(DeviceType::kCPU, int32_t)
REGISTER_PAD_KERNEL(DeviceType::kCPU, int64_t)
REGISTER_PAD_KERNEL(DeviceType::kCPU, int8_t)

template<DeviceType device_type, typename T>
class PadGradKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  PadGradKernel() = default;
  ~PadGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    size_t out_bytes_size = dx->shape().elem_cnt() * GetSizeOfDataType(dx->data_type());
    void* dst = dx->mut_dptr();
    Memset<device_type>(ctx->device_ctx(), dst, 0, out_bytes_size);

    if ((dy->shape().NumAxes() > 0 && dy->shape().elem_cnt() == 0)
        || (dx->shape().NumAxes() > 0 && dx->shape().elem_cnt() == 0)) {
      // if input/output is 0-shape tensor, than do nothing and return
      return;
    }

    const auto& padding_before = ctx->Attr<std::vector<int64_t>>("padding_before");
    const auto& padding_after = ctx->Attr<std::vector<int64_t>>("padding_after");
    const int64_t ndims = dy->shape().NumAxes();

    DimVector dst_pos_vec(ndims, 0);
    DimVector src_pos_vec(padding_before.cbegin(), padding_before.cend());
    DimVector pad_before_vec(padding_before.cbegin(), padding_before.cend());
    DimVector pad_after_vec(padding_after.cbegin(), padding_after.cend());

    for (int i = 0; i < ndims; ++i) {
      if (src_pos_vec[i] < 0) {
        dst_pos_vec[i] -= src_pos_vec[i];
        src_pos_vec[i] = 0;
      }
    }

    DimVector extent_vec(ndims, 0);
    for (int i = 0; i < extent_vec.size(); ++i) {
      if (dy->shape().At(i) < dx->shape().At(i)) {
        extent_vec[i] = dy->shape().At(i);
      } else {
        extent_vec[i] = dx->shape().At(i);
        if (pad_before_vec[i] < 0) { extent_vec[i] = extent_vec[i] + pad_before_vec[i]; }
        if (pad_after_vec[i] < 0) { extent_vec[i] = extent_vec[i] + pad_after_vec[i]; }
      }
    }
    std::unique_ptr<primitive::CopyNd> primitive =
        primitive::NewPrimitive<primitive::CopyNdFactory>(device_type, ndims);
    CHECK(primitive);
    primitive->Launch(ctx->stream_ctx(), dy->data_type(), ndims, dst, dx->shape().ptr(),
                      dst_pos_vec.data(), dy->dptr(), dy->shape().ptr(), src_pos_vec.data(),
                      extent_vec.data());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_PAD_GRAD_KERNEL(dev, dtype)            \
  REGISTER_USER_KERNEL("pad_grad")                      \
      .SetCreateFn<PadGradKernel<dev, dtype>>()         \
      .SetIsMatchedHob((user_op::HobDeviceTag() == dev) \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

#ifdef WITH_CUDA
REGISTER_PAD_GRAD_KERNEL(DeviceType::kGPU, double)
REGISTER_PAD_GRAD_KERNEL(DeviceType::kGPU, float)
REGISTER_PAD_GRAD_KERNEL(DeviceType::kGPU, float16)
REGISTER_PAD_GRAD_KERNEL(DeviceType::kGPU, int32_t)
REGISTER_PAD_GRAD_KERNEL(DeviceType::kGPU, int64_t)
REGISTER_PAD_GRAD_KERNEL(DeviceType::kGPU, int8_t)
#endif
REGISTER_PAD_GRAD_KERNEL(DeviceType::kCPU, double)
REGISTER_PAD_GRAD_KERNEL(DeviceType::kCPU, float)
REGISTER_PAD_GRAD_KERNEL(DeviceType::kCPU, int32_t)
REGISTER_PAD_GRAD_KERNEL(DeviceType::kCPU, int64_t)
REGISTER_PAD_GRAD_KERNEL(DeviceType::kCPU, int8_t)

}  // namespace oneflow
