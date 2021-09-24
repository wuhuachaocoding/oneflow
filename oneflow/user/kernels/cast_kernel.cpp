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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/primitive/include/cast.h"
#include "oneflow/core/primitive/cuda/cuda_graph_support.h"
#include "oneflow/user/kernels/op_kernel_state_wrapper.h"

namespace oneflow {

namespace user_op {

namespace {

template<typename Context>
std::unique_ptr<primitive::Cast> NewCastPrimitive(Context* ctx) {
  const DataType in_data_type = ctx->TensorDesc4ArgNameAndIndex("in", 0)->data_type();
  const DataType out_data_type = ctx->TensorDesc4ArgNameAndIndex("out", 0)->data_type();
  return primitive::NewPrimitive<primitive::CastFactory>(ctx->device_type(), in_data_type,
                                                         out_data_type);
}

using CastKernelState = OpKernelStateWrapper<std::unique_ptr<primitive::Cast>>;

class CastKernel final : public OpKernel, public user_op::CudaGraphSupport {
 public:
  CastKernel() = default;
  ~CastKernel() = default;

  std::shared_ptr<OpKernelState> CreateOpKernelState(KernelInitContext* ctx) const override {
    return std::make_shared<CastKernelState>(NewCastPrimitive(ctx));
  }

 private:
  bool IsCudaGraphSupported(KernelInitContext* ctx, OpKernelState* state) const override {
    return IsCudaGraphPrimitive(CHECK_NOTNULL(dynamic_cast<CastKernelState*>(state))->Get().get());
  }
  void Compute(KernelComputeContext* ctx, OpKernelState* state) const override {
    const Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    Tensor* output_tenor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t elem_cnt = input_tensor->shape().elem_cnt();
    CHECK_EQ(output_tenor->shape().elem_cnt(), elem_cnt);
    CHECK_NOTNULL(dynamic_cast<CastKernelState*>(state))
        ->Get()
        ->Launch(ctx->stream_ctx(), input_tensor->dptr(), output_tenor->mut_dptr(), elem_cnt);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

hob::HobContextGetter<user_op::KernelRegContext, bool> CastPrimitiveExists() {
  return user_op::HobCtxGetter<bool>(
      "CastPrimitiveExists",
      [](const user_op::KernelRegContext& ctx) { return NewCastPrimitive(&ctx).operator bool(); });
}

REGISTER_USER_KERNEL("cast").SetCreateFn<CastKernel>().SetIsMatchedHob(CastPrimitiveExists()
                                                                       == true);
REGISTER_USER_KERNEL("cast_like")
    .SetCreateFn<CastKernel>()
    .SetIsMatchedHob(CastPrimitiveExists() == true);

}  // namespace

}  // namespace user_op

}  // namespace oneflow
