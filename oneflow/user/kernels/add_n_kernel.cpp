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
#include "oneflow/core/primitive/include/add.h"
#include "oneflow/core/primitive/cuda/cuda_graph_support.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/user/kernels/op_kernel_state_wrapper.h"

namespace oneflow {

namespace user_op {

namespace {

template<typename Context>
std::unique_ptr<primitive::Add> NewAddPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("out", 0)->data_type();
  return primitive::NewPrimitive<primitive::AddFactory>(ctx->device_type(), data_type);
}

using AddKernelState = OpKernelStateWrapper<std::unique_ptr<primitive::Add>>;

class AddNKernel : public OpKernel, public CudaGraphSupport {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AddNKernel);
  AddNKernel() = default;
  ~AddNKernel() override = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  std::shared_ptr<OpKernelState> CreateOpKernelState(KernelInitContext* ctx) const override {
    return std::make_shared<AddKernelState>(NewAddPrimitive(ctx));
  }

 private:
  bool IsCudaGraphSupported(KernelInitContext* ctx, OpKernelState* state) const override {
    return IsCudaGraphPrimitive(CHECK_NOTNULL(dynamic_cast<AddKernelState*>(state))->Get().get());
  }
  void Compute(KernelComputeContext* ctx, OpKernelState* state) const override {
    Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const DataType data_type = out->data_type();
    const size_t count = out->shape().elem_cnt();
    size_t in_num = ctx->inputs().size();
    std::vector<const void*> srcs(in_num);
    for (size_t i = 0; i < in_num; ++i) {
      const Tensor* in_i = ctx->Tensor4ArgNameAndIndex("in", i);
      CHECK_EQ(in_i->shape().elem_cnt(), count);
      CHECK_EQ(in_i->data_type(), data_type);
      srcs[i] = in_i->template dptr();
    }
    CHECK_NOTNULL(dynamic_cast<AddKernelState*>(state))
        ->Get()
        ->Launch(ctx->stream_ctx(), srcs.data(), in_num, out->mut_dptr(), count);
  }
};

hob::HobContextGetter<KernelRegContext, bool> AddPrimitiveExists() {
  return HobCtxGetter<bool>("AddPrimitiveExists", [](const KernelRegContext& ctx) {
    return NewAddPrimitive(&ctx).operator bool();
  });
}

REGISTER_USER_KERNEL("add_n")
    .SetCreateFn<AddNKernel>()
    .SetIsMatchedHob(AddPrimitiveExists() == true)
    .SetInplaceProposalFn([](const InferContext&,
                             const AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
      OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));
      return Maybe<void>::Ok();
    });

}  // namespace

}  // namespace user_op

}  // namespace oneflow
