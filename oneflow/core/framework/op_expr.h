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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_EXPR_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_EXPR_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/user_op_conf.pb.h"
#include "oneflow/core/framework/user_op_registry.h"
#include "oneflow/core/framework/arg_tuple.h"

namespace oneflow {
namespace one {

class OpExprGradFunctionIf;
class OpExprGradClosure;

class OpExpr {
 public:
  virtual ~OpExpr() = default;
  virtual const std::string& op_type_name() const = 0;

  virtual int input_size() const = 0;
  virtual int output_size() const = 0;

  virtual Maybe<bool> IsGradDisabled() const = 0;

  virtual Maybe<OpExprGradClosure> GetOrCreateOpGradClosure() const = 0;

 protected:
  OpExpr() = default;
};

class BuiltinOpExpr : public OpExpr {
 public:
  explicit BuiltinOpExpr(const std::string& op_name, const std::vector<std::string>& indexed_ibns,
                         const std::vector<std::string>& indexed_obns);

  virtual ~BuiltinOpExpr() = default;

  const std::string& op_name() const { return op_name_; }

  int input_size() const override { return input_arg_tuple_->size(); }
  int output_size() const override { return output_arg_tuple_->size(); }

  const std::shared_ptr<const ArgTuple>& input_arg_tuple() const { return input_arg_tuple_; }
  const std::shared_ptr<const ArgTuple>& output_arg_tuple() const { return output_arg_tuple_; }

  const std::vector<std::string>& indexed_ibns() const { return input_arg_tuple_->indexed_bns(); }
  const std::vector<std::string>& indexed_obns() const { return output_arg_tuple_->indexed_bns(); }
  const std::vector<std::pair<std::string, int32_t>>& indexed_input_pairs() const {
    return input_arg_tuple_->indexed_arg_name_and_index();
  }
  const std::vector<std::pair<std::string, int32_t>>& indexed_output_pairs() const {
    return output_arg_tuple_->indexed_arg_name_and_index();
  }

  virtual Maybe<void> BuildOpConf(OperatorConf* op_conf, const AttrMap& attrs) const = 0;

 protected:
  std::string op_name_;
  std::shared_ptr<const ArgTuple> input_arg_tuple_;
  std::shared_ptr<const ArgTuple> output_arg_tuple_;
};

class TensorMeta;

template<typename ProtoType>
class BuiltinOpExprImpl : public BuiltinOpExpr {
 public:
  static Maybe<BuiltinOpExprImpl<ProtoType>> New(const std::string& op_name, ProtoType&& op_proto,
                                                 const std::vector<std::string>& indexed_ibns,
                                                 const std::vector<std::string>& indexed_obns) {
    return std::shared_ptr<BuiltinOpExprImpl<ProtoType>>(
        new BuiltinOpExprImpl<ProtoType>(op_name, std::move(op_proto), indexed_ibns, indexed_obns));
  }

  virtual ~BuiltinOpExprImpl() = default;

  const ProtoType& proto() const { return op_proto_; }
  ProtoType* mutable_proto() { return &op_proto_; }

  const std::string& op_type_name() const override;

  Maybe<bool> IsGradDisabled() const override;

  Maybe<OpExprGradClosure> GetOrCreateOpGradClosure() const override;

  Maybe<void> BuildOpConf(OperatorConf* op_conf, const AttrMap& attrs) const override;

 protected:
  explicit BuiltinOpExprImpl(const std::string& op_name, ProtoType&& op_proto,
                             const std::vector<std::string>& indexed_ibns,
                             const std::vector<std::string>& indexed_obns)
      : BuiltinOpExpr(op_name, indexed_ibns, indexed_obns), op_proto_(std::move(op_proto)) {}

  ProtoType op_proto_;
  mutable std::shared_ptr<OpExprGradFunctionIf> op_grad_func_;
};

class StatefulLocalOpKernel;
class ConsistentTensorInferCache;

class UserOpExpr final : public BuiltinOpExprImpl<UserOpConf> {
 public:
  UserOpExpr() = delete;
  virtual ~UserOpExpr() = default;

  static Maybe<UserOpExpr> New(const std::string& op_name, UserOpConf&& op_proto,
                               const std::vector<std::string>& indexed_ibns,
                               const std::vector<std::string>& indexed_obns);

  const AttrMap& base_attrs() const { return base_attrs_; }

  Maybe<StatefulLocalOpKernel> MutKernel4Device(const Device& device) const;

  bool has_device_infer_fn() const { return static_cast<bool>(device_infer_fn_); }
  Maybe<void> InferLogicalShapeAndDType(
      const AttrMap& attrs, const std::string& device_tag,
      const std::function<const TensorMeta*(int32_t)>& TensorMeta4InputIndex,
      const std::function<TensorMeta*(int32_t)>& TensorMeta4OutputIndex) const;
  Maybe<Symbol<Device>> InferDevices(const AttrMap& attrs, const TensorTuple& inputs,
                                     TensorTuple* outputs) const;
  ConsistentTensorInferCache* mut_consistent_tensor_infer_cache() const {
    return consistent_tensor_infer_cache_.get();
  }

 private:
  UserOpExpr(const std::string& op_name, UserOpConf&& proto, const AttrMap& base_attrs,
             const std::vector<std::string>& indexed_ibns,
             const std::vector<std::string>& indexed_obns);
  Maybe<void> Init(const std::shared_ptr<const UserOpExpr>& self);
  AttrMap base_attrs_;
  user_op::TensorDescInferFn shape_infer_fn_;
  user_op::DataTypeInferFn dtype_infer_fn_;
  user_op::DeviceInferFn device_infer_fn_;
  mutable HashMap<Device, std::shared_ptr<StatefulLocalOpKernel>> device2kernel_;
  std::shared_ptr<ConsistentTensorInferCache> consistent_tensor_infer_cache_;
};

class ConsistentToConsistentOpExpr : public OpExpr {
 public:
  virtual ~ConsistentToConsistentOpExpr() = default;

  static Maybe<ConsistentToConsistentOpExpr> New(
      const std::string& op_name, Symbol<cfg::ParallelDistribution> parallel_distribution);
  Maybe<Symbol<cfg::ParallelDistribution>> parallel_distribution() const {
    return parallel_distribution_;
  }
  Maybe<Symbol<cfg::ParallelDistribution>> grad_parallel_distribution() const {
    return grad_parallel_distribution_.value();
  }
  const std::string& op_name() const { return op_name_; }
  const std::string& op_type_name() const override;
  int input_size() const override { return 1; }
  int output_size() const override { return 1; }

  Maybe<bool> IsGradDisabled() const override { return false; }
  Maybe<OpExprGradClosure> GetOrCreateOpGradClosure() const override;

 protected:
  ConsistentToConsistentOpExpr(const std::string& op_name,
                               Symbol<cfg::ParallelDistribution> parallel_distribution);

  std::string op_name_;
  Symbol<cfg::ParallelDistribution> parallel_distribution_;
  Optional<Symbol<cfg::ParallelDistribution>> grad_parallel_distribution_;
  mutable std::shared_ptr<OpExprGradFunctionIf> op_grad_func_;
};

class CastConsistentOpExpr : public OpExpr {
 public:
  virtual ~CastConsistentOpExpr() = default;

  const std::string& op_name() const { return op_name_; }
  int input_size() const override { return 1; }
  int output_size() const override { return 1; }

  Maybe<bool> IsGradDisabled() const override { return false; }

 protected:
  CastConsistentOpExpr(const std::string& op_name);

  std::string op_name_;
  mutable std::shared_ptr<OpExprGradFunctionIf> op_grad_func_;
};

class CastToConsistentOpExpr final : public CastConsistentOpExpr {
 public:
  ~CastToConsistentOpExpr() = default;

  static Maybe<CastToConsistentOpExpr> New(const std::string& op_name);

  const std::string& op_type_name() const override;
  Maybe<OpExprGradClosure> GetOrCreateOpGradClosure() const override;

 private:
  CastToConsistentOpExpr(const std::string& op_name);
};

class CastFromConsistentOpExpr final : public CastConsistentOpExpr {
 public:
  ~CastFromConsistentOpExpr() = default;

  static Maybe<CastFromConsistentOpExpr> New(const std::string& op_name);

  const std::string& op_type_name() const override;
  Maybe<OpExprGradClosure> GetOrCreateOpGradClosure() const override;

 private:
  CastFromConsistentOpExpr(const std::string& op_name);
};
// NOTE(chengcheng): For Lazy nn.Graph Feed/Fetch EagerTensor to/from LazyTensor.
using FeedInputOpExpr = BuiltinOpExprImpl<FeedInputOpConf>;
using FeedVariableOpExpr = BuiltinOpExprImpl<FeedVariableOpConf>;
using FetchOutputOpExpr = BuiltinOpExprImpl<FetchOutputOpConf>;

using VariableOpExpr = BuiltinOpExprImpl<VariableOpConf>;
using CastToMirroredOpExpr = BuiltinOpExprImpl<CastToMirroredOpConf>;
using CastFromMirroredOpExpr = BuiltinOpExprImpl<CastFromMirroredOpConf>;
using DistributeSplitOpExpr = BuiltinOpExprImpl<DistributeSplitOpConf>;
using DistributeCloneOpExpr = BuiltinOpExprImpl<DistributeCloneOpConf>;
using DistributeConcatOpExpr = BuiltinOpExprImpl<DistributeConcatOpConf>;
using DistributeAddOpExpr = BuiltinOpExprImpl<DistributeAddOpConf>;

class SelectFirstOpExpr final : public OpExpr {
 public:
  static Maybe<SelectFirstOpExpr> New() {
    return std::shared_ptr<SelectFirstOpExpr>(new SelectFirstOpExpr());
  }

  const std::string& op_type_name() const override {
    static const std::string kOpTypeName = "select_first";
    return kOpTypeName;
  }

  int input_size() const override {
    UNIMPLEMENTED();
    return 0;
  }

  int output_size() const override { return 1; }

  Maybe<bool> IsGradDisabled() const override { return false; }

  Maybe<OpExprGradClosure> GetOrCreateOpGradClosure() const override;

 private:
  SelectFirstOpExpr() = default;

  mutable std::shared_ptr<OpExprGradFunctionIf> op_grad_func_;
};

class OpExprInterpState;
// TODO(): Finish the class definition of `FunctionOpExpr`.
class FunctionOpExpr : public OpExpr {
 public:
  using FType = std::function<Maybe<void>(const std::shared_ptr<OpExprInterpState>& /*ctx*/,
                                          const TensorTuple& /*inputs or out_grads*/,
                                          TensorTuple* /*outputs or in_grads*/)>;

  FunctionOpExpr(const FType& forward, const FType& backward)
      : OpExpr(), forward_(forward), backward_(backward) {}
  virtual ~FunctionOpExpr() = default;

  const std::string& op_type_name() const override {
    static const std::string& name("function");
    return name;
  }

  int input_size() const override {
    UNIMPLEMENTED();
    return 0;
  }
  int output_size() const override {
    UNIMPLEMENTED();
    return 0;
  }

  FType forward() const { return forward_; }
  FType backward() const { return backward_; }

  std::shared_ptr<const OpExprInterpState> state() const { return state_; }
  std::shared_ptr<OpExprInterpState> mutable_state() { return state_; }

  Maybe<bool> IsGradDisabled() const override { return false; }
  Maybe<OpExprGradClosure> GetOrCreateOpGradClosure() const override { OF_UNIMPLEMENTED(); }

 private:
  FType forward_;
  FType backward_;
  std::shared_ptr<OpExprInterpState> state_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_EXPR_H_
