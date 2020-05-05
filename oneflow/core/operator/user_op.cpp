#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/user_op_util.h"
#include "oneflow/core/framework/op_registration.h"
#include "oneflow/core/framework/kernel_registration.h"
#include "oneflow/core/framework/tensor_desc.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/sbp_context.h"
#include "oneflow/core/framework/batch_axis_context.h"

namespace oneflow {

namespace {

BlobDesc* FindValidBlobDescOfBnsInOp(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const PbRpf<std::string>& bn_in_ops) {
  for (const std::string& bn_in_op : bn_in_ops) {
    BlobDesc* blob_desc = GetBlobDesc4BnInOp(bn_in_op);
    if (blob_desc) { return blob_desc; }
  }
  return nullptr;
}

user_op::TensorDesc GenTensorDescFromBlobDesc(const BlobDesc* blob_desc) {
  BlobDescProto proto;
  blob_desc->ToProto(&proto);
  return user_op::TensorDesc(proto);
}

}  // namespace

class UserOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UserOp);
  UserOp() = default;
  ~UserOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().user_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override;
  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext*, const SbpSignature* sbp_signature,
                                std::function<void(OpContext*)> EnrollOpCtx) const override;

 private:
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override;
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
  Maybe<void> InferBatchAxis(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      const ParallelDesc& parallel_desc, SbpSignatureList* sbp_sig_list) const override;
  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx,
      std::function<const BlobDesc&(const std::string&)> LogicalBlobDesc4BnInOp) const override;

  const user_op::OpRegistrationVal* val_;
};

class UserOpKernelRegContext final : public user_op::KernelRegContext {
 public:
  using ArgVec = std::vector<std::pair<std::string, int32_t>>;

  explicit UserOpKernelRegContext(const UserOp* user_op,
                                  std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx)
      : user_op::KernelRegContext(user_op::UserOpConfWrapper(user_op->op_conf())) {
    const auto& op_conf = user_op->op_conf();
    CHECK(op_conf.has_user_conf());

    device_type_ = op_conf.device_type();
    parallel_ctx_ = parallel_ctx;

    auto InitInOrOut = [&](const PbMap<std::string, UserOpConf::ListString>& arg_map,
                           ArgVec* arg_vec) {
      for (auto it = arg_map.begin(); it != arg_map.end(); ++it) {
        for (int32_t i = 0; i < it->second.s_size(); ++i) {
          arg_vec->emplace_back(std::make_pair(it->first, i));
        }
      }
    };
    InitInOrOut(op_conf.user_conf().input(), &inputs_);
    InitInOrOut(op_conf.user_conf().output(), &outputs_);

    {
#define INSERT_TO_ARG2TENSOR_DESC(prefix)                                                \
  for (const auto& bn : user_op->prefix##_bns()) {                                       \
    const BlobDesc* blob_desc = GetBlobDesc4BnInOp(bn);                                  \
    if (!blob_desc) { continue; }                                                        \
    arg2tensor_desc_.emplace(GenUnRepeatedBn(bn), GenTensorDescFromBlobDesc(blob_desc)); \
  }

      INSERT_TO_ARG2TENSOR_DESC(input)
      INSERT_TO_ARG2TENSOR_DESC(output)
      INSERT_TO_ARG2TENSOR_DESC(tmp)

#undef INSERT_TO_ARG2TENSOR_DESC
    }
  }
  ~UserOpKernelRegContext() = default;

  DeviceType device_type() const override { return device_type_; }
  const ParallelContext& parallel_ctx() const override { return *parallel_ctx_; }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    auto it = arg2tensor_desc_.find(std::make_pair(arg_name, index));
    if (it == arg2tensor_desc_.end()) { return nullptr; }
    return &(it->second);
  }
  const ArgVec& inputs() const override { return inputs_; }
  const ArgVec& outputs() const override { return outputs_; }

 private:
  ArgVec inputs_;
  ArgVec outputs_;
  DeviceType device_type_;
  const ParallelContext* parallel_ctx_;
  HashMap<std::pair<std::string, int32_t>, user_op::TensorDesc> arg2tensor_desc_;
};

class UserOpInferContext : public user_op::InferContext {
 public:
  using ArgVec = std::vector<std::pair<std::string, int32_t>>;

  UserOpInferContext(const OperatorConf& op_conf, const ParallelContext* parallel_ctx,
                     const SbpSignature* sbp_signature,
                     std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp)
      : user_op::InferContext(user_op::UserOpConfWrapper(op_conf)),
        parallel_ctx_(parallel_ctx),
        sbp_signature_(sbp_signature) {
    auto InitInOrOut = [&](const PbMap<std::string, UserOpConf::ListString>& arg_map,
                           ArgVec* arg_vec) {
      for (auto it = arg_map.begin(); it != arg_map.end(); ++it) {
        const std::string& arg_name = it->first;
        for (int32_t i = 0; i < it->second.s_size(); ++i) {
          BlobDesc* blob = GetBlobDesc4BnInOp(GenRepeatedBn(arg_name, i));
          auto key = std::make_pair(arg_name, i);
          arg2tensor_desc_.emplace(key, GenTensorDescFromBlobDesc(blob));
          arg_vec->emplace_back(std::make_pair(arg_name, i));
        }
      }
    };
    InitInOrOut(op_conf.user_conf().input(), &inputs_);
    InitInOrOut(op_conf.user_conf().output(), &outputs_);
  }
  ~UserOpInferContext() = default;

  user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                  int32_t index) override {
    return &(arg2tensor_desc_.at(std::make_pair(arg_name, index)));
  }
  Shape* Shape4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return arg2tensor_desc_.at(std::make_pair(arg_name, index)).mut_shape();
  }
  DataType* Dtype4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return arg2tensor_desc_.at(std::make_pair(arg_name, index)).mut_data_type();
  }
  bool* IsDynamic4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return arg2tensor_desc_.at(std::make_pair(arg_name, index)).mut_is_dynamic();
  }
  bool* IsTensorList4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return arg2tensor_desc_.at(std::make_pair(arg_name, index)).mut_is_tensor_list();
  }

  const ArgVec& inputs() const override { return inputs_; }
  const ArgVec& outputs() const override { return outputs_; }
  const ParallelContext& parallel_ctx() const override { return *parallel_ctx_; };
  const SbpParallel& SbpParallel4ArgNameAndIndex(const std::string& arg_name,
                                                 int32_t index) const override {
    const auto& bn2sbp = sbp_signature_->bn_in_op2sbp_parallel();
    std::string bn = GenRepeatedBn(arg_name, index);
    CHECK(bn2sbp.find(bn) != bn2sbp.end());
    return sbp_signature_->bn_in_op2sbp_parallel().at(bn);
  }

 private:
  ArgVec inputs_;
  ArgVec outputs_;
  const ParallelContext* parallel_ctx_;
  const SbpSignature* sbp_signature_;
  HashMap<std::pair<std::string, int32_t>, user_op::TensorDesc> arg2tensor_desc_;
};

class UserOpSbpContext : public user_op::SbpContext {
 public:
  using ArgVec = std::vector<std::pair<std::string, int32_t>>;

  UserOpSbpContext(const OperatorConf& op_conf, SbpSignatureList* sbp_sig_list,
                   DeviceType device_type, int64_t parallel_num,
                   std::function<Maybe<const BlobDesc*>(const std::string&)> LogicalBlobDesc4Ibn)
      : user_op::SbpContext(user_op::UserOpConfWrapper(op_conf), sbp_sig_list, device_type,
                            parallel_num) {
    const auto& user_op_conf = op_conf.user_conf();
    for (auto it = user_op_conf.input().begin(); it != user_op_conf.input().end(); ++it) {
      const std::string& arg_name = it->first;
      for (int32_t i = 0; i < it->second.s_size(); ++i) {
        const BlobDesc* blob = CHECK_JUST(LogicalBlobDesc4Ibn(GenRepeatedBn(arg_name, i)));
        arg2tensor_desc_.emplace(std::make_pair(arg_name, i), GenTensorDescFromBlobDesc(blob));
        inputs_.emplace_back(std::make_pair(arg_name, i));
      }
    }
    for (auto it = user_op_conf.output().begin(); it != user_op_conf.output().end(); ++it) {
      const std::string& arg_name = it->first;
      for (int32_t i = 0; i < it->second.s_size(); ++i) {
        outputs_.emplace_back(std::make_pair(arg_name, i));
      }
    }
  }
  ~UserOpSbpContext() = default;

  const user_op::TensorDesc& LogicalTensorDesc4InputArgNameAndIndex(
      const std::string& input_arg_name, int32_t index) const override {
    return arg2tensor_desc_.at(std::make_pair(input_arg_name, index));
  }
  const ArgVec& inputs() const override { return inputs_; }
  const ArgVec& outputs() const override { return outputs_; }

 private:
  ArgVec inputs_;
  ArgVec outputs_;
  HashMap<std::pair<std::string, int32_t>, user_op::TensorDesc> arg2tensor_desc_;
};

class UserOpBatchAxisContext : public user_op::BatchAxisContext {
 public:
  using ArgVec = std::vector<std::pair<std::string, int32_t>>;

  UserOpBatchAxisContext(const OperatorConf& op_conf,
                         std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp,
                         std::function<const BlobDesc&(const std::string&)> LogicalBlobDesc4Ibn)
      : user_op::BatchAxisContext(user_op::UserOpConfWrapper(op_conf)) {
    const auto& user_op_conf = op_conf.user_conf();
    for (auto it = user_op_conf.input().begin(); it != user_op_conf.input().end(); ++it) {
      const std::string& arg_name = it->first;
      for (int32_t i = 0; i < it->second.s_size(); ++i) {
        std::string ibn = GenRepeatedBn(arg_name, i);
        const BlobDesc& blob = LogicalBlobDesc4Ibn(ibn);
        arg2tensor_desc_.emplace(std::make_pair(arg_name, i), GenTensorDescFromBlobDesc(&blob));
        arg2batch_axis_.emplace(std::make_pair(arg_name, i), BatchAxis4BnInOp(ibn));
        inputs_.emplace_back(std::make_pair(arg_name, i));
      }
    }
    for (auto it = user_op_conf.output().begin(); it != user_op_conf.output().end(); ++it) {
      const std::string& arg_name = it->first;
      for (int32_t i = 0; i < it->second.s_size(); ++i) {
        arg2batch_axis_.emplace(std::make_pair(arg_name, i),
                                BatchAxis4BnInOp(GenRepeatedBn(arg_name, i)));
        outputs_.emplace_back(std::make_pair(arg_name, i));
      }
    }
  }
  ~UserOpBatchAxisContext() = default;

  const user_op::TensorDesc& LogicalTensorDesc4InputArgNameAndIndex(
      const std::string& input_arg_name, int32_t index) const override {
    return arg2tensor_desc_.at(std::make_pair(input_arg_name, index));
  }
  const ArgVec& inputs() const { return inputs_; }
  const ArgVec& outputs() const { return outputs_; }

  OptInt64* BatchAxis4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return arg2batch_axis_.at(std::make_pair(arg_name, index));
  }

 private:
  ArgVec inputs_;
  ArgVec outputs_;
  HashMap<std::pair<std::string, int32_t>, user_op::TensorDesc> arg2tensor_desc_;
  HashMap<std::pair<std::string, int32_t>, OptInt64*> arg2batch_axis_;
};

void UserOp::InitFromOpConf() {
  CHECK(op_conf().has_user_conf());
  for (const auto& pair : op_conf().user_conf().input()) {
    EnrollRepeatedInputBn(pair.first, pair.second.s_size());
  }
  for (const auto& pair : op_conf().user_conf().output()) {
    EnrollRepeatedOutputBn(pair.first, pair.second.s_size());
  }
  EnrollTmpBn(GenRepeatedBn("tmp_buffer", 0));
  val_ = user_op::LookUpInOpRegistry(op_conf().user_conf().op_type_name());
  if (val_ != nullptr) {
    user_op::GetInputArgModifier GetInputArgModifierFn =
        [&](const std::string& in_arg_name, int32_t in_arg_index) -> user_op::InputArgModifier* {
      std::string ibn = GenRepeatedBn(in_arg_name, in_arg_index);
      if (std::find(input_bns().begin(), input_bns().end(), ibn) != input_bns().end()) {
        return MutInputBlobModifier4Ibn(ibn);
      }
      return nullptr;
    };
    val_->input_arg_modify_fn(GetInputArgModifierFn);
  }
}

Maybe<void> UserOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx,
                                   const SbpSignature* sbp_signature,
                                   std::function<void(OpContext*)> EnrollOpCtx) const {
  JUST(InferOutBlobDescs(GetBlobDesc4BnInOp, parallel_ctx, sbp_signature, EnrollOpCtx));

  // tmp buffer size must be inferred after out shape/dtype
  UserOpInferContext infer_ctx(op_conf(), parallel_ctx, sbp_signature, GetBlobDesc4BnInOp);
  const user_op::KernelRegistrationVal* kernel_reg_val = user_op::LookUpInKernelRegistry(
      op_conf().user_conf().op_type_name(),
      UserOpKernelRegContext(this, GetBlobDesc4BnInOp, parallel_ctx));
  CHECK_OR_RETURN(kernel_reg_val != nullptr)
      << "cannot find op_type: " << op_conf().user_conf().op_type_name() << " in kernel registry !";

  size_t tmp_size = kernel_reg_val->infer_tmp_size_fn(&infer_ctx);
  if (tmp_size > 0) {
    BlobDesc* tmp_buffer_blob = GetBlobDesc4BnInOp(GenRepeatedBn("tmp_buffer", 0));
    CHECK(tmp_buffer_blob != nullptr);
    tmp_buffer_blob->set_data_type(DataType::kChar);
    tmp_buffer_blob->mut_shape() = Shape({static_cast<int64_t>(tmp_size)});
  }

  // get inplace proposal in/out blob pair
  UserOpCtx* op_ctx = new UserOpCtx();
  HashSet<std::string> bn_in_op_unique_check;
  user_op::AddInplaceArgPair AddInplaceArgPairFn =
      [&](const std::string& out_arg_name, int32_t out_arg_index, const std::string& in_arg_name,
          int32_t in_arg_index, bool is_mutable) -> Maybe<void> {
    std::string ibn = GenRepeatedBn(in_arg_name, in_arg_index);
    std::string obn = GenRepeatedBn(out_arg_name, out_arg_index);
    if (is_mutable) {
      op_ctx->mut_inplace_obn2ibn.emplace(obn, ibn);
    } else {
      op_ctx->con_inplace_obn2ibn.emplace(obn, ibn);
    }

    CHECK_OR_RETURN(std::find(input_bns().begin(), input_bns().end(), ibn) != input_bns().end())
        << "Cannot find input_arg_name : " << in_arg_name << " input_arg_index : " << in_arg_index
        << " in op_name: " << op_conf().name();
    CHECK_OR_RETURN(std::find(output_bns().begin(), output_bns().end(), obn) != output_bns().end())
        << "Cannot find output_arg_name : " << out_arg_name
        << " output_arg_index : " << out_arg_index << " in op_name: " << op_conf().name();

    std::string repeated_ibn_err_msg =
        "Cannot repeated set inplace proposal for same intput arg : " + in_arg_name
        + " index : " + std::to_string(in_arg_index) + " in op_name: " + op_conf().name();
    std::string repeated_obn_err_msg =
        "Cannot repeated set inplace proposal for same output arg : " + out_arg_name
        + " index : " + std::to_string(out_arg_index) + " in op_name: " + op_conf().name();
    CHECK_OR_RETURN(bn_in_op_unique_check.insert(ibn).second) << repeated_ibn_err_msg;
    CHECK_OR_RETURN(bn_in_op_unique_check.insert(obn).second) << repeated_obn_err_msg;
    return Maybe<void>::Ok();
  };
  JUST(kernel_reg_val->inplace_proposal_fn(infer_ctx, AddInplaceArgPairFn));
  op_ctx->sbp_sig = *sbp_signature;
  EnrollOpCtx(op_ctx);
  return Maybe<void>::Ok();
}

Maybe<void> UserOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  CHECK_OR_RETURN(val_ != nullptr)
      << "cannot find op_type: " << op_conf().user_conf().op_type_name() << " in op registry!";
  // default method set output blob desc (such as Dtype, is_dynamic, is_tensor_list)
  // set out blob desc attr as first input blob desc (if has)
  BlobDesc* first_in_blob_desc = FindValidBlobDescOfBnsInOp(GetBlobDesc4BnInOp, input_bns());
  if (first_in_blob_desc) {
    for (const std::string& obn : output_bns()) {
      GetBlobDesc4BnInOp(obn)->CopyMetaFrom(*first_in_blob_desc);
    }
  }

  UserOpInferContext infer_ctx(op_conf(), parallel_ctx, sbp_signature, GetBlobDesc4BnInOp);

  JUST(val_->tensor_desc_infer_fn(&infer_ctx));
  for (const auto& pair : infer_ctx.outputs()) {
    BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(GenRepeatedBn(pair.first, pair.second));
    out_blob_desc->set_data_type(*(infer_ctx.Dtype4ArgNameAndIndex(pair.first, pair.second)));
    out_blob_desc->mut_shape() = *(infer_ctx.Shape4ArgNameAndIndex(pair.first, pair.second));
    out_blob_desc->set_is_dynamic(*infer_ctx.IsDynamic4ArgNameAndIndex(pair.first, pair.second));
    out_blob_desc->set_is_tensor_list(
        *infer_ctx.IsTensorList4ArgNameAndIndex(pair.first, pair.second));
  }
  return Maybe<void>::Ok();
}

LogicalBlobId UserOp::ibn2lbi(const std::string& input_bn) const {
  auto pair = GenUnRepeatedBn(input_bn);
  return GenLogicalBlobId(op_conf().user_conf().input().at(pair.first).s(pair.second));
}

LogicalBlobId UserOp::obn2lbi(const std::string& output_bn) const {
  auto pair = GenUnRepeatedBn(output_bn);
  auto ret = GenLogicalBlobId(op_conf().user_conf().output().at(pair.first).s(pair.second));
  CHECK_EQ(ret.op_name(), op_conf().name());
  CHECK_EQ(ret.blob_name(), output_bn);
  return ret;
}

Maybe<void> UserOp::InferBatchAxis(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  CHECK_OR_RETURN(val_ != nullptr)
      << "cannot find op_type: " << op_conf().user_conf().op_type_name() << " in op registry!";
  UserOpBatchAxisContext batch_axis_ctx(op_conf(), BatchAxis4BnInOp, LogicalBlobDesc4Ibn);
  JUST(val_->batch_axis_infer_fn(&batch_axis_ctx));
  return Maybe<void>::Ok();
}

Maybe<void> UserOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    const ParallelDesc& parallel_desc, SbpSignatureList* sbp_sig_list) const {
  CHECK_OR_RETURN(val_ != nullptr)
      << "cannot find op_type: " << op_conf().user_conf().op_type_name() << " in op registry!";
  UserOpSbpContext sbp_ctx(op_conf(), sbp_sig_list, parallel_desc.device_type(),
                           parallel_desc.parallel_num(), LogicalBlobDesc4Ibn);
  JUST(val_->get_sbp_fn(&sbp_ctx));
  // Add Broadcast for source user op tick input
  std::string tick_bn = GenRepeatedBn(user_op::kUserSourceOpTickInputArgName, 0);
  if (val_->op_def.input_size() == 0 && input_bns().size() == 1) {
    CHECK_OR_RETURN(input_bns().Get(0) == tick_bn)
        << "user op_name: " << op_conf().name()
        << " op_type_name: " << op_conf().user_conf().op_type_name()
        << " set ERROR input arg name : " << input_bns().Get(0) << " because NO input in op def";
    for (auto& sbp_sig : *sbp_sig_list->mutable_sbp_signature()) {
      auto* bn2sbp = sbp_sig.mutable_bn_in_op2sbp_parallel();
      if (bn2sbp->find(tick_bn) == bn2sbp->end()) {
        (*bn2sbp)[tick_bn].mutable_broadcast_parallel();
      }
    }
  }
  // Check valid
  for (const auto& sbp_sig : sbp_sig_list->sbp_signature()) {
    const auto& bn2sbp = sbp_sig.bn_in_op2sbp_parallel();
    for (const auto& ibn : input_bns()) {
      auto pair = GenUnRepeatedBn(ibn);
      CHECK_OR_RETURN(bn2sbp.find(ibn) != bn2sbp.end())
          << "In op_name: " << op_conf().name()
          << " op_type_name: " << op_conf().user_conf().op_type_name()
          << ", input_arg_name : " << pair.first << " input_arg_index : " << pair.second
          << " have NOT set sbp signature";
    }
    for (const auto& obn : output_bns()) {
      auto pair = GenUnRepeatedBn(obn);
      CHECK_OR_RETURN(bn2sbp.find(obn) != bn2sbp.end())
          << "In op_name: " << op_conf().name()
          << " op_type_name: " << op_conf().user_conf().op_type_name()
          << ", output_arg_name : " << pair.first << " output_arg_index : " << pair.second
          << " have NOT set sbp signature";
    }
  }
  return Maybe<void>::Ok();
}

void UserOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx,
    std::function<const BlobDesc&(const std::string&)> LogicalBlobDesc4BnInOp) const {
  const UserOpCtx* user_op_ctx = static_cast<const UserOpCtx*>(op_ctx);
  auto user_conf = kernel_conf->mutable_user_conf();
  *(user_conf->mutable_parallel_ctx()) = *parallel_ctx;
  *(user_conf->mutable_sbp_sig()) = user_op_ctx->sbp_sig;
#define BLOB_DESCS_TO_PROTO(prefix)                         \
  for (const auto& bn : prefix##_bns()) {                   \
    BlobDescProto proto;                                    \
    const BlobDesc* blob_desc = GetBlobDesc4BnInOp(bn);     \
    if (!blob_desc) { continue; }                           \
    blob_desc->ToProto(&proto);                             \
    (*user_conf->mutable_bn_in_op2blob_desc())[bn] = proto; \
  }

  BLOB_DESCS_TO_PROTO(input)
  BLOB_DESCS_TO_PROTO(output)
  BLOB_DESCS_TO_PROTO(tmp)

#undef BLOB_DESCS_TO_PROTO
}

REGISTER_OP(OperatorConf::kUserConf, UserOp);

}  // namespace oneflow
