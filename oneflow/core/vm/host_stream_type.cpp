#include "oneflow/core/vm/host_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/mem_instruction.msg.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/thread_ctx.msg.h"
#include "oneflow/core/vm/naive_instruction_status_querier.h"
#include "oneflow/core/vm/mem_buffer_object.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/flat_msg_view.h"

namespace oneflow {
namespace vm {

namespace {

class CudaMallocHostInstructionType final : public InstructionType {
 public:
  CudaMallocHostInstructionType() = default;
  ~CudaMallocHostInstructionType() override = default;

  using stream_type = HostStreamType;

  void Infer(InstrCtx* instr_ctx) const override {
    MemBufferObjectType* mem_buffer_object_type = nullptr;
    size_t size = 0;
    int64_t device_id = 0;
    {
      FlatMsgView<MallocInstruction> view;
      CHECK(view.Match(instr_ctx->instr_msg().operand()));
      size = view->size();
      MirroredObject* mirrored_object = instr_ctx->mut_operand_type(view->mem_buffer());
      mem_buffer_object_type = mirrored_object->Mutable<MemBufferObjectType>();
      device_id = instr_ctx->mut_instr_chain()->stream().thread_ctx().device_id();
    }
    mem_buffer_object_type->set_size(size);
    auto* mem_case = mem_buffer_object_type->mut_mem_case();
    mem_case->mutable_host_mem()->mutable_cuda_pinned_mem()->set_device_id(device_id);
  }
  void Compute(InstrCtx* instr_ctx) const override {
    const MemBufferObjectType* buffer_type = nullptr;
    MemBufferObjectValue* buffer_value = nullptr;
    char* dptr = nullptr;
    {
      FlatMsgView<MallocInstruction> view;
      CHECK(view.Match(instr_ctx->instr_msg().operand()));
      const auto& operand = view->mem_buffer();
      buffer_type = &instr_ctx->mut_operand_type(operand)->Get<MemBufferObjectType>();
      buffer_value = instr_ctx->mut_operand_value(operand)->Mutable<MemBufferObjectValue>();
    }
    CudaCheck(cudaMallocHost(&dptr, buffer_type->size()));
    buffer_value->reset_data(dptr);
  }
};
COMMAND(RegisterInstructionType<CudaMallocHostInstructionType>("CudaMallocHost"));

class MallocInstructionType final : public InstructionType {
 public:
  MallocInstructionType() = default;
  ~MallocInstructionType() override = default;

  using stream_type = HostStreamType;

  void Infer(InstrCtx* instr_ctx) const override {
    MemBufferObjectType* mem_buffer_object_type = nullptr;
    size_t size = 0;
    {
      FlatMsgView<MallocInstruction> view;
      CHECK(view.Match(instr_ctx->instr_msg().operand()));
      size = view->size();
      const auto& operand = view->mem_buffer();
      MirroredObject* mirrored_object = instr_ctx->mut_operand_type(operand);
      mem_buffer_object_type = mirrored_object->Mutable<MemBufferObjectType>();
    }
    mem_buffer_object_type->set_size(size);
    mem_buffer_object_type->mut_mem_case()->mutable_host_mem();
  }
  void Compute(InstrCtx* instr_ctx) const override {
    const MemBufferObjectType* buffer_type = nullptr;
    MemBufferObjectValue* buffer_value = nullptr;
    char* dptr = nullptr;
    {
      FlatMsgView<MallocInstruction> view;
      CHECK(view.Match(instr_ctx->instr_msg().operand()));
      const auto& operand = view->mem_buffer();
      buffer_type = &instr_ctx->mut_operand_type(operand)->Get<MemBufferObjectType>();
      buffer_value = instr_ctx->mut_operand_value(operand)->Mutable<MemBufferObjectValue>();
    }
    dptr = reinterpret_cast<char*>(std::malloc(buffer_type->size()));
    buffer_value->reset_data(dptr);
  }
};
COMMAND(RegisterInstructionType<MallocInstructionType>("Malloc"));
COMMAND(RegisterLocalInstructionType<MallocInstructionType>("LocalMalloc"));

class CudaFreeHostInstructionType final : public InstructionType {
 public:
  CudaFreeHostInstructionType() = default;
  ~CudaFreeHostInstructionType() override = default;

  using stream_type = HostStreamType;

  void Infer(InstrCtx* instr_ctx) const override {
    MirroredObject* type_mirrored_object = nullptr;
    {
      FlatMsgView<FreeInstruction> view;
      CHECK(view.Match(instr_ctx->instr_msg().operand()));
      const auto& operand = view->mem_buffer();
      type_mirrored_object = instr_ctx->mut_operand_type(operand);
      const auto& buffer_type = type_mirrored_object->Get<MemBufferObjectType>();
      CHECK(buffer_type.mem_case().has_host_mem());
      CHECK(buffer_type.mem_case().host_mem().has_cuda_pinned_mem());
    }
    type_mirrored_object->reset_object();
  }
  void Compute(InstrCtx* instr_ctx) const override {
    MirroredObject* value_mirrored_object = nullptr;
    {
      FlatMsgView<FreeInstruction> view;
      CHECK(view.Match(instr_ctx->instr_msg().operand()));
      const auto& operand = view->mem_buffer();
      value_mirrored_object = instr_ctx->mut_operand_value(operand);
    }
    CudaCheck(cudaFreeHost(value_mirrored_object->Mut<MemBufferObjectValue>()->mut_data()));
    value_mirrored_object->reset_object();
  }
};
COMMAND(RegisterInstructionType<CudaFreeHostInstructionType>("CudaFreeHost"));

class FreeInstructionType final : public InstructionType {
 public:
  FreeInstructionType() = default;
  ~FreeInstructionType() override = default;

  using stream_type = HostStreamType;

  void Infer(InstrCtx* instr_ctx) const override { /* do nothing */
    MirroredObject* type_mirrored_object = nullptr;
    {
      FlatMsgView<FreeInstruction> view;
      CHECK(view.Match(instr_ctx->instr_msg().operand()));
      const auto& operand = view->mem_buffer();
      type_mirrored_object = instr_ctx->mut_operand_type(operand);
      const auto& buffer_type = type_mirrored_object->Get<MemBufferObjectType>();
      CHECK(buffer_type.mem_case().has_host_mem());
      CHECK(!buffer_type.mem_case().host_mem().has_cuda_pinned_mem());
    }
    type_mirrored_object->reset_object();
  }
  void Compute(InstrCtx* instr_ctx) const override {
    MirroredObject* value_mirrored_object = nullptr;
    {
      FlatMsgView<FreeInstruction> view;
      CHECK(view.Match(instr_ctx->instr_msg().operand()));
      const auto& operand = view->mem_buffer();
      value_mirrored_object = instr_ctx->mut_operand_value(operand);
    }
    std::free(value_mirrored_object->Mut<MemBufferObjectValue>()->mut_data());
    value_mirrored_object->reset_object();
  }
};
COMMAND(RegisterInstructionType<FreeInstructionType>("Free"));
COMMAND(RegisterLocalInstructionType<FreeInstructionType>("LocalFree"));

}  // namespace

void HostStreamType::InitInstructionStatus(const Stream& stream,
                                           InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(NaiveInstrStatusQuerier) < kInstructionStatusBufferBytes, "");
  NaiveInstrStatusQuerier::PlacementNew(status_buffer->mut_buffer()->mut_data());
}

void HostStreamType::DeleteInstructionStatus(const Stream& stream,
                                             InstructionStatusBuffer* status_buffer) const {
  // do nothing
}

bool HostStreamType::QueryInstructionStatusDone(
    const Stream& stream, const InstructionStatusBuffer& status_buffer) const {
  return NaiveInstrStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

void HostStreamType::Compute(InstrChain* instr_chain) const {
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(instr_chain->mut_instr_ctx_list(), instr_ctx) {
    const auto& instr_type_id = instr_ctx->mut_instr_msg()->instr_type_id();
    CHECK_EQ(instr_type_id.stream_type_id().interpret_type(), InterpretType::kCompute);
    instr_type_id.instruction_type().Compute(instr_ctx);
  }
  auto* status_buffer = instr_chain->mut_status_buffer();
  NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data())->set_done();
}

ObjectMsgPtr<StreamDesc> HostStreamType::MakeRemoteStreamDesc(const Resource& resource,
                                                              int64_t this_machine_id) const {
  auto ret = ObjectMsgPtr<StreamDesc>::New();
  ret->mutable_stream_type_id()->__Init__(LookupStreamType4TypeIndex<HostStreamType>());
  ret->set_num_machines(1);
  ret->set_num_streams_per_machine(1);
  ret->set_num_streams_per_thread(1);
  ret->set_start_parallel_id(this_machine_id);
  return ret;
}

ObjectMsgPtr<StreamDesc> HostStreamType::MakeLocalStreamDesc(const Resource& resource) const {
  auto ret = ObjectMsgPtr<StreamDesc>::New();
  ret->mutable_stream_type_id()->__Init__(LookupStreamType4TypeIndex<HostStreamType>());
  ret->set_num_machines(1);
  ret->set_num_streams_per_machine(1);
  ret->set_num_streams_per_thread(1);
  ret->set_start_parallel_id(0);
  return ret;
}

}  // namespace vm
}  // namespace oneflow
