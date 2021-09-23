#ifndef ONEFLOW_CORE_FRAMEWORK_STREAM_DESCRIPTOR_H_
#define ONEFLOW_CORE_FRAMEWORK_STREAM_DESCRIPTOR_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/optional.h"

namespace oneflow {

inline size_t GetInstructionHighWaterMark() { return 500; }
inline size_t GetInstructionLowWaterMark() { return 200; }
static const size_t kDoubleBufferPoolSize = 2;

class StreamDescriptor final {
 public:
  StreamDescriptor(const StreamDescriptor&) = default;
  StreamDescriptor(StreamDescriptor&&) = delete;
  ~StreamDescriptor() = default;

  static Maybe<const StreamDescriptor*> Find(const std::string& stream_type_name);

  const std::string& stream_type_name() const { return stream_type_name_; }
  const std::string& device_type() const { return device_type_; }
  const Optional<std::string>& shared_transport_stream_type_name() const { return shared_transport_stream_type_name_; }
  const std::string& shared_schedule_stream_type_name() const { return shared_schedule_stream_type_name_; }
  const Optional<std::string>& local_call_instruction_name() const { return local_call_instruction_name_; }
  size_t local_dep_object_pool_size() const { return local_dep_object_pool_size_; }

  StreamDescriptor& operator=(const StreamDescriptor&) = delete;

  bool need_soft_sync_stream() const;

  static Maybe<void> Register(
      const std::string& stream_type_name,
      const std::string& device_type,
      const Optional<std::string>& shared_transport_stream_type_name,
      const std::string& shared_schedule_stream_type_name,
      const Optional<std::string>& local_call_instruction_name,
      size_t local_dep_object_pool_size);

 private:
  StreamDescriptor(
      const std::string& stream_type_name,
      const std::string& device_type,
      const Optional<std::string>& shared_transport_stream_type_name,
      const std::string& shared_schedule_stream_type_name,
      const Optional<std::string>& local_call_instruction_name,
      size_t local_dep_object_pool_size)
    : stream_type_name_(stream_type_name),
      device_type_(device_type),
      shared_transport_stream_type_name_(shared_transport_stream_type_name),
      shared_schedule_stream_type_name_(shared_schedule_stream_type_name),
      local_call_instruction_name_(local_call_instruction_name),
      local_dep_object_pool_size_(local_dep_object_pool_size) {}

  const std::string stream_type_name_;
  const std::string device_type_;
  const Optional<std::string> shared_transport_stream_type_name_;
  const std::string shared_schedule_stream_type_name_;
  const Optional<std::string> local_call_instruction_name_;
  size_t local_dep_object_pool_size_;
};

}

#endif  // ONEFLOW_CORE_FRAMEWORK_STREAM_DESCRIPTOR_H_
