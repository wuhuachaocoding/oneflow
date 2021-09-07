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
#ifndef ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_STATIC_GROUP_COORDINATOR_H_
#define ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_STATIC_GROUP_COORDINATOR_H_

#include "oneflow/core/job/collective_boxing/coordinator.h"

namespace oneflow {

class CollectiveBoxingPlan;

namespace boxing {

namespace collective {

class RequestStore;
class Executor;

class StaticGroupCoordinator : public Coordinator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StaticGroupCoordinator);
  StaticGroupCoordinator() = default;
  ~StaticGroupCoordinator() override = default;

  void Init(std::shared_ptr<RequestStore> request_store,
            std::shared_ptr<Executor> executor) override;
  void AddPlan(const std::vector<int64_t>& job_ids) override;
  void AddRequest(int64_t job_id, int32_t request_id) override;
  void DebugLog() {
    for (const auto& pair : job_id2group_ids_) {
      const int64_t job_id = pair.first;
      LOG(INFO) << "job_id2group_ids_ " << job_id;
      for (const auto& group_id : pair.second) { LOG(INFO) << "group id:" << group_id; }
      LOG(INFO) << "job_id2request_id2group_id_ " << job_id;
      for (const auto& request_id2group_id : job_id2request_id2group_id_.at(job_id)) {
        LOG(INFO) << "request_id2group_id :" << request_id2group_id;
      }
      LOG(INFO) << "job_id2request_id2index_in_group_ " << job_id;
      for (const auto& request_id2index_in_group : job_id2request_id2index_in_group_.at(job_id)) {
        LOG(INFO) << "request_id2index_in_group :" << request_id2index_in_group;
      }
      LOG(INFO) << "job_id2group_states_ " << job_id;
      for (const auto& group_state : job_id2group_states_.at(job_id)) {
        LOG(INFO) << "group_state index2is_ready size :" << group_state.index2is_ready.size();
      }
    }
  }

 private:
  void DumpSummary(const int64_t job_id) const;

  std::shared_ptr<RequestStore> request_store_;
  std::shared_ptr<Executor> executor_;
  HashMap<int64_t, std::vector<int32_t>> job_id2group_ids_;
  HashMap<int64_t, std::vector<int32_t>> job_id2request_id2group_id_;
  HashMap<int64_t, std::vector<int32_t>> job_id2request_id2index_in_group_;
  HashMap<int64_t, std::vector<std::vector<int32_t>>> job_id2group_id2request_ids_;

  struct GroupState {
    explicit GroupState(int32_t group_size) : index2is_ready(group_size), ready_request_count(0) {}

    void AddReadyRequest(int32_t index);
    bool IsReady() const;
    void Reset();

    std::vector<bool> index2is_ready;
    int32_t ready_request_count;
  };
  std::mutex mutex_;
  HashMap<int64_t, std::vector<GroupState>> job_id2group_states_;
  int64_t current_job_id_ = -1;
  int64_t current_group_idx_in_job_ = -1;
};

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_STATIC_GROUP_COORDINATOR_H_
