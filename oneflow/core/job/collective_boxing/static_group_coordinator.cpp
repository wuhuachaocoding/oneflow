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
#include "oneflow/core/job/collective_boxing/static_group_coordinator.h"
#include "oneflow/core/job/collective_boxing/executor.h"
#include "oneflow/core/job/collective_boxing/request_store.h"
#include "oneflow/core/graph/boxing/collective_boxing_util.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/common/str_util.h"

namespace oneflow {

namespace boxing {

namespace collective {

namespace {

void SortRequestIdsByOrder(const int64_t job_id, RequestStore* request_store,
                           std::vector<int32_t>* requests) {
  std::sort(requests->begin(), requests->end(), [job_id, request_store](int32_t a, int32_t b) {
    return request_store->MutRequestEntry(job_id, a)->desc().order()
           < request_store->MutRequestEntry(job_id, b)->desc().order();
  });
}

}  // namespace

void StaticGroupCoordinator::Init(std::shared_ptr<RequestStore> request_store,
                                  std::shared_ptr<Executor> executor) {
  request_store_ = request_store;
  executor_ = executor;
}

void StaticGroupCoordinator::AddPlan(const std::vector<int64_t>& job_ids) {
  const auto& GetRequestDesc = [&](int64_t job_id, int32_t request_id) -> const RequestDesc& {
    return request_store_->MutRequestEntry(job_id, request_id)->desc();
  };

  for (const auto& job_id : job_ids) {
    std::vector<int32_t> request_ids;
    const int32_t request_count = request_store_->RequestCount4Job(job_id);
    for (int64_t request_id = 0; request_id < request_count; ++request_id) {
      auto* request_entry = request_store_->MutRequestEntry(job_id, request_id);
      if (request_entry->HasRankOnThisNode()) { request_ids.push_back(request_id); }
    }
    SortRequestIdsByOrder(job_id, request_store_.get(), &request_ids);
    CHECK(std::adjacent_find(request_ids.begin(), request_ids.end(),
                             [&](int32_t a, int32_t b) {
                               return GetRequestDesc(job_id, a).dependency_depth()
                                      > GetRequestDesc(job_id, b).dependency_depth();
                             })
          == request_ids.end());
    std::vector<int32_t>& group_ids = job_id2group_ids_[job_id];
    std::vector<GroupState>& group_states = job_id2group_states_[job_id];
    std::vector<int32_t>& request_id2group_id = job_id2request_id2group_id_[job_id];
    std::vector<int32_t>& request_id2index_in_group = job_id2request_id2index_in_group_[job_id];
    std::vector<std::vector<int32_t>>& group_id2request_ids = job_id2group_id2request_ids_[job_id];
    request_id2group_id.resize(request_count);
    request_id2index_in_group.resize(request_count);
    executor_->GroupRequests(
        job_id, request_ids, [&](int64_t job_id, std::vector<int32_t>&& group) {
          const int32_t group_id = group_states.size();
          group_states.emplace_back(group.size());
          group_ids.push_back(group_id);
          for (int32_t idx_in_group = 0; idx_in_group < group.size(); ++idx_in_group) {
            const int32_t request_id = group.at(idx_in_group);
            request_id2group_id.at(request_id) = group_id;
            request_id2index_in_group.at(request_id) = idx_in_group;
          }
          group_id2request_ids.push_back(group);
        });
    if (group_states.size() != 0) { DumpSummary(job_id); }
  }
  DebugLog();
}

void StaticGroupCoordinator::AddRequest(int64_t job_id, int32_t request_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (current_job_id_ == -1) {
    current_job_id_ = job_id;
    current_group_idx_in_job_ = 0;
  } else {
    CHECK_EQ(current_job_id_, job_id);
  }
  const auto& request_id2group_id_it = job_id2request_id2group_id_.find(job_id);
  CHECK(request_id2group_id_it != job_id2request_id2group_id_.end());
  const std::vector<int32_t>& request_id2group_id = request_id2group_id_it->second;
  auto group_states_it = job_id2group_states_.find(job_id);
  CHECK(group_states_it != job_id2group_states_.end());
  std::vector<GroupState>& group_states = group_states_it->second;
  const auto& request_id2index_in_group_it = job_id2request_id2index_in_group_.find(job_id);
  CHECK(request_id2index_in_group_it != job_id2request_id2index_in_group_.end());
  const std::vector<int32_t>& request_id2index_in_group = request_id2index_in_group_it->second;
  auto group_id2request_ids_it = job_id2group_id2request_ids_.find(job_id);
  CHECK(group_id2request_ids_it != job_id2group_id2request_ids_.end());
  const std::vector<std::vector<int32_t>>& group_id2request_ids = group_id2request_ids_it->second;

  group_states.at(request_id2group_id.at(request_id))
      .AddReadyRequest(request_id2index_in_group.at(request_id));
  const std::vector<int32_t>& group_ids = job_id2group_ids_.at(current_job_id_);
  int64_t num_launched_groups = 0;
  while (true) {
    const int32_t group_id = group_ids.at(current_group_idx_in_job_);
    auto& group_state = group_states.at(group_id);
    if (group_state.IsReady()) {
      executor_->ExecuteGroupedRequests(current_job_id_, group_id2request_ids.at(group_id));
      group_state.Reset();
      current_group_idx_in_job_ = (current_group_idx_in_job_ + 1) % group_ids.size();
      num_launched_groups += 1;
    } else {
      break;
    }
  }
  if (current_group_idx_in_job_ == 0 && num_launched_groups > 0) {
    current_job_id_ = -1;
    current_group_idx_in_job_ = -1;
  }
}

void StaticGroupCoordinator::DumpSummary(const int64_t job_id) const {
  if (!Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) { return; }
  auto group_ls = TeePersistentLogStream::Create(StrCat("boxing/collective/job_", job_id));
  const auto& group_id2request_ids_it = job_id2group_id2request_ids_.find(job_id);
  CHECK(group_id2request_ids_it != job_id2group_id2request_ids_.end());
  for (int32_t group_id = 0; group_id < job_id2group_states_.at(job_id).size(); ++group_id) {
    group_ls << "group id: " << std::to_string(group_id) << "\n";
    for (const int32_t request_id : group_id2request_ids_it->second.at(group_id)) {
      group_ls->Write(request_store_->MutRequestEntry(job_id, request_id)->desc());
    }
  }
}

void StaticGroupCoordinator::GroupState::AddReadyRequest(int32_t index) {
  CHECK(!index2is_ready.at(index));
  CHECK(index2is_ready.at(index) = true);
  ready_request_count += 1;
}

bool StaticGroupCoordinator::GroupState::IsReady() const {
  return ready_request_count == index2is_ready.size();
}

void StaticGroupCoordinator::GroupState::Reset() {
  ready_request_count = 0;
  std::fill(index2is_ready.begin(), index2is_ready.end(), false);
}

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow
