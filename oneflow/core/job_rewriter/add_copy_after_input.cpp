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

#include "oneflow/core/job_rewriter/add_copy_after_input.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

Maybe<void> AddCopyAfterInput(const OpGraph& op_graph, JobBuilder* job_builder) {
  HashMap<std::string, OperatorConf> op_name2op_conf;
  op_graph.ForEachNode([&](const OpNode* op_node) {
    const OperatorConf& op_conf = op_node->op().op_conf();
    if (!(op_conf.has_variable_conf() || op_conf.has_input_conf())) { return; }
    const LogicalBlobId& out_lbi = op_node->op().BnInOp2Lbi("out");
    const std::string& out_lbn = GenLogicalBlobName(out_lbi);
    OperatorConf proxy_op_conf;
    {
      proxy_op_conf.set_name("System-Boxing-Proxy-" + NewUniqueId());
      auto* copy_conf = proxy_op_conf.mutable_copy_conf();
      copy_conf->set_in(out_lbn);
      copy_conf->set_out("out");
      proxy_op_conf.set_scope_symbol_id(op_conf.scope_symbol_id());
    }
    const std::string& proxy_lbn =
        GenLogicalBlobName(proxy_op_conf.name(), proxy_op_conf.copy_conf().out());
    bool need_proxy_node = false;
    for (const OpEdge* out_edge : op_node->out_edges()) {
      const OpNode* consumer = out_edge->dst_node();
      const std::string& consumer_op_name = consumer->op().op_name();
      if (op_node->parallel_desc() != consumer->parallel_desc()) {
        if (op_name2op_conf.find(consumer_op_name) == op_name2op_conf.end()) {
          op_name2op_conf[consumer_op_name] = consumer->op().op_conf();
        }
        OperatorConf& consumer_op_conf = op_name2op_conf.at(consumer_op_name);
        for (const std::string& ibn : consumer->op().input_bns()) {
          if (consumer->op().BnInOp2Lbi(ibn) == out_lbi) {
            const auto& old_val =
                ReplaceInputLbnInOpCustomizedConf(&consumer_op_conf, ibn, proxy_lbn);
            CHECK_EQ(out_lbn, old_val);
          }
        }
        need_proxy_node = true;
      }
    }
    if (need_proxy_node) {
      job_builder->AddOps(op_node->parallel_desc().parallel_conf(), {proxy_op_conf});
    }
  });
  for (const auto& pair : op_name2op_conf) { job_builder->MutOpsOnlyOnce({pair.second}); }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
