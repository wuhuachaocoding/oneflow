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
#include "oneflow/user/kernels/uniform_generator.h"

// TODO(bowenc): support int uniform and uniform with range

namespace oneflow {

template<typename T>
void UniformGenerator<DeviceType::kCPU>::Generate(DeviceCtx* device_ctx, const int64_t elem_cnt,
                                                  T* dptr) {
  CHECK_GE(elem_cnt, 0);
  std::uniform_real_distribution<T> random_distribution(GetZeroVal<T>(), GetOneVal<T>());
  for (int64_t i = 0; i < elem_cnt; ++i) { dptr[i] = random_distribution(generator_->engine()); }
}

#define INITIATE_CPU_UNIFORM_GENERATOR(T, typeproto)                                    \
  template void UniformGenerator<DeviceType::kCPU>::Generate<T>(DeviceCtx * device_ctx, \
                                                                const int64_t elem_cnt, T* dptr);

OF_PP_FOR_EACH_TUPLE(INITIATE_CPU_UNIFORM_GENERATOR, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow