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
#ifndef ONEFLOW_CORE_COMMON_MEMORY_COPY_ND_DESC_H_
#define ONEFLOW_CORE_COMMON_MEMORY_COPY_ND_DESC_H_

#include "oneflow/core/common/nd_index.h"

namespace oneflow {

struct MemoryCopyNdDesc {
  Shape dst_shape;
  Shape src_shape;
  NdIndex dst_pos;
  NdIndex src_pos;
  Shape extent;
  DataType data_type;

  MemoryCopyNdDesc CreateDimReducedDesc() const {
    MemoryCopyNdDesc reduced;
    DimVector dst_shape_vec;
    DimVector src_shape_vec;
    DimVector dst_pos_vec;
    DimVector src_pos_vec;
    DimVector extent_vec;
    FOR_RANGE(int64_t, i, 0, extent.NumAxes()) {
      if (dst_shape.At(i) == src_shape.At(i) && dst_shape.At(i) == extent.At(i)
          && dst_pos.At(i) == 0 && src_pos.At(i) == 0 && i != 0) {
        dst_shape_vec.back() *= extent.At(i);
        src_shape_vec.back() *= extent.At(i);
        dst_pos_vec.back() *= extent.At(i);
        src_pos_vec.back() *= extent.At(i);
        extent_vec.back() *= extent.At(i);
      } else {
        dst_shape_vec.push_back(dst_shape.At(i));
        src_shape_vec.push_back(src_shape.At(i));
        dst_pos_vec.push_back(dst_pos.At(i));
        src_pos_vec.push_back(src_pos.At(i));
        extent_vec.push_back(extent.At(i));
      }
    }
    reduced.dst_shape = Shape(dst_shape_vec);
    reduced.src_shape = Shape(src_shape_vec);
    reduced.dst_pos = NdIndex(dst_pos_vec);
    reduced.src_pos = NdIndex(src_pos_vec);
    reduced.extent = Shape(extent_vec);
    reduced.data_type = data_type;
    return reduced;
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_ND_INDEX_H_
