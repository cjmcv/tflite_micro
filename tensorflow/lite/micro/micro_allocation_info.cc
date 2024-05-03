/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/micro_allocation_info.h"

#include <algorithm>

#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

namespace {
constexpr char kOfflineMemAllocMetadata[] = "OfflineMemoryAllocation";
constexpr int kUninitializedLifetime = -1;
}  // namespace

// // Mark the given Allocation info as first created at the specified allocation
// // scope count. Only the first creation must be recorded since the allocation
// // scope count monotonically increases throughout the lifetime marking process.
// void AllocationInfoBuilder::UpdateFirstCreated(AllocationInfo* current,
//                                                int allocation_scope_count) {
//   TFLITE_DCHECK(current->first_created <= allocation_scope_count);
//   if (current->first_created == kUninitializedLifetime) {
//     current->first_created = allocation_scope_count;
//   }
// }

// // Mark the given AllocationInfo as last used at the specified allocation scope
// // count. Update the last used marker every time, since the allocation scope
// // count monotonically increases through the lifetime marking process.
// void AllocationInfoBuilder::UpdateLastUsed(AllocationInfo* current,
//                                            int allocation_scope_count) {
//   TFLITE_DCHECK(current->last_used <= allocation_scope_count);
//   current->last_used = allocation_scope_count;
// }

// // TfLiteStatus AllocationInfoBuilder::MarkSubgraphLifetimesIfNecessary(
// //     const Operator* op, internal::ScratchBufferRequest* scratch_buffer_requests,
// //     ScratchBufferHandle* scratch_buffer_handles,
// //     SubgraphAllocations* allocations) {
// //   int first_subgraph_index = -1;
// //   int second_subgraph_index = -1;
// //   const OperatorCode* opcode =
// //       model_->operator_codes()->Get(op->opcode_index());
// //   switch (opcode->builtin_code()) {
// //     case BuiltinOperator_IF: {
// //       first_subgraph_index =
// //           op->builtin_options_as_IfOptions()->then_subgraph_index();
// //       second_subgraph_index =
// //           op->builtin_options_as_IfOptions()->else_subgraph_index();
// //       break;
// //     }
// //     case BuiltinOperator_CALL_ONCE: {
// //       first_subgraph_index =
// //           op->builtin_options_as_CallOnceOptions()->init_subgraph_index();
// //       break;
// //     }
// //     case BuiltinOperator_WHILE: {
// //       first_subgraph_index =
// //           op->builtin_options_as_WhileOptions()->cond_subgraph_index();
// //       second_subgraph_index =
// //           op->builtin_options_as_WhileOptions()->body_subgraph_index();
// //       break;
// //     }
// //     default: {
// //       break;
// //     }
// //   }
// //   if (first_subgraph_index != -1) {
// //     // Enter a new allocation scope for each subgraph.
// //     allocation_scope_count_++;
// //     TF_LITE_ENSURE_STATUS(
// //         MarkAllocationLifetimes(first_subgraph_index, scratch_buffer_requests,
// //                                 scratch_buffer_handles, allocations));
// //   }
// //   if (second_subgraph_index != -1) {
// //     // Enter a new allocation scope for each subgraph.
// //     allocation_scope_count_++;
// //     TF_LITE_ENSURE_STATUS(
// //         MarkAllocationLifetimes(second_subgraph_index, scratch_buffer_requests,
// //                                 scratch_buffer_handles, allocations));
// //   }
// //   return kTfLiteOk;
// // }

// TfLiteStatus AllocationInfoBuilder::CreateAllocationInfo(
//     int scratch_buffer_request_count) {
//   size_t subgraph_offsets_length = model_->subgraphs()->size() * sizeof(size_t);
//   info_.subgraph_offsets =
//       reinterpret_cast<size_t*>(non_persistent_allocator_->AllocateTemp(
//           subgraph_offsets_length, alignof(size_t)));
//   if (info_.subgraph_offsets == nullptr) {
//     MicroPrintf(
//         "Failed to allocate memory for memory planning, %d bytes required",
//         subgraph_offsets_length);
//     return kTfLiteError;
//   }
//   size_t tensor_count = 0;
//   for (size_t subgraph_idx = 0; subgraph_idx < model_->subgraphs()->size();
//        subgraph_idx++) {
//     // Add all tensors in each subgraph to the AllocationInfo array. Even weight
//     // tensors are added but marked with needs_allocating = false. Including all
//     // tensors in the graph here simplifies logic.
//     info_.subgraph_offsets[subgraph_idx] = tensor_count;
//     tensor_count += model_->subgraphs()->Get(subgraph_idx)->tensors()->size();
//   }
//   info_.tensor_count = tensor_count;

//   // Scratch buffer allocations follow tensor allocations, so the scratch offset
//   // is equal to the number of tensor allocations.
//   info_.scratch_offset = tensor_count;
//   info_.allocation_info_count = tensor_count + scratch_buffer_request_count;
//   info_.scratch_buffer_count = scratch_buffer_request_count;
//   size_t bytes = sizeof(AllocationInfo) * info_.allocation_info_count;

//   // Allocate an array of AllocationInfo structs from the temp section. This
//   // struct will be used by AllocationInfoBuilder to find buffer usage.
//   info_.allocation_info = reinterpret_cast<AllocationInfo*>(
//       non_persistent_allocator_->AllocateTemp(bytes, alignof(AllocationInfo)));
//   if (info_.allocation_info == nullptr) {
//     MicroPrintf(
//         "Failed to allocate memory for memory planning, %d bytes required",
//         bytes);
//     return kTfLiteError;
//   }
//   return kTfLiteOk;
// }

// TfLiteStatus AllocationInfoBuilder::FreeAllocationInfo() {
//   non_persistent_allocator_->DeallocateTemp(
//       reinterpret_cast<uint8_t*>(info_.allocation_info));
//   non_persistent_allocator_->DeallocateTemp(
//       reinterpret_cast<uint8_t*>(info_.subgraph_offsets));
//   return kTfLiteOk;
// }

// // Get offline tensors allocation plan. See
// // micro/docs/memory_management.md for more info.
// TfLiteStatus AllocationInfoBuilder::GetOfflinePlannedOffsets(
//     const int32_t** offline_planner_offsets) {
//   if (model_->metadata()) {
//     for (size_t i = 0; i < model_->metadata()->size(); ++i) {
//       auto metadata = model_->metadata()->Get(i);

//       if (metadata->name()) {
//         const size_t metadata_name_size = metadata->name()->size();

//         if ((strncmp(metadata->name()->c_str(), kOfflineMemAllocMetadata,
//                      std::min(metadata_name_size,
//                               strlen(kOfflineMemAllocMetadata))) == 0) &&
//             metadata_name_size == strlen(kOfflineMemAllocMetadata)) {
//           const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers =
//               model_->buffers();
//           auto* buffer = (*buffers)[metadata->buffer()];
//           auto* array = buffer->data();
//           const uint32_t* metadata_buffer =
//               reinterpret_cast<const uint32_t*>(array->data());
//           const size_t nbr_tensors = static_cast<size_t>(metadata_buffer[2]);
//           *offline_planner_offsets =
//               reinterpret_cast<const int32_t*>(&metadata_buffer[3]);

//           if (info_.tensor_count != nbr_tensors) {
//             MicroPrintf(
//                 "Nbr of offline buffer offsets (%d) in metadata "
//                 "not equal nbr tensors (%d)\n",
//                 nbr_tensors, info_.tensor_count);
//             return kTfLiteError;
//           }
//         }
//       }
//     }
//   }
//   return kTfLiteOk;
// }

}  // namespace tflite
