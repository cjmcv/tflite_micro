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

#include "tensorflow/lite/micro/micro_allocator.h"

#include <cstddef>
#include <cstdint>
#include <new>

// #include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/compatibility.h"
// #include "tensorflow/lite/micro/flatbuffer_utils.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"
#include "tensorflow/lite/micro/memory_planner/linear_memory_planner.h"
#include "tensorflow/lite/micro/memory_planner/micro_memory_planner.h"
#include "tensorflow/lite/micro/micro_allocation_info.h"
#include "tensorflow/lite/micro/micro_arena_constants.h"
#include "tensorflow/lite/micro/micro_log.h"
// #include "tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.h"
// #include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

namespace {

// Maximum number of scratch buffer requests per operator. Operator kernels that
// request more than this value will receive an exception.
constexpr size_t kMaxScratchBuffersPerOp = 12;

// Sentinel value used as a placeholder to mark a ScratchBufferRequest request
// needs a node id assignment.
constexpr int kUnassignedScratchBufferRequestIndex = -1;

const TfLiteIntArray kZeroLengthIntArray = {};

MicroMemoryPlanner* CreateMemoryPlanner(
    MemoryPlannerType memory_planner_type,
    IPersistentBufferAllocator* memory_allocator) {
  MicroMemoryPlanner* memory_planner = nullptr;
  uint8_t* memory_planner_buffer = nullptr;

  switch (memory_planner_type) {
    case MemoryPlannerType::kLinear: {
      memory_planner_buffer = memory_allocator->AllocatePersistentBuffer(
          sizeof(LinearMemoryPlanner), alignof(LinearMemoryPlanner));
      memory_planner = new (memory_planner_buffer) LinearMemoryPlanner();
      break;
    }
    case MemoryPlannerType::kGreedy: {
      memory_planner_buffer = memory_allocator->AllocatePersistentBuffer(
          sizeof(GreedyMemoryPlanner), alignof(GreedyMemoryPlanner));
      memory_planner = new (memory_planner_buffer) GreedyMemoryPlanner();
      break;
    }
  }
  return memory_planner;
}


IPersistentBufferAllocator* CreatePersistentArenaAllocator(uint8_t* buffer_head,
                                                           size_t buffer_size) {
  // Align the actually used area by the tail because persistent buffer grows
  // from the bottom to top.
  uint8_t* aligned_buffer_tail =
      AlignPointerDown(buffer_head + buffer_size, MicroArenaBufferAlignment());
  size_t aligned_buffer_size = aligned_buffer_tail - buffer_head;
  PersistentArenaBufferAllocator tmp =
      PersistentArenaBufferAllocator(buffer_head, aligned_buffer_size);

  // Allocate enough bytes from the buffer to create a
  // SingleArenaBufferAllocator. The new instance will use the current adjusted
  // tail buffer from the tmp allocator instance.
  uint8_t* allocator_buffer =
      tmp.AllocatePersistentBuffer(sizeof(PersistentArenaBufferAllocator),
                                   alignof(PersistentArenaBufferAllocator));
  // Use the default copy constructor to populate internal states.
  return new (allocator_buffer) PersistentArenaBufferAllocator(tmp);
}

// NonPersistentBufferAllocator instance is created in the persistent buffer
// because it has to be persistent to keep track of the non-persistent buffer
// information.
INonPersistentBufferAllocator* CreateNonPersistentArenaAllocator(
    uint8_t* buffer_head, size_t buffer_size,
    IPersistentBufferAllocator* persistent_buffer_allocator) {
  uint8_t* allocator_buffer =
      persistent_buffer_allocator->AllocatePersistentBuffer(
          sizeof(NonPersistentArenaBufferAllocator),
          alignof(NonPersistentArenaBufferAllocator));
  // Align the actually used area by the head because persistent buffer grows
  // from the head to bottom.
  uint8_t* aligned_buffer_head =
      AlignPointerUp(buffer_head, MicroArenaBufferAlignment());
  size_t aligned_buffer_size = buffer_head + buffer_size - aligned_buffer_head;

  INonPersistentBufferAllocator* non_persistent_buffer_allocator =
      new (allocator_buffer) NonPersistentArenaBufferAllocator(
          aligned_buffer_head, aligned_buffer_size);
  return non_persistent_buffer_allocator;
}

}  // namespace

namespace internal {

// // Returns a pointer to any buffer associated with the flatbuffer tensor. Can
// // return nullptr if no buffer is found.
// void* GetFlatbufferTensorBuffer(
//     const tflite::Tensor& flatbuffer_tensor,
//     const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers) {
//   // We need to figure out where the actual contents of this tensor are stored
//   // in memory. We'll check to see if there's a serialized buffer (pretty much
//   // the same as a constant op in TensorFlow) associated with this tensor first,
//   // and if there is update the runtime structure to point to its location in
//   // memory.
//   // First see if there's any buffer information in the serialized tensor.
//   // TODO(b/170379532): Add better unit tests to validate flatbuffer values.
//   void* out_buffer = nullptr;
//   if (auto* buffer = (*buffers)[flatbuffer_tensor.buffer()]) {
//     // If we've found a buffer, does it have any data?
//     if (auto* array = buffer->data()) {
//       // If it has any data, is the data size larger than zero?
//       if (array->size()) {
//         // We've found a buffer with valid data, so update the runtime tensor
//         // data structure to point to it.
//         out_buffer = const_cast<void*>(static_cast<const void*>(array->data()));
//       }
//     }
//     // TODO(petewarden): It's not clear in what circumstances we could have a
//     // buffer in the serialized tensor, but it doesn't have any data in it. Is
//     // that a validly-generated file, and if so what does it mean, or is it an
//     // error condition? It would be good to tighten up the specification to make
//     // it less ambiguous.
//   }
//   return out_buffer;
// }

}  // namespace internal


MicroAllocator::MicroAllocator(SingleArenaBufferAllocator* memory_allocator,
                               MicroMemoryPlanner* memory_planner)
    : non_persistent_buffer_allocator_(memory_allocator),
      persistent_buffer_allocator_(memory_allocator),
      memory_planner_(memory_planner),
      model_is_allocating_(false) {}

MicroAllocator::MicroAllocator(
    IPersistentBufferAllocator* persistent_buffer_allocator,
    INonPersistentBufferAllocator* non_persistent_buffer_allocator,
    MicroMemoryPlanner* memory_planner)
    : non_persistent_buffer_allocator_(non_persistent_buffer_allocator),
      persistent_buffer_allocator_(persistent_buffer_allocator),
      memory_planner_(memory_planner),
      model_is_allocating_(false) {}

MicroAllocator::~MicroAllocator() {}

MicroAllocator* MicroAllocator::Create(uint8_t* tensor_arena, size_t arena_size,
                                       MicroMemoryPlanner* memory_planner) {
  uint8_t* aligned_arena =
      AlignPointerUp(tensor_arena, MicroArenaBufferAlignment());
  size_t aligned_arena_size = tensor_arena + arena_size - aligned_arena;
  SingleArenaBufferAllocator* memory_allocator =
      SingleArenaBufferAllocator::Create(aligned_arena, aligned_arena_size);

  return Create(memory_allocator, memory_planner);
}

MicroAllocator* MicroAllocator::Create(uint8_t* tensor_arena, size_t arena_size,
                                       MemoryPlannerType memory_planner_type) {
  uint8_t* aligned_arena =
      AlignPointerUp(tensor_arena, MicroArenaBufferAlignment());
  size_t aligned_arena_size = tensor_arena + arena_size - aligned_arena;
  SingleArenaBufferAllocator* memory_allocator =
      SingleArenaBufferAllocator::Create(aligned_arena, aligned_arena_size);

  // By default create GreedyMemoryPlanner.
  // If a different MemoryPlanner is needed, use the other api.
  MicroMemoryPlanner* memory_planner =
      CreateMemoryPlanner(memory_planner_type, memory_allocator);

  return Create(memory_allocator, memory_planner);
}

MicroAllocator* MicroAllocator::Create(
    SingleArenaBufferAllocator* memory_allocator,
    MicroMemoryPlanner* memory_planner) {
  TFLITE_DCHECK(memory_allocator != nullptr);
  TFLITE_DCHECK(memory_planner != nullptr);

  uint8_t* allocator_buffer = memory_allocator->AllocatePersistentBuffer(
      sizeof(MicroAllocator), alignof(MicroAllocator));
  MicroAllocator* allocator = new (allocator_buffer)
      MicroAllocator(memory_allocator, memory_allocator, memory_planner);
  return allocator;
}

MicroAllocator* MicroAllocator::Create(uint8_t* persistent_tensor_arena,
                                       size_t persistent_arena_size,
                                       uint8_t* non_persistent_tensor_arena,
                                       size_t non_persistent_arena_size,
                                       MemoryPlannerType memory_planner_type) {
  TFLITE_DCHECK(persistent_tensor_arena != nullptr);
  TFLITE_DCHECK(non_persistent_tensor_arena != nullptr);
  TFLITE_DCHECK(persistent_tensor_arena != non_persistent_tensor_arena);

  IPersistentBufferAllocator* persistent_buffer_allocator =
      CreatePersistentArenaAllocator(persistent_tensor_arena,
                                     persistent_arena_size);
  INonPersistentBufferAllocator* non_persistent_buffer_allocator =
      CreateNonPersistentArenaAllocator(non_persistent_tensor_arena,
                                        non_persistent_arena_size,
                                        persistent_buffer_allocator);

  // TODO(b/297821738): this should be changed to CreateMemoryPlanner if
  // possible once  it's figured out why it breaks the HifiMini Build
  uint8_t* memory_planner_buffer = nullptr;
  MicroMemoryPlanner* memory_planner = nullptr;

  if (memory_planner_type == MemoryPlannerType::kGreedy) {
    memory_planner_buffer =
        persistent_buffer_allocator->AllocatePersistentBuffer(
            sizeof(GreedyMemoryPlanner), alignof(GreedyMemoryPlanner));
    memory_planner = new (memory_planner_buffer) GreedyMemoryPlanner();
  } else if (memory_planner_type == MemoryPlannerType::kLinear) {
    memory_planner_buffer =
        persistent_buffer_allocator->AllocatePersistentBuffer(
            sizeof(LinearMemoryPlanner), alignof(LinearMemoryPlanner));
    memory_planner = new (memory_planner_buffer) LinearMemoryPlanner();
  }

  uint8_t* micro_allocator_buffer =
      persistent_buffer_allocator->AllocatePersistentBuffer(
          sizeof(MicroAllocator), alignof(MicroAllocator));
  MicroAllocator* allocator = new (micro_allocator_buffer)
      MicroAllocator(persistent_buffer_allocator,
                     non_persistent_buffer_allocator, memory_planner);
  return allocator;
}

void* MicroAllocator::AllocatePersistentBuffer(size_t bytes) {
  return persistent_buffer_allocator_->AllocatePersistentBuffer(
      bytes, MicroArenaBufferAlignment());
}

TfLiteStatus MicroAllocator::RequestScratchBufferInArena(size_t bytes,
                                                         int subgraph_idx,
                                                         int* buffer_idx) {
  // All scratch buffer requests are stored in the head section of the arena
  // when a model is in the prepare phase. First align a scratch buffer request
  // pointer to the start of the head:
  internal::ScratchBufferRequest* requests = GetScratchBufferRequests();

  // Count the number of requested scratch buffers for the current node:
  size_t current_node_request_count = 0;
  for (size_t i = 0; i < scratch_buffer_request_count_; ++i) {
    if (requests[i].node_idx == kUnassignedScratchBufferRequestIndex) {
      ++current_node_request_count;
    }
  }

  // First, ensure that the per-kernel request has not exceeded the limit:
  if (current_node_request_count >= kMaxScratchBuffersPerOp) {
    MicroPrintf("Scratch buffer request exeeds limit per operator (%d)",
                kMaxScratchBuffersPerOp);
    return kTfLiteError;
  }

  // Initialize and assign values for the request at the current index:
  internal::ScratchBufferRequest* current_request =
      &requests[scratch_buffer_request_count_];
  *current_request = {};
  // Assign -1 as a sentinel value that will be updated when the node finishes
  // allocating:
  current_request->bytes = bytes;
  current_request->node_idx = kUnassignedScratchBufferRequestIndex;
  current_request->subgraph_idx = subgraph_idx;

  // Assign the current request index to the out-param:
  *buffer_idx = scratch_buffer_request_count_;

  // Bump the request count to prepare for the next request:
  ++scratch_buffer_request_count_;
  return kTfLiteOk;
}

TfLiteStatus MicroAllocator::FinishPrepareNodeAllocations(int node_id) {
  // When a node has finished preparing, all temp allocations performed by the
  // kernel should be cleaned up:
  TF_LITE_ENSURE_STATUS(ResetTempAllocations());

  // Find and update any new scratch buffer requests for the current node:
  internal::ScratchBufferRequest* requests = GetScratchBufferRequests();

  for (size_t i = 0; i < scratch_buffer_request_count_; ++i) {
    // A request with a node_idx of -1 is a sentinel value used to indicate this
    // was a new request for the current node. The allocator finally knows the
    // node index at this point. Assign the value and update the list of new
    // requests so the head section can be adjusted to allow for the next kernel
    // to allocate at most kMaxScratchBuffersPerOp requests:
    if (requests[i].node_idx == kUnassignedScratchBufferRequestIndex) {
      requests[i].node_idx = node_id;
    }
  }

  // Ensure that the head is re-adjusted to allow for another at-most
  // kMaxScratchBuffersPerOp scratch buffer requests in the next operator:
  TF_LITE_ENSURE_STATUS(non_persistent_buffer_allocator_->ResizeBuffer(
      scratch_buffer_head_,
      sizeof(internal::ScratchBufferRequest) *
          (scratch_buffer_request_count_ + kMaxScratchBuffersPerOp),
      alignof(internal::ScratchBufferRequest)));

  return kTfLiteOk;
}

size_t MicroAllocator::used_bytes() const {
  return non_persistent_buffer_allocator_->GetNonPersistentUsedBytes() +
         persistent_buffer_allocator_->GetPersistentUsedBytes();
}

// TfLiteTensor* MicroAllocator::AllocatePersistentTfLiteTensor(
//     const Model* model, const SubgraphAllocations* subgraph_allocations,
//     int tensor_index, int subgraph_index) {
//   const SubGraph* subgraph = model->subgraphs()->Get(subgraph_index);
//   TFLITE_DCHECK(subgraph != nullptr);

//   // This value is allocated from persistent arena space. It is guaranteed to be
//   // around for the lifetime of the application.
//   TfLiteTensor* tensor = AllocatePersistentTfLiteTensorInternal();

//   if (tensor == nullptr) {
//     MicroPrintf("Failed to allocate memory for persistent TfLiteTensor");
//     return nullptr;
//   }

//   // // Populate any fields from the flatbuffer, since this TfLiteTensor struct is
//   // // allocated in the persistent section of the arena, ensure that additional
//   // // allocations also take place in that section of the arena.
//   // if (PopulateTfLiteTensorFromFlatbuffer(
//   //         model, tensor, tensor_index, subgraph_index,
//   //         /*allocate_temp=*/false) != kTfLiteOk) {
//   //   MicroPrintf(
//   //       "Failed to populate a persistent TfLiteTensor struct "
//   //       "from flatbuffer data!");
//   //   return nullptr;
//   // }

//   if (subgraph_allocations != nullptr) {
//     // Tensor buffers that are allocated at runtime (e.g. non-weight buffers)
//     // and not located in the flatbuffer are stored on the pre-allocated list of
//     // TfLiteEvalTensors structs. These structs are the source of truth, simply
//     // point the corresponding buffer to the new TfLiteTensor data value.
//     tensor->data.data =
//         subgraph_allocations[subgraph_index].tensors[tensor_index].data.data;
//     // TfLiteEvalTensor structs must also be the source of truth for the
//     // TfLiteTensor dims.
//     tensor->dims =
//         subgraph_allocations[subgraph_index].tensors[tensor_index].dims;
//   }
//   return tensor;
// }

void MicroAllocator::DeallocateTempTfLiteTensor(TfLiteTensor* tensor) {
  TFLITE_DCHECK(tensor != nullptr);

  if (tensor->quantization.type == kTfLiteAffineQuantization) {
    TFLITE_DCHECK(tensor->quantization.params != nullptr);
    TfLiteAffineQuantization* quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            tensor->quantization.params);

    non_persistent_buffer_allocator_->DeallocateTemp(
        reinterpret_cast<uint8_t*>(quantization->zero_point));
    non_persistent_buffer_allocator_->DeallocateTemp(
        reinterpret_cast<uint8_t*>(quantization));
  }

  // Clear the data in case someone still access tensor arena by mistake
  tensor->quantization.type = kTfLiteNoQuantization;
  tensor->quantization.params = nullptr;
  tensor->data.data = nullptr;
  tensor->dims = nullptr;
  non_persistent_buffer_allocator_->DeallocateTemp(
      reinterpret_cast<uint8_t*>(tensor));
}

// TfLiteTensor* MicroAllocator::AllocateTempTfLiteTensor(
//     const Model* model, const SubgraphAllocations* subgraph_allocations,
//     int tensor_index, int subgraph_index) {
//   const SubGraph* subgraph = model->subgraphs()->Get(subgraph_index);
//   TFLITE_DCHECK(subgraph != nullptr);

//   // This value is allocated from temporary arena space. It is guaranteed to be
//   // around for at least the scope of the calling function. Since this struct
//   // allocation takes place in temp space, no need to own or cleanup.
//   TfLiteTensor* tensor = reinterpret_cast<TfLiteTensor*>(
//       non_persistent_buffer_allocator_->AllocateTemp(sizeof(TfLiteTensor),
//                                                      alignof(TfLiteTensor)));

//   // // Populate any fields from the flatbuffer, since this TfLiteTensor struct is
//   // // allocated in the temp section of the arena, ensure that additional
//   // // allocations also take place in that section of the arena.
//   // if (PopulateTfLiteTensorFromFlatbuffer(model, tensor, tensor_index,
//   //                                        subgraph_index,
//   //                                        /*allocate_temp=*/true) != kTfLiteOk) {
//   //   MicroPrintf(
//   //       "Failed to populate a temp TfLiteTensor struct from flatbuffer data!");
//   //   return nullptr;
//   // }

//   if (subgraph_allocations != nullptr) {
//     // Tensor buffers that are allocated at runtime (e.g. non-weight buffers)
//     // and not located in the flatbuffer are stored on the pre-allocated list of
//     // TfLiteEvalTensors structs. These structs are the source of truth, simply
//     // point the corresponding buffer to the new TfLiteTensor data value.
//     tensor->data.data =
//         subgraph_allocations[subgraph_index].tensors[tensor_index].data.data;
//     // TfLiteEvalTensor structs must also be the source of truth for the
//     // TfLiteTensor dims.
//     tensor->dims =
//         subgraph_allocations[subgraph_index].tensors[tensor_index].dims;
//   }
//   return tensor;
// }

uint8_t* MicroAllocator::AllocateTempBuffer(size_t size, size_t alignment) {
  return non_persistent_buffer_allocator_->AllocateTemp(size, alignment);
}

void MicroAllocator::DeallocateTempBuffer(uint8_t* buffer) {
  non_persistent_buffer_allocator_->DeallocateTemp(buffer);
}

TfLiteStatus MicroAllocator::ResetTempAllocations() {
  return non_persistent_buffer_allocator_->ResetTempAllocations();
}

bool MicroAllocator::IsAllTempDeallocated() {
  return non_persistent_buffer_allocator_->IsAllTempDeallocated();
}

// TfLiteStatus MicroAllocator::AllocateVariables(
//     const SubGraph* subgraph, TfLiteEvalTensor* eval_tensors,
//     const int32_t* offline_planner_offsets) {
//   for (size_t i = 0; i < subgraph->tensors()->size(); ++i) {
//     auto* tensor = subgraph->tensors()->Get(i);
//     if (tensor->is_variable()) {
//       if (offline_planner_offsets == nullptr ||
//           offline_planner_offsets[i] == kOnlinePlannedBuffer) {
//         size_t buffer_size;
//         TF_LITE_ENSURE_STATUS(
//             TfLiteEvalTensorByteLength(&eval_tensors[i], &buffer_size));

//         eval_tensors[i].data.data =
//             persistent_buffer_allocator_->AllocatePersistentBuffer(
//                 buffer_size, MicroArenaBufferAlignment());

//         if (eval_tensors[i].data.data == nullptr) {
//           MicroPrintf("Failed to allocate variable tensor of size %d",
//                       buffer_size);
//           return kTfLiteError;
//         }
//       }
//     }
//   }
//   return kTfLiteOk;
// }

TfLiteTensor* MicroAllocator::AllocatePersistentTfLiteTensorInternal() {
  return reinterpret_cast<TfLiteTensor*>(
      persistent_buffer_allocator_->AllocatePersistentBuffer(
          sizeof(TfLiteTensor), alignof(TfLiteTensor)));
}

TfLiteStatus MicroAllocator::AllocateScratchBufferHandles(
    ScratchBufferHandle** scratch_buffer_handles, size_t handle_count) {
  TFLITE_DCHECK(scratch_buffer_handles != nullptr);

  if (scratch_buffer_request_count_ == 0) {
    // No scratch buffer requests were requested during model allocation.
    return kTfLiteOk;
  }

  // Allocate a consecutive block of memory store the scratch buffer handles.
  // This alignment ensures quick lookup during inference time for the model:
  *scratch_buffer_handles = reinterpret_cast<ScratchBufferHandle*>(
      persistent_buffer_allocator_->AllocatePersistentBuffer(
          sizeof(ScratchBufferHandle) * handle_count,
          alignof(ScratchBufferHandle)));

  return kTfLiteOk;
}

TfLiteStatus MicroAllocator::InitScratchBufferData() {
  // A model is preparing to allocate resources, ensure that scratch buffer
  // request counter is cleared:
  scratch_buffer_request_count_ = 0;

  // All requests will be stored in the head section. Each kernel is allowed at
  // most kMaxScratchBuffersPerOp requests. Adjust the head to reserve at most
  // that many requests to begin:
  scratch_buffer_head_ =
      non_persistent_buffer_allocator_->AllocateResizableBuffer(
          sizeof(internal::ScratchBufferRequest) * kMaxScratchBuffersPerOp,
          alignof(internal::ScratchBufferRequest));
  if (scratch_buffer_head_ == nullptr) {
    return kTfLiteError;
  }

  return kTfLiteOk;
}

internal::ScratchBufferRequest* MicroAllocator::GetScratchBufferRequests() {
  return reinterpret_cast<internal::ScratchBufferRequest*>(AlignPointerUp(
      scratch_buffer_head_, alignof(internal::ScratchBufferRequest)));
}

// TfLiteBridgeBuiltinDataAllocator* MicroAllocator::GetBuiltinDataAllocator() {
//   return builtin_data_allocator_;
// }

}  // namespace tflite
