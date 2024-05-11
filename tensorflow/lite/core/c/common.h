/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
// WARNING: Users of TensorFlow Lite should not include this file directly, but
// should instead include "third_party/tensorflow/lite/c/common.h".
// Only the TensorFlow Lite implementation itself should include this file
// directly.

/// This file defines common C types and APIs for implementing operations,
/// delegates and other constructs in TensorFlow Lite. The actual operations and
/// delegates can be defined using C++, but the interface between the
/// interpreter and the operations are C.
///
/// Summary of abstractions:
/// * `TF_LITE_ENSURE` - self-sufficient error checking
/// * `TfLiteStatus` - status reporting
/// * `TfLiteIntArray` - stores tensor shapes (dims),
/// * `TfLiteContext` - allows an op to access the tensors
/// * `TfLiteTensor` - tensor (a multidimensional array)
/// * `TfLiteNode` - a single node or operation
/// * `TfLiteRegistration` - the implementation of a conceptual operation.
/// * `TfLiteDelegate` - allows delegation of nodes to alternative backends.
///
/// Some abstractions in this file are created and managed by Interpreter.
///
/// NOTE: The order of values in these structs are "semi-ABI stable". New values
/// should be added only to the end of structs and never reordered.
///
// clang-format off
// NOLINTBEGIN(whitespace/line_length)
/// \note Users of TensorFlow Lite should use
/// \code
/// #include "tensorflow/lite/core/c/common.h"
/// \endcode
/// to access the APIs documented on this page.
// NOLINTEND(whitespace/line_length)
// clang-format on

// IWYU pragma: private, include "third_party/tensorflow/lite/c/common.h"

#ifndef TENSORFLOW_LITE_CORE_C_COMMON_H_
#define TENSORFLOW_LITE_CORE_C_COMMON_H_

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "tensorflow/lite/core/c/c_api_types.h"  // IWYU pragma: export

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// clang-format off
// NOLINTBEGIN(whitespace/line_length)
/** \defgroup common lite/c/common.h
 *  @{
 */
// NOLINTEND(whitespace/line_length)
// clang-format on

// Forward declare so dependent structs and methods can reference these types
// prior to the struct definitions.
struct TfLiteContext;
// struct TfLiteDelegate;
struct TfLiteRegistration;
// struct TfLiteOpaqueDelegateBuilder;

#define kTfLiteOptionalTensor (-1)

/// Fixed size list of integers. Used for dimensions and inputs/outputs tensor
/// indices
typedef struct TfLiteIntArray {
  int size;

#if defined(_MSC_VER)
  // Context for why this is needed is in http://b/189926408#comment21
  int data[1];
#elif (!defined(__clang__) && defined(__GNUC__) && __GNUC__ == 6 && \
       __GNUC_MINOR__ >= 1) ||                                      \
    defined(HEXAGON) ||                                             \
    (defined(__clang__) && __clang_major__ == 7 && __clang_minor__ == 1)
  // gcc 6.1+ have a bug where flexible members aren't properly handled
  // https://github.com/google/re2/commit/b94b7cd42e9f02673cd748c1ac1d16db4052514c
  int data[0];
#else
  int data[];
#endif
} TfLiteIntArray;

/// Given the size (number of elements) in a TfLiteIntArray, calculate its size
/// in bytes.
size_t TfLiteIntArrayGetSizeInBytes(int size);


/// Check if two intarrays are equal. Returns 1 if they are equal, 0 otherwise.
int TfLiteIntArrayEqual(const TfLiteIntArray* a, const TfLiteIntArray* b);

/// Fixed size list of floats. Used for per-channel quantization.
typedef struct TfLiteFloatArray {
  int size;
#if defined(_MSC_VER)
  // Context for why this is needed is in http://b/189926408#comment21
  float data[1];
#elif (!defined(__clang__) && defined(__GNUC__) && __GNUC__ == 6 && \
       __GNUC_MINOR__ >= 1) ||                                      \
    defined(HEXAGON) ||                                             \
    (defined(__clang__) && __clang_major__ == 7 && __clang_minor__ == 1)
  // gcc 6.1+ have a bug where flexible members aren't properly handled
  // https://github.com/google/re2/commit/b94b7cd42e9f02673cd748c1ac1d16db4052514c
  float data[0];
#else
  float data[];
#endif
} TfLiteFloatArray;

// Since we must not depend on any libraries, define a minimal subset of
// error macros while avoiding names that have pre-conceived meanings like
// assert and check.

// Try to make all reporting calls through TF_LITE_KERNEL_LOG rather than
// calling the context->ReportError function directly, so that message strings
// can be stripped out if the binary size needs to be severely optimized.
#ifndef TF_LITE_STRIP_ERROR_STRINGS
#define TF_LITE_KERNEL_LOG(context, ...)            \
  do {                                              \
    (context)->ReportError((context), __VA_ARGS__); \
  } while (false)

#define TF_LITE_MAYBE_KERNEL_LOG(context, ...)        \
  do {                                                \
    if ((context) != nullptr) {                       \
      (context)->ReportError((context), __VA_ARGS__); \
    }                                                 \
  } while (false)
#else  // TF_LITE_STRIP_ERROR_STRINGS
#define ARGS_UNUSED(...) (void)sizeof(#__VA_ARGS__)
#define TF_LITE_KERNEL_LOG(context, ...) ARGS_UNUSED(__VA_ARGS__)
#define TF_LITE_MAYBE_KERNEL_LOG(context, ...) ARGS_UNUSED(__VA_ARGS__)
#endif  // TF_LITE_STRIP_ERROR_STRINGS

/// Check whether value is true, and if not return kTfLiteError from
/// the current function (and report the error string msg).
#define TF_LITE_ENSURE_MSG(context, value, ...)                \
  do {                                                         \
    if (!(value)) {                                            \
      TF_LITE_KERNEL_LOG((context), __FILE__ " " __VA_ARGS__); \
      return kTfLiteError;                                     \
    }                                                          \
  } while (0)

/// Check whether the value `a` is true, and if not return kTfLiteError from
/// the current function, while also reporting the location of the error.
#define TF_LITE_ENSURE(context, a)                                      \
  do {                                                                  \
    if (!(a)) {                                                         \
      TF_LITE_KERNEL_LOG((context), "%s:%d %s was not true.", __FILE__, \
                         __LINE__, #a);                                 \
      return kTfLiteError;                                              \
    }                                                                   \
  } while (0)

#define TF_LITE_ENSURE_STATUS(a) \
  do {                           \
    const TfLiteStatus s = (a);  \
    if (s != kTfLiteOk) {        \
      return s;                  \
    }                            \
  } while (0)

/// Check whether the value `a == b` is true, and if not return kTfLiteError
/// from the current function, while also reporting the location of the error.
/// `a` and `b` may be evaluated more than once, so no side effects or
/// extremely expensive computations should be done.
///
/// NOTE: Use TF_LITE_ENSURE_TYPES_EQ if comparing TfLiteTypes.
#define TF_LITE_ENSURE_EQ(context, a, b)                                   \
  do {                                                                     \
    if ((a) != (b)) {                                                      \
      TF_LITE_KERNEL_LOG((context), "%s:%d %s != %s (%d != %d)", __FILE__, \
                         __LINE__, #a, #b, (a), (b));                      \
      return kTfLiteError;                                                 \
    }                                                                      \
  } while (0)

#define TF_LITE_ENSURE_TYPES_EQ(context, a, b)                             \
  do {                                                                     \
    if ((a) != (b)) {                                                      \
      TF_LITE_KERNEL_LOG((context), "%s:%d %s != %s (%s != %s)", __FILE__, \
                         __LINE__, #a, #b, TfLiteTypeGetName(a),           \
                         TfLiteTypeGetName(b));                            \
      return kTfLiteError;                                                 \
    }                                                                      \
  } while (0)

#define TF_LITE_ENSURE_NEAR(context, a, b, epsilon)                          \
  do {                                                                       \
    auto delta = ((a) > (b)) ? ((a) - (b)) : ((b) - (a));                    \
    if (delta > epsilon) {                                                   \
      TF_LITE_KERNEL_LOG((context), "%s:%d %s not near %s (%f != %f)",       \
                         __FILE__, __LINE__, #a, #b, static_cast<double>(a), \
                         static_cast<double>(b));                            \
      return kTfLiteError;                                                   \
    }                                                                        \
  } while (0)

#define TF_LITE_ENSURE_OK(context, status) \
  do {                                     \
    const TfLiteStatus s = (status);       \
    if ((s) != kTfLiteOk) {                \
      return s;                            \
    }                                      \
  } while (0)

/// Half precision data type compatible with the C99 definition.
typedef struct TfLiteFloat16 {
  uint16_t data;
} TfLiteFloat16;

/// Return the name of a given type, for error reporting purposes.
const char* TfLiteTypeGetName(TfLiteType type);

/// SupportedQuantizationTypes.
typedef enum TfLiteQuantizationType {
  /// No quantization.
  kTfLiteNoQuantization = 0,
  /// Affine quantization (with support for per-channel quantization).
  /// Corresponds to TfLiteAffineQuantization.
  kTfLiteAffineQuantization = 1,
} TfLiteQuantizationType;

/// Structure specifying the quantization used by the tensor, if-any.
typedef struct TfLiteQuantization {
  /// The type of quantization held by params.
  TfLiteQuantizationType type;
  /// Holds an optional reference to a quantization param structure. The actual
  /// type depends on the value of the `type` field (see the comment there for
  /// the values and corresponding types).
  void* params;
} TfLiteQuantization;

/// Parameters for asymmetric quantization across a dimension (i.e per output
/// channel quantization).
/// quantized_dimension specifies which dimension the scales and zero_points
/// correspond to.
/// For a particular value in quantized_dimension, quantized values can be
/// converted back to float using:
///     `real_value = scale * (quantized_value - zero_point)`
typedef struct TfLiteAffineQuantization {
  TfLiteFloatArray* scale;
  TfLiteIntArray* zero_point;
  int32_t quantized_dimension;
} TfLiteAffineQuantization;

/// A union of pointers that points to memory for a given tensor.
///
/// Do not access these members directly, if possible, use
/// `GetTensorData<TYPE>(tensor)` instead, otherwise only access `.data`, as
/// other members are deprecated.
typedef union TfLitePtrUnion {
  int32_t* i32;
  uint32_t* u32;
  int64_t* i64;
  uint64_t* u64;
  float* f;
  TfLiteFloat16* f16;
  double* f64;
  char* raw;
  const char* raw_const;
  uint8_t* uint8;
  bool* b;
  int16_t* i16;
  uint16_t* ui16;
  int8_t* int8;
  /// Only use this member.
  void* data;
} TfLitePtrUnion;

/// Memory allocation strategies.
///  * `kTfLiteMmapRo`: Read-only memory-mapped data, or data externally
///        allocated.
///  * `kTfLiteArenaRw`: Arena allocated with no guarantees about persistence,
///        and available during eval.
///  * `kTfLiteArenaRwPersistent`: Arena allocated but persistent across eval,
///  and only available during eval.
///  * `kTfLiteDynamic`: Allocated during eval, or for string tensors.
///  * `kTfLitePersistentRo`: Allocated and populated during prepare. This is
///        useful for tensors that can be computed during prepare and treated
///        as constant inputs for downstream ops (also in prepare).
///  * `kTfLiteCustom`: Custom memory allocation provided by the user. See
///        TfLiteCustomAllocation below.
///  * `kTfLiteVariantObject`: Allocation is an arbitrary type-erased C++
///  object.
///        Allocation and deallocation are done through `new` and `delete`.
typedef enum TfLiteAllocationType {
  kTfLiteMemNone = 0,
  kTfLiteMmapRo,
  kTfLiteArenaRw,
  kTfLiteArenaRwPersistent,
  kTfLiteDynamic,
  kTfLitePersistentRo,
  kTfLiteCustom,
  kTfLiteVariantObject,
} TfLiteAllocationType;

/// A tensor in the interpreter system which is a wrapper around a buffer of
/// data including a dimensionality (or NULL if not currently defined).

// NOTE: This flag is opt-in only at compile time.
//
// Specific reduced TfLiteTensor struct for TF Micro runtime. This struct
// contains only the minimum fields required to initialize and prepare a micro
// inference graph. The fields in this struct have been ordered from
// largest-to-smallest for optimal struct sizeof.
//
// This struct does not use:
// - allocation
// - buffer_handle
// - data_is_stale
// - delegate
// - dims_signature
// - name
// - sparsity
typedef struct TfLiteTensor {
  // TODO(b/155784997): Consider consolidating these quantization fields:
  // Quantization information. Replaces params field above.
  TfLiteQuantization quantization;

  // Quantization information.
  TfLiteQuantizationParams params;

  // A union of data pointers. The appropriate type should be used for a typed
  // tensor based on `type`.
  TfLitePtrUnion data;

  // A pointer to a structure representing the dimensionality interpretation
  // that the buffer should have. NOTE: the product of elements of `dims`
  // and the element datatype size should be equal to `bytes` below.
  TfLiteIntArray* dims;

  // The number of bytes required to store the data of this Tensor. I.e.
  // (bytes of each element) * dims[0] * ... * dims[n-1].  For example, if
  // type is kTfLiteFloat32 and dims = {3, 2} then
  // bytes = sizeof(float) * 3 * 2 = 4 * 3 * 2 = 24.
  size_t bytes;

  // The data type specification for data stored in `data`. This affects
  // what member of `data` union should be used.
  TfLiteType type;

  // How memory is mapped
  //  kTfLiteMmapRo: Memory mapped read only.
  //  i.e. weights
  //  kTfLiteArenaRw: Arena allocated read write memory
  //  (i.e. temporaries, outputs).
  TfLiteAllocationType allocation_type;

  // True if the tensor is a variable.
  bool is_variable;
} TfLiteTensor;

// Specific reduced TfLiteNode struct for TF Micro runtime. This struct contains
// only the minimum fields required to represent a node.
//
// This struct does not use:
// - delegate
// - intermediates
// - temporaries
typedef struct TfLiteNode {
  // Inputs to this node expressed as indices into the simulator's tensors.
  TfLiteIntArray* inputs;

  // Outputs to this node expressed as indices into the simulator's tensors.
  TfLiteIntArray* outputs;

  // intermediate tensors to this node expressed as indices into the simulator's
  // tensors.
  TfLiteIntArray* intermediates;

  // Opaque data provided by the node implementer through `Registration.init`.
  void* user_data;

  // Opaque data provided to the node if the node is a builtin. This is usually
  // a structure defined in builtin_op_data.h
  void* builtin_data;

  // Custom initial data. This is the opaque data provided in the flatbuffer.
  //
  // WARNING: This is an experimental interface that is subject to change.
  const void* custom_initial_data;
  int custom_initial_data_size;
} TfLiteNode;

/// Light-weight tensor struct for TF Micro runtime. Provides the minimal amount
/// of information required for a kernel to run during TfLiteRegistration::Eval.
// TODO(b/160955687): Move this field into TF_LITE_STATIC_MEMORY when TFLM
// builds with this flag by default internally.
typedef struct TfLiteEvalTensor {
  /// A union of data pointers. The appropriate type should be used for a typed
  /// tensor based on `type`.
  TfLitePtrUnion data;

  /// A pointer to a structure representing the dimensionality interpretation
  /// that the buffer should have.
  TfLiteIntArray* dims;

  /// The data type specification for data stored in `data`. This affects
  /// what member of `data` union should be used.
  TfLiteType type;
} TfLiteEvalTensor;

/// `TfLiteContext` allows an op to access the tensors.
///
/// `TfLiteContext` is a struct that is created by the TF Lite runtime
/// and passed to the "methods" (C function pointers) in the
/// `TfLiteRegistration` struct that are used to define custom ops and custom
/// delegate kernels. It contains information and methods (C function pointers)
/// that can be called by the code implementing a custom op or a custom delegate
/// kernel. These methods provide access to the context in which that custom op
/// or custom delegate kernel occurs, such as access to the input and output
/// tensors for that op, as well as methods for allocating memory buffers
/// and intermediate tensors, etc.
///
/// See also `TfLiteOpaqueContext`, which is an more ABI-stable equivalent.
typedef struct TfLiteContext {
  /// Number of tensors in the context.
  size_t tensors_size;

  /// An array of tensors in the interpreter context (of length `tensors_size`)
  TfLiteTensor* tensors;

  /// opaque full context ptr (an opaque c++ data structure)
  void* impl_;

  /// Request that an error be reported with format string msg.
  void (*ReportError)(struct TfLiteContext*, const char* msg, ...);

  /// Number of threads that are recommended to subsystems like gemmlowp and
  /// eigen.
  int recommended_num_threads;

  /// Pointer to the op-level profiler, if set; nullptr otherwise.
  void* profiler;

  /// Allocate persistent buffer which has the same life time as the
  /// interpreter. Returns `nullptr` on failure. The memory is allocated from
  /// heap for TFL, and from tail in TFLM. This method is only available in
  /// `Init` or `Prepare` stage.
  ///
  /// WARNING: This is an experimental interface that is subject
  /// to change.
  void* (*AllocatePersistentBuffer)(struct TfLiteContext* ctx, size_t bytes);

  /// Request a scratch buffer in the arena through static memory planning.
  /// This method is only available in `Prepare` stage and the buffer is
  /// allocated by the interpreter between Prepare and Eval stage. In `Eval`
  /// stage, `GetScratchBuffer` API can be used to fetch the address.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus (*RequestScratchBufferInArena)(struct TfLiteContext* ctx,
                                              size_t bytes, int* buffer_idx);

  /// Get the scratch buffer pointer.
  /// This method is only available in Eval stage.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  void* (*GetScratchBuffer)(struct TfLiteContext* ctx, int buffer_idx);

  /// This method provides a preview of post-delegation partitioning. Each
  /// TfLiteDelegateParams in the referenced array corresponds to one instance
  /// of the delegate kernel. Example usage:
  ///
  ///     TfLiteIntArray* nodes_to_replace = ...;
  ///     TfLiteDelegateParams* params_array;
  ///     int num_partitions = 0;
  ///     TF_LITE_ENSURE_STATUS(context->PreviewDelegatePartitioning(
  ///        context, delegate, nodes_to_replace, &params_array,
  ///        &num_partitions));
  ///     for (int idx = 0; idx < num_partitions; idx++) {
  ///        const auto& partition_params = params_array[idx];
  ///        ...
  ///     }
  ///
  /// NOTE: The context owns the memory referenced by partition_params_array. It
  /// will be cleared with another call to PreviewDelegatePartitioning, or after
  /// TfLiteDelegateParams::Prepare returns.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  // TfLiteStatus (*PreviewDelegatePartitioning)(
  //     struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
  //     TfLiteDelegateParams** partition_params_array, int* num_partitions);

  /// Returns a TfLiteTensor struct for a given index.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  ///
  /// WARNING: This method may not be available on all platforms.
  TfLiteTensor* (*GetTensor)(const struct TfLiteContext* context,
                             int tensor_idx);

  /// Returns a TfLiteEvalTensor struct for a given index.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  ///
  /// WARNING: This method may not be available on all platforms.
  TfLiteEvalTensor* (*GetEvalTensor)(const struct TfLiteContext* context,
                                     int tensor_idx);

} TfLiteContext;


#ifdef __cplusplus
}  // extern "C"

#include <utility>

#endif  // __cplusplus
#endif  // TENSORFLOW_LITE_CORE_C_COMMON_H_
