# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

ifneq (3.82,$(firstword $(sort $(MAKE_VERSION) 3.82)))
  $(error "Requires make version 3.82 or later (current is $(MAKE_VERSION))")
endif

####################
## Option
# 构建动态库
IS_BUILD_DYNAMIC := 0
# 是否编译测试代码
IS_BUILD_TEST := 0
# 是否编译signal模块
IS_BUILD_SIGNAL := 0
# kiss fft
IS_BUILD_KISS_FFT := 0

# root directory of tensorflow
TENSORFLOW_ROOT :=
RELATIVE_MAKEFILE_DIR := tensorflow/lite/micro/tools/make
MAKEFILE_DIR := $(TENSORFLOW_ROOT)$(RELATIVE_MAKEFILE_DIR)


# Try to figure out the host system
HOST_OS :=
ifeq ($(OS),Windows_NT)
  HOST_OS = windows
else
  UNAME_S := $(shell uname -s)
  ifeq ($(UNAME_S),Linux)
    HOST_OS := linux
  endif
endif

# Determine the host architecture, with any ix86 architecture being labelled x86_32
HOST_ARCH := $(shell if uname -m | grep -Eq 'i[345678]86'; then echo x86_32; else echo $(shell uname -m); fi)

# Override these on the make command line to target a specific architecture. For example:
# make -f tensorflow/lite/Makefile TARGET=rpi TARGET_ARCH=armv7l
TARGET := $(HOST_OS)
TARGET_ARCH := $(HOST_ARCH)

# Default compiler and tool names:
TOOLCHAIN := gcc
CXX_TOOL := g++
CC_TOOL := gcc
AR_TOOL := ar

# This is the downloads directory inside the makefiles directory
DOWNLOADS_DIR := $(MAKEFILE_DIR)/downloads

INCLUDES := \
-I. \
-I$(DOWNLOADS_DIR) \
-I$(DOWNLOADS_DIR)/gemmlowp \
-I$(DOWNLOADS_DIR)/flatbuffers/include

ifneq ($(TENSORFLOW_ROOT),)
  INCLUDES += -I$(TENSORFLOW_ROOT)
endif

MICROLITE_LIBS := -lm

ifeq ($(TOOLCHAIN), armclang)
  CORE_OPTIMIZATION_LEVEL := -Oz
else
  CORE_OPTIMIZATION_LEVEL := -Os
endif
KERNEL_OPTIMIZATION_LEVEL := -O2
THIRD_PARTY_KERNEL_OPTIMIZATION_LEVEL := -O2

CC_WARNINGS := \
  -Wsign-compare \
  -Wdouble-promotion \
  -Wunused-variable \
  -Wunused-function \
  -Wswitch \
  -Wvla \
  -Wall \
  -Wextra \
  -Wmissing-field-initializers \
  -Wstrict-aliasing \
  -Wno-unused-parameter

ifneq ($(TOOLCHAIN), gcc)
  # GCC can be overly aggressive with shadow warnings, such as warning when a
  # lambda has variable with the same name as a non-captured variable from the
  # enclosing scope. As such, we don't enable shadow warnings on gcc.
  # https://stackoverflow.com/q/66404751
  CC_WARNINGS += -Wshadow
endif

COMMON_FLAGS := \
  -Werror \
  -fno-unwind-tables \
  -ffunction-sections \
  -fdata-sections \
  -fmessage-length=0 \
  -DTF_LITE_STATIC_MEMORY \
  -DTF_LITE_DISABLE_X86_NEON \
  $(CC_WARNINGS) \
  $(ADDITIONAL_DEFINES)

ifeq ($(TARGET), $(HOST_OS))
  # If we are not doing a cross-compilation then -DTF_LITE_USE_CTIME is what we
  # want to have by default.
  COMMON_FLAGS += -DTF_LITE_USE_CTIME
endif

CXXFLAGS := \
  -std=c++11 \
  -fno-rtti \
  -fno-exceptions \
  -fno-threadsafe-statics \
  -Wnon-virtual-dtor \
  $(COMMON_FLAGS)

CCFLAGS := \
  -Wimplicit-function-declaration \
  -std=c11 \
  $(COMMON_FLAGS)

ifeq ($(TOOLCHAIN), gcc)
  # GCC on MacOS uses an LLVM backend so we avoid the additional linker flags
  # that are unsupported with LLVM.
  LDFLAGS += \
    -Wl,--fatal-warnings \
    -Wl,--gc-sections
endif

# override these in the makefile.inc for specific compiler targets
TARGET_TOOLCHAIN_PREFIX :=
TARGET_TOOLCHAIN_ROOT :=

# Specifying BUILD_TYPE=<blah> as part of the make command gives us a few
# options to choose from.
#
# If BUILD_TYPE is not specified, the default build (which should be suitable
# most of the time) has all of the error checking logic at the expense of a
# latency increase of ~5-10% relative to BUILD_TYPE=release_with_logs.
#
# This default build is most suited for usual development and testing as is
# highlighted by the discussion on this github pull request:
# https://github.com/tensorflow/tensorflow/pull/42314#issuecomment-694360567
BUILD_TYPE := default
ifeq ($(BUILD_TYPE), debug)
	# Specifying BUILD_TYPE=debug adds debug symbols to the binary (and makes it
	# larger) and should be used to run a binary with gdb.
	CXXFLAGS += -g
	CCFLAGS  += -g
else ifeq ($(BUILD_TYPE), release)
	# The 'release' build results in the smallest binary (by virtue of removing
	# strings from log messages, DCHECKs ...).
	#
	# The down-side is that we currently do not have a good mechanism to allow
	# for logging that is not related to errors (e.g. profiling information, or
	# logs that help determine if tests pass or fail). As a result, we are unable
	# to run tests or benchmarks with BUILD_TYPE=release (which is a bit
	# counter-intuitive). TODO(b/158205789): A global error reporter might help.
	#
	# For a close approximation of the release build use
	# BUILD_TYPE=release_with_logs.
	CXXFLAGS += -DNDEBUG -DTF_LITE_STRIP_ERROR_STRINGS
	CCFLAGS  += -DNDEBUG -DTF_LITE_STRIP_ERROR_STRINGS
else ifeq ($(BUILD_TYPE), release_with_logs)
	# The latency with BUILD_TYPE=release_with_logs will be close to the 'release'
	# build and there will still be error logs. This build type may be preferable
	# for profiling and benchmarking.
	CXXFLAGS += -DNDEBUG
	CCFLAGS  += -DNDEBUG
else ifeq ($(BUILD_TYPE), no_tf_lite_static_memory)
	# TODO(b/287320282): remove the no_tf_lite_static_memory build.
	#
	# This build should not be used to run any binaries/tests since
	# TF_LITE_STATIC_MEMORY should be defined for all micro builds. However,
	# having a build without TF_LITE_STATIC_MEMORY is useful to catch errors in
	# code that is shared between TfLite Mobile and TfLite Micro. See this issue
	# for more details:
	# https://github.com/tensorflow/tensorflow/issues/43076
	CXXFLAGS := $(filter-out -DTF_LITE_STATIC_MEMORY, $(CXXFLAGS))
	CCFLAGS := $(filter-out -DTF_LITE_STATIC_MEMORY, $(CCFLAGS))

	# We are using C++17 for the no_tf_lite_static_memory_build to make it close
	# to the TfLite bazel build.
	CXXFLAGS := $(filter-out -std=c++11, $(CXXFLAGS))
	CXXFLAGS += -std=c++17
	CCFLAGS := $(filter-out -std=c11, $(CCLAGS))
	CCFLAGS += -std=c17
endif

# This library is the main target for this makefile. It will contain a minimal
# runtime that can be linked in to other programs.
ifeq ($(IS_BUILD_DYNAMIC), 1)
  CCFLAGS += -fPIC
  CXXFLAGS += -fPIC
  MICROLITE_LIB_NAME := libtensorflow-microlite.so
else 
  MICROLITE_LIB_NAME := libtensorflow-microlite.a
endif

# Where compiled objects are stored.
BASE_GENDIR := gen
GENDIR := $(BASE_GENDIR)/$(TARGET)_$(TARGET_ARCH)_$(BUILD_TYPE)/
CORE_OBJDIR := $(GENDIR)obj/core/
KERNEL_OBJDIR := $(GENDIR)obj/kernels/
THIRD_PARTY_KERNEL_OBJDIR := $(GENDIR)obj/third_party_kernels/
THIRD_PARTY_OBJDIR := $(GENDIR)obj/third_party/
GENERATED_SRCS_DIR := $(GENDIR)genfiles/
BINDIR := $(GENDIR)bin/
LIBDIR := $(GENDIR)lib/
PRJDIR := $(GENDIR)prj/

MICROLITE_CC_KERNEL_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/activations.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/activations_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/add.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/add_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/add_n.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/arg_min_max.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/assign_variable.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/batch_matmul.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/batch_to_space_nd.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/broadcast_args.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/broadcast_to.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/call_once.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/cast.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/ceil.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/circular_buffer.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/circular_buffer_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/comparisons.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/concatenation.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/conv.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/conv_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/cumsum.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/depth_to_space.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/depthwise_conv.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/depthwise_conv_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/dequantize.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/dequantize_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/detection_postprocess.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/div.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/elementwise.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/elu.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/embedding_lookup.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/ethosu.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/exp.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/expand_dims.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/fill.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/floor.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/floor_div.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/floor_mod.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/fully_connected.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/fully_connected_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/gather.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/gather_nd.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/hard_swish.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/hard_swish_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/if.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/kernel_runner.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/kernel_util.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/l2norm.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/l2_pool_2d.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/leaky_relu.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/leaky_relu_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/logical.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/logical_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/logistic.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/logistic_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/log_softmax.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/lstm_eval.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/lstm_eval_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/maximum_minimum.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/micro_tensor_utils.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/mirror_pad.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/mul.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/mul_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/neg.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/pack.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/pad.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/pooling.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/pooling_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/prelu.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/prelu_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/quantize.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/quantize_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/read_variable.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/reduce.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/reduce_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/reshape.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/reshape_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/resize_bilinear.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/resize_nearest_neighbor.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/round.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/select.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/shape.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/slice.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/softmax.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/softmax_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/space_to_batch_nd.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/space_to_depth.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/split.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/split_v.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/squared_difference.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/squeeze.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/strided_slice.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/strided_slice_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/sub.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/sub_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/svdf.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/svdf_common.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/tanh.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/transpose.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/transpose_conv.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/unidirectional_sequence_lstm.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/unpack.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/var_handle.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/while.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/zeros_like.cc

# The explicitly specified list of sources and headers that are shared between
# TfLite and TFLM are in the ci/sync_from_upstream_tf.sh script.
TFL_CC_SRCS := \
$(shell find $(TENSORFLOW_ROOT)tensorflow/lite -type d \( -path $(TENSORFLOW_ROOT)tensorflow/lite/experimental -o -path $(TENSORFLOW_ROOT)tensorflow/lite/micro \) -prune -false -o -name "*.cc" -o -name "*.c")

TFL_CC_HDRS := \
$(shell find $(TENSORFLOW_ROOT)tensorflow/lite -type d \( -path $(TENSORFLOW_ROOT)tensorflow/lite/experimental -o -path $(TENSORFLOW_ROOT)tensorflow/lite/micro \) -prune -false -o -name "*.h")

ifneq ($(BUILD_TYPE), no_tf_lite_static_memory)
  EXCLUDED_TFL_CC_SRCS := \
  	$(TENSORFLOW_ROOT)tensorflow/lite/array.cc
  TFL_CC_SRCS := $(filter-out $(EXCLUDED_TFL_CC_SRCS), $(TFL_CC_SRCS))

  EXCLUDED_TFL_CC_HDRS := \
  	$(TENSORFLOW_ROOT)tensorflow/lite/array.h
  TFL_CC_HDRS := $(filter-out $(EXCLUDED_TFL_CC_HDRS), $(TFL_CC_HDRS))
endif

MICROLITE_CC_SRCS := \
$(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/*.cc) \
$(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/arena_allocator/*.cc) \
$(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/memory_planner/*.cc) \
$(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/tflite_bridge/*.cc) \
$(TFL_CC_SRCS)

# TODO(b/165940489): Figure out how to avoid including fixed point
# platform-specific headers.
THIRD_PARTY_CC_HDRS := \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/allocator.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/array.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/base.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/buffer.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/buffer_ref.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/code_generator.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/code_generators.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/default_allocator.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/detached_buffer.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/file_manager.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/flatbuffer_builder.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/flatbuffers.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/flex_flat_util.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/flexbuffers.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/grpc.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/hash.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/idl.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/minireflect.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/reflection.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/reflection_generated.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/registry.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/stl_emulation.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/string.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/struct.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/table.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/util.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/vector.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/vector_downward.h \
$(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/verifier.h \
$(DOWNLOADS_DIR)/flatbuffers/LICENSE \
$(DOWNLOADS_DIR)/gemmlowp/fixedpoint/fixedpoint.h \
$(DOWNLOADS_DIR)/gemmlowp/fixedpoint/fixedpoint_neon.h \
$(DOWNLOADS_DIR)/gemmlowp/fixedpoint/fixedpoint_sse.h \
$(DOWNLOADS_DIR)/gemmlowp/internal/detect_platform.h \
$(DOWNLOADS_DIR)/gemmlowp/LICENSE

# The target-specific makefile must have a name that is exactly
# TARGET_makefile.inc and is only needed for cross-compilation (i.e. when TARGET
# is different from the HOST_OS).
TARGETS_WITHOUT_MAKEFILES := $(HOST_OS)

# This specific string needs to be outputted for a test to be recognized as
# having passed.
TEST_PASS_STRING:='~~~ALL TESTS PASSED~~~'

ifeq ($(IS_BUILD_KISS_FFT), 1)
  INCLUDES += -I$(DOWNLOADS_DIR)/kissfft

  # some kiss fft source file has to be included in header sections because
  # the implemenation of three different resolution fft from one single c file.
  # See http://b/201319430 for additional context.
  THIRD_PARTY_CC_HDRS += \
  $(DOWNLOADS_DIR)/kissfft/COPYING \
  $(DOWNLOADS_DIR)/kissfft/kiss_fft.c \
  $(DOWNLOADS_DIR)/kissfft/kiss_fft.h \
  $(DOWNLOADS_DIR)/kissfft/_kiss_fft_guts.h \
  $(DOWNLOADS_DIR)/kissfft/tools/kiss_fftr.c \
  $(DOWNLOADS_DIR)/kissfft/tools/kiss_fftr.h \

  # Load custom kernels.
  MICROLITE_CC_KERNEL_SRCS += \
  $(TENSORFLOW_ROOT)signal/src/kiss_fft_wrappers/kiss_fft_float.cc \
  $(TENSORFLOW_ROOT)signal/src/kiss_fft_wrappers/kiss_fft_int16.cc \
  $(TENSORFLOW_ROOT)signal/src/kiss_fft_wrappers/kiss_fft_int32.cc

#   MICROLITE_CC_HDRS += \
#   $(wildcard $(TENSORFLOW_ROOT)signal/src/kiss_fft_wrappers/*.h) \

endif

ifeq ($(IS_BUILD_SIGNAL), 1)
  MICROLITE_CC_SIGNAL_KERNEL_SRCS := \
  $(TENSORFLOW_ROOT)signal/micro/kernels/delay.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/energy.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/fft_auto_scale_kernel.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/fft_auto_scale_common.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/filter_bank.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/filter_bank_log.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/filter_bank_square_root.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/filter_bank_square_root_common.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/filter_bank_spectral_subtraction.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/framer.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/irfft.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/rfft.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/stacker.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/overlap_add.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/pcan.cc \
  $(TENSORFLOW_ROOT)signal/micro/kernels/window.cc \
  $(TENSORFLOW_ROOT)signal/src/circular_buffer.cc \
  $(TENSORFLOW_ROOT)signal/src/energy.cc \
  $(TENSORFLOW_ROOT)signal/src/fft_auto_scale.cc \
  $(TENSORFLOW_ROOT)signal/src/filter_bank.cc \
  $(TENSORFLOW_ROOT)signal/src/filter_bank_log.cc \
  $(TENSORFLOW_ROOT)signal/src/filter_bank_square_root.cc \
  $(TENSORFLOW_ROOT)signal/src/filter_bank_spectral_subtraction.cc \
  $(TENSORFLOW_ROOT)signal/src/irfft_float.cc \
  $(TENSORFLOW_ROOT)signal/src/irfft_int16.cc \
  $(TENSORFLOW_ROOT)signal/src/irfft_int32.cc \
  $(TENSORFLOW_ROOT)signal/src/log.cc \
  $(TENSORFLOW_ROOT)signal/src/max_abs.cc \
  $(TENSORFLOW_ROOT)signal/src/msb_32.cc \
  $(TENSORFLOW_ROOT)signal/src/msb_64.cc \
  $(TENSORFLOW_ROOT)signal/src/overlap_add.cc \
  $(TENSORFLOW_ROOT)signal/src/pcan_argc_fixed.cc \
  $(TENSORFLOW_ROOT)signal/src/rfft_float.cc \
  $(TENSORFLOW_ROOT)signal/src/rfft_int16.cc \
  $(TENSORFLOW_ROOT)signal/src/rfft_int32.cc \
  $(TENSORFLOW_ROOT)signal/src/square_root_32.cc \
  $(TENSORFLOW_ROOT)signal/src/square_root_64.cc \
  $(TENSORFLOW_ROOT)signal/src/window.cc

  MICROLITE_CC_KERNEL_SRCS += $(MICROLITE_CC_SIGNAL_KERNEL_SRCS)
endif

ALL_SRCS := \
	$(MICROLITE_CC_SRCS) \
	$(MICROLITE_CC_KERNEL_SRCS) 

MICROLITE_LIB_PATH := $(LIBDIR)$(MICROLITE_LIB_NAME)

CXX := $(TARGET_TOOLCHAIN_ROOT)${TARGET_TOOLCHAIN_PREFIX}${CXX_TOOL}
CC := $(TARGET_TOOLCHAIN_ROOT)${TARGET_TOOLCHAIN_PREFIX}${CC_TOOL}
AR := $(TARGET_TOOLCHAIN_ROOT)${TARGET_TOOLCHAIN_PREFIX}${AR_TOOL}

# The default Makefile target(all) must appear before any target,
# which is compiled if there's no command-line arguments.
all: $(MICROLITE_LIB_PATH)

# Include output directory since example cc files depend on generated headers.
INCLUDES += -I$(GENERATED_SRCS_DIR)
INCLUDES += -I$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)

ifneq ($(IS_BUILD_TEST), 1)
  EXCLUDED_MICROLITE_CC_SRCS := \
    $(TENSORFLOW_ROOT)tensorflow/lite/micro/fake_micro_context.cc \
    $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernel_runner.cc \
    $(TENSORFLOW_ROOT)tensorflow/lite/micro/mock_micro_graph.cc \
  	$(TENSORFLOW_ROOT)tensorflow/lite/micro/test_helpers.cc \
    $(TENSORFLOW_ROOT)tensorflow/lite/micro/test_helper_custom_ops.cc
  MICROLITE_CC_SRCS := $(filter-out $(EXCLUDED_MICROLITE_CC_SRCS), $(MICROLITE_CC_SRCS))
endif

MICROLITE_LIB_OBJS := $(addprefix $(CORE_OBJDIR), \
$(patsubst %.S,%.o,$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(MICROLITE_CC_SRCS)))))

# # 测试无kernel时为多大
# MICROLITE_KERNEL_OBJS := 

MICROLITE_KERNEL_OBJS := $(addprefix $(KERNEL_OBJDIR), \
$(patsubst %.S,%.o,$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(MICROLITE_CC_KERNEL_SRCS)))))

$(CORE_OBJDIR)%.o: %.cc $(THIRD_PARTY_TARGETS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(CORE_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(CORE_OBJDIR)%.o: %.c $(THIRD_PARTY_TARGETS)
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(CORE_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(CORE_OBJDIR)%.o: %.S $(THIRD_PARTY_TARGETS)
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(CORE_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(KERNEL_OBJDIR)%.o: %.cc $(THIRD_PARTY_TARGETS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(KERNEL_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(KERNEL_OBJDIR)%.o: %.c $(THIRD_PARTY_TARGETS)
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(KERNEL_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(KERNEL_OBJDIR)%.o: %.S $(THIRD_PARTY_TARGETS)
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(KERNEL_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

microlite: $(MICROLITE_LIB_PATH)

# Gathers together all the objects we've compiled into a single '.a' archive.
ifeq ($(IS_BUILD_DYNAMIC), 1)
  $(MICROLITE_LIB_PATH): $(MICROLITE_LIB_OBJS) $(MICROLITE_KERNEL_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) -shared -o $(MICROLITE_LIB_PATH) $(MICROLITE_LIB_OBJS) $(MICROLITE_KERNEL_OBJS)
else
  $(MICROLITE_LIB_PATH): $(MICROLITE_LIB_OBJS) $(MICROLITE_KERNEL_OBJS)
	@mkdir -p $(dir $@)
	$(AR) -r $(MICROLITE_LIB_PATH) $(MICROLITE_LIB_OBJS) $(MICROLITE_KERNEL_OBJS)
endif

# ############################################################
# ## Only for testing
ifeq ($(IS_BUILD_TEST), 1)
  include $(MAKEFILE_DIR)/helper_functions.inc

  MICRO_LITE_EXAMPLE_TESTS := $(shell find $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/ -maxdepth 2 -name Makefile.inc)

  MICRO_LITE_BENCHMARKS := $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/tools/benchmarking/Makefile.inc)

  MICRO_LITE_LAYER_BY_LAYER_OUTPUT := $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/tools/Makefile.inc)

  # Load the examples.
  include $(MICRO_LITE_EXAMPLE_TESTS)

  # Load the benchmarks.
  include $(MICRO_LITE_BENCHMARKS)

  $(BINDIR)%_test : $(CORE_OBJDIR)%_test.o $(MICROLITE_LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $@ $< \
	$(MICROLITE_LIB_PATH) $(LDFLAGS) $(MICROLITE_LIBS)

  # Create kernel test targets.
  include $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/Makefile.inc

  # Just build the test targets
  build: $(MICROLITE_BUILD_TARGETS)
endif 
# ############################################################

# snease: Add %.bin rule here since BINDIR is now defined
# These are microcontroller-specific rules for converting the ELF output
# of the linker into a binary image that can be loaded directly.
ifeq ($(TOOLCHAIN), armclang)
  FROMELF := ${TARGET_TOOLCHAIN_ROOT}$(TARGET_TOOLCHAIN_PREFIX)fromelf
  $(BINDIR)%.bin: $(BINDIR)%
		@mkdir -p $(dir $@)
		$(FROMELF) --bin --output=$@ $<
else
  OBJCOPY := ${TARGET_TOOLCHAIN_ROOT}$(TARGET_TOOLCHAIN_PREFIX)objcopy
  $(BINDIR)%.bin: $(BINDIR)%
		@mkdir -p $(dir $@)
		$(OBJCOPY) $< $@ -O binary
endif


list_library_sources:
	@echo $(MICROLITE_CC_SRCS) $(MICROLITE_CC_KERNEL_SRCS)

# list_library_headers:
# 	@echo $(MICROLITE_CC_HDRS)

list_third_party_headers:
	@echo $(THIRD_PARTY_CC_HDRS)

list_generator_dir:
	@echo $(GENERATED_SRCS_DIR)

# Gets rid of all generated files.
clean:
	rm -rf $(BASE_GENDIR)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d
.PRECIOUS: $(BINDIR)%_test

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(ALL_SRCS)))
