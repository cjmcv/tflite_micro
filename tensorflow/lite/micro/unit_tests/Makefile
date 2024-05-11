TFLIE_MICRO_DIR := ../../../../
3RDPARTY_GEMMLOWP_DIR := $(TFLIE_MICRO_DIR)/tensorflow/lite/micro/tools/make/3rdparty/gemmlowp

TFLITE_MOCRO_LIB := -L$(TFLIE_MICRO_DIR)/gen/linux_x86_64_release_with_logs/lib -ltensorflow-microlite

CXX_TOOL := g++
CC_TOOL  := gcc
AR_TOOL  := ar
CXXFLAGS := -Werror -fno-exceptions -fPIC -std=c++11
LDFLAGS  := -lstdc++ -lm 
BUILD    := ./build
OBJ_DIR  := $(BUILD)/objects
APP_DIR  := $(BUILD)/apps
TARGET   := unit_test

CXX := $(CXX_TOOL)
LDFLAGS += $(TFLITE_MOCRO_LIB)

INCLUDE  := -I. \
            -I$(TFLIE_MICRO_DIR) \
			-I$(3RDPARTY_GEMMLOWP_DIR)

# SRC      :=                      \
# 	$(wildcard $(CURDIR)/person_detection_test.cc) \
# 	$(wildcard testdata/*.cc) \
# 	$(wildcard models/*.cc)         \

SRC      :=                      \
	$(wildcard $(CURDIR)/util/fake_micro_context.cc) \
	$(wildcard $(CURDIR)/util/kernel_runner.cc) \
	$(wildcard $(CURDIR)/util/test_helpers.cc)

SRC += 	$(wildcard $(CURDIR)/test_main.cc)

# kernels
SRC += 	$(wildcard $(CURDIR)/test_kernels.cc)

SRC += 	$(wildcard $(CURDIR)/kernels/activations_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/add_n_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/add_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/arg_min_max_test.cc)

SRC += 	$(wildcard $(CURDIR)/kernels/batch_matmul_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/batch_to_space_nd_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/broadcast_args_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/broadcast_to_test.cc)

SRC += 	$(wildcard $(CURDIR)/kernels/cast_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/ceil_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/comparisons_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/concatenation_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/conv_test.cc) \
		$(wildcard $(CURDIR)/kernels/testdata/conv_test_data.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/cumsum_test.cc)

SRC += 	$(wildcard $(CURDIR)/kernels/depth_to_space_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/depthwise_conv_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/dequantize_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/div_test.cc)

SRC += 	$(wildcard $(CURDIR)/kernels/elementwise_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/elu_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/exp_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/expand_dims_test.cc)

SRC += 	$(wildcard $(CURDIR)/kernels/fill_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/floor_div_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/floor_mod_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/floor_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/fully_connected_test.cc)

SRC += 	$(wildcard $(CURDIR)/kernels/gather_nd_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/gather_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/hard_swish_test.cc)

SRC += 	$(wildcard $(CURDIR)/kernels/l2_pool_2d_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/l2norm_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/leaky_relu_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/log_softmax_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/logical_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/logistic_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/lstm_eval_test.cc) \
        $(wildcard $(CURDIR)/kernels/testdata/lstm_test_data.cc)

SRC += 	$(wildcard $(CURDIR)/kernels/maximum_minimum_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/mirror_pad_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/mul_test.cc)

SRC += 	$(wildcard $(CURDIR)/kernels/neg_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/pack_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/pad_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/pooling_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/prelu_test.cc)

SRC += 	$(wildcard $(CURDIR)/kernels/quantize_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/reduce_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/reshape_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/resize_bilinear_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/resize_nearest_neighbor_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/round_test.cc)

SRC += 	$(wildcard $(CURDIR)/kernels/select_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/shape_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/slice_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/softmax_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/space_to_batch_nd_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/space_to_depth_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/split_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/split_v_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/squared_difference_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/squeeze_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/strided_slice_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/sub_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/svdf_test.cc)

SRC += 	$(wildcard $(CURDIR)/kernels/tanh_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/transpose_conv_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/transpose_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/unidirectional_sequence_lstm_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/unpack_test.cc)
SRC += 	$(wildcard $(CURDIR)/kernels/zeros_like_test.cc)

OBJECTS  := $(SRC:%.cc=$(OBJ_DIR)/%.o)
DEPENDENCIES \
         := $(OBJECTS:.o=.d)

all: build $(APP_DIR)/$(TARGET)

$(OBJ_DIR)/%.o: %.cc
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -MMD -o $@

$(APP_DIR)/$(TARGET): $(OBJECTS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $(APP_DIR)/$(TARGET) $^ $(LDFLAGS)

-include $(DEPENDENCIES)

.PHONY: all build clean debug release info

build:
	@mkdir -p $(APP_DIR)
	@mkdir -p $(OBJ_DIR)

debug: CXXFLAGS += -DDEBUG -g
debug: all

release: CXXFLAGS += -O2
release: all

clean:
	-@rm -rvf $(OBJ_DIR)/*
	-@rm -rvf $(APP_DIR)/*

info:
	@echo "[*] Application dir: ${APP_DIR}     "
	@echo "[*] Object dir:      ${OBJ_DIR}     "
	@echo "[*] Sources:         ${SRC}         "
	@echo "[*] Objects:         ${OBJECTS}     "
	@echo "[*] Dependencies:    ${DEPENDENCIES}"