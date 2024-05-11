TFLIE_MICRO_DIR := ../../../../../
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
	$(wildcard $(CURDIR)/main.cc)

# SRC += 	$(wildcard $(CURDIR)/main.cc)


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