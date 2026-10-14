# ag engine. one Makefile, no generator:
#
#   make test     build and run the native tests
#   make torch    compare every op and optimizer against PyTorch (needs torch)
#   make bench    step time against model size
#   make wasm     WebAssembly build into ../web/public/engine (needs emsdk on PATH)
#   make wasm-test  run the moons check through the WASM build under node
#   make wasm-bench the benchmark, compiled to WASM and run under node

CXX      ?= g++
EMCXX    ?= em++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Wpedantic
BUILD    := build
SRC      := $(wildcard src/*.cpp)
OBJ      := $(SRC:src/%.cpp=$(BUILD)/%.o)
WASM_OUT := ../web/public/engine

# every function declared in capi.h is exported, and nothing else. make counts
# parentheses inside $(shell), so the literal one goes through a variable
LPAREN   := (
EXPORTS  := $(shell grep -oE '\bag_[a-z_]+[$(LPAREN)]' include/ag/capi.h | sed -e 's/[$(LPAREN)]//' -e 's/^/_/' | sort -u | paste -sd, -)

.PHONY: all test torch bench wasm wasm-test wasm-bench clean
all: test

$(BUILD)/%.o: src/%.cpp $(wildcard include/ag/*.h*) $(wildcard src/*.hpp) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Iinclude -c $< -o $@

$(BUILD)/test_%.o: tests/test_%.cpp $(wildcard include/ag/*.h*) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Iinclude -c $< -o $@

$(BUILD)/ag_test_%: $(BUILD)/test_%.o $(OBJ)
	$(CXX) -o $@ $^

$(BUILD)/ag_bench: bench/bench.cpp $(OBJ)
	$(CXX) $(CXXFLAGS) -Iinclude -o $@ $^

$(BUILD):
	mkdir -p $(BUILD)

test: $(BUILD)/ag_test_skeleton $(BUILD)/ag_test_engine
	./$(BUILD)/ag_test_skeleton
	./$(BUILD)/ag_test_engine --dump $(BUILD)/cases.json

torch: test
	python tests/torch_check.py $(BUILD)/cases.json

bench: $(BUILD)/ag_bench
	./$(BUILD)/ag_bench

# fixed heap, no growth, no exceptions. malloc returns null instead of
# aborting, so the engine can report OutOfMemory for the big allocations
# (execInit checks first); anything else that runs out is a trap, which the
# web client handles by restarting the worker
wasm:
	mkdir -p $(WASM_OUT)
	$(EMCXX) -std=c++20 -O2 -fno-exceptions -Iinclude $(SRC) -o $(WASM_OUT)/ag.mjs \
	  -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=web,worker,node \
	  -sINITIAL_MEMORY=134217728 -sALLOW_MEMORY_GROWTH=0 -sABORTING_MALLOC=0 -sFILESYSTEM=0 -sSTACK_SIZE=1048576 \
	  -sEXPORTED_FUNCTIONS=$(EXPORTS),_malloc,_free \
	  -sEXPORTED_RUNTIME_METHODS=UTF8ToString,stringToUTF8,lengthBytesUTF8,HEAPU8,HEAP32,HEAPU32,HEAPF32

wasm-test:
	node tests/wasm_moons.mjs

wasm-bench:
	mkdir -p $(BUILD)
	$(EMCXX) -std=c++20 -O2 -fno-exceptions -Iinclude bench/bench.cpp $(SRC) -o $(BUILD)/bench.js \
	  -sENVIRONMENT=node -sINITIAL_MEMORY=134217728
	node $(BUILD)/bench.js

clean:
	rm -rf $(BUILD)
