.RECIPEPREFIX := >

CXX_RV_BARE = riscv64-unknown-elf-g++
CXX_RV_LINUX = riscv64-linux-gnu-g++

SIZE_RV_LINUX = riscv64-linux-gnu-size
OBJDUMP_RV_LINUX = riscv64-linux-gnu-objdump

QEMU_BARE = qemu-riscv64
QEMU_LINUX = qemu-riscv64 -L /usr/riscv64-linux-gnu

BUILD_DIR = build

BASE_FLAGS_BARE = -std=c++17 -Wall -Wextra -Iinclude -march=rv64gcv -mabi=lp64d
BASE_FLAGS_LINUX = -std=c++17 -Wall -Wextra -Iinclude -march=rv64gcv -mabi=lp64d

.PHONY: phase4 run-phase4 size-phase4 vec-report phase6_5 run-phase6_5 clean

$(BUILD_DIR):
> mkdir -p $(BUILD_DIR)

# =============================================================================
# Phase 4 — Compiler optimization sweep
#
# Uses riscv64-linux-gnu-g++ because Phase 4 now uses std::chrono
# for wall-clock time in milliseconds.
# =============================================================================

phase4: $(BUILD_DIR)
> $(CXX_RV_LINUX) $(BASE_FLAGS_LINUX) -O0 tests/phase4_benchmark.cpp -o $(BUILD_DIR)/phase4_O0.elf -lm
> $(CXX_RV_LINUX) $(BASE_FLAGS_LINUX) -O2 tests/phase4_benchmark.cpp -o $(BUILD_DIR)/phase4_O2.elf -lm
> $(CXX_RV_LINUX) $(BASE_FLAGS_LINUX) -O3 tests/phase4_benchmark.cpp -o $(BUILD_DIR)/phase4_O3.elf -lm
> $(CXX_RV_LINUX) $(BASE_FLAGS_LINUX) -Os tests/phase4_benchmark.cpp -o $(BUILD_DIR)/phase4_Os.elf -lm
> $(CXX_RV_LINUX) $(BASE_FLAGS_LINUX) -Ofast tests/phase4_benchmark.cpp -o $(BUILD_DIR)/phase4_Ofast.elf -lm

run-phase4: phase4
> echo "===== O0 ====="
> $(QEMU_LINUX) -cpu rv64,v=true,vlen=128 ./$(BUILD_DIR)/phase4_O0.elf
> echo "===== O2 ====="
> $(QEMU_LINUX) -cpu rv64,v=true,vlen=128 ./$(BUILD_DIR)/phase4_O2.elf
> echo "===== O3 ====="
> $(QEMU_LINUX) -cpu rv64,v=true,vlen=128 ./$(BUILD_DIR)/phase4_O3.elf
> echo "===== Os ====="
> $(QEMU_LINUX) -cpu rv64,v=true,vlen=128 ./$(BUILD_DIR)/phase4_Os.elf
> echo "===== Ofast ====="
> $(QEMU_LINUX) -cpu rv64,v=true,vlen=128 ./$(BUILD_DIR)/phase4_Ofast.elf

size-phase4: phase4
> $(SIZE_RV_LINUX) $(BUILD_DIR)/phase4_O0.elf
> $(SIZE_RV_LINUX) $(BUILD_DIR)/phase4_O2.elf
> $(SIZE_RV_LINUX) $(BUILD_DIR)/phase4_O3.elf
> $(SIZE_RV_LINUX) $(BUILD_DIR)/phase4_Os.elf
> $(SIZE_RV_LINUX) $(BUILD_DIR)/phase4_Ofast.elf

vec-report: $(BUILD_DIR)
> $(CXX_RV_LINUX) $(BASE_FLAGS_LINUX) -O3 -ftree-vectorize -fopt-info-vec-all tests/phase4_benchmark.cpp -o $(BUILD_DIR)/phase4_O3_vec.elf -lm 2> $(BUILD_DIR)/vectorization_report.txt
> $(OBJDUMP_RV_LINUX) -d $(BUILD_DIR)/phase4_O3_vec.elf | grep -c vset > $(BUILD_DIR)/vset_count.txt
> cat $(BUILD_DIR)/vectorization_report.txt
> echo "vset count:"
> cat $(BUILD_DIR)/vset_count.txt

# =============================================================================
# Phase 6.5 — RVV L1 magnitude
#
# Kept with riscv64-unknown-elf-g++ because this phase uses RVV intrinsics
# and the previous rdcycle-based test.
# =============================================================================

phase6_5: $(BUILD_DIR)
> $(CXX_RV_BARE) $(BASE_FLAGS_BARE) -O3 tests/phase6_5_test.cpp -o $(BUILD_DIR)/phase6_5_rvv.elf -lm

run-phase6_5: phase6_5
> echo "===== Phase 6.5 RVV L1 Magnitude | VLEN=128 ====="
> $(QEMU_BARE) -cpu rv64,v=true,vlen=128 ./$(BUILD_DIR)/phase6_5_rvv.elf
> echo "===== Phase 6.5 RVV L1 Magnitude | VLEN=256 ====="
> $(QEMU_BARE) -cpu rv64,v=true,vlen=256 ./$(BUILD_DIR)/phase6_5_rvv.elf
> echo "===== Phase 6.5 RVV L1 Magnitude | VLEN=512 ====="
> $(QEMU_BARE) -cpu rv64,v=true,vlen=512 ./$(BUILD_DIR)/phase6_5_rvv.elf

clean:
> rm -rf $(BUILD_DIR)
