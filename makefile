.RECIPEPREFIX := >

CXX_RV = riscv64-unknown-elf-g++
QEMU = qemu-riscv64

BUILD_DIR = build
BASE_FLAGS = -std=c++17 -Wall -Wextra -Iinclude -march=rv64gcv -mabi=lp64d

.PHONY: phase4 run-phase4 size-phase4 vec-report clean

$(BUILD_DIR):
> mkdir -p $(BUILD_DIR)

phase4: $(BUILD_DIR)
> $(CXX_RV) $(BASE_FLAGS) -O0 tests/phase4_benchmark.cpp -o $(BUILD_DIR)/phase4_O0.elf -lm
> $(CXX_RV) $(BASE_FLAGS) -O2 tests/phase4_benchmark.cpp -o $(BUILD_DIR)/phase4_O2.elf -lm
> $(CXX_RV) $(BASE_FLAGS) -O3 tests/phase4_benchmark.cpp -o $(BUILD_DIR)/phase4_O3.elf -lm
> $(CXX_RV) $(BASE_FLAGS) -Os tests/phase4_benchmark.cpp -o $(BUILD_DIR)/phase4_Os.elf -lm
> $(CXX_RV) $(BASE_FLAGS) -Ofast tests/phase4_benchmark.cpp -o $(BUILD_DIR)/phase4_Ofast.elf -lm

run-phase4: phase4
> echo "===== O0 ====="
> $(QEMU) -cpu rv64,v=true,vlen=128 ./$(BUILD_DIR)/phase4_O0.elf
> echo "===== O2 ====="
> $(QEMU) -cpu rv64,v=true,vlen=128 ./$(BUILD_DIR)/phase4_O2.elf
> echo "===== O3 ====="
> $(QEMU) -cpu rv64,v=true,vlen=128 ./$(BUILD_DIR)/phase4_O3.elf
> echo "===== Os ====="
> $(QEMU) -cpu rv64,v=true,vlen=128 ./$(BUILD_DIR)/phase4_Os.elf
> echo "===== Ofast ====="
> $(QEMU) -cpu rv64,v=true,vlen=128 ./$(BUILD_DIR)/phase4_Ofast.elf

size-phase4: phase4
> riscv64-unknown-elf-size $(BUILD_DIR)/phase4_O0.elf
> riscv64-unknown-elf-size $(BUILD_DIR)/phase4_O2.elf
> riscv64-unknown-elf-size $(BUILD_DIR)/phase4_O3.elf
> riscv64-unknown-elf-size $(BUILD_DIR)/phase4_Os.elf
> riscv64-unknown-elf-size $(BUILD_DIR)/phase4_Ofast.elf

vec-report: $(BUILD_DIR)
> $(CXX_RV) $(BASE_FLAGS) -O3 -ftree-vectorize -fopt-info-vec-all tests/phase4_benchmark.cpp -o $(BUILD_DIR)/phase4_O3_vec.elf -lm 2> $(BUILD_DIR)/vectorization_report.txt
> riscv64-unknown-elf-objdump -d $(BUILD_DIR)/phase4_O3_vec.elf | grep -c vset > $(BUILD_DIR)/vset_count.txt
> cat $(BUILD_DIR)/vectorization_report.txt
> echo "vset count:"
> cat $(BUILD_DIR)/vset_count.txt

clean:
> rm -rf $(BUILD_DIR)
