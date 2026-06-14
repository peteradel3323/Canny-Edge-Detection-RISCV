CC = riscv64-unknown-elf-g++
CFLAGS = -O3 -Wall
QEMU = qemu-riscv64
TARGET = canny_app
SRC = src/main.cpp src/canny.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	$(QEMU) $(TARGET)

clean:
	rm -f $(TARGET)