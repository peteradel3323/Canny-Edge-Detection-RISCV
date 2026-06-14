#include "canny.h"

void grayscale(unsigned char* input, unsigned char* output, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        output[i] = input[i]; 
    }
}