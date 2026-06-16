#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include "canny.h"

// 1. اختبار الصورة الموحدة الإضاءة
TEST(GaussianBlurTest, UniformImageInvariant) {
    const int width = 16;
    const int height = 16;
    std::vector<uint8_t> input(width * height, 128);
    std::vector<uint8_t> output(width * height, 0);

    gaussian_blur_separable<uint8_t, int32_t, int16_t>(input.data(), output.data(), width, height);

    for (int y = 2; y < height - 2; ++y) {
        for (int x = 2; x < width - 2; ++x) {
            EXPECT_NEAR(output[y * width + x], 128, 1);
        }
    }
}

// 2. اختبار الصورة السوداء بالكامل
TEST(GaussianBlurTest, AllBlackImageProducesBlack) {
    const int width = 16;
    const int height = 16;
    std::vector<uint8_t> input(width * height, 0);
    std::vector<uint8_t> output(width * height, 255);

    gaussian_blur_separable<uint8_t, int32_t, int16_t>(input.data(), output.data(), width, height);

    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(output[i], 0);
    }
}

// 3. اختبار النبضة وتماثل انتشار التمويه
TEST(GaussianBlurTest, ImpulseSymmetry) {
    const int width = 15;
    const int height = 15;
    const int cx = 7;
    const int cy = 7;

    std::vector<uint8_t> input(width * height, 0);
    input[cy * width + cx] = 255;

    std::vector<uint8_t> output(width * height, 0);
    gaussian_blur_separable<uint8_t, int32_t, int16_t>(input.data(), output.data(), width, height);

    for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -3; dx <= 3; ++dx) {
            uint8_t val1 = output[(cy + dy) * width + (cx + dx)];
            uint8_t val_mirror_x = output[(cy + dy) * width + (cx - dx)];
            uint8_t val_mirror_y = output[(cy - dy) * width + (cx + dx)];
            uint8_t val_diagonal = output[(cy + dx) * width + (cx + dy)];

            EXPECT_EQ(val1, val_mirror_x);
            EXPECT_EQ(val1, val_mirror_y);
            EXPECT_EQ(val1, val_diagonal);
        }
    }
}
