#include "canny.h"
#include <fstream>
#include <iostream>
#include <vector> 

// ==========================================
// 1. تعريف الثوابت والمكروهات وتكوين سلاسل التوسيع (Phase 6.3 Widening)
// ==========================================

#include <riscv_vector.h>

#if GAUSS_LMUL == 4
    // التوسيع لـ LMUL=4 يتوقف عند m8 (8-بت m4 إلى 16-بت m8) لمنع تجاوز الحد الأقصى للمسجلات العتادية
    typedef vuint8m4_t  vuint8_t;
    typedef vuint16m8_t vint16_t; 
    typedef vuint16m8_t vint32_t; // محاكاة الـ 32 بت باستخدام 16 بت بدون إشارة لتفادي m16 مفقودة
    typedef vbool2_t    vbool_t;

    #define rvv_vsetvl_e8 __riscv_vsetvl_e8m4
    #define rvv_vle8 __riscv_vle8_v_u8m4
    // FIX: correct unsigned widening-convert intrinsic name (was __riscv_vwcvtu_x_v_u16m8)
    // No reinterpret needed here: both input and output are unsigned.
    #define rvv_vwcvt_16(v, vl) __riscv_vwcvtu_x_x_v_u16m8(v, vl)
    
    #define rvv_vle16 __riscv_vle16_v_u16m8
    #define rvv_vse16 __riscv_vse16_v_u16m8
    #define rvv_vadd __riscv_vadd_vv_u16m8
    #define rvv_vmul __riscv_vmul_vx_u16m8
    
    #define rvv_vwmul(v, x, vl) __riscv_vmul_vx_u16m8(v, x, vl)
    #define rvv_vwmacc(acc, x, v, vl) __riscv_vadd_vv_u16m8(acc, __riscv_vmul_vx_u16m8(v, x, vl), vl)
    #define rvv_vsra __riscv_vsrl_vx_u16m8
    
    #define rvv_vncvt_8 __riscv_vncvt_x_x_w_u8m4  
    #define rvv_vse8 __riscv_vse8_v_u8m4
    
    typedef uint16_t TempPixelT;
#elif GAUSS_LMUL == 2
    typedef vuint8m2_t  vuint8_t;
    typedef vint16m4_t  vint16_t; 
    typedef vint32m8_t  vint32_t; 
    typedef vbool4_t    vbool_t;  

    #define rvv_vsetvl_e8 __riscv_vsetvl_e8m2
    #define rvv_vle8 __riscv_vle8_v_u8m2
    // FIX: __riscv_vwcvt_x_x_v_i16m4 requires a SIGNED vint8m2_t input,
    // but our loaded data is vuint8m2_t (unsigned). Reinterpret the bit
    // pattern as signed before widening (values 0-255 stay correct).
    #define rvv_vwcvt_16(v, vl) __riscv_vwcvt_x_x_v_i16m4(__riscv_vreinterpret_v_u8m2_i8m2(v), vl)
    
    #define rvv_vle16 __riscv_vle16_v_i16m4
    #define rvv_vse16 __riscv_vse16_v_i16m4
    #define rvv_vadd __riscv_vadd_vv_i16m4
    #define rvv_vmul __riscv_vmul_vx_i16m4
    
    #define rvv_vwmul __riscv_vwmul_vx_i32m8
    #define rvv_vwmacc __riscv_vwmacc_vx_i32m8
    #define rvv_vsra __riscv_vsra_vx_i32m8
    #define rvv_vmax __riscv_vmax_vx_i32m8
    #define rvv_vmin __riscv_vmin_vx_i32m8
    
    #define rvv_vncvt_16 __riscv_vncvt_x_x_w_i16m4
    #define rvv_vncvt_8 __riscv_vncvt_x_x_w_i8m2
    #define rvv_vreinterpret_u8(v) __riscv_vreinterpret_v_i8m2_u8m2(v)
    #define rvv_vse8 __riscv_vse8_v_u8m2

    typedef int16_t TempPixelT;
#else // الافتراضي GAUSS_LMUL == 1
    typedef vuint8m1_t  vuint8_t;
    typedef vint16m2_t  vint16_t;
    typedef vint32m4_t  vint32_t;
    typedef vbool8_t    vbool_t;  

    #define rvv_vsetvl_e8 __riscv_vsetvl_e8m1
    #define rvv_vle8 __riscv_vle8_v_u8m1
    // FIX: __riscv_vwcvt_x_x_v_i16m2 requires a SIGNED vint8m1_t input,
    // but our loaded data is vuint8m1_t (unsigned). Reinterpret the bit
    // pattern as signed before widening (values 0-255 stay correct).
    #define rvv_vwcvt_16(v, vl) __riscv_vwcvt_x_x_v_i16m2(__riscv_vreinterpret_v_u8m1_i8m1(v), vl)
    
    #define rvv_vle16 __riscv_vle16_v_i16m2
    #define rvv_vse16 __riscv_vse16_v_i16m2
    #define rvv_vadd __riscv_vadd_vv_i16m2
    #define rvv_vmul __riscv_vmul_vx_i16m2
    
    #define rvv_vwmul __riscv_vwmul_vx_i32m4
    #define rvv_vwmacc __riscv_vwmacc_vx_i32m4
    #define rvv_vsra __riscv_vsra_vx_i32m4
    #define rvv_vmax __riscv_vmax_vx_i32m4
    #define rvv_vmin __riscv_vmin_vx_i32m4
    
    #define rvv_vncvt_16 __riscv_vncvt_x_x_w_i16m2
    #define rvv_vncvt_8 __riscv_vncvt_x_x_w_i8m1
    #define rvv_vreinterpret_u8(v) __riscv_vreinterpret_v_i8m1_u8m1(v)
    #define rvv_vse8 __riscv_vse8_v_u8m1

    typedef int16_t TempPixelT;
#endif

const int16_t GAUSSIAN_KERNEL_5X5[5][5] = {
    {1,  4,  7,  4, 1},
    {4, 16, 26, 16, 4},
    {7, 26, 41, 26, 7},
    {4, 16, 26, 16, 4},
    {1,  4,  7,  4, 1}
};

const int32_t GAUSSIAN_SUM = 273; 
const int16_t GAUSSIAN_KERNEL_1D[5] = {1, 4, 6, 4, 1};
const int32_t GAUSSIAN_SUM_1D = 16; 

// ==========================================
// 2. دوال المعالجة المساعدة
// ==========================================
void grayscale(const unsigned char* input, unsigned char* output, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        output[i] = input[i]; 
    }
}

bool read_raw_image(const char* filename, int width, int height, unsigned char*& data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }
    data = new unsigned char[width * height];
    file.read(reinterpret_cast<char*>(data), width * height);
    file.close();
    return true;
}

bool write_raw_image(const char* filename, int width, int height, const unsigned char* data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing " << filename << std::endl;
        return false;
    }
    file.write(reinterpret_cast<const char*>(data), width * height);
    file.close();
    return true;
}

// ==========================================
// 3. دالة التمويه الغاوسي بـ الحسابات الموسعة المباشرة (Phase 6.3)
// ==========================================
template <class PixelT, class AccumulatorT, class KernelT>
void gaussian_blur_separable(const PixelT* __restrict input, PixelT* __restrict output, size_t width, size_t height) {
    
    size_t padded_width = width + 4;
    size_t padded_height = height + 4;
    
    // الميزة هنا: حجز مصفوفة الإدخال كـ 8-بت ومصفوفة المنتصف كـ 16-بت فقط لتوفير الكاش والذاكرة
    uint8_t* padded_input   = new uint8_t[padded_width * padded_height]();
    TempPixelT* temp_padded = new TempPixelT[padded_width * padded_height]();

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            padded_input[(y + 2) * padded_width + (x + 2)] = static_cast<uint8_t>(input[y * width + x]);
        }
    }

    // ----------------------------------------------------------------------
    // التمريرة الأفقية (Horizontal Pass) -> تفريد تحميل 8-بت وتوسيع ديناميكي لـ 16-بت
    // ----------------------------------------------------------------------
    for (size_t y = 2; y < height + 2; ++y) {
        size_t x = 2;
        size_t num_cols = width;
        
        while (num_cols > 0) {
            size_t vl = rvv_vsetvl_e8(num_cols);
            
            // 1. تحميل عتادي سريع ومباشر لبكسلات 8-بت غير موسعة بذاكرة الرام
            vuint8_t l0_8 = rvv_vle8(padded_input + y * padded_width + x - 2, vl);
            vuint8_t l1_8 = rvv_vle8(padded_input + y * padded_width + x - 1, vl);
            vuint8_t l2_8 = rvv_vle8(padded_input + y * padded_width + x    , vl);
            vuint8_t l3_8 = rvv_vle8(padded_input + y * padded_width + x + 1, vl);
            vuint8_t l4_8 = rvv_vle8(padded_input + y * padded_width + x + 2, vl);
            
            // 2. التوسيع الفوري على مستوى المسجلات إلى 16-بت (Widening)
            vint16_t l0 = rvv_vwcvt_16(l0_8, vl);
            vint16_t l1 = rvv_vwcvt_16(l1_8, vl);
            vint16_t l2 = rvv_vwcvt_16(l2_8, vl);
            vint16_t l3 = rvv_vwcvt_16(l3_8, vl);
            vint16_t l4 = rvv_vwcvt_16(l4_8, vl);
            
            // 3. الحسابات الرياضية الآمنة لفلتر الأبعاد الاحادية
            vint16_t v_sum = l0; 
            v_sum = rvv_vadd(v_sum, rvv_vmul(l1, 4, vl), vl);
            v_sum = rvv_vadd(v_sum, rvv_vmul(l2, 6, vl), vl);
            v_sum = rvv_vadd(v_sum, rvv_vmul(l3, 4, vl), vl);
            v_sum = rvv_vadd(v_sum, l4, vl); 
            
            // تخزين الخرج المؤقت (مصفوفة 16-بت مدمجة)
            rvv_vse16(temp_padded + y * padded_width + x, v_sum, vl);
            
            x += vl;
            num_cols -= vl;
        }
    }

    // ----------------------------------------------------------------------
    // التمريرة الرأسية (Vertical Pass) -> التوسيع والضرب التراكمي المباشر إلى 32-بت
    // ----------------------------------------------------------------------
    for (size_t y = 2; y < height + 2; ++y) {
        size_t x = 2;
        size_t num_cols = width;
        
        while (num_cols > 0) {
            size_t vl = rvv_vsetvl_e8(num_cols); // استخدام نفس تماثل الطول لضمان توافق الحلقات
            
            // 1. تحميل قيم الـ 16 بت المتوسطة
            vint16_t v0 = rvv_vle16(temp_padded + (y - 2) * padded_width + x, vl);
            vint16_t v1 = rvv_vle16(temp_padded + (y - 1) * padded_width + x, vl);
            vint16_t v2 = rvv_vle16(temp_padded + y * padded_width + x, vl);
            vint16_t v3 = rvv_vle16(temp_padded + (y + 1) * padded_width + x, vl);
            vint16_t v4 = rvv_vle16(temp_padded + (y + 2) * padded_width + x, vl);
            
            // 2. استخدام الضرب والتراكم الموسع المباشر الموفر للمسجلات العتادية (vwmul & vwmacc)
            vint32_t v_sum = rvv_vwmul(v0, 1, vl);
            v_sum = rvv_vwmacc(v_sum, 4, v1, vl);
            v_sum = rvv_vwmacc(v_sum, 6, v2, vl);
            v_sum = rvv_vwmacc(v_sum, 4, v3, vl);
            v_sum = rvv_vwmacc(v_sum, 1, v4, vl);
            
            // 3. معالجة وتصغير حجم البيانات النهائي
            vint32_t v_div = rvv_vsra(v_sum, 8, vl); // القسمة على 256
            
            #if GAUSS_LMUL != 4
                // القص لتفادي التشويه (فقط لـ LMUL 1 و 2 الحسابية الموقعة)
                vint32_t v_clipped = rvv_vmax(v_div, 0, vl);
                v_clipped = rvv_vmin(v_clipped, 255, vl);
                
                // تقزيم متتالي من 32 بت -> 16 بت -> 8 بت غير موقعة
                vint16_t v16 = rvv_vncvt_16(v_clipped, vl);
                vuint8_t v8 = rvv_vreinterpret_u8(rvv_vncvt_8(v16, vl));
            #else
                // لـ LMUL=4 يتم التقزيم المباشر دون الحاجة لمعالجة الإشارة لأن النطاق آمن
                vuint8_t v8 = rvv_vncvt_8(v_div, vl);
            #endif
            
            // تخزين البكسلات النهائية الصافية 8-بت
            rvv_vse8(reinterpret_cast<uint8_t*>(output + (y - 2) * width + (x - 2)), v8, vl);
            
            x += vl;
            num_cols -= vl;
        }
    }

    delete[] padded_input;
    delete[] temp_padded;
}

// =================================================================
// 4. الاستنساخ الصريح لقوالب الكومبيلر لتطابق الاستدعاء الخارجي
// =================================================================
template void gaussian_blur_separable<uint8_t, int32_t, int16_t>(
    const uint8_t* __restrict input, uint8_t* __restrict output, size_t width, size_t height
);
template void gaussian_blur_separable<uint8_t, uint16_t, int16_t>(
    const uint8_t* __restrict input, uint8_t* __restrict output, size_t width, size_t height
);