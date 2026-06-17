import numpy as np
import matplotlib.pyplot as plt

def save_comparison(file1, file2, width, height, output_name='data/comparison.png'):
    # قراءة البيانات
    img1 = np.fromfile(file1, dtype=np.uint8).reshape((height, width))
    img2 = np.fromfile(file2, dtype=np.uint8).reshape((height, width))

    # إعداد النافذة
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    ax[0].imshow(img1, cmap='gray')
    ax[0].set_title('Processed')
    ax[0].axis('off')

    ax[1].imshow(img2, cmap='gray')
    ax[1].set_title(' Original')
    ax[1].axis('off')

    # حفظ الصورة بدلاً من عرضها
    plt.savefig(output_name)
    print(f"تم حفظ المقارنة بنجاح في ملف: {output_name}")

# تشغيل
save_comparison('output_separable_blur.raw', 'output_2d_blur.raw', 1216, 704)
