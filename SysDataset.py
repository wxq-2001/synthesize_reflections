import cv2
import numpy as np
import random
import os


def random_crop(image, crop_width, crop_height):
    original_height, original_width = image.shape[:2]
    if crop_width > original_width or crop_height > original_height:
        raise ValueError("Crop size must be smaller than the original image size.")
    left = random.randint(0, original_width - crop_width)
    top = random.randint(0, original_height - crop_height)
    right = left + crop_width
    bottom = top + crop_height
    return image[top:bottom, left:right]


def normalize_image(image):
    return image.astype(np.float32) / 255.0


def compute_mean_above_one(image):
    masked_image = np.where(image > 1, image, 0)
    count = np.sum(image > 1)
    if count == 0:
        return 0
    return np.sum(masked_image) / count


def create_image_B(input_image, ratio):
    height, width, channels = input_image.shape
    shift_cols = int(width * ratio)
    B = np.zeros_like(input_image)
    B[:, shift_cols:] = input_image[:, :width - shift_cols]
    B[:, :shift_cols] = 0
    return B


def create_blend_mask(shape, strength=1.5):
    h, w = shape[:2]
    y, x = np.ogrid[:h, :w]
    center_x, center_y = w / 2, h / 2
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    mask = 1 - (distance / max_distance)
    return (mask ** strength).astype(np.float32)


def combine_images(B, R, left, right, shift_ratio):
    normalized_B = normalize_image(B)

    # 随机翻转与裁剪
    if random.random() < 0.5:
        if random.random() < 0.5:
            R = cv2.flip(R, 0)  # 上下翻转
        if random.random() < 0.5:
            R = cv2.flip(R, 1)  # 左右翻转
        min_dim = min(R.shape[:2])
        R = random_crop(R, min_dim, min_dim)

    R = cv2.resize(R, (512, 512))
    rr = create_image_B(R, shift_ratio)

    normalized_lr = normalize_image(R)
    normalized_rr = normalize_image(rr)
    normalized_left = normalize_image(left)
    normalized_right = normalize_image(right)

    attenuation_factor = 0.5
    mask = create_blend_mask(normalized_lr.shape, strength=1.5)
    alpha = np.stack([mask] * 3, axis=2)

    I = normalized_lr * normalized_left + normalized_B
    mean_above_one_left = compute_mean_above_one(I)
    I = normalized_rr * normalized_right + normalized_B
    mean_above_one_right = compute_mean_above_one(I)

    R2_l = normalized_lr - attenuation_factor * (mean_above_one_left - 1)
    R2_r = normalized_rr - attenuation_factor * (mean_above_one_right - 1)

    R3_l = np.clip(R2_l * alpha, 0, 1)
    R3_r = np.clip(R2_r * alpha, 0, 1)

    result = normalized_left * R3_l + normalized_right * R3_r + normalized_B
    result = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)


# ✅ 自动匹配 jpg/png 文件（关键）
def find_matching_file(folder, base_name):
    for ext in ['.jpg', '.png', '.jpeg']:
        candidate = os.path.join(folder, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None


# ✅ 批量处理主函数
def batch_process(base_path_B, base_path_R, base_path_left, base_path_right, save_path, shift_ratio=0):
    os.makedirs(save_path, exist_ok=True)

    file_names = sorted([f for f in os.listdir(base_path_B) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    for name in file_names:
        name_no_ext = os.path.splitext(name)[0]

        path_B = find_matching_file(base_path_B, name_no_ext)
        path_R = find_matching_file(base_path_R, name_no_ext)
        path_left = find_matching_file(base_path_left, name_no_ext)
        path_right = find_matching_file(base_path_right, name_no_ext)

        if not (path_B and path_R and path_left and path_right):
            print(f"⚠️ 跳过缺失文件：{name}")
            continue

        B = cv2.imread(path_B)
        R = cv2.imread(path_R)
        left = cv2.imread(path_left)
        right = cv2.imread(path_right)

        if B is None or R is None or left is None or right is None:
            print(f"❌ 读取失败：{name}")
            continue

        target_size = (512, 512)
        B = cv2.resize(B, target_size)
        left = cv2.resize(left, target_size)
        right = cv2.resize(right, target_size)

        result = combine_images(B, R, left, right, shift_ratio)

        save_name = os.path.join(save_path, name_no_ext + "_result.png")
        cv2.imwrite(save_name, result)
        print(f"✅ 已保存结果：{save_name}")


# ===============================
# 批量处理入口
# ===============================
if __name__ == "__main__":
    base_path_B = r"E:\synthesize_reflections\gt"
    base_path_R = r"E:\synthesize_reflections\R"
    base_path_left = r"E:\synthesize_reflections\left"
    base_path_right = r"E:\synthesize_reflections\right"
    save_path = r"E:\synthesize_reflections\input"

    # 可调整右移比例
    shift_ratio = 0

    batch_process(base_path_B, base_path_R, base_path_left, base_path_right, save_path, shift_ratio)
