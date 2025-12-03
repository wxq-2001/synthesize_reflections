import cv2
import numpy as np
import random


def random_crop(image, crop_width, crop_height):
    original_height, original_width = image.shape[:2]
    if crop_width > original_width or crop_height > original_height:
        raise ValueError("Crop size must be smaller than the original image size.")
    left = random.randint(0, original_width - crop_width)
    top = random.randint(0, original_height - crop_height)
    right = left + crop_width
    bottom = top + crop_height
    cropped_image = image[top:bottom, left:right]
    return cropped_image

def normalize_image(image):
    return image.astype(np.float32) / 255.0

def apply_gaussian_blur(image, kernel_size, sigma):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def compute_mean_above_one(image):
    masked_image = np.where(image > 1, image, 0)
    count = np.sum(image > 1)
    if count == 0:
        return 0
    return np.sum(masked_image) / count

def create_image_A(input_image, ratio):
    height, width, channels = input_image.shape
    start_col = int(width * (1 - ratio))
    A = input_image.copy()
    A[:, start_col:] = 0
    return A

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

    # 随机上下、左右翻转和裁剪
    if random.random() < 0.5:
        if random.random() < 0.5:
            R = cv2.flip(R, 0)  # 上下翻转
        if random.random() < 0.5:
            R = cv2.flip(R, 1)  # 左右翻转

        # 裁剪
        min_dim = min(R.shape[:2])
        R = random_crop(R, min_dim, min_dim)

    R = cv2.resize(R, (512, 512))
    cv2.imwrite('D:\\real_sys_data\\result\R15.png', R)

    rr = create_image_B(R, shift_ratio)
    cv2.imwrite('D:\\real_sys_data\\result\\R_shift15.png', rr)
    normalized_lr = normalize_image(R)
    normalized_rr = normalize_image(rr)
    normalized_left = normalize_image(left)
    normalized_right = normalize_image(right)

    attenuation_factor = 0.8  # 控制亮度，越小反光越弱
    mask = create_blend_mask(normalized_lr.shape, strength=1.5)
    alpha = np.stack([mask] * 3, axis=2)  # 扩展为三通道 alpha 混合

    I = normalized_lr * normalized_left + normalized_B
    mean_above_one_left = compute_mean_above_one(I)
    I = normalized_rr * normalized_right + normalized_B
    mean_above_one_right = compute_mean_above_one(I)

    R2_l = normalized_lr - attenuation_factor * (mean_above_one_left - 1)
    R2_r = normalized_rr - attenuation_factor * (mean_above_one_right - 1)

    R3_l = np.clip(R2_l * alpha, 0, 1)
    R3_r = np.clip(R2_r * alpha, 0, 1)

    # R2_l = normalized_lr - attenuation_factor * (mean_above_one_left - 1)
    # R2_l = np.maximum(normalized_lr - attenuation_factor * (mean_above_one_left - 1), 0)
    # R3_l = np.clip(R2_l, 0, 1)
    # R2_r = normalized_rr - attenuation_factor * (mean_above_one_right - 1)
    # R3_r = np.clip(R2_r, 0, 1)

    result = normalized_left * R3_l + normalized_right * R3_r + normalized_B
    result = np.clip(result, 0, 1)

    # result = normalized_left * R3_l + normalized_right * R3_r + normalized_B
    # result = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)

# 读取图片
B = cv2.imread('D:\\real_sys_data\\real_face\\15.jpg')
R = cv2.imread('D:\\real_sys_data\\R\\16.jpg')
left = cv2.imread('D:\\real_sys_data\\left\\15.png')
right = cv2.imread('D:\\real_sys_data\\right\\15.png')

# Resize 所有图像为统一大小
target_size = (512, 512)
B = cv2.resize(B, target_size)
left = cv2.resize(left, target_size)
right = cv2.resize(right, target_size)

# 指定右移比例，例如1/2、1/3、1/4
shift_ratio = 0  # 这里可以更改为你想要的比例


# 合成图片
result = combine_images(B, R, left, right, shift_ratio)
cv2.imwrite('D:\\real_sys_data\\result\\result15.png', result)

# 显示结果
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()



