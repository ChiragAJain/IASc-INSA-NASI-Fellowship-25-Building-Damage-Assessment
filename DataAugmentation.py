import os
import cv2
import numpy as np
from tqdm import tqdm
from albumentations import (HorizontalFlip, VerticalFlip, RandomRotate90, Rotate,RandomBrightnessContrast, GaussianBlur, RandomCrop,Resize, Compose)
from albumentations.pytorch import ToTensorV2


PATCH_DIR = "postdisaster-patches"
TARGET_CLASSES = ['minor-damage', 'major-damage', 'destroyed']
IMAGE_SIZE = 128
AUG_PER_IMAGE = 5  

transform = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomRotate90(p=0.5),
    Rotate(limit=15, p=0.5),
    RandomBrightnessContrast(p=0.3),
    GaussianBlur(p=0.2),
    RandomCrop(height=IMAGE_SIZE, width=IMAGE_SIZE, p=0.3),
    Resize(IMAGE_SIZE, IMAGE_SIZE),
])

def augment_image(img_path, save_dir, base_name, aug_count=5):
    image = cv2.imread(img_path)
    if image is None:
        print(f"Could not read image: {img_path}")
        return 0

    h, w, _ = image.shape
    if h < 128 or w < 128:
        dynamic_transform = Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomRotate90(p=0.5),
            Rotate(limit=15, p=0.5),
            RandomBrightnessContrast(p=0.3),
            GaussianBlur(p=0.2),
            Resize(128, 128),
        ])
    else:
        dynamic_transform = transform 
    count = 0
    for i in range(aug_count):
        augmented = dynamic_transform(image=image)["image"]
        aug_name = f"{base_name}_aug{i:02d}.png"
        cv2.imwrite(os.path.join(save_dir, aug_name), augmented)
        count += 1
    return count
def main():
    total_aug = 0
    for cls in TARGET_CLASSES:
        class_dir = os.path.join(PATCH_DIR, cls)
        img_files = [f for f in os.listdir(class_dir) if f.endswith(".png")]
        print(f"\nAugmenting {len(img_files)} images from class: {cls}")

        for img_file in tqdm(img_files, desc=f"Processing {cls}"):
            img_path = os.path.join(class_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            total_aug += augment_image(img_path, class_dir, base_name, AUG_PER_IMAGE)

    print(f"\nTotal augmented images created: {total_aug}")

if __name__ == "__main__":
    main()
