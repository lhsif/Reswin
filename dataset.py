import torch
from torch.utils.data import Dataset
from scipy import ndimage, interpolate
import numpy as np
import os
import random
import shutil
import re
import matplotlib.pyplot as plt
from config import config
from preprocessing import process_scan


def rotate(volume, max_angle=20):
    angle = random.uniform(-max_angle, max_angle)
    rotated_volume = np.zeros_like(volume)

    for i in range(volume.shape[0]):
        rotated_volume[i] = ndimage.rotate(volume[i], angle, reshape=False)

    return np.clip(rotated_volume, 0, 1)


def translate(volume, max_shift=0.2):
    depth, height, width = volume.shape
    shift_y = int(height * random.uniform(-max_shift, max_shift))
    shift_x = int(width * random.uniform(-max_shift, max_shift))

    translated_volume = np.zeros_like(volume)
    for i in range(depth):
        translated_volume[i] = np.roll(volume[i], (shift_y, shift_x), axis=(0, 1))

    return translated_volume


def add_noise(volume, noise_type='gaussian', mean=0, var=0.005):
    if noise_type == 'gaussian':
        noise = np.random.normal(mean, np.sqrt(var), volume.shape)
    elif noise_type == 'salt_pepper':
        noise = np.random.choice([0, 1], size=volume.shape, p=[1 - var, var])
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    return np.clip(volume + noise, 0, 1)


def elastic_deformation(volume, alpha=1000, sigma=30):
    # Generate the coordinate grid
    shape = volume.shape
    y, x, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')

    # Generate Gaussian-smoothed random fields
    dy = np.random.randn(*shape) * alpha
    dx = np.random.randn(*shape) * alpha
    dz = np.zeros_like(x)  # No deformation along the z-axis

    # Smooth the displacement fields
    dy = ndimage.gaussian_filter(dy, sigma)
    dx = ndimage.gaussian_filter(dx, sigma)
    dz = ndimage.gaussian_filter(dz, sigma)

    # Apply the displacement fields
    indices = (
        np.reshape(y + dy, (-1, 1)),
        np.reshape(x + dx, (-1, 1)),
        np.reshape(z + dz, (-1, 1)),
    )
    deformed_volume = ndimage.map_coordinates(volume, indices, order=1).reshape(shape)

    return deformed_volume


augmentation_list = [
    rotate,
    translate,
    add_noise,
    elastic_deformation
]


def apply_augmentations(image, num_augmentations):
    selected_augmentations = random.sample(augmentation_list, num_augmentations)

    for augmentation in selected_augmentations:
        image = augmentation(image)

    return image


def split_dataset(folders, data_root, train_ratio=0.8, seed=None):
    if seed is not None:
        random.seed(seed)

    train_folders = []
    val_folders = []

    for folder in folders:
        folder_path = os.path.join(data_root, folder)

        train_folder = os.path.join(folder_path, 'train')
        val_folder = os.path.join(folder_path, 'val')

        if os.path.exists(train_folder):
            shutil.rmtree(train_folder)

        if os.path.exists(val_folder):
            shutil.rmtree(val_folder)

        os.makedirs(train_folder)
        os.makedirs(val_folder)

        files = [f for f in os.listdir(folder_path) if f.endswith('.nii.gz')]

        random.shuffle(files)

        num_train = int(len(files) * train_ratio)
        train_files = files[:num_train]
        val_files = files[num_train:]

        for train_file in train_files:
            src = os.path.join(folder_path, train_file)
            dst = os.path.join(train_folder, train_file)
            shutil.copy(src, dst)

        for val_file in val_files:
            src = os.path.join(folder_path, val_file)
            dst = os.path.join(val_folder, val_file)
            shutil.copy(src, dst)

        train_folders.append(train_folder)
        val_folders.append(val_folder)

    return train_folders, val_folders


class NiiDataset(Dataset):
    def __init__(self, folders, task="binary", transform=None, torchio_transform=None, split='train'):
        self.transform = transform
        self.torchio_transform = torchio_transform
        self.files = []

        for folder in folders:
            # Get the base folder name before adding the split subfolder
            base_folder_name = os.path.basename(os.path.dirname(folder))

            if task == "binary":
                if "CT-0" in base_folder_name:
                    label = 0
                elif "CT-1" in base_folder_name:
                    label = 1
            elif task == "multiclass":
                match = re.search(r"CT-(\d+)", base_folder_name)
                if match:
                    label = int(match.group(1))
                else:
                    raise ValueError(f"Invalid folder name format: {base_folder_name}")

            for filename in os.listdir(folder):
                if filename.endswith(".nii.gz"):
                    file_path = os.path.join(folder, filename)
                    self.files.append((file_path, label))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx, apply_torchio_transform=True):
        file_path, label = self.files[idx]

        img = process_scan(file_path)

        if apply_torchio_transform and self.torchio_transform:
            img = apply_augmentations(img, 2)

        img = img[np.newaxis, :, :, :].astype(np.float32)
        img = np.repeat(img, 3, axis=0)

        img_tensor = torch.from_numpy(img)
        return img_tensor, label

    def get_all_labels(self):
        return [label for _, label in self.files]


def visualize_augmentation(dataset, idx, seed=None):
    if seed is not None:
        random.seed(seed)

    original_image, _ = dataset.__getitem__(idx, apply_torchio_transform=False)
    augmented_image, _ = dataset.__getitem__(idx, apply_torchio_transform=True)

    middle_slice_index = original_image.shape[1] // 2

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    original_image_single_channel = original_image[0, middle_slice_index, :, :]
    augmented_image_single_channel = augmented_image[0, middle_slice_index, :, :]

    axes[0].imshow(original_image_single_channel, cmap='gray')
    axes[0].set_title(f"Original Image (slice {middle_slice_index})")
    axes[1].imshow(augmented_image_single_channel, cmap='gray')
    axes[1].set_title(f"Augmented Image (slice {middle_slice_index})")

    plt.tight_layout()
    plt.show()

# # 根据任务选择数据集
# folders = ["CT-0", "CT-1"]
# train_folders, val_folders = split_dataset(folders, train_ratio=config["train_ratio"], seed=None,
#                                            data_root="./data")
# # torchio_transform = get_torchio_transforms()
# train_dataset = NiiDataset(train_folders, transform=None, torchio_transform=True, split='train',
#                            task=config["task"])
# val_dataset = NiiDataset(val_folders, transform=None, torchio_transform=False, split='val',
#                          task=config["task"])
#
# # 随机选择一个图像索引
# random_idx = np.random.randint(len(train_dataset))
#
# # 可视化原始和增强图像
# visualize_augmentation(train_dataset, random_idx)
