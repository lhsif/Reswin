import nibabel as nib
import numpy as np
from scipy import ndimage


def read_nifti_file(filepath):
    """读取和加载数据"""
    # 读取文件
    scan = nib.load(filepath)
    # 获取原始数据
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """数据归一化"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """跨 z 轴调整大小"""
    # 设置所需的深度
    desired_depth = 32
    desired_width = 224
    desired_height = 224
    # 获取当前深度
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # 计算深度因子
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # 旋转
    img = ndimage.rotate(img, 90, reshape=False)
    # 跨z轴调整大小
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)

    # 交换轴以获得 (depth, height, width) 格式
    img = np.transpose(img, (2, 0, 1))

    return img


def process_scan(path):
    """读取和调整数据大小"""
    # 读取扫描文件
    volume = read_nifti_file(path)
    # 归一化
    volume = normalize(volume)
    # 调整宽度、高度和深度
    volume = resize_volume(volume)
    return volume
