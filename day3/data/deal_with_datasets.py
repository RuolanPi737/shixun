import os
import shutil
from sklearn.model_selection import train_test_split
import random
#form prl file
# 设置随机种子以确保可重复性sss
random.seed(42)

# 数据集路径ssss
dataset_dir = r'D:\shixun-main\day3\data\Images'  # 替换为你的数据集路径
train_dir = r'D:\shixun-main\day3\data\images\train'  # 训练集输出路径
val_dir = r'D:\shixun-main\day3\data\images\val'  # 验证集输出路径

# 划分比例......
train_ratio = 0.7

# 创建训练集和验证集目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 遍历每个类别文件夹aaaa
for class_name in os.listdir(dataset_dir):
    if class_name not in ["train", "val"] and os.path.isdir(os.path.join(dataset_dir, class_name)):
        class_path = os.path.join(dataset_dir, class_name)

        # 获取该类别下的所有图片aaaa
        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

        # 确保图片路径包含类别文件夹
        images = [os.path.join(class_name, img) for img in images]

        # 划分训练集和验证集sssss
        train_images, val_images = train_test_split(images, train_size=train_ratio, random_state=42)

        # 创建类别子文件夹sss
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        # 移动训练集图片ssss
        for img in train_images:
            src = os.path.join(dataset_dir, img)
            dst = os.path.join(train_dir, img)
            shutil.move(src, dst)

        # 移动验证集图片
        for img in val_images:
            src = os.path.join(dataset_dir, img)
            dst = os.path.join(val_dir, img)
            shutil.move(src, dst)

        # 删除原始类别文件夹（现在应该为空）ssss
        shutil.rmtree(class_path)