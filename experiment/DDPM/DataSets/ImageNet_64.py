import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ImagenetDataset(Dataset):
    """
    来自 https://www.kaggle.com/datasets/ayaroshevskiy/downsampled-imagenet-64x64/data 数据集
    """
    def __init__(self, image_size=64):
        super().__init__()

        # 定义三个文件夹的路径 (使用 r"" 防止转义字符问题)
        self.folders = [
            r"D:\py\DATA\ImageNet-64x64\train_64x64\train_64x64",
            r"D:\py\DATA\ImageNet-64x64\valid_64x64\valid_64x64",
        ]

        self.image_paths = []

        print("🔍 Scanning image files...")
        # 遍历所有文件夹，收集图片路径
        for folder in self.folders:
            if not os.path.exists(folder):
                print(f"⚠️ Warning: Folder not found: {folder}")
                continue

            for filename in os.listdir(folder):
                # 检查常见的图片后缀
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.image_paths.append(os.path.join(folder, filename))

        print(f"✅ Found {len(self.image_paths)} images in total.")

        # 定义预处理流程
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # 确保尺寸绝对正确
            transforms.RandomHorizontalFlip(p=0.5),  # 数据增强：50%概率水平翻转
            transforms.ToTensor(),  # 转为 Tensor，范围 [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化到 [-1, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        try:
            # 打开图片并转为 RGB (防止某些图片是 RGBA 或 Grayscale 导致报错)
            img = Image.open(path).convert("RGB")

            # 应用变换
            img = self.transform(img)

            return img

        except Exception as e:
            print(f"❌ Error loading image: {path}, Error: {e}")
            # 如果这张图坏了，就递归读取下一张图，防止训练中断
            return self.__getitem__((idx + 1) % len(self))


# ================= 测试代码 =================
if __name__ == "__main__":
    # 简单的测试脚本，看看能不能读出来数据
    dataset = ImagenetDataset()

    if len(dataset) > 0:
        img = dataset[0]
        print("\n--- Sample Info ---")
        print(f"Image Shape: {img.shape}")  # 应该是 [3, 64, 64]
        print(f"Value Range: min={img.min():.2f}, max={img.max():.2f}")  # 应该是 -1 到 1 之间
        print("Type:", img.dtype)

        # 如果你想看看处理后的图片长什么样 (反归一化保存一张试试)
        from torchvision.utils import save_image

        # 还原到 [0, 1] 用于保存查看
        save_img = (img + 1) / 2
        save_image(save_img, "../test_cat_sample.png")
        print("✅ Sample image saved to 'test_cat_sample.png'")
    else:
        print("❌ No images found. Please check your paths.")