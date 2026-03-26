# src/datasets/transforms.py

from torchvision import transforms


def build_basic_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet
            std=[0.229, 0.224, 0.225],
        ),
    ])