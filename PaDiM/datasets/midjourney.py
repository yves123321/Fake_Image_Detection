import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

CLASS_NAMES = [
"tench",
"goldfish",
"great_white_shark",
"tiger_shark",
"hammerhead",
"electric_ray",
"sting_ray",
"cock",
"hen",
"ostrich"]

class MidjourneyDataset(Dataset):

    def __init__(self, root, class_name="tench", is_train=True, test_ratio=0.3):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.root = root
        self.is_train = is_train

        nature_dir = os.path.join(root, class_name, "train", "nature")
        ai_dir = os.path.join(root, class_name, "train", "ai")

        nature_imgs = [os.path.join(nature_dir, f) for f in os.listdir(nature_dir)]
        ai_imgs = [os.path.join(ai_dir, f) for f in os.listdir(ai_dir)]

        random.shuffle(nature_imgs)
        random.shuffle(ai_imgs)

        split_nature = int(len(nature_imgs) * (1 - test_ratio))
        split_ai = int(len(ai_imgs) * (1 - test_ratio))

        if is_train:
            # train uniquement nature
            self.images = nature_imgs[:split_nature]
            self.labels = [0] * len(self.images)

        else:
            test_nature = nature_imgs[split_nature:]
            test_ai = ai_imgs[split_ai:]

            self.images = test_nature + test_ai
            self.labels = [0]*len(test_nature) + [1]*len(test_ai)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = self.images[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, label
