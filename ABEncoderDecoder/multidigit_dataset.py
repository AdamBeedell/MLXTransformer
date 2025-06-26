import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Token definitions
SOS_TOKEN = 10
EOS_TOKEN = 11
PAD_TOKEN = 12
MAX_SEQ_LEN = 6  # <sos> d1 d2 d3 d4 <eos>

class MultiDigitDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.img_dir = os.path.join(data_dir, split, "images")
        self.label_file = os.path.join(data_dir, split, "labels.txt")
        self.transform = transform
        self.samples = []

        with open(self.label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                img_name = parts[0]
                label = [int(d) for d in parts[1:]]

                img_path = os.path.join(self.img_dir, img_name)
                if os.path.exists(img_path):
                    self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {self.label_file}!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        padded = [SOS_TOKEN] + label + [EOS_TOKEN]
        while len(padded) < MAX_SEQ_LEN:
            padded.append(PAD_TOKEN)
        tgt_input = torch.tensor(padded[:-1])
        tgt_output = torch.tensor(padded[1:])
        return image, tgt_input, tgt_output