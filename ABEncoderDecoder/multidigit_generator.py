import os
import random
import numpy as np
from PIL import Image
import torch
from torchvision import datasets
from torchvision.transforms import ToPILImage, ToTensor
from collections import Counter
import matplotlib.pyplot as plt

# --- Config ---
MNIST_PATH = "./.data"
OUTPUT_PATH = "./.data/multidigit_mnist"
MAX_DIGITS = 4
CANVAS_SIZE = 96  # output image will be 64x64
DIGIT_SIZE = 28
PADDING = 2
DIGIT_DATASET_SPLIT = 'training'  # 'training' or 'testing'

os.makedirs(f"{OUTPUT_PATH}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_PATH}/labels", exist_ok=True)

# --- Load MNIST dataset ---
mnist = datasets.MNIST(root=MNIST_PATH, train=(DIGIT_DATASET_SPLIT=='training'), download=True)
to_pil = ToPILImage()
to_tensor = ToTensor()

# --- Compose multiple digits onto a canvas ---
def generate_canvas(digits, labels):
    canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
    positions = []
    used_boxes = []
    
    for i, (digit_img, label) in enumerate(zip(digits, labels)):
        placed = False
        attempt = 0
        while not placed and attempt < 50:
            # Random top-left corner
            x = random.randint(0, CANVAS_SIZE - DIGIT_SIZE)
            y = random.randint(0, CANVAS_SIZE - DIGIT_SIZE)
            new_box = [x, y, x + DIGIT_SIZE, y + DIGIT_SIZE]

            # Check overlap
            if all((x2 <= x or x1 >= x + DIGIT_SIZE or y2 <= y or y1 >= y + DIGIT_SIZE)
                   for x1, y1, x2, y2 in used_boxes):
                canvas.paste(digit_img, (x, y))
                used_boxes.append(new_box)
                positions.append(((x + DIGIT_SIZE//2, y + DIGIT_SIZE//2), label))
                placed = True
            attempt += 1

    return canvas, positions

# --- Generate synthetic data ---
NUM_SAMPLES = 100000
digit_count_tracker = Counter()
example_images = {}

with open(f"{OUTPUT_PATH}/labels/labels.txt", "w") as label_file:
    for idx in range(NUM_SAMPLES):
        num_digits = random.randint(1, MAX_DIGITS)
        indices = [random.randint(0, len(mnist) - 1) for _ in range(num_digits)]
        digits = [mnist[i][0] for i in indices]
        labels = [mnist[i][1] for i in indices]

        canvas, positions = generate_canvas(digits, labels)

        image_path = f"{OUTPUT_PATH}/images/img_{idx:05d}.png"
        canvas.save(image_path)

        label_str = " ".join(str(label) for _, label in sorted(positions, key=lambda pos: pos[0][0]))
        label_file.write(f"img_{idx:05d}.png {label_str}\n")

        digit_count_tracker[num_digits] += 1
        if num_digits not in example_images:
            example_images[num_digits] = canvas.copy()

print(f"Generated {NUM_SAMPLES} synthetic multi-digit images in {OUTPUT_PATH}")

plt.bar(digit_count_tracker.keys(), digit_count_tracker.values())
plt.xlabel('Number of Digits')
plt.ylabel('Count')
plt.title('Distribution of Digit Counts in Synthetic Dataset')
plt.savefig(f"{OUTPUT_PATH}/digit_distribution.png")

for count, img in example_images.items():
    img.save(f"{OUTPUT_PATH}/example_{count}_digits.png")