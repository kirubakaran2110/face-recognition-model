# visualize_dataset.py
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import random

# Define same transform used in training
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# Load train split
dataset = datasets.ImageFolder("data/vggface2_split/train", transform=transform)
classes = dataset.classes
print(f"Total identities: {len(classes)}")
print(f"Total images: {len(dataset)}")

# Show random 5 identities
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for ax in axs:
    idx = random.randint(0, len(dataset)-1)
    img, label = dataset[idx]
    ax.imshow(img.permute(1, 2, 0))
    ax.set_title(classes[label], fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.show()
