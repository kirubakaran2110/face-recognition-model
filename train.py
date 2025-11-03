# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import os

# ArcFace loss layer
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.cos_m = torch.cos(torch.tensor(m))
        self.sin_m = torch.sin(torch.tensor(m))
        self.th = torch.cos(torch.tensor(3.14159) - m)
        self.mm = torch.sin(torch.tensor(3.14159) - m) * m

    def forward(self, embeddings, labels):
        cosine = nn.functional.linear(nn.functional.normalize(embeddings), nn.functional.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transformations
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Dataset loading
data_dir = "data/vggface2_split"
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

num_classes = len(train_dataset.classes)

# Model setup
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 512)
model = model.to(device)

arcface_loss = ArcFaceLoss(512, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(model.parameters()) + list(arcface_loss.parameters()), lr=1e-4)

# Training loop
for epoch in range(3):  # adjust epochs
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/3"):
        imgs, labels = imgs.to(device), labels.to(device)
        embeddings = model(imgs)
        logits = arcface_loss(embeddings, labels)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/3], Loss: {running_loss/len(train_loader):.4f}")

# Save model
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/resnet50_arcface.pth")
print("✅ Model saved at checkpoints/resnet50_arcface.pth")
