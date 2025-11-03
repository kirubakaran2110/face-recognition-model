import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms, models
from sklearn.preprocessing import normalize

# =========================
# 🔧 CONFIGURATION
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "checkpoints/resnet50_arcface.pth"
val_dir = "data/vggface2_split/val"  # make sure this matches your folder

# =========================
# 🧩 Model definition
# =========================
class FaceNet(nn.Module):
    def __init__(self, embedding_size=512):
        super(FaceNet, self).__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        base_model.fc = nn.Identity()
        self.backbone = base_model
        self.embedding = nn.Linear(2048, embedding_size)

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        x = nn.functional.normalize(x)
        return x


# =========================
# 🧠 Load Model
# =========================
model = FaceNet(embedding_size=512).to(device)
checkpoint = torch.load(model_path, map_location=device)

# Handle backbone prefix mismatch if needed
if not any(k.startswith("backbone.") for k in checkpoint.keys()):
    print("⚙️  Adjusting checkpoint key names (adding 'backbone.' prefix)...")
    new_state = {}
    for k, v in checkpoint.items():
        if k.startswith("fc."):
            new_state["embedding." + k.split("fc.")[-1]] = v
        else:
            new_state["backbone." + k] = v
    checkpoint = new_state

model.load_state_dict(checkpoint, strict=False)
model.eval()
print(f"✅ Loaded model from {model_path}")

# =========================
# 🧾 Transform
# =========================
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# =========================
# 🧠 Embedding Extraction
# =========================
def extract_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor).cpu().numpy().flatten()
    return emb


# =========================
# 🏗️ Build Gallery Embeddings
# =========================
print("🔍 Building embeddings (1 image per identity)...")

identities = sorted(os.listdir(val_dir))
gallery_embeddings = []
gallery_labels = []

for person in tqdm(identities):
    person_dir = os.path.join(val_dir, person)
    img_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.png'))]
    if len(img_files) == 0:
        continue
    img_path = os.path.join(person_dir, img_files[0])
    emb = extract_embedding(img_path)
    gallery_embeddings.append(emb)
    gallery_labels.append(person)

gallery_embeddings = np.array(gallery_embeddings)

print(f"✅ Total identities used: {len(gallery_labels)}")

# =========================
# 📊 Embedding Stats
# =========================
print("\nEmbedding sample stats:")
print("Shape:", gallery_embeddings.shape)
print("Mean:", np.mean(gallery_embeddings))
print("Std:", np.std(gallery_embeddings))

# =========================
# 🧮 Compute Similarity
# =========================
print("🧮 Computing similarity matrix...")
gallery_embeddings = normalize(gallery_embeddings)
similarity = np.matmul(gallery_embeddings, gallery_embeddings.T)

# =========================
# 🎯 Compute Accuracy
# =========================
top1_correct = 0
top5_correct = 0
n = len(gallery_labels)

for i in range(n):
    sims = similarity[i]
    sorted_idx = np.argsort(-sims)  # descending
    top1 = sorted_idx[0]
    top5 = sorted_idx[:5]
    
    if gallery_labels[top1] == gallery_labels[i]:
        top1_correct += 1
    if gallery_labels[i] in [gallery_labels[j] for j in top5]:
        top5_correct += 1

top1_acc = 100 * top1_correct / n
top5_acc = 100 * top5_correct / n

print(f"\n✅ Top-1 Accuracy: {top1_acc:.2f}%")
print(f"✅ Top-5 Accuracy: {top5_acc:.2f}%")
