import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
from tqdm import tqdm

# -----------------------------
# Model Definition (same as training)
# -----------------------------
class FaceNet(nn.Module):
    def __init__(self, embedding_size=512):
        super(FaceNet, self).__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, embedding_size)

    def forward(self, x):
        return self.backbone(x)


# -----------------------------
# Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_image(img_path):
    image = Image.open(img_path).convert("RGB")
    return transform(image).unsqueeze(0)


# -----------------------------
# Load Model Checkpoint
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "checkpoints/resnet50_arcface.pth"

model = FaceNet(embedding_size=512)
state_dict = torch.load(model_path, map_location=device)

# Fix: add "backbone." prefix if missing
if not any(k.startswith("backbone.") for k in state_dict.keys()):
    print("⚙️  Adjusting checkpoint key names (adding 'backbone.' prefix)...")
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[f"backbone.{k}"] = v
    state_dict = new_state_dict

model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()

print("✅ Model loaded successfully from:", model_path)


# -----------------------------
# Build Gallery Embeddings
# -----------------------------
gallery_dir = "data/vggface2_split/val"

gallery_embeddings = []
gallery_labels = []

print("Building gallery embeddings...")
for person_name in tqdm(os.listdir(gallery_dir)):
    person_folder = os.path.join(gallery_dir, person_name)
    if not os.path.isdir(person_folder):
        continue

    img_files = [f for f in os.listdir(person_folder) if f.lower().endswith(('.jpg', '.png'))]
    if len(img_files) == 0:
        continue

    img_path = os.path.join(person_folder, img_files[0])
    img_tensor = load_image(img_path).to(device)

    with torch.no_grad():
        emb = model(img_tensor)
        emb = F.normalize(emb, p=2, dim=1)

    gallery_embeddings.append(emb)
    gallery_labels.append(person_name)

gallery_embeddings = torch.cat(gallery_embeddings)
print(f"✅ Gallery built with {len(gallery_labels)} identities.")


# -----------------------------
# Test Inference
# -----------------------------
test_image_path = "data/vggface2_split/val/n000002/0009_01.jpg"



img_tensor = load_image(test_image_path).to(device)

with torch.no_grad():
    test_emb = model(img_tensor)
    test_emb = F.normalize(test_emb, p=2, dim=1)

similarities = F.cosine_similarity(test_emb, gallery_embeddings)
best_idx = torch.argmax(similarities).item()

predicted_name = gallery_labels[best_idx]
confidence = similarities[best_idx].item()

print("\n🔍 Inference Result:")
print(f"Input Image: {test_image_path}")
print(f"Predicted Identity: {predicted_name}")
print(f"Confidence: {confidence:.4f}")
