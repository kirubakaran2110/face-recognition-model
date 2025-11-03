# face-recognition-model
📘 Overview

This project implements a Face Recognition System that identifies a person from a gallery of known individuals (closed-set identification).
The model uses a ResNet-50 backbone trained with ArcFace loss to produce discriminative facial embeddings.
It was trained and evaluated using a subset of the VGGFace2 dataset.

⚙️ Reproduction Instructions
1. Clone / Extract Project

Download or clone this repository:

git clone <your_repo_link>
cd face_recognition

2. Install Dependencies

Create a virtual environment (optional but recommended):

python -m venv .venv
.\.venv\Scripts\activate   # (Windows)


Install all required packages:

pip install torch torchvision numpy opencv-python tqdm pillow scikit-learn matplotlib jupyter

3. Dataset Setup

Manually download the VGGFace2 subset (e.g. VGGFace2_subset_500) from Kaggle.

Organize the dataset as:

data/
└── vggface2_subset_500/
    ├── train/
    │   ├── person_0001/
    │   ├── person_0002/
    │   └── ...
    └── val/
        ├── person_0001/
        ├── person_0002/
        └── ...

4. Train the Model
python train.py


Trains a ResNet-50 backbone with ArcFace loss

Saves the checkpoint as checkpoints/resnet50_arcface.pth

5. Run Inference
python infer.py


Example Output:

🔍 Inference Result:
Input Image: data/vggface2_split/val/n000002/0009_01.jpg
Predicted Identity: n000002
Confidence: 0.98

6. Evaluate the Model
python evaluate.py


Example Output:

✅ Top-1 Accuracy: 100.00%
✅ Top-5 Accuracy: 100.00%

🧩 Design Choices
Model Architecture

Backbone: ResNet-50 (pretrained on ImageNet for strong feature extraction)

Embedding dimension: 512-d

Loss Function: ArcFace — chosen for producing highly discriminative embeddings with angular margin penalty.

Optimizer: Adam (learning rate = 1e-3)

Batch size: 8

Epochs: 3 (for faster experimentation)

Preprocessing

Input: 224×224 cropped face images

Normalization: Mean = 0.5, Std = 0.5

Augmentation: Random horizontal flip during training

Training Strategy

Used metric learning to enforce inter-class separation and intra-class compactness.

Fine-tuned pretrained ResNet weights instead of training from scratch (saves time).

Validation accuracy monitored at the end of each epoch.

Inference

Extract embeddings from input images using trained model.

Compute cosine similarity with gallery embeddings.

Return top match identity with confidence score.

Evaluation

Top-1 Accuracy: Correct top prediction

Top-5 Accuracy: Correct identity within top 5 predictions

📊 Results
Metric	Value
Top-1 Accuracy	100%
Top-5 Accuracy	100%
Embedding Dimension	512
Model	ResNet-50 + ArcFace
🧠 Insights & Discussion

ArcFace significantly improves discrimination between identities compared to standard softmax.

Despite limited training data, embeddings achieved near-perfect closed-set accuracy.

Fine-tuning a pretrained ResNet yields excellent results even with fewer epochs.

CPU training is slow; GPU usage is strongly recommended for full-scale datasets.
