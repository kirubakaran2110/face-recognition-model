# 🧠 Face Recognition Model

This project implements a **Face Recognition System** that identifies a person from a gallery of known individuals (**closed-set identification**).  
The model uses a **ResNet-50** backbone trained with **ArcFace loss** to produce discriminative facial embeddings.  
It was trained and evaluated using a subset of the **VGGFace2 dataset**.

---

## ⚙️ Reproduction Instructions

### 1️⃣ Clone / Extract Project
Download or clone this repository:
```bas
git clone <your_repo_link>
cd face_recognition
```
2️⃣ Install Dependencies
```
python -m venv .venv
.venv\Scripts\activate   # (Windows)
```

3️⃣ Dataset Setup

Manually download the VGGFace2 subset (e.g. VGGFace2_subset_500) from Kaggle.
Organize it as follows:

```
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
```

📦 Dataset Source:

Kaggle subset → VGGFace2 Subset 500 

4️⃣ Train the Model
```
python train.py
```
Trains a ResNet-50 backbone with ArcFace loss.

Saves the checkpoint automatically as:
```
checkpoints/resnet50_arcface.pth
```
5️⃣ Run Inference
```
python infer.py
```
Example Output:
```
🔍 Inference Result:
Input Image: data/vggface2_split/val/n000002/0009_01.jpg
Predicted Identity: n000002
Confidence: 0.98
```
6️⃣ Evaluate the Model
```
python evaluate.py
```
Example Output:
```
✅ Top-1 Accuracy: 100.00%
✅ Top-5 Accuracy: 100.00%
```
🧩 Design Choices
Model Architecture
Component	Description
Backbone	ResNet-50 (pretrained on ImageNet)
Embedding Dim	512
Loss Function	ArcFace (angular margin-based softmax)
Optimizer	Adam (lr = 1e-3)
Batch Size	8
Epochs	3 (for faster experimentation)

Preprocessing

1.Input size: 224×224 cropped faces
2.Normalization: Mean = 0.5, Std = 0.5
3.Augmentation: Random horizontal flip during training

Training Strategy

1.Used metric learning to enforce inter-class separation and intra-class compactness
2.Fine-tuned pretrained ResNet-50 weights instead of training from scratch
3.Validation accuracy monitored after every epoch
4.Model checkpoint saved after each training run

Inference Logic

Extract embeddings from input image
Compute cosine similarity with gallery embeddings
Return top-1 predicted identity and confidence score

📊 Results
Metric	Value
Top-1 Accuracy	100%
Top-5 Accuracy	100%
Embedding Dimension	512
Model	ResNet-50 + ArcFace
🧠 Insights & Discussion

1.ArcFace provides angular margin separation, improving inter-class discrimination.

2.Even with limited data, fine-tuning achieved near-perfect accuracy on a closed-set evaluation.

3.ResNet-50 backbone leveraged pretrained ImageNet weights for strong feature extraction.

4.CPU training is significantly slower — GPU usage is highly recommended for scalability.

5.Real-world deployment can integrate webcam/video input for live face recognition.

📁 Folder Structure
```
face_recognition/
├── data/
│   ├── vggface2_subset_500/
│   │   ├── train/
│   │   └── val/
│   └── README.txt
│
├── checkpoints/
│   ├── resnet50_arcface.pth   # Generated after training
│   └── README.txt
│
├── train.py                   # Model training script
├── infer.py                   # Face identification demo
├── evaluate.py                # Model evaluation (Top-1/Top-5)
├── experiments.ipynb          # Experiment & visualization notebook
├── requirements.txt
└── README.md
```
💡 Future Improvements

1.Integrate real-time webcam inference

2.Add face alignment & detection (MTCNN/RetinaFace)

3.Visualize embeddings using t-SNE or PCA plots

4.Train on larger VGGFace2 / CelebA datasets

5.Convert to ONNX or TensorRT for faster inference


🏁 Author

👤 Kirubakaran P


