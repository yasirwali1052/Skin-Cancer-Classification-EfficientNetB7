# Skin Cancer Classification using EfficientNetB7

A deep learning project that applies **Transfer Learning (EfficientNetB7)** to classify skin cancer images into **Malignant (1)** and **Benign (0)** categories. The model is trained on the Kaggle dataset and implemented in **Google Colab** using **TensorFlow/Keras**.

## Dataset

**Source:** [Skin Cancer Malignant vs Benign](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign)

**Classes:**
- **Malignant** → Cancerous (Label = 1)  
- **Benign** → Non-cancerous (Label = 0)  

**Dataset Structure:**
```
data/
└── train/
    ├── malignant/
    └── benign/
```

## Workflow

### 1. Data Preprocessing
- Images resized to **224×224**
- Pixel values normalized to **[0,1]**
- Labels converted to binary (0 = benign, 1 = malignant)

### 2. Train/Validation Split
- **85% Training**  
- **15% Validation**

### 3. Model Architecture
- Base: **EfficientNetB7 (pre-trained, frozen)**
- Layers:
  - Global Average Pooling
  - Dense (256) + BatchNorm + Dropout
  - Dense (256) + BatchNorm
  - Output: Sigmoid (binary classification)

### 4. Training Setup
- **Optimizer:** Adam  
- **Loss:** Binary Crossentropy  
- **Metric:** AUC  
- **Batch Size:** 32  
- **Epochs:** 5  

### 5. Prediction
- Load the saved model (`.h5`)  
- Preprocess new image  
- Predict → Output: *Malignant* or *Benign*  

## How to Run

### Clone Repository
```bash
git clone https://github.com/yasirwali1052/Skin-Cancer-Classification-EfficientNetB7.git
cd Skin-Cancer-Classification-EfficientNetB7
```

### Open in Google Colab
Upload the notebook or copy the code into Colab.

### Install Dependencies
```bash
pip install tensorflow pillow matplotlib seaborn
```

### Train Model
```python
history = model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=1)
```

### Save Model
```python
model.save("skin_cancer_model_rgb.h5")
```

### Predict on New Image
```python
img = preprocess_image("/content/sample.jpg")
pred = model.predict(img)
print("Malignant" if pred[0][0] > 0.5 else "Benign")
```

## Results

- ✅ Achieved high **AUC score** within a few epochs using transfer learning
- ✅ Model shows strong generalization performance
- ✅ EfficientNetB7 proves effective even with limited training time

