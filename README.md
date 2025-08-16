

```markdown
# 🩺 Skin Cancer Classification using EfficientNetB7

This project uses **Transfer Learning (EfficientNetB7)** to classify **skin cancer images** into two categories: **Malignant (1)** and **Benign (0)**.  
The model is trained on the Kaggle dataset and implemented in **Google Colab** with TensorFlow/Keras.  

---

## 📂 Dataset
Dataset used: [Skin Cancer Malignant vs Benign](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign)  
- **Malignant** = Cancerous  
- **Benign** = Non-cancerous  

Dataset structure:
```

data/
└── train/
├── malignant/
└── benign/

````

---

## ⚙️ Project Workflow
1. **Data Loading & Preprocessing**
   - Images resized to `224x224`
   - Normalized pixel values `[0,1]`
   - Labels: malignant=1, benign=0

2. **Train/Validation Split**
   - 85% training, 15% validation

3. **Model Architecture**
   - Pre-trained **EfficientNetB7 (frozen base)**
   - Global Average Pooling
   - Dense + BatchNorm + Dropout layers
   - Final Sigmoid output for binary classification

4. **Training**
   - Optimizer: `Adam`
   - Loss: `Binary Crossentropy`
   - Metric: `AUC`
   - Batch size: 32, Epochs: 5

5. **Prediction**
   - Load saved model (`.h5`)
   - Preprocess new image → Predict malignant/benign

---

## 🚀 How to Run

### 1️⃣ Clone Repository
```bash
git clone https://github.com/yasirwali1052/Skin-Cancer-Classification-EfficientNetB7.git
cd Skin-Cancer-Classification-EfficientNetB7
````

### 2️⃣ Open in Google Colab

Upload notebook or copy code into Colab.

### 3️⃣ Install Dependencies

```bash
pip install tensorflow pillow matplotlib seaborn
```

### 4️⃣ Train Model

```python
history = model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=1)
```

### 5️⃣ Save Model

```python
model.save("skin_cancer_model_rgb.h5")
```

### 6️⃣ Predict on New Image

```python
img = preprocess_image("/content/sample.jpg")
pred = model.predict(img)
print("Malignant" if pred[0][0] > 0.5 else "Benign")
```

---

## 📊 Results

* Pre-trained **EfficientNetB7** achieved high **AUC score** within few epochs.
* Model generalizes well with transfer learning despite limited training time.

---



```

---

👉 Do you want me to also **write the GitHub `requirements.txt` and a Colab notebook `.ipynb` setup** so you can directly push and run the repo?
```
