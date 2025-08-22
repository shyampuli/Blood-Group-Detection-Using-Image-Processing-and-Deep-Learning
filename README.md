# Blood-Group-Detection-Using-Image-Processing-and-Deep-Learning
Developed a novel system for automatic blood group detection using image processing and deep learning techniques.

# 🩸 Blood Group Detection using Deep Learning

This project implements a **Convolutional Neural Network (CNN)** with **transfer learning** (InceptionV3/ResNet/MobileNet from TensorFlow Hub) to classify blood group images into **8 classes**:

- A+
- A-
- B+
- B-
- AB+
- AB-
- O+
- O-

---

## 🚀 Steps in the Project

### 1. Data Preprocessing
- Extracted dataset from `Blood_Grp_Detection.zip`.
- Merged all subfolder images into a single folder (`all_images/`) with labels embedded in filenames.
- Created **class labels**:
A+ = 0, A- = 1, B+ = 2, B- = 3,
AB+ = 4, AB- = 5, O+ = 6, O- = 7
- Resized all images to `(224,224,3)` for CNN compatibility.
- Converted `.BMP` images into NumPy arrays.

### 2. Train-Test Split
- Training images: **4800**
- Testing images: **1200**
- Normalized pixel values to `[0,1]`.

### 3. Model Architecture
- Pre-trained base model: **InceptionV3 (TensorFlow Hub)**
- Custom head:
- Dense (1024, ReLU)
- Dropout (0.5)
- Dense (8, output layer)
- Loss: `SparseCategoricalCrossentropy`
- Optimizer: `Adam`

### 4. Training
- Epochs: 5 (for demo; can be increased with EarlyStopping).
- Accuracy: ~26% (low due to few epochs & imbalance).
- Saved model as `75%.h5`.

### 5. Evaluation
```python
score, acc = model.evaluate(X_test, Y_test)
print("Test Loss:", score)
print("Test Accuracy:", acc)

### 6. Prediction System

Upload an image → Preprocess → Predict blood group.

Example:

Input: /content/cluster_4_4867.BMP
Output: AB+ Blood Group
