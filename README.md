# Pizza vs Not Pizza — Binary Image Classification Using Custom CNN

This project implements a **binary image classification pipeline** using a **custom Convolutional Neural Network (CNN)** to distinguish between **pizza** and **non-pizza** images.
The dataset is sourced from **Kaggle**, and the focus is on **training from scratch, data augmentation, and generalization**, rather than transfer learning or leaderboard optimization.

---

## 📌 Problem Statement

Food image classification is challenging due to:

* High intra-class variability (different pizza styles, toppings, lighting)
* Visual overlap between food categories
* Limited labeled data

The goal of this project is to evaluate how well a **custom CNN**, trained from scratch, can learn discriminative visual features for a binary food classification task.

---

## 📂 Dataset

* **Source:** Kaggle (Food Image Dataset)
* **Classes:**

  * Pizza
  * Not Pizza
* Dataset verified for correct directory structure and label integrity prior to training.

---

## 🔧 Data Preprocessing

* Images resized to a fixed input resolution
* Pixel values normalized using rescaling
* Directory-based loading using `tf.keras.image_dataset_from_directory`

---

## 🔁 Data Augmentation

Data augmentation was used as a **generalization strategy** to handle visual variability.

Applied transformations:

* Random horizontal flips
* Random rotations
* Zoom and cropping
* Contrast variation

Augmentation was integrated directly into the model pipeline using Keras preprocessing layers.

---

## 🧠 Model Architecture

* Custom CNN designed from scratch using Keras
* Architecture includes:

  * Convolutional layers for feature extraction
  * Max pooling for spatial downsampling
  * Batch normalization for training stability
  * Dropout for regularization
  * Fully connected layers for classification

**Output Layer**

* Sigmoid activation
* Binary Cross-Entropy loss

**Model Size**

* ~700K trainable parameters
* Balanced for small-to-medium dataset size

---

## ⚙️ Training & Optimization

* Optimizer: Adam
* Loss function: Binary Cross-Entropy
* Trained with validation monitoring to assess generalization
* Emphasis on avoiding overfitting rather than maximizing depth

---

## 📊 Results

* **Validation Accuracy:** **78.5%**

This performance reflects:

* Training from scratch without pretrained features
* High visual variability in food images
* Limited dataset size

Accuracy was evaluated as a learning and experimentation benchmark, not as a production-level result.

---

## 🛠️ Tech Stack

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib

---

## 📚 Key Learnings

* Custom CNNs can learn meaningful features for food images but are sensitive to data variability
* Data augmentation plays a critical role in improving robustness
* Architectural simplicity helps control overfitting on limited datasets
* Transfer learning would likely improve performance for this task

---

## 🚀 Future Improvements

* Apply transfer learning (VGG, ResNet, MobileNet)
* Improve class balance and dataset size
* Perform hyperparameter tuning
* Extend to multi-class food classification

---


