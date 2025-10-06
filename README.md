That's a solid plant species classification project using **EfficientNetV2B0**\! Here is a structured **README.md** file suitable for GitHub, explaining the code cell-by-cell.

# 🌿 Plant Species Classification with EfficientNetV2B0

This project demonstrates a deep learning approach for classifying plant species using a **Transfer Learning** technique with the **EfficientNetV2B0** model, a state-of-the-art Convolutional Neural Network (CNN).

## 🚀 Project Overview

The goal is to accurately classify plant images into their respective species. The process involves:

1.  **Setting up the environment** and downloading the dataset.
2.  **Loading and preparing** the image dataset.
3.  Applying **data augmentation** to improve model robustness.
4.  Building a classification model using a **pre-trained EfficientNetV2B0** base.
5.  Training the model in two phases: **Feature Extraction (Freezing)** and **Fine-Tuning (Unfreezing)**.
6.  Evaluating the final model performance.

## 🛠️ Setup and Prerequisites

This project was developed in a **Google Colab** environment, which is reflected in the code's environment setup steps.

### Imports

This cell imports all necessary libraries for the project.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import EfficientNetV2B0 # Imported later, but good to list here
```

### Unzip Dataset

These steps handle the installation of necessary tools (`gdown`) and the downloading/unzipping of the dataset from a specified Google Drive location.

```python
!pip install gdown # Installs gdown for downloading files from Google Drive
# Download the file from Google Drive using the file ID
file_id = '1HL56kxu9y0oaJhfyvkvrOYaszDGQKv_G'
file_name = 'combined_dataset.zip'
!gdown --id {file_id} -O {file_name} # Downloads the zipped dataset

!unzip -q "{file_name}" -d "/content/plant_dataset" # Unzips the dataset into a local directory
```

-----

## ⚙️ Configuration and Data Loading

### Parameters

Sets key constants used throughout the data preparation and modeling steps.

```python
# Parameters
BATCH_SIZE = 32 # Number of images processed per training step
IMG_SIZE = (256, 256) # All images are resized to 256x256 pixels
SEED = 42 # Ensures reproducibility of dataset splits and random operations

data_dir = "/content/plant_dataset/combined_dataset" # Path to the unzipped images
```

### Load Dataset

The `tf.keras.utils.image_dataset_from_directory` utility is used to efficiently load images from the directory structure, automatically labeling them based on the folder names. The dataset is split into training (80%) and validation (20%) sets.

```python
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Detected {num_classes} species:", class_names) # Prints the discovered class labels
```

### Data Augmentation

A `keras.Sequential` model is created to apply random, on-the-fly transformations to the training images. This helps the model learn a more generalized set of features, making it less susceptible to overfitting.

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2), # Rotate up to 20%
    layers.RandomZoom(0.2), # Zoom in/out up to 20%
    layers.RandomBrightness(0.2), # Adjust brightness up to 20%
])

AUTOTUNE = tf.data.AUTOTUNE
# Optimize dataset loading with caching, shuffling, and prefetching
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

-----

## 🧠 Model Building and Training

### Build EfficientNet Model

This section sets up the **Transfer Learning** architecture using `EfficientNetV2B0`, pre-trained on the massive **ImageNet** dataset.

  * **Freezing:** Initially, the weights of the `base_model` are set to `trainable = False`. This is the **Feature Extraction** stage, where the model learns to classify using the powerful features already learned by EfficientNet, while only training the new classification layers.
  * **Head Layers:** A new classification head is added, including `GlobalAveragePooling2D` to condense features, `Dropout` to prevent overfitting, and a final `Dense` layer with `softmax` activation for classification into the `num_classes`.

<!-- end list -->

```python
base_model = EfficientNetV2B0(include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,))
base_model.trainable = False  # Freeze base layers for feature extraction

# Define the full model architecture
inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = base_model(x, training=False) # Pass augmented images through frozen base
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-2),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()
```

### Callbacks

**Callbacks** are functions executed during training to automate common tasks and improve performance.

```python
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True), # Stops training if validation loss doesn't improve for 5 epochs
    keras.callbacks.ModelCheckpoint("efficientnet_plants.keras", save_best_only=True), # Saves the model with the best validation accuracy
    keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.2) # Reduces learning rate if validation loss plateaus for 3 epochs
]
```

### Train the Model (Feature Extraction Phase)

The initial training phase. Since the base model is frozen, training is fast and focuses on getting the new classification head's weights optimal.

```python
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20, # Initial max epochs
    callbacks=callbacks
)
# ... evaluation and plotting code for this phase ...
```

### Fine-Tuning

This is the second, crucial stage of Transfer Learning.

  * **Unfreezing:** The `base_model.trainable = True` command unfreezes the EfficientNet weights.
  * **Lower Learning Rate:** The model is recompiled with a much lower learning rate (`1e-3`). This allows the model to gently adjust the pre-trained weights to be more specific to the plant classification task without destroying the valuable learned features.

<!-- end list -->

```python
base_model.trainable = True # Unfreeze the base model layers
model.compile(
    optimizer=keras.optimizers.Adam(1e-3), # Use a lower learning rate
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

fine_tune_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10, # Additional epochs for fine-tuning
    callbacks=callbacks
)
```

-----

## 📊 Evaluation and Visualization

### Evaluate

The final evaluation of the model on the validation set after both training phases are complete.

```python
val_loss, val_acc = model.evaluate(val_ds)
print(f"Validation Accuracy: {val_acc:.2%}")
```

### Visualize Training Curves

Plots are generated to visualize the training and validation accuracy over all epochs (Feature Extraction + Fine-Tuning). This helps diagnose **overfitting** (large gap between train and validation accuracy) or **underfitting** (both accuracies are low).

```python
plt.figure(figsize=(12,5))
# Combines the history from both training phases for a continuous plot
plt.plot(history.history['accuracy'] + fine_tune_history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'] + fine_tune_history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```
