
🟩 1. Imports

```
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import EfficientNetV2B0
```

🔑 What’s happening:
We’re bringing in ready-made tools and libraries:

TensorFlow & Keras → the deep-learning framework that lets us build and train the model.

Layers → building blocks (like LEGO pieces) for the network.

Matplotlib → for drawing graphs.

os → to handle file paths.

EfficientNetV2B0 → the advanced image-recognition model we’re going to reuse.

💬 How to explain:

“We start by importing tools we need—like you open MS Word to write, Excel to handle tables. TensorFlow and Keras help us build and train the brain of our app. Matplotlib draws charts, and EfficientNet is a pre-trained brain we’ll reuse to save time.”


🟩 2. Downloading and Unzipping the Dataset

```
!pip install gdown
file_id = '1HL56kxu9y0oaJhfyvkvrOYaszDGQKv_G'
file_name = 'combined_dataset.zip'
!gdown --id {file_id} -O {file_name}
!unzip -q "{file_name}" -d "/content/plant_dataset"
```

🔑 What’s happening:

gdown lets us download a file from Google Drive using its ID.

We download a ZIP file that has all the plant images.

We unzip (extract) the images into a folder called plant_dataset.

💬 How to explain:

“We download the collection of plant photos and unzip them so our program can read them. It’s like downloading a folder of photos from Google Drive and opening it on the computer.”


🟩 3. Setting Parameters

```
BATCH_SIZE = 32
IMG_SIZE = (256, 256)
SEED = 42
data_dir = "/content/plant_dataset/combined_dataset"
```

🔑 What’s happening:

BATCH_SIZE: number of photos the model looks at in one step.

IMG_SIZE: we resize every photo to 256×256 pixels so they’re all uniform.

SEED: fixed random seed → keeps the split of train/validation data the same each time (helps reproducibility).

data_dir: folder location of images.

💬 How to explain:

“We set a few settings: how many pictures we show the model at a time, what size we make them, and a seed so the random shuffling stays consistent.”


🟩 4. Loading the Dataset

```
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="training",
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="validation",
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Detected {num_classes} species:", class_names)
```

🔑 What’s happening:

TensorFlow automatically reads the images from folders and assigns labels based on the folder names.

We split the data: 80% for training the model, 20% for validation (testing as we go).

We save the species names and count.

💬 How to explain:

“The code goes into the folders, reads all the images, and automatically knows which species each photo belongs to by the folder name. We keep 80% to teach the model and 20% aside to check how well it’s learning.”


🟩 5. Data Augmentation

```
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomBrightness(0.2),
])
```

🔑 What’s happening:
We create slightly changed versions of each photo:

Flip it sideways, rotate, zoom in/out, change brightness.

💬 How to explain:

“To help the model recognize plants in different angles, lighting, or zoom levels, we automatically create variations of each photo. This way the model becomes more robust and not confused by minor changes.”


🟩 6. Speeding up Data Loading

```
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

🔑 What’s happening:

Cache: keeps images in memory to avoid re-reading from disk.

Shuffle: mixes images randomly so the model doesn’t learn patterns from order.

Prefetch: loads next batch while the current one is training → faster training.

💬 How to explain:

“We make the loading of pictures faster and mix them up so the model doesn’t just memorize the order.”


🟩 7. Building the Model

```
base_model = EfficientNetV2B0(include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,))
base_model.trainable = False  # freeze base model

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
```

🔑 What’s happening:

We load EfficientNetV2B0 that was already trained on millions of general images (ImageNet).

Freeze base_model: keep its learned knowledge, don’t change it yet.

Add our own small layers (like a new head) to classify our specific plant species.

Compile model: choose optimizer (how it learns), loss function, and metric (accuracy).

💬 How to explain:

“We start with a pre-trained brain (EfficientNet) that already knows about shapes, edges, colors from millions of images. We freeze it so we don’t disturb that knowledge, then add our own small decision-making layers to recognize our plant species. We tell the model how to learn and what to measure.”


🟩 8. Callbacks

```
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("efficientnet_plants.keras", save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.2)
]
```

🔑 What’s happening:

EarlyStopping: stop training if it stops improving → saves time and avoids overfitting.

ModelCheckpoint: save the best version of the model.

ReduceLROnPlateau: lower the learning rate if progress stalls → fine tune gently.

💬 How to explain:

“These are automatic helpers. If the model stops improving, it stops training early, saves the best version, and slows down the learning rate when needed to keep improving steadily.”


🟩 9. Training Phase 1 – Feature Extraction

```
history = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)
```

🔑 What’s happening:

The model learns only the new top layers, keeping the base frozen.

We train for up to 20 rounds (epochs), but it can stop earlier if EarlyStopping triggers.

💬 How to explain:

“We first train just the new part of the model to adjust it for our plant photos, keeping the old knowledge untouched.”


🟩 10. Fine-Tuning Phase 2

```
base_model.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
fine_tune_history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)
```

🔑 What’s happening:

We unfreeze the base model so it can learn too, but use a much smaller learning rate to avoid wiping out its good knowledge.

This makes the model adjust its older features to fit plants better.

💬 How to explain:

“Once the new layers are stable, we let the full brain learn together—slowly and carefully—to fine-tune it specifically for plants.”


🟩 11. Evaluation
bash
```
val_loss, val_acc = model.evaluate(val_ds)
print(f"Validation Accuracy: {val_acc:.2%}")
```

🔑 What’s happening:
We test the final model on the 20% validation set it hasn’t seen during training to measure real-world accuracy.

💬 How to explain:

“We finally check how well the model recognizes plant species on photos it never saw before.”


🟩 12. Visualization
bash
```
plt.figure(figsize=(12,5))
plt.plot(history.history['accuracy'] + fine_tune_history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'] + fine_tune_history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

🔑 What’s happening:
We draw a graph showing accuracy over each epoch for both training and validation sets.
Helps us see improvement and detect if the model overfitted.

💬 How to explain:

“We draw a chart to show how accuracy improved during both phases and to confirm the model is learning properly.”
