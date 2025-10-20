---

# 🌿 Plant Species Classification App

This app predicts the **species of a plant** from an uploaded image using a **pre-trained EfficientNet model**.
It’s built with **Streamlit** for the user interface and **TensorFlow** for the deep learning model.

---

## ⚙️ How It Works (Step-by-Step)

### **1. User Uploads an Image**

* When you open the app, you’ll see an option to **upload a plant image** (`.jpg`, `.jpeg`, `.png`).
* Once uploaded, the image is displayed on the screen for preview.

### **2. Model Loading**

* The app initializes a `PlantPredictor` object from **`pipeline.py`**.
* Inside this class:

  * The trained **EfficientNet model** (`plant_species_model.keras`) is loaded using TensorFlow.
  * The file **`class_names.txt`** is also read — it contains all the plant species names the model can recognize.
* The app uses Streamlit’s `@st.cache_resource` decorator to **load the model only once**, making future predictions much faster.

### **3. Image Preprocessing**

Before sending the uploaded image to the model:

* The image is **resized to 256×256 pixels** (the same size used during training).
* It’s converted to a **NumPy array** and reshaped to match the model’s expected input shape.
* **No normalization** (dividing by 255) is applied because the model was trained without it.

This ensures the uploaded image is compatible with the trained model’s input format.

### **4. Model Prediction**

* The preprocessed image array is passed into the model:

  ```python
  preds = self.model.predict(img_array)
  ```
* The model outputs a list of probabilities — one for each possible plant species.

### **5. Interpreting the Prediction**

* The app picks the species with the **highest probability**:

  ```python
  predicted_class = self.class_names[np.argmax(preds)]
  ```
* It also calculates the **confidence percentage**:

  ```python
  confidence = float(np.max(preds)) * 100
  ```
* These two values (predicted class and confidence) are returned to the Streamlit app.

### **6. Displaying the Output**

* Streamlit then displays the result clearly on screen:

  ```
  ✅ Predicted Species: Strelitzia Reginae
  📊 Confidence: 97.45%
  ```
* This means the model is 97.45% sure that the uploaded image belongs to *Strelitzia Reginae*.

---

## 🔄 Behind the Scenes Summary

| Step | What Happens                     | File Responsible |
| ---- | -------------------------------- | ---------------- |
| 1️⃣  | User uploads a plant image       | `app.py`         |
| 2️⃣  | Model and class names are loaded | `pipeline.py`    |
| 3️⃣  | Image resized and converted      | `pipeline.py`    |
| 4️⃣  | Model predicts species           | `pipeline.py`    |
| 5️⃣  | Result shown on web UI           | `app.py`         |

---

## 💡 Key Components

### **`app.py`**

* Builds the web interface using Streamlit.
* Lets the user upload an image.
* Loads the `PlantPredictor` once (cached for efficiency).
* Displays the predicted species and confidence score.

### **`pipeline.py`**

* Contains the **`PlantPredictor`** class.
* Handles:

  * Model loading
  * Image preprocessing
  * Prediction logic
* Returns both **predicted species** and **confidence percentage** to the Streamlit app.

---

## ▶️ How to Run the App

### **1. Install Required Packages**

Make sure you have Python installed, then run:

```bash
pip install streamlit tensorflow pillow numpy
```

### **2. Place Required Files**

Ensure these files are in the same directory:

```
app.py
pipeline.py
plant_species_model.keras
class_names.txt
```

### **3. Run the App**

Start the Streamlit app:

```bash
streamlit run app.py
```

### **4. Use the App**

* Upload any plant image.
* Click **🔍 Predict**.
* See the **predicted species name** and **confidence score** instantly.

---

## 🧠 In Short

The workflow is simple:

```
Upload → Preprocess → Predict → Display
```

This clean, modular setup separates the **model logic** (in `pipeline.py`) from the **user interface** (in `app.py`), making it easy to understand, maintain, and extend.

---

Would you like me to make this into a downloadable `README.md` file (ready for your GitHub repo)?
