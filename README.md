# Projectile Point Image Classifier (Beginner Tutorial)

This tutorial will guide you through building a machine learning classifier for North American projectile points using Python and TensorFlow, all inside Google Colab. By the end, you will have a fully trained and testable image classification model that runs in your browser with no installation required. You will be able to upload a new image of a projectile point and receive a prediction of its type (for example, “Clovis” or “Dalton”), along with a confidence score.

Why is this useful? Archaeologists, museum professionals, and students often work with large collections of projectile points, and identifying or labeling them can be time-consuming. A classifier does not replace expert judgment, but it can assist in organizing data, exploring typological comparisons, or helping students learn through hands-on experience with real datasets. Because everything runs in Google Colab (a free, cloud-based notebook environment), there is no need to install software or configure complex environments. You simply follow the steps, run the code, and see the results.

More importantly, this tutorial is about making AI approachable. For many in archaeology and heritage fields, machine learning can feel overly technical or inaccessible. This guide is designed to make the process clear, transparent, and beginner-friendly. You will see how the model learns, where it performs well, and where it might need improvement. The goal is not just to build a working classifier, but to help you understand what is happening behind the scenes so you can use these tools critically and confidently in your own research or teaching.

> **Note**: While this tutorial focuses on projectile points, the same workflow can be used to classify many other types of archaeological or heritage objects. For example, you could train a model to recognize decorated ceramics, identify other lithic tool types, distinguish colonial pipe stems, or sort beads by material or period. The approach is flexible and can be adapted to any object type as long as you have labeled images that reflect the variation you want the model to learn.

## What You’ll Learn

- Organize a dataset of projectile point images into training, validation, and test sets.  
- Load and prepare image data in Google Colab using TensorFlow.  
- Build a convolutional neural network (CNN) for image classification.  
- Train the model and track its progress across multiple epochs.  
- Visualize accuracy and loss to understand how well the model is learning.  
- Evaluate performance on a held-out test set for honest results.  
- Create a simple interactive demo with Gradio to test the model in your browser.  
- Save and reload your trained model for future use.  
- Develop confidence in reading ML metrics like accuracy, loss, overfitting, and underfitting.  


## Step 1 — Connect Google Drive and Upload Your Dataset

We’ll start by mounting your Google Drive so the notebook can access your image dataset. Make sure your projectile point images are stored in labeled folders — one folder per type (e.g., `clovis`, `folsom`, `dalton`).

You’ll also want to upload a separate “held-out” test folder (`points_heldout`) with the same structure. This test set will not be used during training — we’ll use it later to evaluate how well the model generalizes.

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Step 2 — Set Up Paths and Check Your Images

Let’s tell Colab where to find your dataset. Your images should be organized by class: for example, all Clovis images in one folder, all Folsom images in another, and so on. The folder names become the labels your model will learn to recognize.

We’ll also count how many images are in each class folder. This helps catch issues early — like if you forgot to upload something or if a folder is empty.

📝 **Note**: A quick check like this is often called a **sanity check** — it’s just making sure things look normal before training.

```python
import os, glob
from pathlib import Path

DRIVE = "/content/drive/MyDrive"
DATA_DIR = f"{DRIVE}/points"
TEST_DIR = f"{DRIVE}/points_heldout"

def list_classes(root):
    return sorted([p.name for p in Path(root).glob("*") if p.is_dir()])

def count_images(root):
    counts = {}
    for cls in list_classes(root):
        n = len(glob.glob(str(Path(root)/cls/"*")))
        counts[cls] = n
    return counts

print("Train/Val classes:", list_classes(DATA_DIR))
print("Test classes:", list_classes(TEST_DIR))
print("Train/Val counts:", count_images(DATA_DIR))
print("Test counts:", count_images(TEST_DIR))
```

## Step 3 — Create Our Image Dataset

Now we’ll load the images and turn them into something TensorFlow understands. This step creates a "dataset object" from your folders, resizes the images, labels them based on folder names, and prepares them for training.

We’ll also split the images into **training** and **validation** groups. Training images help the model learn; validation images help us measure how well it's learning (without bias).

📝 **Note**: The **validation set** is a portion of data set aside to check the model’s performance during training. It’s never used to teach the model — just to test it as it learns.

```python
import tensorflow as tf

IMG_SIZE = 224
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE)
```

## Step 4 — Normalize and Prepare the Data

Images are made of pixel values ranging from 0 to 255. Neural networks work better when those numbers are smaller and consistent — so we scale (or “normalize”) them to a range between 0 and 1.

We’ll also add caching and prefetching. These speed up training by making sure the next batch of images is always ready when the model needs it.

📝 **Note**:
- **Normalization** = Scaling values to a standard range, often 0 to 1.
- **Prefetching** = Loading data ahead of time so it’s ready when needed.
- **Caching** = Storing data in memory so it doesn’t have to be reprocessed repeatedly.

```python
AUTOTUNE = tf.data.AUTOTUNE

def prep(ds):
    return ds.map(lambda x, y: (x / 255.0, y)).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

train_ds = prep(train_ds)
val_ds = prep(val_ds)
```

## Step 5 — Set Up the Model Using Transfer Learning

Instead of training a model from scratch (which takes tons of data and time), we’ll use a pre-trained model called **MobileNetV2**. It was trained on millions of images and knows how to recognize general features like edges, shapes, and textures.

We’ll use it as the “backbone” of our model and then add a few new layers on top that are specific to our projectile point types. This approach is called **transfer learning** — and it's a powerful shortcut.

**Note**:
- **Transfer Learning** = Reusing a model trained on one task to jump-start learning on a new task.
- **Frozen layers** = Parts of the model that won’t be updated during training.

```python
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet')

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(train_ds.class_names), activation='softmax')
])

model.summary()
```

## Step 6 — Compile the Model

Now that our model architecture is ready, we need to "compile" it — this just means telling TensorFlow how to train the model.

**Note**:
- **Loss function** = A measure of how wrong the model’s predictions are. Lower is better.
- **Optimizer** = An algorithm that adjusts the model to reduce loss.
- **Metrics** = Stats we track during training (like accuracy).

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
```

## Step 7 — Train the Model

Time to let the model learn! Training is the process where the model looks at your labeled images, makes predictions, compares them to the correct answers, and adjusts itself to do better next time.

 **Note**:
- **Epoch** = One complete pass through the training dataset.
- **Validation** = Testing the model on unseen data to check if it’s learning useful patterns or just memorizing.

```python
EPOCHS = 10

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS)
```

## Step 8 — Visualize Model Performance

We’ll plot **accuracy** and **loss** for both the training and validation sets. This lets us see if the model is overfitting (doing well on training but poorly on validation) or underfitting (not doing well on either).

**Note**:
- **Overfitting** = When a model memorizes training data but can’t generalize to new data.
- **Underfitting** = When a model hasn’t learned enough patterns — poor performance overall.

```python
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.legend()
plt.title('Loss')

plt.show()
```

## Step 9 — Evaluate on Held-Out Test Set

Now let’s test it on a completely new set of images it’s never encountered before.

**Note**:
- **Held-out test set** = A final set of data kept separate from all training and validation. Used to test the model at the very end.
- **Evaluate** = Run the model on new data and return metrics like accuracy and loss.

```python
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE)

test_ds = test_ds.map(lambda x, y: (x / 255.0, y))

loss, accuracy = model.evaluate(test_ds)
print("Test accuracy:", accuracy)
```

## Step 10 — Build an Interactive App with Gradio

We’ll use **Gradio**, a tool that turns Python functions into simple web interfaces. You’ll be able to upload an image and instantly get a prediction from your model.

**Note**:
- **Gradio** = A Python library that builds web apps from your code — perfect for testing and demos.
- **Interface** = A user-friendly screen that lets someone interact with a function or model.

```python
import numpy as np
import gradio as gr

class_names = train_ds.class_names

def predict(img):
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.expand_dims(img / 255.0, 0)
    pred = model.predict(img)[0]
    return {class_names[i]: float(pred[i]) for i in range(len(class_names))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Projectile Point Classifier",
    description="Upload an image of a projectile point to see the model’s top predictions.")

demo.launch(share=True)
```

## Step 11 — Save the Trained Model

Saving your model means you can reuse it later without retraining.

**Note**:
- **.keras file** = A file that stores everything about your model — architecture, weights, training config, etc.
- **Serialization** = The process of turning a model into a file that can be saved and reloaded later.

```python
save_path = f"{DRIVE}/projectile_point_model.keras"
model.save(save_path)
print("Model saved to:", save_path)
```

## Step 12 — Load a Saved Model and Use It

Let’s load the model back and test that it works.

📝 **Note**:
- **Deserialization** = Loading a model file and turning it back into a usable model in your code.

```python
from tensorflow import keras

# Projectile Point Image Classifier (Beginner Tutorial)

This tutorial walks you step-by-step through building an image classifier that identifies types of projectile points (Clovis, Folsom, etc.) using Python and TensorFlow — all within Google Colab.

You don’t need any prior coding or machine learning experience. The tutorial explains what each step does, why it matters, and how to validate that it’s working.

## Step 1 — Connect Google Drive and Upload Your Dataset

We’ll start by mounting your Google Drive so the notebook can access your image dataset. Make sure your projectile point images are stored in labeled folders — one folder per type (e.g., `clovis`, `folsom`, `dalton`).

You’ll also want to upload a separate “held-out” test folder (`points_heldout`) with the same structure. This test set will not be used during training — we’ll use it later to evaluate how well the model generalizes.

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Step 2 — Set Up Paths and Check Your Images

Let’s tell Colab where to find your dataset. Your images should be organized by class: for example, all Clovis images in one folder, all Folsom images in another, and so on. The folder names become the labels your model will learn to recognize.

We’ll also count how many images are in each class folder. This helps catch issues early — like if you forgot to upload something or if a folder is empty.

**Note**: A quick check like this is often called a **sanity check** — it’s just making sure things look normal before training.

```python
import os, glob
from pathlib import Path

DRIVE = "/content/drive/MyDrive"
DATA_DIR = f"{DRIVE}/points"
TEST_DIR = f"{DRIVE}/points_heldout"

def list_classes(root):
    return sorted([p.name for p in Path(root).glob("*") if p.is_dir()])

def count_images(root):
    counts = {}
    for cls in list_classes(root):
        n = len(glob.glob(str(Path(root)/cls/"*")))
        counts[cls] = n
    return counts

print("Train/Val classes:", list_classes(DATA_DIR))
print("Test classes:", list_classes(TEST_DIR))
print("Train/Val counts:", count_images(DATA_DIR))
print("Test counts:", count_images(TEST_DIR))
```

## Step 3 — Create Our Image Dataset

Now we’ll load the images and turn them into something TensorFlow understands. This step creates a "dataset object" from your folders, resizes the images, labels them based on folder names, and prepares them for training.

We’ll also split the images into **training** and **validation** groups. Training images help the model learn; validation images help us measure how well it's learning (without bias).

**Note**: The **validation set** is a portion of data set aside to check the model’s performance during training. It’s never used to teach the model — just to test it as it learns.

```python
import tensorflow as tf

IMG_SIZE = 224
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE)
```

## Step 4 — Normalize and Prepare the Data

Images are made of pixel values ranging from 0 to 255. Neural networks work better when those numbers are smaller and consistent — so we scale (or “normalize”) them to a range between 0 and 1.

We’ll also add caching and prefetching. These speed up training by making sure the next batch of images is always ready when the model needs it.

**Note**:
- **Normalization** = Scaling values to a standard range, often 0 to 1.
- **Prefetching** = Loading data ahead of time so it’s ready when needed.
- **Caching** = Storing data in memory so it doesn’t have to be reprocessed repeatedly.

```python
AUTOTUNE = tf.data.AUTOTUNE

def prep(ds):
    return ds.map(lambda x, y: (x / 255.0, y)).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

train_ds = prep(train_ds)
val_ds = prep(val_ds)
```

## Step 5 — Set Up the Model Using Transfer Learning

Instead of training a model from scratch (which takes tons of data and time), we’ll use a pre-trained model called **MobileNetV2**. It was trained on millions of images and knows how to recognize general features like edges, shapes, and textures.

We’ll use it as the “backbone” of our model and then add a few new layers on top that are specific to our projectile point types. This approach is called **transfer learning** — and it's a powerful shortcut.

**Note**:
- **Transfer Learning** = Reusing a model trained on one task to jump-start learning on a new task.
- **Frozen layers** = Parts of the model that won’t be updated during training.

```python
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet')

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(train_ds.class_names), activation='softmax')
])

model.summary()
```

## Step 6 — Compile the Model

Now that our model architecture is ready, we need to "compile" it — this just means telling TensorFlow how to train the model.

**Note**:
- **Loss function** = A measure of how wrong the model’s predictions are. Lower is better.
- **Optimizer** = An algorithm that adjusts the model to reduce loss.
- **Metrics** = Stats we track during training (like accuracy).

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
```

## Step 7 — Train the Model

Time to let the model learn! Training is the process where the model looks at your labeled images, makes predictions, compares them to the correct answers, and adjusts itself to do better next time.

**Note**:
- **Epoch** = One complete pass through the training dataset.
- **Validation** = Testing the model on unseen data to check if it’s learning useful patterns or just memorizing.

```python
EPOCHS = 10

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS)
```

## Step 8 — Visualize Model Performance

We’ll plot **accuracy** and **loss** for both the training and validation sets. This lets us see if the model is overfitting (doing well on training but poorly on validation) or underfitting (not doing well on either).


**Note**:
- **Overfitting** = When a model memorizes training data but can’t generalize to new data.
- **Underfitting** = When a model hasn’t learned enough patterns — poor performance overall.

```python
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.legend()
plt.title('Loss')

plt.show()
```

## Step 9 — Evaluate on Held-Out Test Set

Now let’s test it on a completely new set of images it’s never encountered before.

**Note**:
- **Held-out test set** = A final set of data kept separate from all training and validation. Used to test the model at the very end.
- **Evaluate** = Run the model on new data and return metrics like accuracy and loss.

```python
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE)

test_ds = test_ds.map(lambda x, y: (x / 255.0, y))

loss, accuracy = model.evaluate(test_ds)
print("Test accuracy:", accuracy)
```

## Step 10 — Build an Interactive App with Gradio

We’ll use **Gradio**, a tool that turns Python functions into simple web interfaces. You’ll be able to upload an image and instantly get a prediction from your model.

**Note**:
- **Gradio** = A Python library that builds web apps from your code — perfect for testing and demos.
- **Interface** = A user-friendly screen that lets someone interact with a function or model.

```python
import numpy as np
import gradio as gr

class_names = train_ds.class_names

def predict(img):
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.expand_dims(img / 255.0, 0)
    pred = model.predict(img)[0]
    return {class_names[i]: float(pred[i]) for i in range(len(class_names))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Projectile Point Classifier",
    description="Upload an image of a projectile point to see the model’s top predictions.")

demo.launch(share=True)
```

## Step 11 — Save the Trained Model

Saving your model means you can reuse it later without retraining.

**Note**:
- **.keras file** = A file that stores everything about your model — architecture, weights, training config, etc.
- **Serialization** = The process of turning a model into a file that can be saved and reloaded later.

```python
save_path = f"{DRIVE}/projectile_point_model.keras"
model.save(save_path)
print("Model saved to:", save_path)
```

## Step 12 — Load a Saved Model and Use It

Let’s load the model back and test that it works.

**Note**:
- **Deserialization** = Loading a model file and turning it back into a usable model in your code.

```python
from tensorflow import keras

loaded_model = keras.models.load_model(save_path)

img_path = glob.glob(f"{TEST_DIR}/**/*.jpg", recursive=True)[0]
img = tf.keras.preprocessing.image.load_img(img_path)
img_array = tf.keras.preprocessing.image.img_to_array(img)
result = predict(img_array)
print("Predicted:", result)
```loaded_model = keras.models.load_model(save_path)

img_path = glob.glob(f"{TEST_DIR}/**/*.jpg", recursive=True)[0]
img = tf.keras.preprocessing.image.load_img(img_path)
img_array = tf.keras.preprocessing.image.img_to_array(img)
result = predict(img_array)
print("Predicted:", result)
```


## Running This Tutorial *Locally* (Optional, Beginner Friendly)

You don’t have to use Google Colab—you can also run this on your own computer. This section walks you through that from zero, step by step. No assumptions, plenty of explanations.

### What we’re going to do
1) Make sure you have **Python** installed.  
2) Create a **virtual environment** (a private “sandbox” for this project’s packages).  
3) Install everything from `requirements.txt`.  
4) Open the notebook and run it, just like in Colab.

> **Note:** A *virtual environment* (often “venv”) is a private folder that holds the exact Python packages your project needs, without messing with your computer’s global setup. Think of it like giving this project its own toolbox so it doesn’t mix up tools with other projects.

### Step A — Check you have Python
- On Windows: open **Command Prompt** (press Start, type “cmd”).  
- On macOS: open **Terminal** (Spotlight → type “Terminal”).  
- On Linux: open your terminal.

Type:
```bash
python --version
```
You should see something like `Python 3.10.x` (anything 3.8+ is fine).

> **Note:** *CLI* means *Command-Line Interface*—just a window where you type commands instead of clicking buttons. It’s normal in programming and gives you precise control.

If you get an error or a very old version:
- Install Python from https://python.org (choose “Add to PATH” on Windows when asked).  
- Close and reopen your terminal after installing.

### Step B — Create a project folder and move into it
Pick a folder where you want this project to live:
```bash
mkdir projectile-point-classifier
cd projectile-point-classifier
```

### Step C — Create a virtual environment
```bash
# Windows (Command Prompt)
python -m venv .venv

# macOS / Linux
python3 -m venv .venv
```
Activate it (turn it on):
```bash
# Windows (Command Prompt)
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```
Your prompt should now show `(.venv)`—that means you’re working “inside” the sandbox.

### Step D — Get the files into this folder
Download and unzip the repo zip you got from ChatGPT (or clone from GitHub if you push it later). Make sure you can see:
- `README.md`
- `Projectile_Point_Classifier_Colab.ipynb`
- `requirements.txt`
- `CITATION.cff`

### Step E — Install the requirements
With the virtual environment **activated**, run:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
This fetches the tools your project needs, like TensorFlow, scikit-learn, matplotlib, Pillow, and Gradio.

>**Note:** *pip* is Python’s package manager—it downloads and installs libraries for you.

### Step F — Open and run the notebook
You have two easy options:

**Option 1: VS Code (recommended for beginners)**
1) Install **Visual Studio Code** (VS Code) from https://code.visualstudio.com/  
2) Open your project folder in VS Code (`File → Open Folder…`).  
3) Install the **Python** and **Jupyter** extensions (VS Code will suggest them).  
4) Open `Projectile_Point_Classifier_Colab.ipynb` in VS Code and click “Run All” (or run cells one by one).

**Option 2: Jupyter Notebook in your browser**
```bash
pip install jupyter
jupyter notebook
```
Your browser will open at a local address (something like `http://localhost:8888`). Click the notebook file to open it, then run cells top-to-bottom.

>**Note:** *localhost* just means “your own computer.” Jupyter runs a tiny local server to show notebooks in your browser.

### Step G — GPU or no GPU?
Running locally with **no GPU** is completely fine for this tutorial—just slower. If you do have an NVIDIA GPU and want to use it:
- You’ll need the right **CUDA** and **cuDNN** versions (NVIDIA’s libraries for GPU math) that match your TensorFlow version.  
- This setup is more advanced; if it sounds intimidating, skip it. CPU training works for our small dataset.

>**Note:** *CUDA/cuDNN* are NVIDIA’s special toolkits that let TensorFlow use the GPU. Setting them up is a one-time chore; many beginners learn it later.

### Common questions (and answers)
**Q: My terminal says “command not found: python”.**  
A: On macOS/Linux you might need `python3` instead of `python`. Try:
```bash
python3 --version
```

**Q: How do I “deactivate” the virtual environment?**  
A: Just type:
```bash
deactivate
```

**Q: The notebook can’t find my images.**  
A: Check your paths at the top of the notebook (`DATA_DIR` and `TEST_DIR`). Make sure folder names match exactly and that each class has its own folder with images inside.

**Q: TensorFlow install was slow or showed lots of messages. Is that normal?**  
A: Yes. TensorFlow is a big library. As long as the install completes and your imports work (`import tensorflow as tf`), you’re good.

**Q: I got an “Out of Memory” error.**  
A: Try lowering the batch size in the notebook (e.g., `BATCH = 16`), close other heavy apps, or run in Colab where GPUs are available.

---
