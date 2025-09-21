# Projectile Point Classifier — Google Colab Edition (Beginner Friendly)

In this tutorial, you’ll teach a computer to recognize different types of projectile points from photos. We’ll use **Google Colab** (a free tool that runs code in your browser) and **TensorFlow** (a toolbox for teaching computers to spot patterns in data).  

The workflow looks like this:  
1. Load your images from Google Drive.  
2. Build and train a model.  
3. Evaluate how well it works.  
4. Save it so you don’t lose your work.  
5. (Optional) Create a tiny demo interface so others can test it too.  

**Before you start:**  
Organize your images in Google Drive using “one folder per class.”  

Example structure (names are examples, yours may differ):  

- `MyDrive/points/Clovis/`  
- `MyDrive/points/Folsom/`  
- `MyDrive/points/Dalton/`  
- `MyDrive/points/Kirk/`  

That folder is for training and validation.  

Also create a **held-out test set** you’ll never use during training:  

- `MyDrive/points_heldout/Clovis/`  
- `MyDrive/points_heldout/Folsom/`  
- `MyDrive/points_heldout/Dalton/`  
- `MyDrive/points_heldout/Kirk/`  

This test set is how we honestly check what the model learned.  

> 💡 **Note:** Think of the held-out test set like giving a student a brand-new exam. If they do well on problems they’ve never seen, you know they truly understand.  

---

## Step 0 — Check our setup

When you launch a Google Colab notebook, you’re basically borrowing a temporary computer hosted in Google’s cloud. Sometimes Colab gives you only a CPU (the regular kind of processor found in laptops and desktops), and other times you’re lucky enough to get a GPU (a graphics card). GPUs were originally designed for handling 3D graphics in games, but because they can perform thousands of calculations in parallel, they’re now the go-to hardware for training AI models. If you get one, your training will run noticeably faster. If not, don’t worry — the tutorial still works, it’ll just take more time.  

We’ll also install **Gradio** right away. Gradio is a neat Python library that lets us wrap our model in a simple web interface. Later, once your model is trained, you’ll be able to upload an image and instantly see predictions in a little app without writing any web code. Think of it as giving your model a “front door” that anyone can walk up to and use.  

**What:** Detect GPU and install Gradio.  
**Why:** GPU makes training faster; Gradio makes sharing/testing easier.  
**Validate:** You should see TensorFlow’s version printed, and either a GPU listed or an empty list (no GPU is fine, just slower).  

```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

!pip -q install gradio==4.44.0
```

---

## Step 1 — Connect to Google Drive

Colab notebooks are temporary: when you close your session, files in `/content` vanish. To avoid losing your dataset or trained model, we connect to Google Drive. This is like plugging in an external hard drive to your Colab computer. Everything in Drive persists, so it’s the perfect place to store both your input images and the finished model.  

When you run the command below, a little popup will ask for permission. Once granted, Colab will create a folder called `/content/drive` that acts as the bridge between your notebook and your Google Drive. From there, you can read and write files just as if they were on your computer.  

**What:** Mount Google Drive.  
**Why:** Lets Colab read your images and save files that survive after the session ends.  
**Validate:** You’ll be asked for permission. After approving, you should see `/content/drive` created.  

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Step 2 — Point to our folders and check our images

Before we dive into machine learning, we need to double-check that our data is set up correctly. Machine learning models are picky: if your folders are misnamed or if one class has way fewer images than another, training results will suffer. Each folder name becomes the **label** the model learns, so consistency is crucial.  

We’ll also count how many images are in each folder. This is the programming equivalent of making sure your lab bench is stocked with the right chemicals before starting an experiment. If something looks off now, it’s much easier to fix before we hit “train.”  

**What:** Define dataset paths and print class names with image counts.  
**Why:** Ensures Colab can find your images and that every class has at least a few pictures before training.  
**Validate:** The printed output should list your class names and show a non-zero number of images for each.  

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

print("Training/Validation classes:", list_classes(DATA_DIR))
print("Held-out test classes:", list_classes(TEST_DIR))
print("Training/Validation counts:", count_images(DATA_DIR))
print("Held-out test counts:", count_images(TEST_DIR))
```

## Step 3 — Create Our Image Dataset

Now we’ll load the images and turn them into something TensorFlow understands. This step creates a "dataset object" from your folders, resizes the images, labels them based on folder names, and prepares them for training.

We’ll also split the images into **training** and **validation** groups. Training images help the model learn; validation images help us measure how well it's learning (without bias).

### What this step does:
- Loads images from your folders and resizes them to 224×224 pixels.
- Automatically assigns labels based on folder names.
- Splits the data into 80% for training and 20% for validation.

> 💬 **Jargon explained:**  
> **Validation set** = A portion of data set aside to check the model’s performance during training. It’s never used to teach the model — just to test it as it learns.

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


## Step 4 — Normalize and Prepare the Data

Images are made of pixel values ranging from 0 to 255. Neural networks work better when those numbers are smaller and consistent — so we scale (or “normalize”) them to a range between 0 and 1.

We’ll also add caching and prefetching. These speed up training by making sure the next batch of images is always ready when the model needs it.

### What this step does:
- Scales all pixel values to be between 0 and 1.
- Enables performance boosts using prefetching and caching.

> 💬 **Jargon explained:**  
> **Normalization** = Scaling values to a standard range, often 0 to 1.  
> **Prefetching** = Loading data ahead of time so it’s ready when needed.  
> **Caching** = Storing data in memory so it doesn’t have to be reprocessed repeatedly.

```python
AUTOTUNE = tf.data.AUTOTUNE

def prep(ds):
    return ds.map(lambda x, y: (x / 255.0, y)).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

train_ds = prep(train_ds)
val_ds = prep(val_ds)

## Step 5 — Set Up the Model Using Transfer Learning

Instead of training a model from scratch (which takes tons of data and time), we’ll use a pre-trained model called **MobileNetV2**. It was trained on millions of images and knows how to recognize general features like edges, shapes, and textures.

We’ll use it as the “backbone” of our model and then add a few new layers on top that are specific to our projectile point types. This approach is called **transfer learning** — and it's a powerful shortcut.

### What this step does:
- Loads a pre-trained MobileNetV2 model.
- Freezes its layers so we don’t accidentally retrain them.
- Adds new layers for our specific classification task.

> 💬 **Jargon explained:**  
> **Transfer Learning** = Reusing a model trained on one task to jump-start learning on a new task.  
> **Frozen layers** = Parts of the model that won’t be updated during training.

```python
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet')

base_model.trainable = False  # freeze the base model

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(train_ds.class_names), activation='softmax')
])

model.summary()

## Step 6 — Compile the Model

Now that our model architecture is ready, we need to "compile" it — this just means telling TensorFlow how to train the model. We’ll specify:

- **Loss function**: How the model knows it’s making mistakes.
- **Optimizer**: How it updates itself to make fewer mistakes.
- **Metrics**: What we want to track during training (like accuracy).

We’ll use common defaults that work well for multi-class image classification tasks.

### What this step does:
- Defines how the model learns and what it pays attention to.
- Prepares it to start training.

> 💬 **Jargon explained:**  
> **Loss function** = A measure of how wrong the model’s predictions are. Lower is better.  
> **Optimizer** = An algorithm that adjusts the model to reduce loss.  
> **Metrics** = Stats we track during training (like accuracy).

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

## Step 7 — Train the Model

Time to let the model learn! Training is the process where the model looks at your labeled images, makes predictions, compares them to the correct answers, and adjusts itself to do better next time. This happens over multiple passes through the data, called **epochs**.

We’ll store the training progress in a variable called `history`, which we’ll use later to make graphs of how well the model learned.

### What this step does:
- Trains the model for 10 full passes through the dataset (10 epochs).
- Tracks loss and accuracy on both training and validation sets.

> 💬 **Jargon explained:**  
> **Epoch** = One complete pass through the training dataset.  
> **Validation** = Testing the model on unseen data to check if it’s learning useful patterns or just memorizing.

```python
EPOCHS = 10

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS)

## Step 8 — Visualize Model Performance

After training, we want to know: how well did it go? Was the model improving over time, or did it start to struggle? Visualizing the training process helps us answer these questions.

We’ll plot **accuracy** and **loss** for both the training and validation sets. This lets us see if the model is overfitting (doing well on training but poorly on validation) or underfitting (not doing well on either).

### What this step does:
- Creates simple line plots showing how accuracy and loss changed over time.
- Helps us decide whether the model trained well or needs tweaking.

> 💬 **Jargon explained:**  
> **Overfitting** = When a model memorizes training data but can’t generalize to new data.  
> **Underfitting** = When a model hasn’t learned enough patterns — poor performance overall.

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

## Step 9 — Evaluate on Held-Out Test Set

So far, we’ve only tested the model on validation data — data it has “seen” during training. Now let’s test it on a completely new set of images it’s never encountered before. This is our **held-out test set** — it gives us the clearest picture of how well the model will work in the real world.

We’ll use the same image loading function we used earlier, and then evaluate the model’s accuracy on this fresh data.

### What this step does:
- Loads the test images.
- Measures accuracy using completely unseen data.

> 💬 **Jargon explained:**  
> **Held-out test set** = A final set of data kept separate from all training and validation. Used to test the model at the very end.  
> **Evaluate** = Run the model on new data and return metrics like accuracy and loss.

```python
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE)

test_ds = test_ds.map(lambda x, y: (x / 255.0, y))

loss, accuracy = model.evaluate(test_ds)
print("Test accuracy:", accuracy)

## Step 10 — Build an Interactive App with Gradio

Let’s make this model easy to use — even for someone who doesn’t know Python or machine learning. We’ll use **Gradio**, a tool that turns Python functions into simple web interfaces. You’ll be able to upload an image and instantly get a prediction from your model.

This is a great way to demo your project, share it with others, or even build a tool for researchers or museum staff.

### What this step does:
- Defines a prediction function that takes an image and returns a class label.
- Uses Gradio to wrap that function in a friendly interface.
- Launches a live demo that you can use or share.

> 💬 **Jargon explained:**  
> **Gradio** = A Python library that builds web apps from your code — perfect for testing and demos.  
> **Interface** = A user-friendly screen that lets someone interact with a function or model.

```python
import numpy as np
import gradio as gr

class_names = train_ds.class_names

def predict(img):
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.expand_dims(img / 255.0, 0)  # normalize and batch
    pred = model.predict(img)[0]
    return {class_names[i]: float(pred[i]) for i in range(len(class_names))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Projectile Point Classifier",
    description="Upload an image of a projectile point to see the model’s top predictions.")

demo.launch(share=True)

## Step 11 — Save the Trained Model

If you’re happy with your model, you’ll want to save it. That way, you don’t have to retrain from scratch every time — you can load it later and immediately start making predictions or build new tools around it.

We’ll save it in your Google Drive so it’s always available, even after this Colab session ends.

### What this step does:
- Saves the model in TensorFlow’s `.keras` format.
- Stores it inside your connected Google Drive.

> 💬 **Jargon explained:**  
> **.keras file** = A file that stores everything about your model — architecture, weights, training config, etc.  
> **Serialization** = The process of turning a model into a file that can be saved and reloaded later.

```python
save_path = f"{DRIVE}/projectile_point_model.keras"
model.save(save_path)
print("Model saved to:", save_path)

## Step 12 — Load a Saved Model and Use It

If you've previously saved your model, you can load it back into memory and start using it right away — no need to retrain. This is useful if you're returning to the project later or want to use the model in another notebook or app.

We'll also test that it works by making a prediction, just like before.

### What this step does:
- Loads the `.keras` model file from Google Drive.
- Uses it to make a prediction on a new image.

> 💬 **Jargon explained:**  
> **Deserialization** = Loading a model file and turning it back into a usable model in your code.

```python
from tensorflow import keras

# Load the saved model
loaded_model = keras.models.load_model(save_path)

# Try a test prediction (re-using the Gradio function)
img_path = glob.glob(f"{TEST_DIR}/**/*.jpg", recursive=True)[0]
img = tf.keras.preprocessing.image.load_img(img_path)
img_array = tf.keras.preprocessing.image.img_to_array(img)
result = predict(img_array)
print("Predicted:", result)


## Running This Tutorial *Locally* (Optional, Beginner Friendly)

You don’t have to use Google Colab—you can also run this on your own computer. This section walks you through that from zero, step by step. No assumptions, plenty of explanations.

### What we’re going to do
1) Make sure you have **Python** installed.  
2) Create a **virtual environment** (a private “sandbox” for this project’s packages).  
3) Install everything from `requirements.txt`.  
4) Open the notebook and run it, just like in Colab.

> 💡 **Note (Jargon):** A *virtual environment* (often “venv”) is a private folder that holds the exact Python packages your project needs, without messing with your computer’s global setup. Think of it like giving this project its own toolbox so it doesn’t mix up tools with other projects.

### Step A — Check you have Python
- On Windows: open **Command Prompt** (press Start, type “cmd”).  
- On macOS: open **Terminal** (Spotlight → type “Terminal”).  
- On Linux: open your terminal.

Type:
```bash
python --version
```
You should see something like `Python 3.10.x` (anything 3.8+ is fine).

> 💡 **Note (Jargon):** *CLI* means *Command-Line Interface*—just a window where you type commands instead of clicking buttons. It’s normal in programming and gives you precise control.

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

> 💡 **Note (Jargon):** *pip* is Python’s package manager—it downloads and installs libraries for you.

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

> 💡 **Note (Jargon):** *localhost* just means “your own computer.” Jupyter runs a tiny local server to show notebooks in your browser.

### Step G — GPU or no GPU?
Running locally with **no GPU** is completely fine for this tutorial—just slower. If you do have an NVIDIA GPU and want to use it:
- You’ll need the right **CUDA** and **cuDNN** versions (NVIDIA’s libraries for GPU math) that match your TensorFlow version.  
- This setup is more advanced; if it sounds intimidating, skip it. CPU training works for our small dataset.

> 💡 **Note (Jargon):** *CUDA/cuDNN* are NVIDIA’s special toolkits that let TensorFlow use the GPU. Setting them up is a one-time chore; many beginners learn it later.

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
