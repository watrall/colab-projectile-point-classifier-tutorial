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

## What You’ll Need

- A Google account (required for using Google Colab and Google Drive).  
- Access to Google Colab (free to use in your browser with a Google account).  
- A small set of rights-cleared projectile point images (start with around 50 images).  
- Image quality matters. Clear, well-lit, in-focus images make it easier for the model to learn distinguishing patterns such as fluting, base shape, or shoulder angles.  
- Include variation: different lighting, backgrounds, and orientations. A model trained only on clean catalog shots may struggle with field photos.  
- More images are always better. Large and varied datasets usually produce more accurate models, but even 50 images will let you complete this tutorial successfully.  
- No installations required. Everything runs directly in the browser through Colab, and your files will be stored in Google Drive.  

## Understanding Computer Vision

At its core, computer vision is about teaching a computer to recognize patterns in images. It does not “understand” archaeology. Instead, it identifies statistical fingerprints such as edges, shapes, contours, colors, and textures that correlate with the labels you provide. For example, if you show it dozens of labeled Clovis points, it begins to notice what they have in common and how they differ from Dalton or Folsom points.

A helpful analogy is training a graduate student. At first, they may not be able to tell two types apart. After studying many labeled examples, they start to notice features such as flute length, shoulder angle, or base shape. They do not have innate knowledge; they learn through exposure and feedback. A computer vision model works in the same way.

This distinction is important for building confidence. The system is not “deciding” what an object is. It is calculating probabilities based on patterns it has seen. For example, a result like “Clovis (0.85)” means the model estimates there is an 85% chance the image matches the profile of a Clovis point. You, as the researcher, still hold interpretive authority to accept, question, or refine the result.

Finally, performance depends heavily on the quality and diversity of your dataset. If your images vary in lighting, backgrounds, materials, and angles, the model becomes more robust. If your dataset only includes pristine catalog photographs, the model may struggle with field photos. The key is to use AI critically, treating it as a transparent assistant that you can test, evaluate, and trust.

## Why TensorFlow in Colab

When people hear about AI, they often think of it as a “black box” that is complicated or inaccessible. The good news is that TensorFlow, paired with Google Colab, makes machine learning transparent and approachable. Colab provides a free cloud-based notebook where you can write and run Python code in your browser. TensorFlow is the library that actually builds and trains the neural network.

The key strength of TensorFlow is flexibility. It allows you to design and train a model from the ground up. Unlike pre-built services that may only recognize everyday objects like cars or coffee cups, TensorFlow lets you define your own categories such as Clovis, Dalton, or Folsom and train the model specifically on your dataset. This puts control in your hands as a researcher or educator.

Colab complements TensorFlow by removing technical barriers. There is no need to install software or manage hardware. Everything runs in the cloud, and Google even provides free GPU acceleration to make training faster. Together, Colab and TensorFlow give you the power to experiment with machine learning in a way that is beginner-friendly but still powerful enough for research and teaching.

Most importantly, TensorFlow in Colab supports openness and transparency. You can see the exact code that trains your model, examine performance metrics like accuracy and loss, and refine your dataset step by step. This openness makes it easier to think of AI as a scientific tool that you can evaluate and critique, rather than a mysterious system that produces unquestionable results.

## Understanding Bias in Machine Learning

Like any tool, machine learning is shaped by the data it is trained on. If the dataset is unbalanced or limited, the model will reflect those limitations. This is what we mean when we talk about bias in machine learning.

There are two main types of bias to keep in mind:

- **Dataset bias**: If you include many images of Dalton points but only a handful of Clovis points, the model will get much better at recognizing Daltons than Clovis. The system learns what you show it.  
- **Model bias**: TensorFlow models are built on general training techniques that were originally optimized on large image datasets. This background influences which visual patterns the model finds easier to learn.

Being aware of bias is important because it helps you interpret results critically. A model’s prediction is not a statement of fact. It is a reflection of patterns in the data it has seen. By evaluating your model’s performance and adjusting your dataset, you can reduce bias and make the results more reliable.

**Note**: In archaeology and heritage work, bias also intersects with ethics. Collections often reflect colonial histories, selective preservation, or curatorial choices. Always approach training data with transparency and responsibility, and avoid presenting the output of an AI model as objective truth.

## Ethics, Rights, Permissions, and Cultural Property

When working with museum collections and archaeological records, it is important to remember that data is never “just data.” These are records of objects, histories, and in some cases, living cultural knowledge.

- **Copyright and permissions**: Make sure you have the right to use the images you are training your model on.  
- **Cultural patrimony**: Many artifacts in museums are part of the cultural patrimony of descendant or source communities. Treat them with respect and, where appropriate, consultation.  
- **Colonial histories**: Be transparent about the colonial contexts that shaped many collections.  
- **Privacy and sensitivity**: Be cautious about including site locations, collector names, or donor histories.  
- **Bias and framing**: Remember that typologies are scholarly constructs, not objective truths.  

This tutorial is designed as a teaching and research tool. It should be used responsibly, in collaboration with institutions and communities, and never as a substitute for expertise, consultation, or ethical stewardship.

### Responsible Sharing

Once you have built a working classifier, think carefully about how you share it. Transparency prevents misuse and builds trust among scholars, institutions, and communities.  

- Clearly describe the size, sources, and limitations of your dataset.  
- State that the classifier is an educational and research tool, not a replacement for expert analysis.  
- Avoid uploading sensitive images or sharing site-specific information that could put collections or heritage at risk.  
- Document the scope of your model so users understand what it can and cannot do.  

By being upfront about these limitations and responsibilities, you help ensure that machine learning is used as a constructive and ethical assistant in archaeology and heritage research.



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
