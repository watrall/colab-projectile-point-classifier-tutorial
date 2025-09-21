# Projectile Point Classifier — Google Colab  (Beginner Friendly)

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
