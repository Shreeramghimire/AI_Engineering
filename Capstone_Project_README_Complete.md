# Capstone Project: Satellite Image Classification for Fertilizer Sales Forecasting
### Agricultural vs. Non-Agricultural: CNNs and Vision Transformers, built in both Keras and PyTorch

> **About this repository:** a capstone project on **geospatial AI and binary image classification**. **Part 1** covers data handling (loading methods, efficient pipelines, augmentation). **Part 2** builds the same CNN in Keras and PyTorch. **Part 3** compares the two fairly. **Part 4** adds Vision Transformers and transfer learning. **Part 5** integrates and evaluates the CNN-ViT hybrids. Every part is tied to a hands-on lab notebook, and the document is written both as a **learning reference** (to memorize the key ideas) and as a **guide for others** to follow.

## Table of Contents

- **[Project at a Glance](#project-at-a-glance)**
- **[Results at a Glance](#results-at-a-glance)**
- **[Why This Project?](#why-this-project)**
- **[Project Goals](#project-goals)**
- **[How the Project Is Organized](#how-the-project-is-organized)**
- **[The Dataset](#the-dataset)**
- **[Conventions Used in All Models](#conventions-used-in-all-models)**
- **[Part 1: Data Handling](#part-1-data-handling)**
  - [1.1 Data Loading Methods](#11-data-loading-methods)
  - [1.2 Building Efficient Data Pipelines](#12-building-efficient-data-pipelines)
  - [1.3 Data Augmentation in PyTorch](#13-data-augmentation-in-pytorch)
  - [1.4 Lessons from the Labs](#14-lessons-from-the-labs)
  - [1.5 Part 1 Summary](#15-part-1-summary)
- **[Part 2: CNN Classifier in Keras and PyTorch](#part-2-cnn-classifier-in-keras-and-pytorch)**
  - [2.1 Problem Framing](#21-problem-framing)
  - [2.2 Hyperparameters](#22-hyperparameters)
  - [2.3 Data Pipeline and Augmentation in the CNN Labs](#23-data-pipeline-and-augmentation-in-the-cnn-labs)
  - [2.4 Model Architecture](#24-model-architecture)
  - [2.5 Training Workflow](#25-training-workflow)
  - [2.6 What We Learned from Building the Same CNN Twice](#26-what-we-learned-from-building-the-same-cnn-twice)
- **[Part 3: Comparative Analysis of Keras and PyTorch Models](#part-3-comparative-analysis-of-keras-and-pytorch-models)**
  - [3.1 Why Compare, and What Can Differ](#31-why-compare-and-what-can-differ)
  - [3.2 What Must Be Standardized](#32-what-must-be-standardized)
  - [3.3 Evaluation Metrics](#33-evaluation-metrics)
  - [3.4 Results of the Lab Comparison](#34-results-of-the-lab-comparison)
  - [3.5 Why These Numbers Are Not Test Scores](#35-why-these-numbers-are-not-test-scores)
  - [3.6 How to Interpret a Comparison](#36-how-to-interpret-a-comparison)
  - [3.7 Practical Recommendations: Keras or PyTorch?](#37-practical-recommendations-keras-or-pytorch)
  - [3.8 Part 3 Summary](#38-part-3-summary)
- **[Part 4: Vision Transformers and Transfer Learning](#part-4-vision-transformers-and-transfer-learning)**
  - [4.1 From CNNs to Transformers](#41-from-cnns-to-transformers)
  - [4.2 ViT Architecture](#42-vit-architecture)
  - [4.3 The Hybrid CNN + ViT](#43-the-hybrid-cnn--vit)
  - [4.4 Transfer Learning](#44-transfer-learning)
  - [4.5 Implementation in Keras (Lab 07)](#45-implementation-in-keras-lab-07)
  - [4.6 Implementation in PyTorch (Lab 08)](#46-implementation-in-pytorch-lab-08)
  - [4.7 Keras vs. PyTorch for Vision Transformers](#47-keras-vs-pytorch-for-vision-transformers)
  - [4.8 Design Issues Worth Knowing](#48-design-issues-worth-knowing)
  - [4.9 Part 4 Summary](#49-part-4-summary)
- **[Part 5: Integration and Evaluation of the CNN-ViT Hybrids](#part-5-integration-and-evaluation-of-the-cnn-vit-hybrids)**
  - [5.1 The Workflow](#51-the-workflow)
  - [5.2 Results](#52-results)
  - [5.3 Speed and Resource Usage](#53-speed-and-resource-usage)
  - [5.4 Part 5 Summary](#54-part-5-summary)
- **[Project Conclusions](#project-conclusions)**
  - [All Four Models Side by Side](#all-four-models-side-by-side)
  - [Key Findings](#key-findings)
  - [Pitfalls Found Along the Way](#pitfalls-found-along-the-way)
  - [Limitations](#limitations)
  - [Next Steps](#next-steps)
- **[Appendix A: Data and Setup](#appendix-a-data-and-setup)**
- **[Appendix B: Reproducibility Checklist](#appendix-b-reproducibility-checklist)**
- **[Appendix C: Suggested Presentation Outline](#appendix-c-suggested-presentation-outline)**
- **[Appendix D: Acknowledgements](#appendix-d-acknowledgements)**

---

## Project at a Glance

| Item | Description |
|---|---|
| **Problem** | Binary image classification: **agricultural** vs. **non-agricultural** land from satellite images |
| **Business use** | Estimate agricultural land to support **fertilizer sales forecasting** |
| **Data** | `images_dataSAT`: **6,000** RGB images of **64×64** pixels, **3,000 per class**, in two folders (`class_0_non_agri`, `class_1_agri`). File names follow the Sentinel-2 Level-2A naming pattern. Source and license: to be confirmed (Appendix A) |
| **Models** | Four models: a **CNN** in Keras, a **CNN** in PyTorch, and a **hybrid CNN + Vision Transformer (ViT)** in each framework |
| **Skills practiced** | Data pipelines, data augmentation, CNNs, vision transformers, transfer learning, hyperparameter tuning, evaluation (accuracy, precision, recall, F1, ROC-AUC, confusion matrix), framework comparison, geospatial AI |
| **Hands-on practice** | Nine lab notebooks (listed below), one for each step of the project |
| **Deliverables** | Notebooks, reusable code, saved results, and this README |

## Results at a Glance

All four models reach 98.5% or higher on every metric (the lowest value is the Keras CNN's recall of 0.9850). The numbers below come from the lab notebooks, where **all 6,000 images were evaluated, including the images each model was trained on**. They show that the models learned the task, but they are **not** held-out test scores (see Section 3.5).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| CNN, Keras | 0.9925 | 1.0000 | 0.9850 | 0.9924 | 1.0000 |
| CNN, PyTorch | 0.9988 | 0.9983 | 0.9993 | 0.9988 | 1.0000 |
| CNN + ViT hybrid, Keras | 0.9958 | 0.9990 | 0.9927 | 0.9958 | 0.9998 |
| CNN + ViT hybrid, PyTorch | 0.9990 | 0.9990 | 0.9990 | 0.9990 | 1.0000 |

Precision, recall, and F1 are for the positive class (`agricultural`).

## Why This Project?

**The business problem.** Fertilizer sellers need to know where agricultural land is, and how much of it, to forecast demand. Satellite imagery covers large regions cheaply, but it is far too large to label by hand. A classifier that labels each image tile as agricultural or non-agricultural turns raw imagery into an estimate of farmland that a sales-forecasting model can use.

> **Scope:** the forecasting model itself is outside this repository. This project delivers the **classifier** that supplies its input.

**Why precision comes first.** Which error is worse depends on the use case:

| Error | What happens | Effect on the forecast |
|---|---|---|
| **False positive** (non-agricultural tile labeled agricultural) | Farmland is over-counted | Fertilizer demand is **over-forecast** |
| **False negative** (agricultural tile labeled non-agricultural) | Farmland is missed | Fertilizer demand is **under-forecast** |

We report **precision** first, and always alongside recall, F1, and ROC-AUC, so the trade-off stays visible.

## Project Goals

| # | Goal | Why it matters |
|---|---|---|
| 1 | **Validate the models** | Two independent implementations should reach comparable results. If they disagree, one of them probably has a bug, so each model cross-checks the other. |
| 2 | **Understand the distinct workflows** | Knowing how Keras and PyTorch differ (data loading, augmentation, training loop, evaluation) gives us wider career flexibility. |
| 3 | **Compare fairly** | Identify what must be standardized (data, model design, evaluation) so that any difference can be attributed to the framework. |
| 4 | **Go beyond CNNs** | Add a Vision Transformer on top of the CNN (a hybrid model) and use transfer learning, then evaluate it the same way. |

## How the Project Is Organized

| Part | Topic | Question it answers | Hands-on lab notebooks |
|---|---|---|---|
| **Part 1** | Data handling | How do we load, batch, and augment images efficiently? | 01, 02, 03 |
| **Part 2** | CNN classifier in Keras and PyTorch | How do we build, train, and tune the same CNN in two frameworks? | 04, 05 |
| **Part 3** | Comparative analysis of Keras and PyTorch models | What must be standardized for a fair comparison, and which metrics do we use? | 06 |
| **Part 4** | Vision Transformers and transfer learning | How do we add global context to a CNN with a transformer? | 07, 08 |
| **Part 5** | Integration and evaluation of the CNN-ViT hybrids | How do the hybrid models compare with each other and with the CNNs? | 09 |

```text
satellite-agri-classifier/
├── README.md
├── requirements.txt
├── data/                       # NOT committed; see Appendix A for download steps
├── notebooks/
│   ├── 01_compare_memory_vs_generator_loading.ipynb
│   ├── 02_data_loading_augmentation_keras.ipynb
│   ├── 03_data_loading_augmentation_pytorch.ipynb
│   ├── 04_cnn_keras_classifier.ipynb
│   ├── 05_cnn_pytorch_classifier.ipynb
│   ├── 06_comparative_analysis_keras_vs_pytorch.ipynb
│   ├── 07_cnn_vit_keras.ipynb
│   ├── 08_cnn_vit_pytorch.ipynb
│   └── 09_cnn_vit_integration_evaluation.ipynb
├── src/
│   ├── data_pipeline.py        # generators, Dataset classes, tf.data helpers
│   ├── models_keras.py
│   ├── models_pytorch.py
│   └── evaluate.py             # ONE shared evaluation script for all models
└── results/
    ├── models/                 # saved weights (not committed if large)
    ├── predictions/            # y_true and probabilities for each model
    └── figures/                # confusion matrices, ROC curves, augmentation samples
```

## The Dataset

| Property | Value |
|---|---|
| **Name** | `images_dataSAT` |
| **Size** | 6,000 images (about 20 MB archive) |
| **Image format** | JPEG, RGB, **64×64** pixels |
| **Classes** | `class_0_non_agri` (non-cultivable land) and `class_1_agri` (cultivable land) |
| **Class balance** | 3,000 per class (perfectly balanced) |
| **Folder layout** | `images_dataSAT/class_0_non_agri/*.jpg` and `images_dataSAT/class_1_agri/*.jpg` |
| **Split used in the labs** | 80% training (4,800) and 20% validation (1,200). **No separate test set is held out** |

Because the classes are balanced, accuracy is not misleading here, but we still report precision, recall, F1, and ROC-AUC so that the results carry over to imbalanced data.

## Conventions Used in All Models

| Convention | Value |
|---|---|
| **Positive class** | `agricultural` = **1**, `non_agricultural` = 0. The `class_0_` and `class_1_` folder prefixes make this automatic in both frameworks |
| **Image size** | 64×64×3 |
| **Pixel scale** | Float values in `[0, 1]` before any normalization |
| **Decision threshold** | 0.5 |
| **Evaluation** | One shared script (`src/evaluate.py`) applied to saved predictions |

---

## Part 1: Data Handling

> **Hands-on labs for this part:** notebooks `01` (memory-based vs. generator-based loading), `02` (Keras loading and augmentation), and `03` (PyTorch loading and augmentation).

Good models start with good data pipelines. This part covers **how images get from disk into the model** (loading), **how to build the pipeline efficiently** (custom generators vs. built-in utilities), and **how to make the training data more varied** (data augmentation in PyTorch).

```text
Files on disk → decode → resize → augment (training only) → batch → prefetch → model
```

### 1.1 Data Loading Methods

There are two basic ways to get images into a model.

#### Bulk loading (memory-based)

**Idea:** read **all** images into one big array in RAM, then slice batches from it.

```python
import numpy as np
from PIL import Image

def load_all(paths, size=(64, 64)):
    # uint8 array of shape (N, H, W, 3), held entirely in memory
    return np.stack([np.asarray(Image.open(p).convert("RGB").resize(size)) for p in paths])

X_train = load_all(train_paths)
model.fit(X_train, y_train, batch_size=32, epochs=30, validation_data=(X_val, y_val))
```

#### Sequential loading (generator-based)

**Idea:** read images **batch by batch, on demand**. Only the current batch lives in memory.

```python
import numpy as np
from PIL import Image

def batch_generator(paths, labels, batch_size=32, size=(64, 64), shuffle=True):
    idx = np.arange(len(paths))
    while True:                                   # endless: Keras stops after steps_per_epoch
        if shuffle:
            np.random.shuffle(idx)
        for start in range(0, len(idx), batch_size):
            batch = idx[start:start + batch_size]
            x = np.stack([np.asarray(Image.open(paths[i]).convert("RGB").resize(size))
                          for i in batch]).astype("float32") / 255.0
            yield x, labels[batch]

steps = int(np.ceil(len(train_paths) / 32))
model.fit(batch_generator(train_paths, y_train), steps_per_epoch=steps, epochs=30)
```

#### Memory-based vs. generator-based

| Aspect | Bulk (memory-based) | Sequential (generator-based) |
|---|---|---|
| **Memory use** | Whole dataset in RAM | One batch (plus a small buffer) |
| **Startup time** | Slow (everything is read first) | Fast (starts immediately) |
| **Speed per epoch** | Fast (no disk reads after loading) | Slower, unless loading overlaps with training |
| **Shuffling** | Easy and global | Needs care (shuffle the index list each epoch) |
| **Augmentation** | Applied to arrays, or pre-computed copies | Applied on the fly, a new random version every epoch |
| **Scales to large data** | No | Yes |
| **Code complexity** | Very simple | A little more code |
| **Best for** | Small datasets that fit comfortably in RAM | Large datasets, or when augmenting on the fly |

**How much memory does bulk loading need?**

$$\text{bytes} = N \times H \times W \times C \times \text{bytes per value}$$

| Example (illustrative sizes) | `uint8` (1 byte) | `float32` (4 bytes) |
|---|---:|---:|
| **Our dataset:** 6,000 images of 64×64×3 | ≈ 74 MB | ≈ 295 MB |
| Illustration: 100,000 images of 256×256×3 | ≈ 19.7 GB | ≈ 78.6 GB |

> **Tip:** keep images as `uint8` in memory and convert to `float32` one batch at a time.

**What we measured in the lab.** Reading all 3,000 non-agricultural images into one NumPy array took about **1.3 seconds** and about **37 MB** (as `uint8`). For a dataset this small, bulk loading is perfectly fine. Generator-based loading becomes essential when the data no longer fits comfortably in memory, or when we want to augment images on the fly. The lab built the sequential version by listing and sorting the file paths first (`os.listdir`), and opening each image only when it was needed.

### 1.2 Building Efficient Data Pipelines

An efficient pipeline keeps the GPU busy: while the model trains on batch *k*, the next batches are already being loaded and prepared. We build and compare two approaches.

| | **Approach A: custom data generators** | **Approach B: built-in utilities** |
|---|---|---|
| **Keras** | `keras.utils.PyDataset` (or a Python generator) | `keras.utils.image_dataset_from_directory` + `tf.data` |
| **PyTorch** | `torch.utils.data.Dataset` + `DataLoader` | `torchvision.datasets.ImageFolder` + `DataLoader` |

#### Approach A: custom data generators

**Keras: a class-based generator (`PyDataset`).** It knows its own length and is safe for parallel loading, which a plain `yield` generator is not.

```python
import numpy as np
import keras
from PIL import Image

class SatelliteSequence(keras.utils.PyDataset):
    def __init__(self, paths, labels, batch_size=32, image_size=(64, 64), shuffle=True, **kwargs):
        super().__init__(**kwargs)               # kwargs: workers, use_multiprocessing, max_queue_size
        self.paths, self.labels = np.array(paths), np.array(labels)
        self.batch_size, self.image_size, self.shuffle = batch_size, image_size, shuffle
        self.on_epoch_end()

    def __len__(self):                           # number of batches per epoch
        return int(np.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, idx):                  # one batch, loaded from disk on demand
        batch = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        x = np.stack([np.asarray(Image.open(self.paths[i]).convert("RGB").resize(self.image_size))
                      for i in batch]).astype("float32")
        return x, self.labels[batch].astype("float32")

    def on_epoch_end(self):                      # reshuffle after every epoch
        self.indices = np.arange(len(self.paths))
        if self.shuffle:
            np.random.shuffle(self.indices)
```

**PyTorch: a custom `Dataset` and a `DataLoader`.** This is also the pattern used for augmentation in Section 1.3.

```python
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class SatelliteDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)

train_loader = DataLoader(SatelliteDataset(train_paths, y_train, train_tf),
                          batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
```

#### Approach B: built-in utilities

**Keras: `image_dataset_from_directory`.** It reads images from folders, resizes them, batches them, and returns a `tf.data.Dataset`.

```text
images_dataSAT/
├── class_0_non_agri/*.jpg
└── class_1_agri/*.jpg
```

```python
import keras
import tensorflow as tf

IMAGE_SIZE, BATCH_SIZE = (64, 64), 8          # the lab uses a small batch to inspect samples; training uses 128

train_ds = keras.utils.image_dataset_from_directory(
    "images_dataSAT",
    labels="inferred",                         # labels come from the folder names
    label_mode="int",                          # 0 = class_0_non_agri, 1 = class_1_agri
    validation_split=0.2,                      # 20% of the images are reserved for validation
    subset="training",
    seed=1337,                                 # the same seed in both calls keeps the subsets disjoint
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)
val_ds = keras.utils.image_dataset_from_directory(
    "images_dataSAT", labels="inferred", label_mode="int",
    validation_split=0.2, subset="validation", seed=1337,
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
)
print(train_ds.class_names)                    # ['class_0_non_agri', 'class_1_agri']
```

The utility reports `Found 6000 files belonging to 2 classes. Using 4800 files for training.` (and 1,200 for validation).

> **Label order:** class folders are sorted alphabetically, so `class_0_non_agri` becomes 0 and `class_1_agri` becomes 1. The `class_0_` and `class_1_` prefixes make the positive class (agricultural = 1) automatic. With other folder names, set `class_names` explicitly in Keras, or check `class_to_idx` in PyTorch.

> **Caching caveat:** `.cache()` speeds up small datasets, but on a batched dataset it freezes the batch composition after the first epoch. Use it only when the data fits in memory, and put random augmentation *after* the cache (for example, as layers inside the model).

> **Deprecated:** the older `ImageDataGenerator` is deprecated in Keras. Prefer `image_dataset_from_directory` plus preprocessing layers for new code.

**PyTorch: `ImageFolder` + `DataLoader`.** Same folder layout, one line to load:

```python
from torchvision import datasets
train_ds = datasets.ImageFolder("data/train", transform=train_tf)
print(train_ds.class_to_idx)      # {'class_0_non_agri': 0, 'class_1_agri': 1}
```

> **Check:** `ImageFolder` also sorts class folders alphabetically. In the lab, `class_to_idx` printed `{'class_0_non_agri': 0, 'class_1_agri': 1}`, which is the mapping we want.

#### Efficiency settings side by side

| Concern | Keras / `tf.data` | PyTorch |
|---|---|---|
| **Parallel decoding** | Handled by the utility (`num_parallel_calls=tf.data.AUTOTUNE` in custom `map` calls) | `DataLoader(num_workers=N)` |
| **Overlap loading with training** | `.prefetch(tf.data.AUTOTUNE)` | Worker processes prefetch batches (`prefetch_factor`) |
| **Caching** | `.cache()` | No built-in cache; cache decoded images inside the `Dataset` |
| **Faster transfer to the GPU** | Automatic | `pin_memory=True` |
| **Shuffling** | `shuffle=True` | `shuffle=True` in `DataLoader` |
| **Multiprocessing for custom loaders** | `PyDataset(workers=..., use_multiprocessing=True)` | `num_workers > 0` |

#### Custom generators vs. built-in utilities

| Criterion | Custom generator / `Dataset` | Built-in utility |
|---|---|---|
| **Control** | Full: any file format, any logic | Limited to the supported layout |
| **Code to write** | More | Very little |
| **Speed** | As good as our implementation | Usually well optimized |
| **Reusing a manifest (`train.csv`)** | Natural | Needs the folder layout |
| **Best for** | Custom splits, extra metadata, unusual formats | Standard folder-per-class image data |

**The labs reached the same conclusion in both frameworks:** the built-in utilities (`image_dataset_from_directory`, `ImageFolder`) are the right choice for almost all standard image-classification tasks. Custom generators and `Dataset` classes are worth writing for learning, for non-standard layouts (labels in a CSV or database), or when we need fine-grained control. A plain single-threaded Python generator can become an input bottleneck for the GPU, while `tf.data` (with `.prefetch()`) and `DataLoader` (with `num_workers`) are built to avoid that.

#### Pipeline checklist

- [ ] **Split first, augment later:** create train/val/test splits before any augmentation
- [ ] **Stratify** the split so class balance is the same in each set
- [ ] **Never augment** validation or test data
- [ ] **Compute normalization statistics on the training set only** (if we use mean/std)
- [ ] Set **`shuffle=True` for training only**; keep `shuffle=False` for val/test so predictions line up with labels
- [ ] Use the **same split files** in both frameworks

### 1.3 Data Augmentation in PyTorch

**What it is.** Data augmentation creates new, slightly different versions of training images (flipped, rotated, re-lit) on the fly. The label stays the same, but the model sees more variety, which **reduces overfitting** and improves generalization.

| | **Online augmentation** | **Offline augmentation** |
|---|---|---|
| **When** | Every time an image is loaded | Once, before training |
| **Storage** | None | Augmented copies saved to disk |
| **Variety** | A new random version every epoch | A fixed set of versions |
| **Used here** | Yes (in `Dataset.__getitem__`) | No |

#### The transforms

```python
import torch
from torchvision.transforms import v2

train_tf = v2.Compose([
    v2.ToImage(),
    v2.Resize((64, 64)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(degrees=15),
    v2.ColorJitter(contrast=0.1),
    v2.ToDtype(torch.float32, scale=True),     # uint8 [0, 255] → float32 [0, 1]
])

eval_tf = v2.Compose([                          # NO random transforms for validation and test
    v2.ToImage(),
    v2.Resize((64, 64)),
    v2.ToDtype(torch.float32, scale=True),
])
```

`train_tf` goes into the training `Dataset`; `eval_tf` goes into the validation and test datasets.

#### Augmentation settings used in the labs

| | Keras data lab (02) | PyTorch data lab (03) | CNN training labs (04, 05) |
|---|---|---|---|
| **Flips** | `RandomFlip("horizontal_and_vertical")` | Horizontal (p = 0.5) and vertical (p = 0.2) | Horizontal only |
| **Rotation** | `RandomRotation(0.2)` (±72°) | `RandomRotation(45)` (±45°) | ±40° |
| **Other** | None | None | Shear 0.2; Keras adds shifts of 0.2 and zoom of 0.2 |
| **Normalization** | None (pixels 0 to 255) | `Normalize` to `[-1, 1]` | Keras: divide by 255. PyTorch: ImageNet mean and std |
| **API** | Preprocessing layers inside a `tf.data` pipeline | `torchvision.transforms` (v1 API) | `ImageDataGenerator` and `transforms` |

The labs use the classic `torchvision.transforms` API. The `transforms.v2` version shown above is its modern equivalent.

#### Which transforms suit satellite images?

| Transform | What it simulates | Suitable for satellite tiles? |
|---|---|---|
| `RandomHorizontalFlip`, `RandomVerticalFlip` | Different orientation | **Yes.** Overhead imagery usually has no canonical "up" |
| `RandomRotation` | Different orientation or sensor angle | **Yes.** Small angles leave empty corners (filled black by default) |
| `ColorJitter` | Lighting, season, atmosphere | **Use mildly.** Color is a key cue for vegetation, so strong changes (especially hue) could change the true label |
| `RandomResizedCrop` | Different scale and framing | **Carefully.** A crop may remove the evidence for the label |
| `GaussianBlur`, noise | Haze, sensor noise | Optional |
| `Normalize` | Not augmentation: standardizes the inputs | Apply **identically** to train, val, and test |

#### Checking and reproducing augmentation

```python
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open(sample_path).convert("RGB")
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for ax in axes.ravel():
    ax.imshow(train_tf(img).permute(1, 2, 0))      # (C, H, W) → (H, W, C)
    ax.axis("off")
plt.savefig("results/figures/augmentation_examples.png", bbox_inches="tight")
```

- **Always look at augmented samples** before training. Wrong settings (too much rotation or color shift) are easy to spot visually.
- **Seed everything** (`torch.manual_seed(42)`) and give `DataLoader` workers a `worker_init_fn` or a seeded `generator`, so augmentation is reproducible.

> **Connection to Part 2:** Section 2.3 puts the Keras and PyTorch augmentation settings of the CNN labs side by side.

### 1.4 Lessons from the Labs

Working through the labs exposed four pitfalls that are easy to miss. We note them here because they affect later results.

| # | Pitfall | What happens | Fix |
|---|---|---|---|
| 1 | **`cache()` after augmentation** (Keras lab) | The lab chains `.map(augment)` and then `.cache()`. The random augmentation is computed once and then frozen, so every epoch after the first sees the **same** augmented images | Put `.cache()` **before** the augmentation step, or apply augmentation as layers inside the model |
| 2 | **Rotation units differ** | `RandomRotation(0.2)` in Keras means a fraction of a full turn (±72°), while `RandomRotation(45)` in PyTorch means degrees | Convert units when matching settings across frameworks (Keras `degrees / 360`) |
| 3 | **Shared `transform` after `random_split`** (PyTorch labs) | `random_split` returns two views of the **same** dataset object. Setting `val_dataset.dataset.transform` therefore changes the transform for the **training** subset too, so training runs **without augmentation** | Build two `ImageFolder` datasets (one per transform) and split the indices once (see Section 2.3) |
| 4 | **Augmenting the validation data** (Keras CNN labs) | The same `ImageDataGenerator` (with augmentation) is used for both the training and the validation generators | Use a second generator with `rescale` only for validation (see Section 2.3) |

Pitfalls 3 and 4 are revisited in Part 2 and Part 3, because they also break the "same preprocessing" requirement of a fair comparison.

### 1.5 Part 1 Summary

| Topic | Key takeaway |
|---|---|
| **Bulk loading** | Simple and fast, but memory grows with the dataset |
| **Sequential (generator) loading** | Constant memory, scales to large data, ideal for on-the-fly augmentation |
| **Custom generators** | Maximum control (`PyDataset`, `Dataset` + `DataLoader`) |
| **Built-in utilities** | Least code (`image_dataset_from_directory`, `ImageFolder`) |
| **Augmentation in PyTorch** | `torchvision.transforms.v2` in the training `Dataset` only; choose transforms that keep the label true |
| **Lab pitfalls** | Cache before augmenting, convert rotation units, never share one `transform` between train and validation, never augment validation data |

---

## Part 2: CNN Classifier in Keras and PyTorch

> **Hands-on labs for this part:** notebooks `04` (train and evaluate a Keras CNN) and `05` (implement and test a PyTorch CNN).

In this part we build a convolutional neural network (CNN) **twice**, once in Keras and once in PyTorch, using the same dataset and the same overall recipe. The numbers below come from the lab runs. Part 3 then asks how fairly the two can be compared.

### 2.1 Problem Framing

| Item | Choice |
|---|---|
| **Task** | Binary classification of a 64×64 RGB tile: agricultural (1) or non-agricultural (0) |
| **Input** | Image tensor of 64×64×3 |
| **Output** | A score for each image, turned into a class by a decision rule |

The two labs set up the output layer and loss differently:

| | **Keras lab** | **PyTorch lab** |
|---|---|---|
| **Output layer** | 1 neuron with sigmoid: the probability of "agricultural" | 2 neurons with raw logits: one score per class |
| **Loss** | `binary_crossentropy` | `nn.CrossEntropyLoss()` |
| **Prediction** | Probability above 0.5 means agricultural | `argmax` over the two logits |
| **Probability of "agricultural"** | The model output | `softmax(logits)[:, 1]` |

Both are valid binary classifiers (a softmax over two logits is equivalent to a sigmoid of their difference), but they are **different parameterizations**. Part 3 lists this among the items to align for a strict comparison.

### 2.2 Hyperparameters

Hyperparameters are the settings we choose **before** training. They fall into three groups.

| Group | Hyperparameters | What they control |
|---|---|---|
| **Optimization** | Optimizer, learning rate, momentum, learning-rate schedule | How the weights are updated |
| **Regularization** | Dropout, weight decay, augmentation, early stopping, batch normalization | How strongly we prevent overfitting |
| **Training budget** | Batch size, number of epochs | How much and how often the model learns |

#### Hyperparameters used in the labs

| Hyperparameter | Keras CNN lab | PyTorch CNN lab |
|---|---|---|
| **Image size** | 64×64×3 | 64×64×3 |
| **Batch size** | 128 (38 steps per epoch) | 128 (38 steps per epoch) |
| **Learning rate** | 0.001 | 0.001 |
| **Optimizer** | Adam | Adam |
| **Epochs** | 3 (demo run), with a 20-epoch model provided | 3 (demo run), with a 20-epoch model provided |
| **Dropout** | 0.4 after each of 6 dense blocks | 0.4 after the one dense block |
| **Batch normalization** | After every conv and dense block | After every conv block and the dense block |
| **Weight initialization** | He uniform | PyTorch default |
| **Checkpointing** | `ModelCheckpoint` on validation loss (minimum) | `torch.save` when validation loss improves |
| **Random seed** | 7331 (Python, NumPy, TensorFlow) | 42 (Python, NumPy, PyTorch, deterministic cuDNN, DataLoader workers) |
| **Hardware** | CPU | CPU |

The labs train for only 3 epochs "for convenience". A realistic run uses 15 or more epochs.

#### Optimization hyperparameters

| Hyperparameter | What it controls | Common starting point | Keras | PyTorch |
|---|---|---|---|---|
| **Optimizer** | The update rule (SGD, Adam, AdamW, RMSprop) | Adam | `keras.optimizers.Adam` | `torch.optim.Adam` |
| **Learning rate** | Step size. Too high: unstable or diverging. Too low: slow training | 1e-3 for Adam | `learning_rate=1e-3` | `lr=1e-3` |
| **Momentum** (SGD only) | How much past gradients are carried forward | 0.9 | `SGD(momentum=0.9)` | `SGD(momentum=0.9)` |
| **LR schedule** | Lowers the learning rate when progress stalls | Halve after 2 epochs without improvement | `ReduceLROnPlateau` callback | `lr_scheduler.ReduceLROnPlateau` |

#### Regularization hyperparameters

| Hyperparameter | What it does | Common starting point | Keras | PyTorch |
|---|---|---|---|---|
| **Dropout rate** | Randomly switches off a fraction of neurons during training | 0.3 to 0.5 | `Dropout(0.4)` | `nn.Dropout(0.4)` |
| **Weight decay (L2)** | Penalizes large weights | 1e-4 | `kernel_regularizer=l2(1e-4)` | `weight_decay=2e-4` (Keras `l2(λ)` matches PyTorch `weight_decay=2λ`) |
| **Data augmentation** | Adds variety to the training images | Flips, small rotation | Preprocessing layers or `ImageDataGenerator` | `transforms` |
| **Early stopping** | Stops when validation loss stops improving | Patience 5 | `EarlyStopping` | Manual loop |
| **Batch normalization** | Stabilizes training, mild regularizer | After each block | `BatchNormalization()` | `nn.BatchNorm2d`, `nn.BatchNorm1d` |

In both frameworks, the dropout rate is the probability of **dropping** a neuron.

#### Batch size and epoch count

| Hyperparameter | Effect when **smaller** | Effect when **larger** |
|---|---|---|
| **Batch size** | Noisier gradients (can regularize), more updates per epoch, slower per epoch | Smoother gradients, faster per epoch, more memory; may need a higher learning rate |
| **Epoch count** | Underfitting: the model has not converged yet | Overfitting: validation loss rises while training loss keeps falling |

$$\text{steps per epoch} = \left\lceil \frac{N_{\text{train}}}{\text{batch size}} \right\rceil = \left\lceil \frac{4800}{128} \right\rceil = 38$$

#### Tuning strategy

1. **Tune on the validation set only.** A held-out test set is used **once**, at the end.
2. Use the **same search space and the same budget** for both frameworks.
3. Start with the learning rate and batch size, then dropout and weight decay.
4. Pick the best **common** configuration for the final comparison.

| Hyperparameter | Example values |
|---|---|
| Learning rate | 1e-2, 1e-3, 1e-4 |
| Batch size | 32, 64, 128 |
| Dropout | 0.3, 0.4, 0.5 |

**Tuning log template:**

| Run | Framework | Optimizer | LR | Batch size | Dropout | Epochs run | Val loss | Val precision | Val F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Keras | Adam | 1e-3 | 128 | 0.4 | TBD | TBD | TBD | TBD |
| 2 | PyTorch | Adam | 1e-3 | 128 | 0.4 | TBD | TBD | TBD | TBD |

### 2.3 Data Pipeline and Augmentation in the CNN Labs

| | **Keras lab** | **PyTorch lab** |
|---|---|---|
| **Loader** | `ImageDataGenerator.flow_from_directory` | `ImageFolder` + `random_split` + `DataLoader` (`num_workers=2`) |
| **Split** | `validation_split=0.2`, taken by **sorted file order** (not random): 4,800 / 1,200 | `random_split` 80 / 20 (random): 4,800 / 1,200 |
| **Scaling** | Divide by 255 (`rescale=1./255`) | `ToTensor()`, then `Normalize` with ImageNet mean and std |
| **Augmentation** | Rotation 40°, width and height shift 0.2, shear 0.2, zoom 0.2, horizontal flip, fill `nearest` | `RandomRotation(40)`, `RandomHorizontalFlip()`, `RandomAffine(0, shear=0.2)` |
| **Labels** | `class_mode="binary"` | Integer class index |
| **Reproducibility** | Seeds for Python, NumPy, TensorFlow | `set_seed()` for Python, NumPy, PyTorch, cuDNN, plus `worker_init_fn` |

The two settings differ in several ways (loader, split, scaling, augmentation), which matters for the comparison in Part 3. On top of that, **each lab has a bug that silently changes what it trains or validates on**:

| Lab | The bug | Consequence |
|---|---|---|
| **Keras** | One `ImageDataGenerator` (with augmentation) is used for both the training and the validation generators | The validation images are **augmented** too, so validation scores are noisier and not comparable with clean data |
| **PyTorch** | `val_dataset.dataset.transform = val_transform` changes the transform of the shared `full_dataset` | The **training** set also uses `val_transform`, so training runs **without augmentation** |

**Fix for Keras:** use a second generator for validation that only rescales (the same `validation_split` keeps the subsets consistent).

```python
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40,
                                   width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, fill_mode="nearest",
                                   validation_split=0.2)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)     # no augmentation

train_generator = train_datagen.flow_from_directory(
    dataset_path, target_size=(64, 64), batch_size=128, class_mode="binary", subset="training")
validation_generator = val_datagen.flow_from_directory(
    dataset_path, target_size=(64, 64), batch_size=128, class_mode="binary",
    subset="validation", shuffle=False)
```

**Fix for PyTorch:** create two datasets, one per transform, and split the **indices** once.

```python
from torch.utils.data import Subset

train_full = datasets.ImageFolder(dataset_path, transform=train_transform)
val_full   = datasets.ImageFolder(dataset_path, transform=val_transform)

g = torch.Generator().manual_seed(42)
perm = torch.randperm(len(train_full), generator=g).tolist()
n_train = int(0.8 * len(perm))
train_dataset = Subset(train_full, perm[:n_train])        # augmented
val_dataset   = Subset(val_full,   perm[n_train:])        # clean
```

> **Note:** `ImageDataGenerator` is deprecated in Keras. The labs use it for convenience; new code should use `image_dataset_from_directory` with preprocessing layers (Section 1.2).

### 2.4 Model Architecture

Both labs build a **six-block convolutional feature extractor** and then a classifier head. The convolution blocks are identical; the heads are not.

| Stage | Keras CNN | PyTorch CNN |
|---|---|---|
| **Conv blocks 1 to 6** | `Conv2D` 5×5, `same` padding, ReLU; filters 32, 64, 128, 256, 512, 1024; each followed by `MaxPooling2D(2)` and `BatchNormalization` | Same: `Conv2d` 5×5 (`padding=2`), ReLU, `MaxPool2d(2)`, `BatchNorm2d` |
| **Pooling** | `GlobalAveragePooling2D` | `AdaptiveAvgPool2d(1)` + `Flatten` |
| **Head** | **Six** dense blocks (64, 128, 256, 512, 1024, 2048), each with ReLU, `BatchNormalization`, and `Dropout(0.4)` | **One** dense block: `Linear(1024, 2048)`, ReLU, `BatchNorm1d`, `Dropout(0.4)` |
| **Output** | `Dense(1, sigmoid)` | `Linear(2048, 2)` (logits) |
| **Parameters** | **20,352,897** (20,340,801 trainable + 12,096 non-trainable), as printed by `model.summary()` | **19,575,042** trainable (our count) |

A 64×64 input passes through six poolings, so the feature map shrinks 64 → 32 → 16 → 8 → 4 → 2 → **1**. After the last conv block the tensor is 1×1×1024. (This detail matters for the Vision Transformer in Part 4.)

#### Keras

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, BatchNormalization,
                                     GlobalAveragePooling2D, Dense, Dropout)
from tensorflow.keras.initializers import HeUniform

def build_keras_cnn(img_w=64, img_h=64, n_channels=3):
    layers = []
    for i, filters in enumerate([32, 64, 128, 256, 512, 1024]):            # 6 convolution blocks
        extra = {"input_shape": (img_w, img_h, n_channels)} if i == 0 else {}
        layers += [Conv2D(filters, (5, 5), activation="relu", padding="same",
                          kernel_initializer=HeUniform(), **extra),
                   MaxPooling2D(2, 2), BatchNormalization()]
    layers.append(GlobalAveragePooling2D())
    for units in [64, 128, 256, 512, 1024, 2048]:                           # 6 dense blocks
        layers += [Dense(units, activation="relu", kernel_initializer=HeUniform()),
                   BatchNormalization(), Dropout(0.4)]
    layers.append(Dense(1, activation="sigmoid"))
    return Sequential(layers)
```

#### PyTorch

```python
import torch.nn as nn

channels = [3, 32, 64, 128, 256, 512, 1024]

def conv_block(c_in, c_out):
    return [nn.Conv2d(c_in, c_out, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(c_out)]

model = nn.Sequential(
    *[layer for i in range(6) for layer in conv_block(channels[i], channels[i + 1])],
    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    nn.Linear(1024, 2048), nn.ReLU(), nn.BatchNorm1d(2048), nn.Dropout(0.4),
    nn.Linear(2048, 2),
)
```

#### How the two styles differ

| Aspect | Keras | PyTorch |
|---|---|---|
| **Style** | Declarative: a list of layers in `Sequential` | Object-oriented or `Sequential`, with explicit shapes |
| **Input channels / features** | Inferred from the previous layer | Given explicitly (`Conv2d(3, 32, ...)`, `Linear(1024, 2048)`) |
| **Activation** | An argument of the layer (`activation="relu"`) | A separate module (`nn.ReLU()`) |
| **Padding** | `padding="same"` | `padding=2` (explicit number) |
| **Initialization** | `kernel_initializer=HeUniform()` per layer | Default initializer unless we set one |
| **Layer order** | Conv (with ReLU) → pool → batch norm | Conv → ReLU → pool → batch norm (equivalent, since ReLU and max pooling commute) |

### 2.5 Training Workflow

| Step | Keras | PyTorch |
|---|---|---|
| **Configure training** | `model.compile(optimizer, loss, metrics)` | Create `criterion` and `optimizer` objects |
| **Training loop** | `model.fit(...)` | Write the loop: forward, loss, `zero_grad`, `backward`, `step` |
| **Train / eval modes** | Automatic | `model.train()` and `model.eval()` |
| **Validation** | `validation_data=...` | A second loop under `torch.no_grad()` |
| **Best model** | `ModelCheckpoint` callback | `torch.save(...)` inside the loop |
| **Device placement** | Automatic | `.to(device)` for the model and every batch |
| **Progress** | Built-in progress bar | `tqdm` |
| **Amount of code** | Less | More, but full control over every step |

#### Keras

```python
model = build_keras_cnn()
model.compile(optimizer=Adam(learning_rate=0.001),
              loss="binary_crossentropy",
              metrics=["accuracy"])

checkpoint_cb = ModelCheckpoint("ai_capstone_keras_best_model.model.keras",
                                monitor="val_loss", mode="min", save_best_only=True, verbose=1)

fit = model.fit(train_generator, epochs=3,
                validation_data=validation_generator,
                callbacks=[checkpoint_cb])
```

#### PyTorch

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
best_loss = float("inf")

for epoch in range(3):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            val_loss += criterion(model(images), labels).item()
    avg_val_loss = val_loss / len(val_loader)

    if avg_val_loss < best_loss:                         # keep the best checkpoint
        best_loss = avg_val_loss
        torch.save(model.state_dict(), "ai_capstone_pytorch_state_dict.pth")
```

#### Results of the lab runs (3 epochs, CPU)

**PyTorch CNN** (about 200 seconds per epoch):

| Epoch | Train loss | Val loss | Train acc | Val acc |
|---:|---:|---:|---:|---:|
| 1 | 0.2512 | 0.0494 | 0.9371 | 0.9775 |
| 2 | 0.0170 | **0.0094** | 0.9940 | **0.9975** |
| 3 | 0.0365 | 0.0481 | 0.9892 | 0.9842 |

**Keras CNN:** the saved notebook output shows only epoch 1 (about 276 seconds, 38 steps): training accuracy 0.9148 and loss 0.2002, validation accuracy 0.6442 and loss 1.9708. This is an early, unsettled epoch (the lab's validation data is also augmented, see Section 2.3), and the later epochs are not visible in the output.

Both CNNs learn the task quickly. Validation loss is best at epoch 2 for PyTorch and rises slightly in epoch 3, which is why a checkpoint on validation loss matters.

> **Pitfall: evaluating the wrong weights.** The PyTorch lab's final accuracy (0.9842) equals the **last** epoch's validation accuracy, not the best epoch's (0.9975), because the best checkpoint was saved but never reloaded. The Keras lab follows the same pattern. Always reload the best checkpoint before the final evaluation (`model.load_state_dict(torch.load(...))` in PyTorch, `load_model(...)` or `restore_best_weights=True` in Keras).

### 2.6 What We Learned from Building the Same CNN Twice

| Observation | Keras | PyTorch |
|---|---|---|
| **Speed to a first model** | Fast: `compile` and `fit` | Slower: we write the loop |
| **Control** | Limited inside `fit` | Full control of every step, which makes custom behavior and debugging easy |
| **Silent bugs we hit** | Augmented validation data | Shared `transform` disabling augmentation |
| **Both** | Evaluated last-epoch weights instead of the best checkpoint | |

The common lesson is that **a high-level API hides decisions, and a low-level API hides bugs in our own code**. Either way, we have to inspect what the data pipeline actually delivers (shapes, labels, a visual check of augmented samples) before trusting the numbers.

---

## Part 3: Comparative Analysis of Keras and PyTorch Models

> **Hands-on lab for this part:** notebook `06` (comparative analysis of the Keras and PyTorch models).

**Learning objectives**

1. Identify what must be standardized in the **dataset**, the **model design**, and the **evaluation** to make a fair comparison.
2. Explain the **evaluation metrics** used to compare the models.

### 3.1 Why Compare, and What Can Differ

Keras and PyTorch can both build and train powerful deep learning models. They differ in design philosophy, workflow, and tooling, and those differences can affect performance, reproducibility, and deployment. A complete comparison can look at several dimensions: **predictive performance, speed, resource efficiency, flexibility, ease of use**, and **how well the model generalizes** to new data.

For the predictive part, the recipe is simple. Evaluate both models on the **same test set**, with the **same metrics**, after matching the preprocessing, batch size, and architecture. Run each model to get predictions, then compute the metrics from the true labels. If it is relevant, also compare inference speed and resource use. This way, a difference in results points to the **framework or implementation**, not to an inconsistency in the experiment.

### 3.2 What Must Be Standardized

| Area | Requirement | Why it matters |
|---|---|---|
| **Dataset** | Identical train, validation, and test splits | Different splits mean the models are trained and judged on different images |
| **Model design** | Identical architecture: same number, type, and order of layers | Otherwise we compare designs, not frameworks |
| **Preprocessing** | Same resizing, normalization, and augmentation | Different inputs change what the model learns |
| **Hyperparameters** | Same optimizer, learning rate, batch size, and number of epochs | Training budget and step size directly affect the result |
| **Randomness** | Fixed random seeds for every source of randomness | Makes runs repeatable |
| **Output and loss** | Equivalent output activation and loss (for example, sigmoid with binary cross-entropy for binary classification) | The model should optimize the same objective |
| **Evaluation** | Same held-out test set and the same metric code | Differences must come from the models, not from how we measured them |

#### Fairness audit of our two CNN labs

We checked each requirement against what the Keras lab (04) and the PyTorch lab (05) actually did.

| Requirement | Keras lab | PyTorch lab | Matched? |
|---|---|---|:---:|
| Same splits | 80/20 by sorted file order | 80/20 random | No |
| Identical architecture | 6 conv blocks, 6 dense blocks, 1 sigmoid output | 6 conv blocks, 1 dense block, 2 logits | No |
| Same preprocessing | Divide by 255; shifts, shear, and zoom; validation also augmented | ImageNet mean and std; rotation, flip, shear; training not augmented (bug) | No |
| Same hyperparameters | Adam, 0.001, batch 128, 3 epochs | Adam, 0.001, batch 128, 3 epochs | **Yes** |
| Fixed seeds | Yes (7331) | Yes (42) | **Yes** (values differ) |
| Equivalent output and loss | Sigmoid and binary cross-entropy | Two logits and cross-entropy | No |
| Same held-out test set | Evaluated on all 6,000 images | Evaluated on all 6,000 images | No |
| Same metric code | One scikit-learn function for both | One scikit-learn function for both | **Yes** |

Only **3 of the 8** requirements are fully met. The comparison in the lab is still useful as a **workflow demonstration**, but the differences it shows cannot be credited to the frameworks. The next table is the plan for a strictly fair re-run.

#### Alignment plan for a fair re-run

| What to align | How |
|---|---|
| **Splits** | Create one stratified split (for example 70 / 15 / 15 for train, validation, test), save it as CSV files, and load those files in both frameworks |
| **Architecture** | Use the same head in both: for example, copy the PyTorch head (`Linear(1024, 2048)`, batch norm, dropout) into Keras, and check that both report the same parameter count |
| **Output and loss** | Use one logit with a sigmoid and binary cross-entropy in both (`BCEWithLogitsLoss` in PyTorch) |
| **Initialization** | Use He initialization in both (`kaiming_uniform_` or `kaiming_normal_` in PyTorch) |
| **Preprocessing** | Same scaling in both (for example, divide by 255 and use the same mean and std) and the same augmentation, applied to **training data only** (Section 2.3) |
| **Evaluation** | Reload the best checkpoint, evaluate on the **test** split only, and use the shared metric script |
| **Repeats** | Train with three seeds per framework and report mean ± standard deviation |

#### Framework defaults that still differ

Even with matching settings, these defaults differ. We set each one explicitly:

| Difference | Keras default | PyTorch default | What we do |
|---|---|---|---|
| **Initial weights** | Glorot uniform, zero biases | Kaiming-uniform variant, uniform biases | Use He initialization in both |
| **Adam epsilon** | 1e-7 | 1e-8 | Set the same value in both |
| **Convolution padding** | `"valid"` | `0` | Set explicitly (`"same"` or `padding=2`) |
| **Rotation fill** | Reflect | Black (0) | Set Keras `fill_mode="constant"` |
| **Resize** | Bilinear, no antialiasing | Bilinear, antialiasing on for tensors | Pre-resize the images once (our images are already 64×64) |
| **L2 or weight decay** | `l2(λ)` adds λ·Σw² to the loss | `weight_decay=λ′` adds λ′·w to the gradient | Use `λ′ = 2λ`, or `AdamW` in both |
| **Batch norm momentum** | 0.99 | 0.1 (equals 0.9 in Keras terms) | Set equivalent values |
| **Data layout** | Channels last (NHWC) | Channels first (NCHW) | No effect on results; remember it when comparing shapes |
| **Random number generators** | Different | Different | Runs are never bit-identical, so repeat with several seeds |

### 3.3 Evaluation Metrics

To compare models we use the **same metrics**, computed by **one shared script** from each model's saved probabilities, so every model is judged by exactly the same code. The positive class is **agricultural**. The lab's `model_metrics` function does this with scikit-learn.

#### The confusion matrix

| | **Predicted agricultural** | **Predicted non-agricultural** |
|---|---|---|
| **Actually agricultural** | True positive (TP) | False negative (FN) |
| **Actually non-agricultural** | False positive (FP) | True negative (TN) |

Every other metric below is computed from these four counts (or, for ROC and AP, from the probabilities directly).

#### The metrics

| Metric | Formula | The question it answers | Watch out for |
|---|---|---|---|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | How often is the model right overall? | Misleading when classes are imbalanced |
| **Precision** | $\frac{TP}{TP + FP}$ | Of the tiles predicted agricultural, how many really are? | Ignores missed farmland |
| **Recall** (sensitivity, TPR) | $\frac{TP}{TP + FN}$ | Of the real agricultural tiles, how many did we find? | Ignores false alarms |
| **F1 score** | $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | One number that balances precision and recall | Hides which of the two is weak |
| **Log loss** | $-\frac{1}{N}\sum \big[y\log p + (1-y)\log(1-p)\big]$ | How well calibrated are the predicted probabilities? Lower is better | Punishes confident mistakes heavily |
| **False positive rate** | $\frac{FP}{FP + TN}$ | Of the non-agricultural tiles, how many were wrongly flagged? | Used for the ROC curve |
| **ROC-AUC** | Area under the TPR-vs-FPR curve | How well does the model rank agricultural tiles above non-agricultural ones, at **any** threshold? | Can look optimistic under heavy class imbalance |
| **Average precision (AP)** | Area under the precision-recall curve | How good are precision and recall across thresholds, focused on the positive class? | Depends on class balance |
| **IoU (Jaccard index)** | $\frac{TP}{TP + FP + FN}$ | How much do the predicted and true agricultural sets overlap? | Ignores true negatives |
| **mAP** | Mean of AP over classes | AP averaged over the classes | See the note below |

> **About IoU and mAP.** These two metrics come from **object detection and segmentation**, where IoU measures the overlap between predicted and true boxes or masks, and mAP averages precision over classes (and, in detection, over IoU thresholds). For **image-level binary classification** they reduce to simpler quantities: IoU becomes the Jaccard index of the positive class (formula above), and mAP becomes the average of AP over the two classes. We report them for completeness, but **precision, recall, F1, and ROC-AUC are the primary metrics** here. They become truly central if the project is extended to segmenting farmland pixel by pixel.

#### ROC and AUC

The **ROC curve** plots the true positive rate (recall) against the false positive rate while the decision threshold moves from 1 to 0.

| ROC-AUC | Meaning |
|---|---|
| 0.5 | No better than random guessing |
| 0.5 to 1.0 | Better ranking; closer to 1 is better |
| 1.0 | Perfect ranking |

AUC can be read as: *the probability that a randomly chosen agricultural tile gets a higher score than a randomly chosen non-agricultural tile.* Unlike precision and recall, it does **not** depend on the threshold.

#### Precision and the decision threshold

Precision and recall depend on the threshold. Raising it makes the model more cautious:

| Threshold | Precision | Recall | Effect on the forecast |
|---|---|---|---|
| **Lower** | Falls | Rises | Fewer missed farms, more false alarms (risk of over-forecasting) |
| **Higher** | Rises | Falls | Fewer false alarms, more missed farms (risk of under-forecasting) |

Because we prioritize precision, we can choose the threshold **on the validation set** for a target precision, then apply it unchanged to the test set:

```python
import numpy as np
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_val_true, y_val_prob)
ok = np.where(precision[:-1] >= 0.95)[0]               # thresholds that reach 95% precision
chosen_threshold = thresholds[ok[0]] if len(ok) else 0.5   # smallest one keeps recall highest
```

#### The shared evaluation script

```python
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score, jaccard_score)

def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
        "iou_jaccard": jaccard_score(y_true, y_pred),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }

results = {}
for name in ["keras", "pytorch"]:
    df = pd.read_csv(f"results/{name}/predictions.csv")
    results[name] = evaluate(df["y_true"].values, df["y_prob"].values)
print(pd.DataFrame(results))
```

```python
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

fig, ax = plt.subplots()
for name in ["keras", "pytorch"]:
    df = pd.read_csv(f"results/{name}/predictions.csv")
    RocCurveDisplay.from_predictions(df["y_true"], df["y_prob"], name=name, ax=ax)
fig.savefig("results/figures/roc_keras_vs_pytorch.png", bbox_inches="tight")
```

#### Which metric to look at when

| Question | Metric |
|---|---|
| Are we over-counting farmland? | **Precision** |
| Are we missing farmland? | **Recall** |
| Do we need one balanced number? | **F1** |
| Is the model good regardless of threshold? | **ROC-AUC** and **AP** |
| Where exactly are the mistakes? | **Confusion matrix** |
| Is the model good overall (balanced classes only)? | **Accuracy** |

### 3.4 Results of the Lab Comparison

The lab downloads two **pre-trained 20-epoch CNNs** (one per framework) and evaluates them with the same code. Keras rescales pixels by 1/255 with `ImageDataGenerator`; PyTorch uses `ImageFolder` with ImageNet normalization. Both read the images in a fixed order (no shuffling), and both are evaluated on **all 6,000 images**.

| Metric | Keras CNN | PyTorch CNN |
|---|---:|---:|
| Accuracy | 0.9925 | **0.9988** |
| Precision (agricultural) | **1.0000** | 0.9983 |
| Recall (agricultural) | 0.9850 | **0.9993** |
| F1 score | 0.9924 | **0.9988** |
| ROC-AUC | 1.0000 | 1.0000 |
| Log loss | 0.0247 | **0.0024** |

**Confusion matrices** (rows are the true class, `agri` = positive):

| Keras CNN | Predicted non-agri | Predicted agri |
|---|---:|---:|
| **Actually non-agri** | 3000 (TN) | 0 (FP) |
| **Actually agri** | 45 (FN) | 2955 (TP) |

| PyTorch CNN | Predicted non-agri | Predicted agri |
|---|---:|---:|
| **Actually non-agri** | 2995 (TN) | 5 (FP) |
| **Actually agri** | 2 (FN) | 2998 (TP) |

**Reading the results:**

- **Keras CNN:** never raises a false alarm (precision 1.0000), but it misses 45 agricultural tiles (recall 0.9850). All of its 45 errors are false negatives, so it tends to **under-count farmland**.
- **PyTorch CNN:** makes only 7 errors in total (5 false positives, 2 false negatives), so it is more balanced and has the higher accuracy, recall, and F1.
- **ROC-AUC** is 1.0000 for both: ranked by probability, the two classes are almost perfectly separable. The Keras model's mistakes come from the 0.5 threshold, not from poor ranking, so choosing another threshold on a validation set could reduce its false negatives.
- For our **precision-first** goal, the Keras CNN looks best on paper, but see the next section before drawing conclusions.

> **Correction to the lab's answer key:** the lab states that the PyTorch model has 5 false negatives. With `agri` as the positive class, the matrix shows **5 false positives** (non-agri tiles predicted agri) and **2 false negatives**. In scikit-learn's confusion matrix the rows are the true labels.

### 3.5 Why These Numbers Are Not Test Scores

The lab evaluates both models on **all 6,000 images**. Each model was trained on about 80% of them (4,800 images), so at most **1,200 images (20%)** were unseen by a given model, and the two frameworks split the data differently, so their unseen images are not even the same ones. The scores therefore describe how well the models fit data they have mostly seen, and they are **optimistic** estimates of performance on new satellite tiles. The lab's own summary says the same and recommends repeating the evaluation with a labeled validation set.

| Question | Answer |
|---|---|
| Can we say Keras or PyTorch is **better** from these numbers? | **No.** The recipes differ (Section 3.2), the test data overlaps the training data, and a few images change the metrics |
| Can we say both models **learned the task**? | **Yes.** Both reach 98.5% or higher on every metric, with a ROC-AUC of about 1 |
| What do we need for a real comparison? | A held-out test set, an aligned recipe, and several seeds (Section 3.2) |

**How big is one error?** On a test set of 1,200 images, one image is worth 0.083 percentage points. When accuracies are around 99%, differences of a few images are within noise. A paired bootstrap on the same test set gives an interval for the difference:

```python
import numpy as np

def bootstrap_accuracy_difference(y_true, pred_a, pred_b, n_boot=2000, seed=42):
    """95% interval for (accuracy of A) minus (accuracy of B) on the same test images."""
    rng = np.random.default_rng(seed)
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)                       # resample images with replacement
        diffs.append((pred_a[idx] == y_true[idx]).mean() - (pred_b[idx] == y_true[idx]).mean())
    return np.percentile(diffs, [2.5, 97.5])
```

If the interval contains 0, the data do not show a clear difference.

### 3.6 How to Interpret a Comparison

| Observation | Conclusion |
|---|---|
| The difference between frameworks is **smaller than the seed-to-seed variation** (or the bootstrap interval contains 0) | No meaningful difference. Both implementations agree, which **validates the models** (Goal 1) |
| One framework is **clearly and consistently** better | First look for a hidden difference (initialization, augmentation, preprocessing, the table in Section 3.2) before crediting the framework |
| Results differ **a lot** between frameworks | Suspect a bug: check label order, preprocessing, augmentation on validation or test data, `train` / `eval` modes, and which checkpoint was evaluated |

### 3.7 Practical Recommendations: Keras or PyTorch?

| Need | Better fit | Reason |
|---|---|---|
| **Rapid prototyping and deployment** | Keras | Fast to set up, high-level API, easy to deploy for standard tasks |
| **Exploratory work and custom architectures** | PyTorch | Full control and transparency over the model and the training loop |
| **Large-scale distributed training** | Both | Both are capable; the TensorFlow ecosystem offers more out-of-the-box solutions for enterprise deployment |
| **Debugging inside the training loop** | PyTorch | We write and can inspect every step |
| **Least code for a standard classifier** | Keras | `compile` and `fit` replace the manual loop |

### 3.8 Part 3 Summary

- A fair comparison needs the **same test data** and the **same metrics**, with the dataset splits, architecture, preprocessing, hyperparameters, seeds, and output/loss matched.
- When the conditions are controlled, both frameworks reach **comparable predictive performance**.
- The key metrics are **accuracy, precision, recall, F1 score, ROC-AUC, and the confusion matrix**.
- Our labs match only 3 of 8 requirements, and they evaluate on training data, so we treat their numbers as a demonstration and follow the alignment plan for the final comparison.

---

## Part 4: Vision Transformers and Transfer Learning

> **Hands-on labs for this part:** notebooks `07` (CNN + Vision Transformer in Keras) and `08` (CNN + Vision Transformer in PyTorch).

CNNs are very good at **local** patterns. In this part we add a **Vision Transformer (ViT)** on top of our CNN to model **global** relationships, and we reuse the CNN we already trained as the starting point (transfer learning).

### 4.1 From CNNs to Transformers

CNNs extract local features efficiently with convolutional filters and spatial hierarchies. Their weakness is **global context**: to relate two distant regions, a CNN needs many stacked layers or very large kernels.

Vision Transformers take a different view. They treat an image as a **sequence of patches**, much like words in a sentence. Each patch becomes a vector, positional information is added, and transformer encoder blocks with **self-attention** let every patch attend to every other patch, however far apart they are.

| | **CNN** | **Vision Transformer** |
|---|---|---|
| **Sees** | Local neighborhoods, built up layer by layer | All patches at once |
| **Built-in assumptions** (inductive bias) | Locality and translation invariance | Almost none |
| **Data needs** | Works well on smaller datasets | Data-hungry |
| **Compute** | Efficient | Heavier (attention over all patch pairs) |

### 4.2 ViT Architecture

| Step | What happens |
|---|---|
| **1. Patchification** | The image is split into fixed-size patches (for example 16×16 pixels). Each patch is flattened and projected into a higher-dimensional vector |
| **2. Positional encoding** | Transformers have no sense of space, so a position vector is added to each patch embedding to retain where the patch came from |
| **3. Transformer encoder** | Stacked blocks of self-attention and feed-forward layers capture dependencies between all patches |
| **4. Classification head** | A special **[CLS] token** (or a pooled output) is used for the final prediction |

```text
Image → patches → [patch embedding + position] → Transformer block × N → [CLS] or pooled token → classifier
```

**Self-attention** compares every token with every other token and mixes their information according to the similarity:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^{T}}{\sqrt{d}}\right) V$$

Here $Q$, $K$, and $V$ are the query, key, and value matrices computed from the tokens, and $d$ is the size of each attention head. In **multi-head** attention, the embedding is split across several heads so that each head can look for a different kind of relationship.

### 4.3 The Hybrid CNN + ViT

Pure ViTs need large datasets and a lot of compute, because they assume so little about images. A **hybrid** model keeps the strengths of both:

1. The **CNN** processes the image and produces a feature map.
2. The feature map is divided into **patches (tokens)**.
3. Each token is embedded and passed, with **positional encoding**, into the transformer.
4. The transformer outputs are **pooled** and fed to a classifier.

The CNN extracts local features and shrinks the input, so the transformer works on far fewer tokens, which makes it more efficient and more robust on smaller datasets such as ours.

```text
Image (64×64×3) → CNN backbone → feature map → tokens → + position → Transformer blocks → pool → classifier
```

### 4.4 Transfer Learning

**Transfer learning** reuses a model that was already trained, instead of starting from random weights.

| Strategy | What is trained | When it fits |
|---|---|---|
| **Frozen backbone** | Only the new layers on top | Small dataset, quick results |
| **Fine-tuning** | New layers, plus some or all backbone layers (unfrozen) | More data, or a task that differs from the original |

Benefits: **faster training, less data needed, and often better performance**. In our project the backbone is the **CNN we trained ourselves** (Part 2). An ImageNet model is the other common choice:

```python
# Keras: an ImageNet-pretrained backbone without its classifier
cnn_base = keras.applications.ResNet50(include_top=False, weights="imagenet", pooling=None)
```

```python
# PyTorch: an ImageNet-pretrained backbone, frozen, with a new output layer
from torchvision import models
backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for p in backbone.parameters():
    p.requires_grad = False
backbone.fc = nn.Linear(backbone.fc.in_features, 2)
```

> **Token count depends on the backbone.** ResNet50 shrinks the image by a factor of 32, so a 64×64 input gives a 2×2 feature map (4 tokens), while a 224×224 input gives 7×7 (49 tokens). Keep this in mind when choosing the image size and the layer to take features from (Section 4.8).

### 4.5 Implementation in Keras (Lab 07)

**Plan of the lab:** load our trained CNN, pick an intermediate feature map, reshape it into tokens, add learnable positional embeddings, stack transformer blocks, pool, classify. Keras has no ready-made ViT block, so we define two small custom layers.

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

@tf.keras.utils.register_keras_serializable(package="Custom")
class AddPositionEmbedding(layers.Layer):
    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches, self.embed_dim = num_patches, embed_dim
        self.pos = self.add_weight(name="pos_embedding", shape=(1, num_patches, embed_dim),
                                   initializer="random_normal", trainable=True)   # learned positions

    def call(self, tokens):
        return tokens + self.pos

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "embed_dim": self.embed_dim})
        return config


@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads=8, mlp_dim=2048, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim, self.num_heads, self.mlp_dim, self.dropout = embed_dim, num_heads, mlp_dim, dropout
        self.mha = layers.MultiHeadAttention(num_heads, key_dim=embed_dim)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([layers.Dense(mlp_dim, activation="gelu"), layers.Dropout(dropout),
                                        layers.Dense(embed_dim), layers.Dropout(dropout)])

    def call(self, x):
        x = self.norm1(x + self.mha(x, x))          # self-attention + residual connection
        return self.norm2(x + self.mlp(x))          # feed-forward + residual connection

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads,
                       "mlp_dim": self.mlp_dim, "dropout": self.dropout})
        return config


def build_cnn_vit_hybrid(cnn_model, feature_layer_name, num_transformer_layers=4,
                         num_heads=8, mlp_dim=2048, num_classes=2):
    cnn_model.trainable = False                                  # frozen backbone (transfer learning)
    features = cnn_model.get_layer(feature_layer_name).output    # (batch, H, W, C)
    H, W, C = features.shape[1], features.shape[2], features.shape[3]
    x = layers.Reshape((H * W, C))(features)                     # every spatial position becomes a token
    x = AddPositionEmbedding(H * W, C)(x)
    for _ in range(num_transformer_layers):
        x = TransformerBlock(C, num_heads, mlp_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(cnn_model.layers[0].input, outputs, name="CNN_ViT_hybrid")
```

> **Note:** generic tutorials sometimes write `layers.TransformerBlock(...)`. That class does not exist in Keras; it must be defined as above (it is built on `layers.MultiHeadAttention`).

**Configuration used in the lab**

| Setting | Value |
|---|---|
| Backbone | Our trained Keras CNN (loaded with `load_model`), **frozen** |
| Feature layer | `batch_normalization_5` (the output of the last conv block) |
| Transformer | 4 blocks, 8 heads, MLP size 2048 |
| Output | `Dense(2, softmax)` with `categorical_crossentropy` (one-hot labels) |
| Optimizer | Adam, learning rate 0.0001 |
| Data | `ImageDataGenerator` (augmented, `class_mode="categorical"`), batch size 4 |
| Training | 3 epochs of 128 steps each (512 images per epoch), best model saved on validation loss |
| Shape check | A random `(1, 64, 64, 3)` input gives an output of shape `(1, 2)` |

**Lab results (CPU, about 1,130 seconds per epoch)**

| Epoch | Train acc | Train loss | Val acc | Val loss |
|---:|---:|---:|---:|---:|
| 1 | 0.9785 | 0.2710 | 0.9917 | 0.1343 |
| 2 | 0.9785 | 0.1995 | 0.9933 | **0.0589** |
| 3 | 0.9941 | 0.0644 | 0.9883 | 0.1031 |

### 4.6 Implementation in PyTorch (Lab 08)

PyTorch has no ViT block in the core library either, so the lab builds the pieces explicitly: a **patch embedding**, a **multi-head self-attention** module, a **transformer block**, the **ViT** with a [CLS] token, and the **hybrid** model that joins the CNN backbone and the ViT.

```python
class ConvNet(nn.Module):                       # same 6 conv blocks as the CNN in Part 2
    def __init__(self):
        super().__init__()
        ch = [3, 32, 64, 128, 256, 512, 1024]
        blocks = []
        for i in range(6):
            blocks += [nn.Conv2d(ch[i], ch[i + 1], 5, padding=2), nn.ReLU(),
                       nn.MaxPool2d(2), nn.BatchNorm2d(ch[i + 1])]
        self.features = nn.Sequential(*blocks)

    def forward_features(self, x):
        return self.features(x)                 # (B, 1024, H, W)


class PatchEmbed(nn.Module):                    # 1×1 convolution = learned linear projection per position
    def __init__(self, in_ch=1024, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=1)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)           # (B, L, D), with L = H × W


class MHSA(nn.Module):                          # multi-head self-attention
    def __init__(self, dim, heads=8, dropout=0.0):
        super().__init__()
        self.heads, self.scale = heads, (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop, self.proj, self.proj_drop = nn.Dropout(dropout), nn.Linear(dim, dim), nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(B, N, self.heads, -1).transpose(1, 2) for t in (q, k, v)]   # (B, heads, N, d)
        attn = self.attn_drop((q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1))
        return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(B, N, D)))


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(dim), MHSA(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))        # normalize first, then attention, then residual
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    def __init__(self, in_ch=1024, num_classes=2, embed_dim=768, depth=6, heads=8,
                 mlp_ratio=4.0, dropout=0.1, max_tokens=50):
        super().__init__()
        self.patch = PatchEmbed(in_ch, embed_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))              # learnable [CLS] token
        self.pos = nn.Parameter(torch.randn(1, max_tokens, embed_dim))     # learnable positions
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes)

    def forward(self, x):                                                  # x: (B, C, H, W)
        x = self.patch(x)
        B, L, _ = x.shape
        x = torch.cat((self.cls.expand(B, -1, -1), x), dim=1)              # prepend [CLS]
        x = x + self.pos[:, :L + 1]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x)[:, 0])                               # classify from the [CLS] token


class CNN_ViT_Hybrid(nn.Module):
    def __init__(self, num_classes=2, embed_dim=768, depth=6, heads=8):
        super().__init__()
        self.cnn = ConvNet()
        self.vit = ViT(num_classes=num_classes, embed_dim=embed_dim, depth=depth, heads=heads)

    def forward(self, x):
        return self.vit(self.cnn.forward_features(x))
```

**Configuration used in the lab**

| Setting | Value |
|---|---|
| Backbone | The CNN architecture from Part 2 (weights loaded with `load_state_dict(..., strict=False)`) |
| Training mode | **All parameters trained** (end to end) |
| Transformer | Depth 3, 6 heads, embedding size 768, MLP ratio 4, dropout 0.1 |
| Output | 2 logits with `CrossEntropyLoss` |
| Optimizer | Adam, learning rate 0.001 |
| Data | `ImageFolder` with `random_split` 80 / 20, batch size 32, 150 steps per epoch (a full pass) |
| Training | 5 epochs, best model saved on validation loss |

**Lab results (CPU, about 550 seconds per epoch)**

| Epoch | Train acc | Train loss | Val acc | Val loss |
|---:|---:|---:|---:|---:|
| 1 | 0.9467 | 0.2790 | 0.9808 | 0.0400 |
| 2 | 0.9723 | 0.0943 | 0.9925 | **0.0182** |
| 3 | 0.9673 | 0.1038 | **0.9933** | 0.0225 |
| 4 | 0.9781 | 0.0681 | 0.9883 | 0.0458 |
| 5 | 0.9692 | 0.0999 | 0.9867 | 0.0671 |

The best validation loss is at epoch 2, and later epochs show the usual sign of the model starting to overfit.

#### The depth experiment

The lab then trains a **deeper** hybrid (depth 12, 12 heads) with otherwise identical settings:

| | Depth 3, 6 heads | Depth 12, 12 heads |
|---|---:|---:|
| **Best validation loss** | **0.0182** (epoch 2) | 0.0762 (epoch 4) |
| **Validation accuracy across epochs** | 0.981, 0.993, 0.993, 0.988, 0.987 | 0.926, 0.976, **0.703**, 0.979, 0.826 |
| **Time per epoch** | about 550 s | about 950 s |

The deeper model is **slower and unstable**: its validation accuracy swings between 70% and 98%. The lab's own depth cheat sheet suggests a **lower learning rate for deeper models** (0.0005 at depth 12), but this run used 0.001.

| Depth | Heads | Typical dataset size | Suggested learning rate | Notes |
|---:|---:|---|---|---|
| 3 | 6 | under 1,000 | 0.001 | Learns low-level patterns; may underfit |
| 6 | 6 | under 1,000 | 0.001 | Good for simple tasks |
| 12 | 12 | 1,000 to 10,000 | 0.0005 | A common balance |
| 18 | 12 | 10,000 to 100,000 | 0.0003 | High performance on complex tasks |
| 24 | 16 | over 100,000 | 0.0001 | Diminishing returns, overfitting risk |
| 36 | 16 | over 100,000 | 0.0001 | Likely overkill for most tasks |

Treat this table as a **rule of thumb**, not a law. For our 6,000 images and 64×64 inputs, the shallow hybrid was the better choice.

### 4.7 Keras vs. PyTorch for Vision Transformers

| Aspect | Keras (TensorFlow) | PyTorch |
|---|---|---|
| **Ease of use** | High-level, intuitive API, fast prototyping | More verbose, highly flexible |
| **Pretrained models** | Good support through KerasCV and TensorFlow Hub | Extensive: `timm`, `torchvision`, Hugging Face |
| **Customization** | Less flexible for novel research architectures | Full control over architecture, data flow, and training loop |
| **Training loop** | Abstracted (`fit`, callbacks) | Manual and fully customizable |
| **Deployment** | Smooth with TensorFlow Serving and TFLite | More manual, but improving |
| **Community** | Strong in industry and production | Favored in research and academia |

### 4.8 Design Issues Worth Knowing

Reading the two hybrid labs closely shows five issues. They do not stop the models from reaching high scores, but they change what the scores mean.

| # | Issue | Detail | Suggested fix |
|---|---|---|---|
| 1 | **The transformer sees a single token** | A 64×64 input passes through six poolings, so the last conv feature map is **1×1×1024**. In Keras the hybrid therefore has **1 token**; in PyTorch it has 1 token plus the [CLS] token. Self-attention over one token has nothing to compare (its attention weight is trivially 1), so the model cannot use global context. Gains over the CNN come from the extra layers, not from spatial attention | Take features from an **earlier** conv block. The feature map after block 4 is 4×4×256 (**16 tokens**), after block 3 it is 8×8×128 (**64 tokens**). In PyTorch, raise `max_tokens` (it is 50) to fit the tokens plus [CLS] |
| 2 | **Frozen vs. fine-tuned backbone** | The Keras hybrid freezes the CNN; the PyTorch hybrid trains everything | Use the same strategy in both |
| 3 | **Unequal training budgets** | Keras: 3 epochs × 512 images (about a third of one pass over the training set), batch 4, learning rate 0.0001. PyTorch: 5 full epochs, batch 32, learning rate 0.001 | Match batch size, learning rate, and number of passes |
| 4 | **`strict=False` can hide a failed weight load** | The saved CNN weights come from an `nn.Sequential` model (keys such as `0.weight`), while `ConvNet` expects `features.0.weight`. With `strict=False`, a total key mismatch loads **nothing** and raises no error, so the backbone may start from random weights. We did not have the weight file to confirm, so this needs a check | Inspect the result of the load, and remap the keys if needed (code below) |
| 5 | **The data-pipeline bugs from Part 2** | The Keras generator also augments validation data; the PyTorch `random_split` code disables training augmentation | Use the fixes in Section 2.3 |

**Fix for issue 1 (Keras):** take an earlier feature layer.

```python
hybrid_model = build_cnn_vit_hybrid(cnn_model, feature_layer_name="batch_normalization_3",   # 4×4×256 → 16 tokens
                                    num_transformer_layers=4, num_heads=8, mlp_dim=2048, num_classes=2)
```

**Fix for issue 4 (PyTorch):** check what was loaded, and remap the keys if the names differ.

```python
state = torch.load(pytorch_state_dict_path, map_location="cpu")
result = model.cnn.load_state_dict(state, strict=False)
print("missing:", len(result.missing_keys), "| unexpected:", len(result.unexpected_keys))   # we want 0 missing

# if the names differ (Sequential keys '0.weight' vs 'features.0.weight'), remap the conv blocks (layers 0 to 23):
remapped = {f"features.{k}": v for k, v in state.items() if int(k.split(".")[0]) < 24}
model.cnn.load_state_dict(remapped, strict=True)
```

### 4.9 Part 4 Summary

- Vision Transformers treat images as **sequences of patches** and use **self-attention** to model local and global relationships.
- **Patchification** and **positional encoding** are the key steps that let a transformer process images.
- A **hybrid CNN + ViT** combines local feature extraction with global attention and suits smaller datasets.
- **Transfer learning** reuses a trained backbone, either frozen or fine-tuned.
- Keras is faster for prototyping; PyTorch gives more control and transparency.
- In our labs, the ViT sees only one spatial token, so the **choice of feature layer** is the first thing to improve.

---

## Part 5: Integration and Evaluation of the CNN-ViT Hybrids

> **Hands-on lab for this part:** notebook `09` (land classification: CNN-ViT integration and evaluation).

In this last part we take the two **pre-trained hybrid models** (one built in Keras, one in PyTorch), run the same images through both, and evaluate them with the same metrics. The point is a **framework-agnostic workflow**: import a model, feed it the right input format, extract comparable probabilities, and compute the same scores.

### 5.1 The Workflow

| Step | Keras | PyTorch |
|---|---|---|
| **1. Load the model** | `load_model(path, custom_objects={...})`: the file holds the architecture **and** the weights, but our custom layers (`AddPositionEmbedding`, `TransformerBlock`) must be known | Re-create the class definitions, build `CNN_ViT_Hybrid(...)`, then `load_state_dict(...)`: the file holds **weights only** |
| **2. Switch to inference** | Automatic in `predict` | `model.eval()` and `torch.no_grad()` |
| **3. Prepare the input** | `ImageDataGenerator(rescale=1/255)`, `shuffle=False`, `class_mode="categorical"` | `ImageFolder` with `Resize`, `ToTensor`, ImageNet `Normalize`, `DataLoader(shuffle=False)` |
| **4. Predict** | `model.predict(...)` returns softmax **probabilities** `(N, 2)` | The model returns **logits** `(N, 2)`; apply `softmax` ourselves |
| **5. Take the positive class** | `probs[:, 1]` | `softmax(logits)[:, 1]` |
| **6. Evaluate** | The same scikit-learn metric function for both: confusion matrix, classification report, ROC-AUC, log loss | |

```python
# Keras
from tensorflow.keras.models import load_model

keras_vit = load_model("keras_cnn_vit.model.keras",
                       custom_objects={"AddPositionEmbedding": AddPositionEmbedding,
                                       "TransformerBlock": TransformerBlock})
probs = keras_vit.predict(eval_generator)              # (6000, 2) softmax probabilities
y_prob_keras = probs[:, 1]                             # probability of 'agri'
y_true = eval_generator.classes                        # valid because shuffle=False
```

```python
# PyTorch
model = CNN_ViT_Hybrid(num_classes=2, heads=6, depth=3, embed_dim=768).to(device)
result = model.load_state_dict(torch.load("pytorch_cnn_vit.pth", map_location=device), strict=False)
print(result)                                          # check for missing or unexpected keys!
model.eval()

probs = []
with torch.no_grad():
    for x, _ in eval_loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu())
y_prob_torch = torch.cat(probs).numpy()
```

#### Differences that can break a cross-framework evaluation

| Difference | Keras | PyTorch | How to stay safe |
|---|---|---|---|
| **Model file** | Architecture and weights together | Weights only (`state_dict`) | Keep the class definitions in version control |
| **Output** | Probabilities (softmax inside the model) | Logits | Apply `softmax` before computing probabilities |
| **Pixel preprocessing** | Divide by 255 | ImageNet mean and std | Use exactly what the model was trained with |
| **Image order** | Alphabetical, fixed by `shuffle=False` | `ImageFolder` order, fixed by `shuffle=False` | Never shuffle an evaluation loader |
| **Loading** | Strict by default | `strict=False` can hide mismatches | Print and check the result of `load_state_dict` |

### 5.2 Results

Both hybrids are evaluated on **all 6,000 images** (see the caveat in Section 3.5).

| Metric | Keras CNN + ViT | PyTorch CNN + ViT |
|---|---:|---:|
| Accuracy | 0.9958 | **0.9990** |
| Precision (agricultural) | 0.9990 | 0.9990 |
| Recall (agricultural) | 0.9927 | **0.9990** |
| F1 score | 0.9958 | **0.9990** |
| ROC-AUC | 0.9998 | **1.0000** |
| Log loss | 0.0530 | **0.0047** |

| Keras CNN + ViT | Predicted non-agri | Predicted agri |
|---|---:|---:|
| **Actually non-agri** | 2997 (TN) | 3 (FP) |
| **Actually agri** | 22 (FN) | 2978 (TP) |

| PyTorch CNN + ViT | Predicted non-agri | Predicted agri |
|---|---:|---:|
| **Actually non-agri** | 2997 (TN) | 3 (FP) |
| **Actually agri** | 3 (FN) | 2997 (TP) |

#### Hybrid vs. CNN only

| Model | TN | FP | FN | TP | Errors | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Keras CNN | 3000 | 0 | 45 | 2955 | 45 | 0.9925 | 1.0000 | 0.9850 | 0.9924 |
| Keras CNN + ViT | 2997 | 3 | 22 | 2978 | 25 | 0.9958 | 0.9990 | 0.9927 | 0.9958 |
| PyTorch CNN | 2995 | 5 | 2 | 2998 | 7 | 0.9988 | 0.9983 | 0.9993 | 0.9988 |
| PyTorch CNN + ViT | 2997 | 3 | 3 | 2997 | 6 | 0.9990 | 0.9990 | 0.9990 | 0.9990 |

**Reading the results:**

- **PyTorch:** the hybrid and the CNN are essentially tied (6 vs. 7 errors out of 6,000).
- **Keras:** the hybrid halves the false negatives (45 → 22) but adds 3 false positives, so precision drops slightly (1.0000 → 0.9990) while recall rises (0.9850 → 0.9927).
- **Between frameworks:** the PyTorch hybrid has fewer errors (6 vs. 25), but the two were trained with different recipes (frozen vs. end-to-end backbone, batch size, learning rate, number of images seen; Section 4.8), so the framework is not the only explanation.
- **No evidence yet that the ViT helps.** Because the transformer sees a single spatial token (Section 4.8), it cannot add global context, and all of these scores come from data the models have mostly seen. The hybrid should be re-tested with an earlier feature layer and a held-out test set.

The lab's own advice points the same way: repeat the evaluation with a **labeled validation dataset**, compute precision, recall, and F1, and compare **speed and resource usage**.

### 5.3 Speed and Resource Usage

Accuracy is only one dimension. A fair timing needs a warm-up run, repeated measurements, and (on a GPU) a synchronization before reading the clock.

```python
import time
import torch

def time_keras(model, batch, n_runs=20):
    model.predict(batch, verbose=0)                          # warm-up
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model.predict(batch, verbose=0)
    return (time.perf_counter() - t0) / n_runs               # seconds per batch

def time_torch(model, batch, n_runs=20, device="cpu"):
    model.eval()
    batch = batch.to(device)
    with torch.no_grad():
        model(batch)                                         # warm-up
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(batch)
        if device == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_runs
```

Template to fill with measurements on the same machine and the same batch size:

| Measure | Keras CNN | PyTorch CNN | Keras CNN + ViT | PyTorch CNN + ViT |
|---|---|---|---|---|
| Parameters | 20,352,897 | 19,575,042 | TBD | TBD |
| Inference time per batch | TBD | TBD | TBD | TBD |
| Peak memory | TBD | TBD | TBD | TBD |
| Training time per epoch | about 276 s (CPU, epoch 1) | about 200 s (CPU) | about 1,130 s (CPU, 512 images) | about 550 s (CPU, 4,800 images) |

The training times come from different workloads (different batch sizes and numbers of images per epoch), so they are **not** a like-for-like speed comparison.

### 5.4 Part 5 Summary

- A framework-agnostic evaluation needs three things aligned: the **input format**, the **model loading**, and the **output type** (probabilities vs. logits).
- All four models reach 98.5% or higher on every metric, evaluated on data that includes their training images.
- The hybrids match or slightly improve on the CNNs, but the single-token design and the unequal recipes prevent any conclusion about the ViT or about the frameworks.

---

## Project Conclusions

### All Four Models Side by Side

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Errors (of 6,000) |
|---|---:|---:|---:|---:|---:|---:|
| CNN, Keras | 0.9925 | 1.0000 | 0.9850 | 0.9924 | 1.0000 | 45 (0 FP, 45 FN) |
| CNN, PyTorch | 0.9988 | 0.9983 | 0.9993 | 0.9988 | 1.0000 | 7 (5 FP, 2 FN) |
| CNN + ViT, Keras | 0.9958 | 0.9990 | 0.9927 | 0.9958 | 0.9998 | 25 (3 FP, 22 FN) |
| CNN + ViT, PyTorch | 0.9990 | 0.9990 | 0.9990 | 0.9990 | 1.0000 | 6 (3 FP, 3 FN) |

### Key Findings

| # | Finding |
|---|---|
| 1 | **The task is learnable.** Both CNNs and both hybrids separate agricultural from non-agricultural tiles almost perfectly (ROC-AUC about 1.0). Every model reaches 98.5% or higher, although on data it has mostly seen |
| 2 | **We cannot rank the frameworks yet.** The recipes differ in splits, architecture, preprocessing, output and loss, and training budget, and the evaluation overlaps the training data (Section 3.2 and Section 3.5) |
| 3 | **The ViT adds little in the labs, for a known reason.** The transformer receives one spatial token, so it cannot model global context (Section 4.8) |
| 4 | **Workflows differ as expected.** Keras is faster to a first model (`compile` and `fit`); PyTorch gives full control and makes the training loop easy to inspect |
| 5 | **Precision-first view.** The Keras CNN never over-counts farmland (precision 1.0000) but under-counts it (45 misses). The PyTorch models are more balanced. Because ROC-AUC is about 1, choosing the threshold on a validation set can trade these errors off |

### Pitfalls Found Along the Way

| Where | Pitfall | Fix |
|---|---|---|
| Data (Keras) | `cache()` after augmentation freezes the random augmentation | Cache before augmenting (Section 1.4) |
| Data | Different rotation units (fraction of a turn vs. degrees) | Convert units (Section 1.4) |
| Data (PyTorch) | One shared `transform` after `random_split` disables training augmentation | Two datasets, split the indices once (Section 2.3) |
| Data (Keras) | The augmenting generator is also used for validation | A separate rescale-only generator (Section 2.3) |
| Training | The last-epoch model is evaluated instead of the best checkpoint | Reload the best checkpoint (Section 2.5) |
| Evaluation | Metrics computed on all images, including training data | A held-out test set (Section 3.5) |
| Hybrid | The ViT sees a single token | Use an earlier feature map (Section 4.8) |
| Hybrid | `strict=False` can silently skip the backbone weights | Check the result, remap keys (Section 4.8) |

### Limitations

- All reported scores come from the lab runs of 3 to 5 epochs (20 for the downloaded models), a single seed, and no held-out test set.
- The classes are balanced and the images are small (64×64), which makes the task easier than real regional mapping.
- If tile locations are known, random splits can overestimate performance, because neighboring tiles look alike. Splitting by geographic region is a stricter test.
- The fertilizer sales forecast itself is not evaluated here.

### Next Steps

| Step | Goal |
|---|---|
| **Re-run with an aligned recipe** | Shared split files with a test set, identical architecture, same output and loss, same preprocessing, and 3 seeds (Section 3.2) |
| **Fix the hybrid** | Use an earlier feature layer so the transformer sees 16 to 64 tokens, and compare frozen vs. fine-tuned backbones |
| **Choose the threshold for precision** | Pick it on the validation set for a target precision (Section 3.3) |
| **Try an ImageNet backbone** | Transfer learning with ResNet and a larger input size |
| **Add speed and memory measurements** | Complete the table in Section 5.3 |
| **Connect to the forecast** | Feed the estimated agricultural area into the fertilizer sales model |

---

## Appendix A: Data and Setup

| Item | Details |
|---|---|
| **Dataset** | `images_dataSAT`, as used in the labs: 6,000 RGB images (64×64), 3,000 per class, in `class_0_non_agri` and `class_1_agri` |
| **How the labs obtain it** | Each notebook downloads and extracts `images-dataSAT.tar` (about 20 MB) from course storage; the links are in the notebooks |
| **Origin** | The file names follow the Sentinel-2 Level-2A pattern, which suggests the images derive from Sentinel-2 data. **Confirm the dataset's origin and license before publishing** |
| **In the repository** | Raw data is **not** committed. Describe the download steps instead |

**Software versions used in the labs**

```text
tensorflow==2.19        # includes Keras 3
torch==2.8.0
torchvision==0.23.0
scikit-learn==1.7.0
numpy==1.26.4
pandas==2.2.3
matplotlib==3.9.2
tqdm
```

**Run order:** the notebooks in `notebooks/`, from `01` to `09`.

## Appendix B: Reproducibility Checklist

- [ ] Fixed random seeds (for example 42, 43, 44) in Python, NumPy, TensorFlow/Keras, and PyTorch (including DataLoader workers)
- [ ] One set of split files (train, validation, **test**), reused by both frameworks
- [ ] Positive class (`agricultural` = 1) verified in both pipelines (`class_names`, `class_to_idx`)
- [ ] Augmentation applied to **training data only** (check the Keras validation generator and the PyTorch `transform` assignment)
- [ ] Same architecture and parameter count in both frameworks
- [ ] Same output activation and loss; explicit initializers, Adam epsilon, padding, and rotation fill
- [ ] Same hyperparameters: optimizer, learning rate, batch size, and number of epochs
- [ ] Best checkpoint reloaded before the final evaluation
- [ ] Validation and test loaders use `shuffle=False`
- [ ] `load_state_dict(strict=False)` results printed and checked
- [ ] All metrics computed by the same script from saved predictions
- [ ] Test set used **once**, after the hyperparameters were chosen
- [ ] Library versions recorded in `requirements.txt`

## Appendix C: Suggested Presentation Outline

| # | Slide | Key message | Visual |
|---:|---|---|---|
| 1 | Title and problem | Classify satellite tiles as agricultural or not | Example images of both classes |
| 2 | Business value | Farmland estimates feed fertilizer sales forecasts; precision guards against over-forecasting | Error-effect table |
| 3 | Dataset | 6,000 balanced 64×64 images, two classes | Class folders and samples |
| 4 | Loading methods | Memory-based vs. generator-based | Comparison table and memory formula |
| 5 | Efficient pipelines | Custom generators vs. built-in utilities, in both frameworks | Side-by-side code |
| 6 | Augmentation | Label-preserving transforms for satellite images; the lab pitfalls | Grid of augmented samples |
| 7 | CNN architecture | Same convolutional extractor, different heads | Architecture table |
| 8 | Hyperparameters | Optimization, regularization, batch size, epochs | Lab settings table |
| 9 | Fair comparison | What must be standardized, and our audit (3 of 8 met) | Audit table |
| 10 | Metrics | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix | Confusion matrix |
| 11 | CNN results | Keras vs. PyTorch CNN, with the caveat about evaluation data | Metrics table and ROC overlay |
| 12 | Vision Transformers | Patches, positional encoding, attention, hybrid CNN + ViT | Architecture diagram |
| 13 | Hybrid results | All four models side by side; the single-token finding | Comparison table |
| 14 | Lessons and next steps | Pitfalls found, aligned re-run, threshold for precision | Pitfall table |

## Appendix D: Acknowledgements

The hands-on practice in this project follows the lab notebooks of the *AI Capstone Project with Deep Learning* course (IBM, on Coursera). The text, explanations, and code in this README were written and adapted for this project. If notebooks or text from the course are added to a public repository, check the course's terms of use first.
