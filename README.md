# StreamlineML

## Overview

StreamlineML is a Python library designed to streamline machine learning workflows with high-level abstractions, multi-threading capabilities, and efficient data structures. It addresses common issues such as data redundancy, project management complexity, and data security, making the ML development process more efficient and less error-prone.

## Problem Statement

### 1. Data Redundancy
Data redundancy consumes time and memory as users maintain multiple copies to prevent accidental modifications.

### 2. Multiple Projects
Users have to maintain a directory structure in the file explorer to manage projects, which becomes stressful after a point.

### 3. Data Security
Performing file operations is risky due to potential data loss, high expense, and computational complexity, making errors costly.

## Proposed Solution

StreamlineML leverages specific data structures to tackle the problems outlined above:

- **N-ary Tree**: Organizes datasets into test-train splits and stores raw data alongside its necessary metadata.
- **TreeMap (Hashmap of AVL Tree)**: Manages raw data efficiently with operations like insertion, deletion, and searching.
- **Singly-linked Lists and Queues**: Constructs a custom data pipeline featuring an image transformer, facilitating modifications to each image.

## Key Features

- **Data Manager**: Efficiently manages datasets and metadata.
- **Data Loader**: Loads data into the pipeline for processing.
- **Data Transformer**: Applies transformations to data, such as image preprocessing.

## Future Development

StreamlineML will evolve beyond data loading and pipeline creation to offer a full spectrum of machine learning functionalities, including:

- Architectural design
- Modeling
- Training
- Testing
- Generation of evaluation matrices

Ultimately, it aims to become a comprehensive machine learning library, providing users with robust tools to tackle diverse challenges in the field.

## Usage

### Managing Datasets

```python
from MLDevHelper.DataManager import dataset_manager

# Adding datasets
dataset_manager.add_dataset("dataset1", r".\sample_dataset", 0.8, 0.1, 0.1, ["jpg", "jpeg", "png"], balance=True)
dataset_manager.add_dataset("dataset2", r".\sample_dataset", 0.8, 0.1, 0.1, ["jpg", "jpeg", "png"], balance=False)

# Deleting datasets
dataset_manager.delete_dataset("dataset1")
dataset_manager.delete_dataset("dataset2")

# Viewing dataset heads
x, y, z = dataset_manager.show_dataset_heads("dataset1")
print(x)

# Getting dataset splits
train = dataset_manager.get_train_df("dataset1")
print(train["labels"].value_counts())
test_df = dataset_manager.get_test_df("dataset1")
print(test_df["labels"].value_counts())
val_df = dataset_manager.get_val_df("dataset1")
print(val_df["labels"].value_counts())
raw_df = dataset_manager.get_raw_df("dataset1")
print(raw_df["labels"].value_counts())
```

### Transforming Images

```python
from MLDevHelper.Transformer import TransformerList
import cv2  
import matplotlib.pyplot as plt 

# Loading and displaying an image
image, label = train.iloc[0]
image = cv2.imread(rf"{image}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Creating a list of transformations
transformer_list = TransformerList()
transformer_list.create_transformer_list(
    [
        transformer_list.resize_image((200, 200)),
        transformer_list.color_jitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),
        transformer_list.random_rotate_image(angle=30),
        transformer_list.random_horizontal_flip(threshold=0.5),
        transformer_list.random_vertical_flip(threshold=0.5),
    ]
)

# Applying transformations
transformed_image = transformer_list.forward(image)

# Displaying original and transformed images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(transformed_image)
plt.title("Transformed Image")
plt.show()
```

### Loading Data with Transformations

```python
from MLDevHelper.DataLoader import image_loader

# Creating a data loader with transformations
train_loader = image_loader(train, 32, transformer_list, 4, True, 69)

# Displaying a batch of images
for batch in train_loader:
    images = batch["images"] 
    labels = batch["labels"]  
    for image, label in zip(images, labels):
        plt.imshow(image)
        plt.title(label)
        plt.show()
        break
```

### Using HashMaps for Efficient Data Management

```python
from MLDevHelper import hashMaps

# Creating and displaying a hash map
hmap = hashMaps.HashMaps()
hmap.bulk_insert_from_dataframe(train)
hmap.display_map()

# Converting hash map to DataFrame
df = hmap.to_dataframe()
print(df)

# Adding images to dataset
dataset_manager.add_images(images, word_label, "dataset1", True)
```

## Documentation

Detailed documentation is coming soon.

## Contributing

We welcome contributions!

