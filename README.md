# Champion Solution for Track 2 of the IEEE GRSS Data Fusion Contest: All-Weather Building Damage Mapping

We are delighted to share that our paper has been successfully accepted by the IEEE International Geoscience and Remote Sensing Symposium (IGARSS), 2025 (Oral).[Paper Link](https://arxiv.org/abs/2505.04941).

This repository contains the code for our paper:  
**[Building-Guided Pseudo-Label Learning for Cross-Modal Building Damage Mapping](https://arxiv.org/abs/2505.04941)**.


## Code Execution Instructions

Our detection pipeline consists of two main stages:
1. **Single-Class Building Extraction**: Extracts building footprints from pre-disaster optical imagery (located in the `BuildingExtraction` folder).
2. **Building Damage Change Detection**: Identifies building damage by comparing pre- and post-disaster images (located in the `ChangeDetection` folder).

## Step 1: Single-Class Building Extraction
Navigate to the `BuildingExtraction` directory and run the inference scripts for different scenes:
```bash
cd BuildingExtraction
bash inference.sh
```

### Step 2: Building Damage Change Detection
Navigate to the `ChangeDetection` directory and execute the change detection process:
```bash
cd ../ChangeDetection
bash inference.sh  # Performs test-time augmentation for coarse change detection, followed by a connected-regions-based merging strategy for refinement
```

Upon completion, the final detection results will be saved in `./cd_results/post_pred/`. Ensure all required dependencies are installed before executing the scripts.

---

## Training Instructions

We follow a two-stage training strategy:
1. **Single-Class Building Extraction**: Training on pre-disaster optical imagery (`BuildingExtraction` folder).
2. **Building Damage Change Detection**: Training on both pre- and post-disaster images (`ChangeDetection` folder).


### Single-Class Building Extraction Training
#### Dataset Preparation
We divide the official training dataset into two subsets: **train set** and **test set** (refer to `./train_test_set/`).
#### Model Training
We propose three versions of a **Pyramid Vision Transformer (PVT)**-based building extraction model, called **UFPN**, using different backbones: `PVT-v2-b2`, `PVT-v2-b3`, and `PVT-v2-b4`. In total, six models are trained.

During inference, we apply multi-model fusion and test-time augmentation strategies to generate pseudo-labels for buildings. These pseudo-labels include both binary segmentation results and predicted probability maps, which are stored as `.npy` files.

#### Pseudo-Label-Based Training
Additionally, we adopt a **low-uncertainty-based pseudo-label training method** (refer to `train_pseudo_label.py`) to further refine building segmentation results.

### Training Process
Run the following commands to train and validate the models:
```bash
cd BuildingExtraction
bash train.sh  # Train six models
bash test.sh  # Generate predicted probability maps stored in .npy format
bash train_pseudo_post.sh  
bash test_pseudo_post.sh # Perform inference usinglow-uncertainty pseudo-label training and post-processing strategies
```

By following these steps, we obtain refined and high-quality building segmentation results, which serve as the foundation for the subsequent change detection stage.

---

## Building Damage Change Detection

Based on the official training dataset from the **2025 IEEE GRSS DFC**, we adapted **UABCD**, a model introduced in our ISPRS paper: *Overcoming the Uncertainty Challenges in Detecting Building Changes from Remote Sensing Images*. Specifically, we trained a **PVT-v2-b2**-based change detection network using the following commands:

```bash
cd ../ChangeDetection
bash train.sh
```

### Dataset Preparation and Model Training

To ensure robust performance, we evaluated the trained models on the validation dataset (Phase 1) and selected the best-performing checkpoints from epochs **27, 29, 30, 33, 38, and 39**. We then employed **multi-model fusion** and **test-time augmentation** strategies to generate pseudo-labels for multi-class change detection. These pseudo-labels include both change detection results and corresponding probability maps, which are saved as `.npy` files.

During the competition, we observed that the **Hawaii-wildfire dataset** (provided in the training set) exhibited similarities in imagery resolution and building distribution characteristics with the test dataset. Leveraging this insight, we converted the **Hawaii-wildfire dataset** labels into a one-hot encoded `.npy` format and merged them with the probability maps generated in the previous step. This resulted in an enriched training dataset for pseudo-label-based training.

Execute the following command to generate the necessary pseudo-labels and prepare the dataset:

```bash
cd ../ChangeDetection
bash test.sh  # Generates pseudo-labels for the test dataset, converts the Hawaii-wildfire dataset into .npy format, and merges the datasets.
```

### Building-Guided Pseudo-Label-Based Training
Using the merged dataset, we employed a **building-guided low-uncertainty-based pseudo-label training method** (see `train_pseudo_label.py`) to refine the building damage detection results. This approach utilizes feature-level guidance derived from the generated building results, leveraging accurate building labels from the Hawaii-wildfire dataset and building predictions from the Single-Class Building Extraction process for the test dataset, to enhance detection accuracy.

To train and validate the models, run the following commands:
Using the merged dataset, we applied a **building-guided low-uncertainty-based pseudo-label training method** (see `train_pseudo_label.py`) to refine the building damage detection results based on the feature-level guidance by the generated building results (Hawaii-wildfire dataset: correct building labels; test dataset: building results from **Single-Class Building Extraction**).

To train and validate the models, run the following commands:

```bash
cd ../ChangeDetection
bash train_pseudo_post.sh  # Train six models
bash test_pseudo_post.sh  # Perform inference using low-uncertainty pseudo-label training and post-processing strategies
```

> **Note:** Ensure that all necessary dependencies and environment configurations are properly set up before executing the scripts. Besides, change the path of **test_root** to your path.
