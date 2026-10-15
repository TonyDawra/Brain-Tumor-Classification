# Brain Tumor Classification with Deep Learning

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google-Colab-F9AB00?logo=googlecolab&logoColor=white)

A comparative deep learning project for **classifying brain MRI scans into 15 categories** (14 tumor/lesion types and one normal class). The project explores transfer learning with pretrained convolutional neural networks (CNNs), compares them against a custom CNN trained from scratch, and evaluates how well the models generalize to different MRI datasets.

**Best internal test accuracy: 94.41% (EfficientNetB5).**

**[View the notebook](Brain-Tumor-Classification.ipynb)** · **[Open in Google Colab](https://colab.research.google.com/github/TonyDawra/Brain-Tumor-Classification/blob/main/Brain-Tumor-Classification.ipynb)**

> **Research and educational use only.** This project is not a medical diagnostic tool, and its predictions should not be used to make clinical decisions.

## Project objectives

- Build an image-classification pipeline for brain MRI scans.
- Compare **EfficientNetB0**, a simplified/frozen-backbone **EfficientNetB0**, **EfficientNetB5**, **DenseNet121**, and a **custom CNN**.
- Address class imbalance with training-only image augmentation.
- Evaluate performance using **accuracy, macro F1-score, classification reports, and confusion matrices**.
- Test model generalization on external MRI datasets with different class definitions and image characteristics.

## Datasets

### Main training dataset

**[Brain Tumor MRI Images – 44C (Kaggle)](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c)** by `fernando2rad`.

The notebook uses these **15 classes**:

| | | |
|---|---|---|
| Astrocitoma | Carcinoma | Ependimoma |
| Ganglioglioma | Germinoma | Glioblastoma |
| Granuloma | Meduloblastoma | Meningioma |
| Neurocitoma | Oligodendroglioma | Papiloma |
| Schwannoma | Tuberculoma | Normal (`_NORMAL`) |

The source data contains **4,479 images** in the notebook run. The stratified image-level split is:

| Partition | Proportion | Images |
|---|---:|---:|
| Training (before augmentation) | 80% | 3,583 |
| Validation | 12% | 537 |
| Test | 8% | 359 |

After balancing, the training generator reports **4,583 images**. The test generator successfully loads **358 images**, which is the sample count used in the reported internal test evaluation.

### External evaluation datasets

To examine generalization beyond the original dataset, the notebook also evaluates models using:

1. **[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)** (`masoudnickparvar`) — evaluated through a three-class label mapping.
2. **[Brain Tumor MRI Images – 17 Classes](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-17-classes)** (`fernando2rad`) — evaluated using matched class labels and a grouped-label comparison.

External labels are mapped or grouped to align with the project's trained outputs. These results are **not directly comparable** to the original 15-class test accuracy.

## Methodology

The notebook follows this workflow:

1. **Data preparation:** Download the MRI dataset through `kagglehub`, build image paths and labels, and create stratified train/validation/test splits.
2. **Class balancing:** Generate additional training images for underrepresented classes using rotations, shifts, shear, zoom, and horizontal flips; apply a small majority-class reduction where needed. Validation and test images are not augmented.
3. **Preprocessing:** Resize images to **224 × 224 pixels**, use three input channels, and construct Keras image generators with a **batch size of 64**.
4. **Training:** Train four ImageNet-initialized transfer-learning configurations and a custom CNN using **Adamax**, categorical cross-entropy, dropout, and model-specific regularization. The reusable training function also supports **early stopping** and **learning-rate reduction**.
5. **Evaluation:** Track learning curves, calculate classification metrics, and inspect confusion matrices.
6. **External testing:** Assess models on additional MRI collections and study the effect of domain shift and label differences.

### Models compared

| Model | Approach |
|---|---|
| **EfficientNetB0** | ImageNet-pretrained backbone with a custom classification head |
| **EfficientNetB0 (Simplified)** | Frozen ImageNet-pretrained backbone with a smaller classification head |
| **EfficientNetB5** | Larger ImageNet-pretrained EfficientNet backbone |
| **DenseNet121** | ImageNet-pretrained densely connected CNN |
| **Custom CNN** | CNN trained from scratch using convolutional blocks, batch normalization, pooling, and dropout |

## Results

### Internal test-set performance

Results below are taken from the notebook's saved evaluation output (**358 loaded test images**, 15 classes).

| Model | Test accuracy | Macro F1-score | Parameters |
|---|---:|---:|---:|
| **EfficientNetB5** | **94.41%** | **0.9198** | 29.04M |
| EfficientNetB0 | 94.13% | 0.9141 | 4.38M |
| DenseNet121 | 93.02% | 0.9129 | 7.30M |
| EfficientNetB0 (Simplified) | 83.52% | 0.8134 | 4.22M |
| Custom CNN | 75.14% | 0.7281 | 1.21M |

**EfficientNetB5 achieved the highest internal test accuracy**, while the original EfficientNetB0 delivered a similar score with substantially fewer parameters.

### External dataset performance

The notebook includes several different external evaluation protocols. Their results are reported separately to avoid confusing restricted-label accuracy with full 15-class accuracy.

| Model | 3-class external test¹ | 5 matched classes² | Grouped-label test² |
|---|---:|---:|---:|
| EfficientNetB0 | 44.25% | 82.71% | 87.18% |
| EfficientNetB0 (Simplified) | 41.83% | 77.61% | 82.18% |
| EfficientNetB5 | 47.42% | 81.10% | 87.90% |
| **DenseNet121** | 37.92% | **85.16%** | **89.49%** |
| Custom CNN | 37.08% | 76.67% | 74.94% |

¹ Evaluation on the `masoudnickparvar` dataset using **1,200 images** and predictions restricted to three mapped classes.  
² Evaluation on the `fernando2rad` 17-class dataset using **4,158 validated images**. The matched-class experiment evaluates five mapped classes; the grouped experiment aggregates the original classes into broader groups. The grouped report includes a category with zero reference examples, so those grouped metrics require particular care when interpreting.

The external experiments demonstrate that **strong in-distribution results do not guarantee equally strong performance on new MRI collections**. Differences in scan orientation, acquisition conditions, image appearance, and label definitions can produce significant domain shift. DenseNet121 performed best in the matched-class and grouped-label evaluations on the second external dataset.

## How to run

**Recommended environment: Google Colab with a GPU runtime.** The notebook uses Google Colab and Google Drive paths; running it locally would require adapting those paths and the Drive-mount cell.

1. Open the notebook with the **[Google Colab link](https://colab.research.google.com/github/TonyDawra/Brain-Tumor-Classification/blob/main/Brain-Tumor-Classification.ipynb)**.
2. Select **Runtime → Change runtime type → GPU** (if available).
3. Install any missing packages:

   ```python
   !pip install tensorflow kagglehub numpy pandas scikit-learn pillow matplotlib seaborn plotly requests
   ```

4. Mount Google Drive when prompted and run the notebook cells **in order**. The main dataset is downloaded automatically with `kagglehub`.
5. Review the training curves, per-class performance, confusion matrices, and external testing sections.

The notebook writes cached data, augmented images, trained Keras models, and related files under:

```text
/content/drive/MyDrive/brain_tumor_project/
├── saved_data/
├── augmented_images/
└── models/
```

The cached file paths are environment-specific: if you switch runtimes or change the dataset location, rebuild the cache by setting `FORCE_REBUILD_CACHE = True` in the notebook. **Model weights and datasets are not included in this GitHub repository.** Training may require substantial GPU time and Google Drive storage.

## Repository structure

```text
Brain-Tumor-Classification/
├── Brain-Tumor-Classification.ipynb   # Complete experiments and visualizations
└── README.md                         # Project overview and instructions
```

## Limitations and future improvements

- **Generalization:** External datasets show meaningful performance changes due to distribution shift.
- **Label alignment:** External evaluation depends on mapping or grouping class names; not all source categories correspond perfectly.
- **Data splitting:** The notebook uses stratified *image-level* splits; patient-level independence has not been verified. Patient-level splitting would be preferable for medical imaging claims.
- **Clinical validation:** Independent clinical datasets, stronger error analysis, calibration, and expert validation would be required before any real-world clinical use.
- **Reproducibility:** Package versions are not pinned, and saved-data paths are specific to Google Colab/Drive.

Future work could include patient-level validation, more diverse MRI data, systematic hyperparameter optimization, class-specific performance analysis, and model interpretability methods such as Grad-CAM.

## Tools and technologies

**Python · TensorFlow/Keras · scikit-learn · NumPy · pandas · Matplotlib · Seaborn · Plotly · Pillow · KaggleHub · Google Colab**

## Author

**Tony Dawra** — [GitHub @TonyDawra](https://github.com/TonyDawra)

## References

- [Main dataset — Brain Tumor MRI Images 44C](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c)
- [External dataset — Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [External dataset — Brain Tumor MRI Images 17 Classes](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-17-classes)
- [TensorFlow: Transfer learning and fine-tuning](https://www.tensorflow.org/guide/keras/transfer_learning)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://proceedings.mlr.press/v97/tan19a.html)
- [Densely Connected Convolutional Networks](https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html)
