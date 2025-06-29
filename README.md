# Cross-Silo Federated Learning: Addressing Small-Sample, Non-IID Challenges in Healthcare Applications via Synthetic Data Augmentation

A bachelor project focused on implementing and evaluating Federated Learning (FL) techniques for healthcare datasets.

## Abstract
Cross-silo federated learning (FL) offers hospitals a way to jointly train predictive models without centralizing sensitive patient data. However, when individual clients possess only few training examples with imbalanced class distributions, standard FedAvg deteriorates dramatically, resorting to majority-class guessing. We examine this phenomenon on two distinct healthcare datasets: MIMIC-III (tabular time series for in-hospital mortality prediction) and a Brain Tumor MRI image set. The datasets are partitioned across eight clients using a beta distribution with concentration parameters $\alpha \in \{0.1, 1, 10, 100, \infty\}$ to simulate a distributed environment with different levels of heterogeneity. We experiment with both large (up to 16,000 ICU stays; 200 MRIs) and small (800 ICU stays; 48 MRIs) training set sizes. We find that the performance of regular FedAvg degrades significantly under especially non-IID conditions, with AUC-ROC scores dropping to 0.62 on MIMIC-III and 0.48 on Brain Tumor under extreme heterogeneity ($\alpha$ = 0.1), and minority-class recall collapsing to near-zero, illustrating how sparse, skewed data cause clients to default toward majority class predictions. To address these challenges, we propose FedAug: each client locally trains a label-conditional generative model to produce synthetic samples for every class, which are then shared via the central server. Under extreme data heterogeneity, FedAug improved AUC-ROC by up to 10 pp. on MIMIC-III and 38 pp. on the Brain Tumor dataset, and increased Average Precision by 10 and 23 pp., respectively, demonstrating its ability to counteract federated models’ bias toward the majority class. Under IID or moderately heterogeneous settings ($\alpha$ ≥ 10), FedAug does not improve performance substantially, indicating that synthetic augmentation is most beneficial when client data are limited and non-IID. We evaluate synthetic sample generalization and find minimal overfitting and memorization, although more extensive evaluations would be needed to ensure that the synthetic samples do not leak any sensitive information about the local datasets. By restoring discriminative power under sub-optimal conditions without undermining privacy, FedAug paves the way for more reliable, privacy-respecting FL in healthcare.

## Datasets
This project uses two datasets:

### Brain Tumor MRI Dataset
The Brain Tumor dataset is a publicly available collection of brain MRI images from [Kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/data). This dataset contains MRI scans that are divided into two classes: “yes” (156 samples) for images with the presence of a brain tumor and “no” (95 samples) for images without a brain tumor.

The Brain Tumor MRI dataset is preprocessed by loading the “yes” (tumor) and “no” (no tumor) image folders, converting every scan to RGB, and resizing each image to 64 × 64 pixels. Each image is then transformed into a tensor and pixel values are normalized to the [0, 1] range via division by 255. Finally, scans in the “yes” folder were labeled 1 and those in the “no” folder labeled 0, yielding a uniform set of (image_tensor, label) pairs ready for model training.

The prediction task is binary classification for the presence of a Brain Tumor.

### MIMIC-III
The MIMIC-III (Medical Information Mart for Intensive Care III) dataset is a large, publicly available database that contains de-identified electronic health record (EHR) data collected from patients admitted to critical care units at Beth Israel Deaconess Medical Center between 2001 and 2012. The dataset is available via [PhysioNet.org](https://physionet.org/content/mimiciii/1.4/).

The preprocessing of the MIMIC-III dataset follows the preprocessing steps found in [YerevaNN/mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks) for the in-hospital mortality prediction task. The dataset is filtered to adult ICU stays (≥18 years) with at least 48 hours of records and linked across ICUSTAYS, PATIENTS, ADMISSIONS, CHARTEVENTS, LABEVENTS, and OUTPUTEVENTS. Time-series measurements are binned into one-hour intervals over the first 48 hours, taking the latest value per bin, and missing entries are forward-filled (or imputed with global means/modes) with binary masks indicating imputation. Continuous outliers beyond 1.5×IQR are clipped and later imputed, 17 key vitals and labs were extracted (12 continuous, 5 categorical one-hot encoded), and each ICU stay labeled for in-hospital mortality.

The prediction task is binary classification for in-hospital mortality based on vitals of the first 48 hours in the intensive care unit.

**Note**: The data folder is not included in the GitHub repository due to size constraints and privacy concerns. To use this code, you will need to obtain these datasets separately.

## Python Environment Setup

To run this code, you need Python 3.11.9 and the following packages:

```bash
# Create a new virtual environment
python -m venv fl-env
source fl-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

The main dependencies include:
- torch (2.6.0)
- numpy (1.26.4)
- tensorflow (2.18.0)
- scikit-learn (1.6.1)
- prefect (3.2.15)
- pandas (2.2.3)
- and others listed in requirements.txt

## Running the Pipeline

The ML pipeline can be executed using the `pipeline/main.py` script with various command-line arguments defined in `pipeline/arguments.py`.

See the [pipeline README](pipeline/README.md) for more details.

## Paper
The findings of this research are documented in the paper found at [https://drive.google.com/file/d/12N2VapvAXbFYGQooYz1afIvxX3aKc2-E/view?pli=1](https://drive.google.com/file/d/12N2VapvAXbFYGQooYz1afIvxX3aKc2-E/view?pli=1).
