# tfMRI Classifier

This repository contains code for classifying cognitive tasks using functional connectivity features extracted from task-based fMRI data. It used data from the AOMIC dataset (ds002785), specifically the preprocessed outputs from fMRIPrep.

The goal is to evaluate how well models trained on functional connectomes can classify task states, and investigate the contribution of different brain networks to task classification through ablation studies.

## 🧪 Experiments

### 1. Binary Classification
Classify between task and resting state using XGBoost:
```bash
python -m tfmri_classifier.modelling.binary_classifier --task workingmemory
```

### 2. Hyperparameter Tuning
Optimize model performance through grid search:
```bash
python -m tfmri_classifier.modelling.tune_top_classifiers --task workingmemory
```

### 3. Multi-class Classification
Classify between all available tasks:
```bash
python -m tfmri_classifier.modelling.train_multiclass
```

### 4. Network Ablation Study
Assess how classification performance degrades as networks are progressively removed:
```bash
python -m tfmri_classifier.modelling.ablation_study
```

### 5. Task Embedding Analysis
Visualize task relationships using dimensionality reduction (PCA -> t-SNE):
```bash
python -m tfmri_classifier.visualization.plot_task_embedding
```

### Visualize Results
Generate plots for the experiments:
```bash
# Combined ablation plot
python -m tfmri_classifier.visualization.plot_combined_ablation

# Classifier comparison
python -m tfmri_classifier.visualization.plot_classifier_comparison

# Task embedding plot
python -m tfmri_classifier.visualization.plot_task_embedding
```

Results are saved in `data/results/`.

---
## 🛠️ Setup Instructions

We recommend using a virtual environment to keep dependencies isolated.

### 1. Create a new Conda environment

```bash
conda create -n tfmri python=3.10 -y
conda activate tfmri
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

---

## 📥 Data Download

The dataset used is hosted on [OpenNeuro](https://openneuro.org/datasets/ds002785) and can be accessed via AWS S3.

To download the relevant preprocessed data for **all subjects for a given task**, use:

```bash
make <task>
```

For example:
```bash
make workingmemory
```

To download all tasks across all subjects:

```bash
make all
```

This will download files into the following structure:

```bash
tfmri-classifier/
│
├── data/                            # Local data will sync here (gitignored)
│   └── ds002785/
│       └── derivatives/
│           └── fmriprep/
│               └── sub-XXXX/
└── Makefile                         # 🛠 Data sync logic
```

### Prerequisites

Ensure you have the AWS CLI installed (v2 recommended):

```bash
# MacOS
brew install awscli

# Ubuntu / Debian
sudo apt install awscli

# Or use pip
pip install awscli
```

No authentication is required to access OpenNeuro data. You do not need to run aws configure.

### Data Preparation

After downloading the data, you can extract the connectomes for each task using:

```bash
python -m tfmri_classifier.data_prep.extract_connectome --task TASK_NAME
```

Available tasks:
- `anticipation`
- `emomatching`
- `faces`
- `gstroop`
- `restingstate`
- `workingmemory`

You can process a specific subject by adding the `--subject` parameter:

```bash
python -m tfmri_classifier.data_prep.extract_connectome --task workingmemory --subject sub-0001
```

The extracted connectomes will be saved in `data/derivatives/connectomes/`.

---


## 📄 Citations

**AOMIC Dataset**  
Snoek, L., van der Miesen, M. M., Beemsterboer, T., van der Leij, A., Eigenhuis, A., Scholte, H. S. (2021). The Amsterdam Open MRI Collection, a set of multimodal MRI datasets for individual difference analyses. *Scientific Data, 8*, 85. https://doi.org/10.1038/s41597-021-00870-6

**fMRIPrep**  
Esteban, O., Markiewicz, C. J., Blair, R. W., Moodie, C. A., Isik, A. I., Erramuzpe, A., ... Gorgolewski, K. J. (2019). fMRIPrep: a robust preprocessing pipeline for functional MRI. *Nature Methods, 16*, 111–116. https://doi.org/10.1038/s41592-018-0235-4
