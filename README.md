# tfMRI Classifier

This repository contains code for classifying cognitive tasks using functional connectivity features extracted from task-based fMRI data. It used data from the AOMIC dataset (ds002785), specifically the preprocessed outputs from fMRIPrep.

The goal is to evaluate how well models trained on functional connectomes can classify task states, and whether brain networks not typically associated with a task contribute to classification accuracy.

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

---


## 📄 Citations

**AOMIC Dataset**  
Snoek, L., van der Miesen, M. M., Beemsterboer, T., van der Leij, A., Eigenhuis, A., Scholte, H. S. (2021). The Amsterdam Open MRI Collection, a set of multimodal MRI datasets for individual difference analyses. *Scientific Data, 8*, 85. https://doi.org/10.1038/s41597-021-00870-6

**fMRIPrep**  
Esteban, O., Markiewicz, C. J., Blair, R. W., Moodie, C. A., Isik, A. I., Erramuzpe, A., ... Gorgolewski, K. J. (2019). fMRIPrep: a robust preprocessing pipeline for functional MRI. *Nature Methods, 16*, 111–116. https://doi.org/10.1038/s41592-018-0235-4
