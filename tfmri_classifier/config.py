import os

# Path to the package root (i.e., tfmri_classifier/)
PACKAGE_ROOT = os.path.abspath(os.path.dirname(__file__))

# Core data paths (relative to the top-level repo)
PROJECT_ROOT = os.path.abspath(os.path.join(PACKAGE_ROOT, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DERIVATIVES_DIR = os.path.join(DATA_DIR, "derivatives")
FMRIPREP_DIR = os.path.join(DERIVATIVES_DIR, "fmriprep")
CONNECTOMES_DIR = os.path.join(DERIVATIVES_DIR, "connectomes")

# Resources (stored inside the package)
RESOURCES_DIR = os.path.join(PACKAGE_ROOT, "resources")
FREESURFER_LUT = os.path.join(RESOURCES_DIR, "FreeSurferColorLUT.csv")
