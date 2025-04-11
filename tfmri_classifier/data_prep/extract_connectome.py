import os
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm
from tfmri_classifier.config import CONNECTOMES_DIR, DATA_DIR

# Define the complete set of atlas regions in order
ATLAS_REGIONS = [
    # Subcortical structures
    2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 30, 31,
    41, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63, 77, 85,
    # Corpus Callosum segments
    251, 252, 253, 254, 255,
    # Left hemisphere cortical regions
    1000, 1001, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012,
    1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024,
    1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035,
    # Right hemisphere cortical regions
    2000, 2001, 2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
    2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024,
    2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035
]

# Create mapping from region ID to index in connectivity matrix
REGION_TO_INDEX = {region: idx for idx, region in enumerate(ATLAS_REGIONS)}

def extract_mean_time_series(bold_img, atlas_img, mask_img):
    bold_data = bold_img.get_fdata()
    atlas_data = atlas_img.get_fdata()
    mask_data = mask_img.get_fdata()

    # Apply brain mask
    bold_data *= mask_data[..., np.newaxis]

    # Initialize time series array with zeros for all atlas regions
    n_timepoints = bold_data.shape[-1]
    time_series = np.zeros((len(ATLAS_REGIONS), n_timepoints))

    # Extract time series for each region that exists in the data
    for region in ATLAS_REGIONS:
        region_mask = atlas_data == region
        if np.any(region_mask):  # if region exists in this subject
            regional_ts = bold_data[region_mask, :].mean(axis=0)
            time_series[REGION_TO_INDEX[region]] = regional_ts

    return time_series

def compute_connectivity(time_series):
    """Compute connectivity matrix with special handling for zero time series.
    If a region is missing (all zeros), its correlation will be set to 0.
    """
    n_regions = len(ATLAS_REGIONS)
    connectivity = np.zeros((n_regions, n_regions))
    
    # Find non-zero time series
    non_zero = np.any(time_series != 0, axis=1)
    
    if np.any(non_zero):
        # Compute correlation only for non-zero time series
        valid_corr = np.corrcoef(time_series[non_zero])
        
        # Fill in the correlations for non-zero regions
        non_zero_idx = np.where(non_zero)[0]
        for i, idx_i in enumerate(non_zero_idx):
            for j, idx_j in enumerate(non_zero_idx):
                connectivity[idx_i, idx_j] = valid_corr[i, j]
    
    return connectivity

def process_subject_task(sub_dir, task):
    bold_glob = os.path.join(sub_dir, "func", f"*task-{task}*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
    atlas_glob = os.path.join(sub_dir, "func", f"*task-{task}*_space-MNI152NLin2009cAsym_desc-aparcaseg_dseg.nii.gz")
    mask_glob  = os.path.join(sub_dir, "func", f"*task-{task}*_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")

    bold_files = glob(bold_glob)
    atlas_files = glob(atlas_glob)
    mask_files = glob(mask_glob)

    if not bold_files:
        raise FileNotFoundError(f"No BOLD file found for {task} in {sub_dir} with pattern:\n{bold_glob}")
    if not atlas_files:
        raise FileNotFoundError(f"No atlas file found for {task} in {sub_dir} with pattern:\n{atlas_glob}")
    if not mask_files:
        raise FileNotFoundError(f"No brain mask found for {task} in {sub_dir} with pattern:\n{mask_glob}")

    bold_img = nib.load(bold_files[0])
    atlas_img = nib.load(atlas_files[0])
    mask_img = nib.load(mask_files[0])

    ts = extract_mean_time_series(bold_img, atlas_img, mask_img)
    conn_matrix = compute_connectivity(ts)

    return conn_matrix

def main(data_root=os.path.join(DATA_DIR, "ds002785/derivatives/fmriprep"), task="workingmemory", subject=None):
    output_dir = os.path.join(CONNECTOMES_DIR, task)
    os.makedirs(output_dir, exist_ok=True)

    # Save the atlas regions once
    np.save(os.path.join(CONNECTOMES_DIR, "atlas_regions.npy"), np.array(ATLAS_REGIONS))

    if subject:
        subject_dirs = [os.path.join(data_root, subject)]
    else:
        subject_dirs = sorted(glob(os.path.join(data_root, "sub-*")))

    for sub_dir in tqdm(subject_dirs, desc=f"Processing task-{task}"):
        try:
            subject_id = os.path.basename(sub_dir)
            conn_matrix = process_subject_task(sub_dir, task)
            np.save(os.path.join(output_dir, f"{subject_id}_connectome.npy"), conn_matrix)
        except Exception as e:
            print(f"[!] Failed to process {sub_dir}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., workingmemory)")
    parser.add_argument("--subject", type=str, default=None, help="Optional subject ID (e.g., sub-0001)")
    parser.add_argument("--data_root", type=str, default=os.path.join(DATA_DIR, "ds002785/derivatives/fmriprep"))
    args = parser.parse_args()

    main(data_root=args.data_root, task=args.task, subject=args.subject)
