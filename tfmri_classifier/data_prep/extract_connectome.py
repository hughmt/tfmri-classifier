import os
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm

def extract_mean_time_series(bold_img, atlas_img, mask_img):
    bold_data = bold_img.get_fdata()
    atlas_data = atlas_img.get_fdata()
    mask_data = mask_img.get_fdata()

    # Apply brain mask
    bold_data *= mask_data[..., np.newaxis]

    regions = np.unique(atlas_data)
    regions = regions[regions != 0]  # remove background

    time_series = []
    for region in regions:
        region_mask = atlas_data == region
        regional_ts = bold_data[region_mask, :].mean(axis=0)
        time_series.append(regional_ts)

    return np.array(time_series), regions

def compute_connectivity(time_series):
    return np.corrcoef(time_series)

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

    ts, regions = extract_mean_time_series(bold_img, atlas_img, mask_img)
    conn_matrix = compute_connectivity(ts)

    return conn_matrix, regions

def main(data_root="data/ds002785/derivatives/fmriprep", task="workingmemory", subject=None):
    output_dir = f"data/derivatives/connectomes/{task}/"
    os.makedirs(output_dir, exist_ok=True)

    if subject:
        subject_dirs = [os.path.join(data_root, subject)]
    else:
        subject_dirs = sorted(glob(os.path.join(data_root, "sub-*")))

    for sub_dir in tqdm(subject_dirs, desc=f"Processing task-{task}"):
        try:
            subject_id = os.path.basename(sub_dir)
            conn_matrix, regions = process_subject_task(sub_dir, task)
            np.save(os.path.join(output_dir, f"{subject_id}_connectome.npy"), conn_matrix)
            np.save(os.path.join(output_dir, f"{subject_id}_regions.npy"), regions)
        except Exception as e:
            print(f"[!] Failed to process {sub_dir}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., workingmemory)")
    parser.add_argument("--subject", type=str, default=None, help="Optional subject ID (e.g., sub-0001)")
    parser.add_argument("--data_root", type=str, default="data/ds002785/derivatives/fmriprep")
    args = parser.parse_args()

    main(data_root=args.data_root, task=args.task, subject=args.subject)
