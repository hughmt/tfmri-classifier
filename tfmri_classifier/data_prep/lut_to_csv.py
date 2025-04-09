import pandas as pd
from src.config import RESOURCES_DIR, FREESURFER_LUT
import os

# After downloading: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT

def load_lut_as_dataframe(lut_path=FREESURFER_LUT):
    rows = []
    with open(lut_path, 'r') as f:
        for line in f:
            if line.strip() == '' or line.startswith('#'):
                continue
            parts = line.split()
            try:
                idx = int(parts[0])
                label = parts[1]
                r, g, b, a = map(int, parts[2:6])
                rows.append((idx, label, r, g, b, a))
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(rows, columns=["id", "label", "R", "G", "B", "A"])
    return df

# Load and preview
df = load_lut_as_dataframe()
print(df.head())

# Save to CSV in the same resources directory
output_path = os.path.join(RESOURCES_DIR, "FreeSurferColorLUT.csv")
df.to_csv(output_path, index=False)
print(f"Saved LUT CSV to {output_path}")
