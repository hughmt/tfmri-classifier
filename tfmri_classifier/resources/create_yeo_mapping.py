import os
import numpy as np
import pandas as pd
from nilearn import datasets
import nibabel as nib
from nilearn.image import resample_to_img

# Load Yeo 17-network atlas
print("Downloading Yeo atlas...")
yeo = datasets.fetch_atlas_yeo_2011()
yeo_17 = nib.load(yeo['thick_17'])

# Load our DK atlas regions and their labels
dk_regions = pd.read_csv('DK_atlas_regions.csv')

# Define Yeo network labels
yeo_network_labels = {
    1: "Visual-1",
    2: "Visual-2",
    3: "Motor-1",
    4: "Motor-2",
    5: "DA-1",  # Dorsal Attention
    6: "DA-2",
    7: "VA-1",  # Ventral Attention
    8: "VA-2",
    9: "Limbic-1",
    10: "Limbic-2",
    11: "FP-1",  # Frontoparietal
    12: "FP-2",
    13: "DMN-1",  # Default Mode Network
    14: "DMN-2",
    15: "DMN-3",
    16: "DMN-4",
    17: "None"
}

# Save Yeo network labels
network_labels_df = pd.DataFrame([
    {'network_id': k, 'network_name': v} 
    for k, v in yeo_network_labels.items()
])
network_labels_df.to_csv('yeo17_network_labels.csv', index=False)

# Create a template mapping for cortical regions
cortical_mapping = {
    # Left hemisphere approximate mappings based on anatomical knowledge
    'ctx-lh-unknown': 17,  # None
    'ctx-lh-bankssts': 13,  # DMN-1 (temporal)
    'ctx-lh-caudalanteriorcingulate': 7,  # VA-1 (attention)
    'ctx-lh-caudalmiddlefrontal': 11,  # FP-1 (executive)
    'ctx-lh-cuneus': 1,  # Visual-1
    'ctx-lh-entorhinal': 13,  # DMN-1 (memory)
    'ctx-lh-fusiform': 2,  # Visual-2 (face/object)
    'ctx-lh-inferiorparietal': 11,  # FP-1 (attention)
    'ctx-lh-inferiortemporal': 2,  # Visual-2 (object)
    'ctx-lh-isthmuscingulate': 13,  # DMN-1
    'ctx-lh-lateraloccipital': 1,  # Visual-1
    'ctx-lh-lateralorbitofrontal': 9,  # Limbic-1
    'ctx-lh-lingual': 1,  # Visual-1
    'ctx-lh-medialorbitofrontal': 9,  # Limbic-1
    'ctx-lh-middletemporal': 13,  # DMN-1
    'ctx-lh-parahippocampal': 13,  # DMN-1 (memory)
    'ctx-lh-paracentral': 3,  # Motor-1
    'ctx-lh-parsopercularis': 11,  # FP-1 (language)
    'ctx-lh-parsorbitalis': 11,  # FP-1
    'ctx-lh-parstriangularis': 11,  # FP-1 (language)
    'ctx-lh-pericalcarine': 1,  # Visual-1
    'ctx-lh-postcentral': 3,  # Motor-1
    'ctx-lh-posteriorcingulate': 13,  # DMN-1
    'ctx-lh-precentral': 3,  # Motor-1
    'ctx-lh-precuneus': 13,  # DMN-1
    'ctx-lh-rostralanteriorcingulate': 13,  # DMN-1
    'ctx-lh-rostralmiddlefrontal': 11,  # FP-1
    'ctx-lh-superiorfrontal': 11,  # FP-1
    'ctx-lh-superiorparietal': 5,  # DA-1
    'ctx-lh-superiortemporal': 13,  # DMN-1
    'ctx-lh-supramarginal': 7,  # VA-1
    'ctx-lh-frontalpole': 11,  # FP-1
    'ctx-lh-temporalpole': 13,  # DMN-1
    'ctx-lh-transversetemporal': 3,  # Motor-1 (auditory)
    'ctx-lh-insula': 7,  # VA-1
    
    # Right hemisphere (same mappings as left)
    'ctx-rh-unknown': 17,
    'ctx-rh-bankssts': 13,
    'ctx-rh-caudalanteriorcingulate': 7,
    'ctx-rh-caudalmiddlefrontal': 11,
    'ctx-rh-cuneus': 1,
    'ctx-rh-entorhinal': 13,
    'ctx-rh-fusiform': 2,
    'ctx-rh-inferiorparietal': 11,
    'ctx-rh-inferiortemporal': 2,
    'ctx-rh-isthmuscingulate': 13,
    'ctx-rh-lateraloccipital': 1,
    'ctx-rh-lateralorbitofrontal': 9,
    'ctx-rh-lingual': 1,
    'ctx-rh-medialorbitofrontal': 9,
    'ctx-rh-middletemporal': 13,
    'ctx-rh-parahippocampal': 13,
    'ctx-rh-paracentral': 3,
    'ctx-rh-parsopercularis': 11,
    'ctx-rh-parsorbitalis': 11,
    'ctx-rh-parstriangularis': 11,
    'ctx-rh-pericalcarine': 1,
    'ctx-rh-postcentral': 3,
    'ctx-rh-posteriorcingulate': 13,
    'ctx-rh-precentral': 3,
    'ctx-rh-precuneus': 13,
    'ctx-rh-rostralanteriorcingulate': 13,
    'ctx-rh-rostralmiddlefrontal': 11,
    'ctx-rh-superiorfrontal': 11,
    'ctx-rh-superiorparietal': 5,
    'ctx-rh-superiortemporal': 13,
    'ctx-rh-supramarginal': 7,
    'ctx-rh-frontalpole': 11,
    'ctx-rh-temporalpole': 13,
    'ctx-rh-transversetemporal': 3,
    'ctx-rh-insula': 7
}

# Create mapping DataFrame
mapping = []
for _, row in dk_regions.iterrows():
    region_id = row['id']
    region_label = row['label']
    region_type = row['region_type']
    
    if region_type in ['left_hemisphere', 'right_hemisphere']:
        # Use our predefined cortical mappings
        yeo_network = cortical_mapping.get(region_label, 17)  # Default to 17 (None) if not found
        mapping.append({
            'dk_id': region_id,
            'dk_label': region_label,
            'dk_region_type': region_type,
            'yeo_network': yeo_network,
            'yeo_network_name': yeo_network_labels[yeo_network],
            'mapping_method': 'anatomical'
        })
    else:
        # For subcortical and other regions, mark as non-cortical
        mapping.append({
            'dk_id': region_id,
            'dk_label': region_label,
            'dk_region_type': region_type,
            'yeo_network': 17,  # None
            'yeo_network_name': 'None',
            'mapping_method': 'non-cortical'
        })

# Convert to DataFrame and save
mapping_df = pd.DataFrame(mapping)
mapping_df = mapping_df.sort_values('dk_id')
mapping_df.to_csv('dk_to_yeo17_mapping.csv', index=False)

# Print summary
print("\nMapping Summary:")
print("\nBy Region Type:")
print(mapping_df.groupby('dk_region_type')['yeo_network_name'].value_counts().to_string())
print("\nBy Network (cortical regions only):")
cortical_df = mapping_df[mapping_df['dk_region_type'].isin(['left_hemisphere', 'right_hemisphere'])]
print(cortical_df['yeo_network_name'].value_counts().to_string())
