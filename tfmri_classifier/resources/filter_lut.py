import pandas as pd
import os

# Read the full FreeSurfer LUT
lut = pd.read_csv('FreeSurferColorLUT.csv')

# Our atlas regions
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

# Function to determine region type
def get_region_type(region_id):
    if region_id < 251:  # Subcortical structures
        return 'subcortical'
    elif 251 <= region_id <= 255:  # Corpus Callosum
        return 'corpus_callosum'
    elif 1000 <= region_id < 2000:  # Left hemisphere
        return 'left_hemisphere'
    else:  # Right hemisphere
        return 'right_hemisphere'

# Filter the LUT to only include our atlas regions
filtered_lut = lut[lut['id'].isin(ATLAS_REGIONS)].copy()

# Verify we found all regions
missing = set(ATLAS_REGIONS) - set(filtered_lut['id'])
if missing:
    print(f"Warning: Could not find these regions in the LUT: {missing}")

# Add region type column
filtered_lut['region_type'] = filtered_lut['id'].apply(get_region_type)

# Sort by the order in ATLAS_REGIONS
filtered_lut['atlas_order'] = filtered_lut['id'].map({id: idx for idx, id in enumerate(ATLAS_REGIONS)})
filtered_lut = filtered_lut.sort_values('atlas_order').drop('atlas_order', axis=1)

# Save to new CSV
filtered_lut.to_csv('DK_atlas_regions.csv', index=False)
