import pickle

# After downloading:
"""
curl -o src/FreeSurferColorLUT.txt "https://surfer.nmr.mgh.harvard.edu/fswiki/aparcstats2table?action=AttachFile&do=get&target=FreeSurferColorLUT.txt"
"""

def load_freesurfer_lut(txt_path):
    lut = {}
    with open(txt_path, 'r') as f:
        for line in f:
            if line.strip() == '' or line.startswith('#'):
                continue
            parts = line.split()
            try:
                idx = int(parts[0])
                name = parts[1]
                lut[idx] = name
            except:
                continue
    return lut

if __name__ == "__main__":
    txt_path = "src/FreeSurferColorLUT.txt"
    output_path = "src/FreeSurferColorLUT.pkl"

    lut = load_freesurfer_lut(txt_path)
    with open(output_path, "wb") as f:
        pickle.dump(lut, f)

    print(f"Saved LUT with {len(lut)} entries to {output_path}")
