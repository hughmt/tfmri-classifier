import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_connectome(connectome_path, title=None, save_path=None):
    conn = np.load(connectome_path)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conn, cmap='coolwarm', center=0, square=True, cbar_kws={"label": "Pearson r"})
    plt.title(title or os.path.basename(connectome_path).replace("_", " "))
    plt.xlabel("Region")
    plt.ylabel("Region")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

path = "../data/derivatives/connectomes/workingmemory/sub-0001_connectome.npy"
plot_connectome(path, title="Functional Connectivity - sub-0001 (Working Memory)")
