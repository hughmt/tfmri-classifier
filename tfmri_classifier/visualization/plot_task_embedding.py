import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from glob import glob
from tfmri_classifier.config import CONNECTOMES_DIR, RESULTS_DIR
from tfmri_classifier.modelling.train_multiclass import load_connectome

def load_all_data():
    """Load preprocessed connectome data from all tasks."""
    from tfmri_classifier.modelling.prepare_data import load_connectome
    
    # All available tasks
    tasks = [
        'anticipation',
        'emomatching',
        'faces',
        'gstroop',
        'restingstate',
        'workingmemory'
    ]
    
    # Initialize lists to store data
    features_list = []
    labels = []
    subjects = []
    
    # Load data for each task
    for task_idx, task in enumerate(tasks):
        task_dir = os.path.join(CONNECTOMES_DIR, task)
        connectome_files = glob(os.path.join(task_dir, "*_connectome.npy"))
        
        for file_path in connectome_files:
            subject_id = os.path.basename(file_path).split("_")[0]
            # Use the preprocessing from prepare_data.py
            features = load_connectome(file_path)
            
            features_list.append(features)
            labels.append(task)
            subjects.append(subject_id)
    
    return np.array(features_list), np.array(labels), np.array(subjects)

def plot_embedding(X_embedded, labels, save_path, title="PCA — t-SNE"):
    """Create scatter plot of embedded data."""
    plt.figure(figsize=(12, 8))
    
    # Create a color map for tasks
    unique_tasks = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_tasks)))
    task_to_color = dict(zip(unique_tasks, colors))
    
    # Plot each task with a different color
    for task in unique_tasks:
        mask = labels == task
        plt.scatter(
            X_embedded[mask, 0],
            X_embedded[mask, 1],
            c=[task_to_color[task]],
            label=task.upper(),
            alpha=0.6,
            s=30
        )
    
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(title="Task", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load and preprocess data
    print("Loading data...")
    X, labels, subjects = load_all_data()
    
    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # First reduce dimensionality with PCA
    print("Performing PCA...")
    pca = PCA(n_components=50)  # Reduce to 50 dimensions first
    X_pca = pca.fit_transform(X_scaled)
    
    # Then apply t-SNE
    print("Performing t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=2000,
        random_state=42
    )
    X_embedded = tsne.fit_transform(X_pca)
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(RESULTS_DIR, f"{timestamp}_task_embedding")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save the plot
    plot_path = os.path.join(results_dir, "task_embedding.png")
    plot_embedding(X_embedded, labels, plot_path)
    print(f"Plot saved to: {plot_path}")
    
    # Save the embeddings and metadata
    np.save(os.path.join(results_dir, "embeddings.npy"), X_embedded)
    np.save(os.path.join(results_dir, "labels.npy"), labels)
    np.save(os.path.join(results_dir, "subjects.npy"), subjects)
    
    # Save parameters
    params = {
        "pca_components": 50,
        "tsne_perplexity": 30,
        "tsne_iterations": 2000,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist()
    }
    
    with open(os.path.join(results_dir, "parameters.json"), 'w') as f:
        json.dump(params, f, indent=2)

if __name__ == "__main__":
    from datetime import datetime
    import json
    main()
