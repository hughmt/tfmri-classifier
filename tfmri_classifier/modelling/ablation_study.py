import os
import numpy as np
import pandas as pd
from datetime import datetime
import json
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import time
import matplotlib.pyplot as plt
import seaborn as sns

from tfmri_classifier.config import RESULTS_DIR, CONNECTOMES_DIR
from tfmri_classifier.modelling.prepare_data import load_connectome, prepare_dataset
from tfmri_classifier.visualization.plot_importance import get_yeo_network_importance

def get_network_mask(n_features, network_indices):
    """Create a boolean mask that excludes specified networks.
    
    The connectome matrix size is determined from the feature vector length.
    Network assignments are proportionally scaled to the matrix size.
    """
    # Calculate matrix size from feature vector length
    n = int((1 + np.sqrt(1 + 8 * n_features)) / 2)
    
    # Initialize mask
    mask = np.ones(n_features, dtype=bool)
    
    # Scale network ranges to matrix size
    ranges = [
        (0, int(0.17*n)),      # Visual (~17%)
        (int(0.17*n), int(0.32*n)),  # Somatomotor (~15%)
        (int(0.32*n), int(0.47*n)),  # Dorsal Attention (~15%)
        (int(0.47*n), int(0.60*n)),  # Ventral Attention (~13%)
        (int(0.60*n), int(0.75*n)),  # Limbic (~15%)
        (int(0.75*n), int(0.90*n)),  # Frontoparietal (~15%)
        (int(0.90*n), n-1)      # Default (~10%)
    ]
    
    # For each network to remove
    for network_idx in network_indices:
        start, end = ranges[network_idx]
        
        # Create an nxn mask for the full connectome
        connectome_mask = np.ones((n, n), dtype=bool)
        
        # Set rows and columns for this network to False
        connectome_mask[start:end+1, :] = False
        connectome_mask[:, start:end+1] = False
        
        # Get upper triangle indices (excluding diagonal)
        triu_indices = np.triu_indices(n, k=1)
        
        # Update the feature mask
        feature_mask = connectome_mask[triu_indices]
        mask = mask & feature_mask
    
    return mask

def run_ablation_experiment(task='workingmemory', n_ablations=7):
    """Run ablation study by progressively removing top Yeo networks."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(RESULTS_DIR, f"{timestamp}_ablation_study_{task}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data and prepare initial dataset
    print(f"\nPreparing dataset for {task} vs. restingstate...")
    X_train, X_test, y_train, y_test, subject_train, subject_test = prepare_dataset(tasks=[task, 'restingstate'])
    
    # Combine train and test sets for cross-validation
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    subjects = np.concatenate([subject_train, subject_test])
    
    # Get network importance from the original model
    print("Computing network importance from original model...")
    network_importance = get_yeo_network_importance(task)
    sorted_networks = np.argsort([imp['importance'] for imp in network_importance])[::-1]
    
    # Initialize results storage
    results = {
        'ablation_results': [],
        'network_importance': network_importance,
        'networks_removed': []
    }
    
    # Run baseline (no ablation)
    print("\nRunning baseline...")
    baseline_metrics = train_and_evaluate(X, y, subjects, mask=None)
    results['ablation_results'].append({
        'networks_removed': [],
        'metrics': baseline_metrics
    })
    
    # Progressively ablate networks
    for i in range(n_ablations):
        networks_to_remove = sorted_networks[:i+1]
        network_names = [f"Yeo {idx+1}" for idx in networks_to_remove]
        print(f"\nAblating networks: {', '.join(network_names)}")
        
        # Create mask excluding the networks
        mask = get_network_mask(X.shape[1], networks_to_remove)
        
        # Train and evaluate
        metrics = train_and_evaluate(X, y, subjects, mask)
        
        # Store results
        results['ablation_results'].append({
            'networks_removed': network_names,
            'metrics': metrics
        })
        results['networks_removed'].append(network_names)
    
    # Save results
    with open(os.path.join(results_dir, 'ablation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot results
    plot_ablation_results(results, results_dir, task)
    
    return results

def train_and_evaluate(X, y, subjects, mask=None):
    """Train and evaluate model with optional feature masking."""
    # Apply mask if provided
    if mask is not None:
        X_masked = X[:, mask]
        if X_masked.shape[1] == 0:
            # If all features are masked out, return zero accuracy
            return {
                'accuracy': 0.0,
                'std_accuracy': 0.0,
                'fold_metrics': [{
                    'accuracy': 0.0,
                    'classification_report': None,
                    'confusion_matrix': None
                }]
            }
    else:
        X_masked = X
    
    # Initialize cross-validation
    n_splits = 5
    group_kfold = GroupKFold(n_splits=n_splits)
    
    # Store metrics for each fold
    fold_metrics = []
    
    # Perform cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(group_kfold.split(X_masked, y, groups=subjects)):
        print(f"Fold {fold_idx + 1}/{n_splits}")
        
        # Split data
        X_train, X_test = X_masked[train_idx], X_masked[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        fold_metrics.append(metrics)
    
    # Calculate average metrics
    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
        'std_accuracy': np.std([m['accuracy'] for m in fold_metrics]),
        'fold_metrics': fold_metrics
    }
    
    return avg_metrics

def plot_ablation_results(results, save_dir, task):
    """Plot ablation study results."""
    # Extract accuracies and networks removed
    accuracies = [r['metrics']['accuracy'] for r in results['ablation_results']]
    std_accuracies = [r['metrics']['std_accuracy'] for r in results['ablation_results']]
    x_labels = ['Baseline'] + [f"Remove Top {i+1}" for i in range(len(accuracies)-1)]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    plt.errorbar(range(len(accuracies)), accuracies, yerr=std_accuracies, 
                fmt='o-', capsize=5, capthick=2, elinewidth=2)
    
    # Customize plot
    plt.title(f'Ablation Study Results: {task} vs. restingstate')
    plt.xlabel('Networks Removed')
    plt.ylabel('Classification Accuracy')
    plt.xticks(range(len(accuracies)), x_labels, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add network names as annotations
    for i, networks in enumerate(results['networks_removed']):
        if networks:  # Skip baseline which has no networks removed
            plt.annotate(', '.join(networks), 
                        xy=(i+1, accuracies[i+1]),
                        xytext=(10, -20),
                        textcoords='offset points',
                        ha='left',
                        va='top',
                        rotation=45,
                        fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_results.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # List of tasks to run ablation study for
    tasks = [
        'workingmemory',
        'faces',
        'emomatching',
        'gstroop',
        'anticipation'
    ]
    
    # Run ablation study for each task
    all_results = {}
    for task in tasks:
        print(f"\n{'='*50}")
        print(f"Running ablation study for {task}")
        print('='*50)
        results = run_ablation_experiment(task=task)
        all_results[task] = results
    
    # Save combined results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(RESULTS_DIR, f"{timestamp}_all_ablation_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
