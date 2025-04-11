import os
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from xgboost.callback import EarlyStopping
from .prepare_data import prepare_dataset
from tfmri_classifier.config import RESULTS_DIR

def save_confusion_matrix(cm, labels, save_path_base):
    """Save confusion matrix as both visualization and CSV."""
    # Save visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{save_path_base}.png")
    plt.close()
    
    # Save as CSV with labels
    import pandas as pd
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(f"{save_path_base}.csv")

def get_region_labels():
    """Load region labels from FreeSurfer LUT."""
    import pandas as pd
    from tfmri_classifier.config import FREESURFER_LUT
    from tfmri_classifier.data_prep.extract_connectome import ATLAS_REGIONS
    
    # Load FreeSurfer LUT
    lut = pd.read_csv(FREESURFER_LUT)
    
    # Create mapping from region ID to label
    region_labels = {}
    for region in ATLAS_REGIONS:
        label = lut[lut['id'] == region]['label'].values
        region_labels[region] = label[0] if len(label) > 0 else f'Unknown-{region}'
    
    return region_labels

def get_connectivity_feature_labels():
    """Generate labels for connectivity features (region pairs) with Yeo network information."""
    from tfmri_classifier.data_prep.extract_connectome import ATLAS_REGIONS
    import pandas as pd
    from tfmri_classifier.config import RESOURCES_DIR
    
    # Load region labels and Yeo mapping
    region_labels = get_region_labels()
    yeo_mapping = pd.read_csv(os.path.join(RESOURCES_DIR, 'dk_to_yeo17_mapping.csv'))
    yeo_dict = dict(zip(yeo_mapping['dk_label'], yeo_mapping['yeo_network_name']))
    
    feature_labels = []
    feature_info = []
    n_regions = len(ATLAS_REGIONS)
    
    # For each feature (upper triangle of connectivity matrix)
    for i in range(n_regions):
        for j in range(i+1, n_regions):
            region1 = ATLAS_REGIONS[i]
            region2 = ATLAS_REGIONS[j]
            
            # Get region labels
            label1 = region_labels[region1]
            label2 = region_labels[region2]
            
            # Get Yeo networks
            yeo1 = yeo_dict.get(label1, 'Unknown')
            yeo2 = yeo_dict.get(label2, 'Unknown')
            
            # Create label and feature info
            label = f'{label1} ↔ {label2}'
            feature_labels.append(label)
            feature_info.append({
                'feature': label,
                'region1': label1,
                'region2': label2,
                'yeo_network1': yeo1,
                'yeo_network2': yeo2
            })
    
    return feature_labels, feature_info

def save_feature_importance(importances, save_path_base):
    """Save feature importance as both visualization and CSV with descriptive labels and Yeo network information."""
    # Get feature labels and info
    feature_labels, feature_info = get_connectivity_feature_labels()
    
    # Get sorted indices and values
    sorted_indices = np.argsort(importances)[::-1]  # Reverse to get descending order
    sorted_importance = importances[sorted_indices]
    sorted_labels = [feature_labels[i] for i in sorted_indices]
    sorted_info = [feature_info[i] for i in sorted_indices]
    
    # Save visualization of top 20 connections
    plt.figure(figsize=(15, 10))
    top_k = 20
    plt.barh(range(top_k), sorted_importance[:top_k])
    plt.yticks(range(top_k), sorted_labels[:top_k], fontsize=8)
    plt.xlabel('Importance')
    plt.title('Top 20 Most Important Connectivity Features')
    plt.tight_layout()
    plt.savefig(f"{save_path_base}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save full feature importance data with Yeo network information
    import pandas as pd
    importance_df = pd.DataFrame(sorted_info)
    importance_df['importance'] = sorted_importance
    
    # Add network pair column for easier analysis
    importance_df['network_pair'] = importance_df.apply(
        lambda x: f"{x['yeo_network1']} ↔ {x['yeo_network2']}", axis=1
    )
    
    # Save detailed CSV of connections
    importance_df.to_csv(f"{save_path_base}.csv", index=False)
    
    # Create network-level summary
    network_summary = importance_df.groupby('network_pair')['importance'].agg([
        ('total_importance', 'sum'),
        ('mean_importance', 'mean'),
        ('count', 'size')
    ]).sort_values('total_importance', ascending=False)
    
    network_summary.to_csv(f"{save_path_base}_network_summary.csv")
    
    # Create parcellation-level summary
    # First, split each connection into two rows
    parcellation_rows = []
    for _, row in importance_df.iterrows():
        # Add first region
        parcellation_rows.append({
            'parcellation': row['region1'],
            'yeo_network': row['yeo_network1'],
            'importance': row['importance']  # Same importance for both regions
        })
        # Add second region
        parcellation_rows.append({
            'parcellation': row['region2'],
            'yeo_network': row['yeo_network2'],
            'importance': row['importance']  # Same importance for both regions
        })
    
    # Convert to DataFrame and group by parcellation
    parcellation_df = pd.DataFrame(parcellation_rows)
    parcellation_summary = parcellation_df.groupby(['parcellation', 'yeo_network'])['importance'].agg([
        ('total_importance', 'sum'),  # Sum of all connection importances involving this region
        ('connection_count', 'size'),  # Number of connections involving this region
        ('mean_importance', 'mean')  # Average importance of connections involving this region
    ]).reset_index()
    
    # Sort by total importance
    parcellation_summary = parcellation_summary.sort_values('total_importance', ascending=False)
    
    # Save parcellation-level summary
    parcellation_summary.to_csv(f"{save_path_base}_parcellation_summary.csv", index=False)
    
    # Create network-level summary by summing importances for each network
    network_summary = parcellation_df.groupby('yeo_network')['importance'].agg([
        ('total_importance', 'sum'),
        ('connection_count', 'size'),
        ('mean_importance', 'mean')
    ]).sort_values('total_importance', ascending=False)
    
    network_summary.to_csv(f"{save_path_base}_network_summary.csv")
    
    # Create visualization of top parcellations
    plt.figure(figsize=(15, 10))
    top_k = 20
    top_parcellations = parcellation_summary.head(top_k)
    plt.barh(range(top_k), top_parcellations['total_importance'])
    labels = [f"{row['parcellation']}\n({row['yeo_network']})" 
             for _, row in top_parcellations.iterrows()]
    plt.yticks(range(top_k), labels, fontsize=8)
    plt.xlabel('Total Importance')
    plt.title('Top 20 Most Important Brain Regions')
    plt.tight_layout()
    plt.savefig(f"{save_path_base}_parcellations.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create visualization of networks
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(network_summary)), network_summary['total_importance'])
    plt.yticks(range(len(network_summary)), network_summary.index, fontsize=8)
    plt.xlabel('Total Importance')
    plt.title('Importance by Yeo Network')
    plt.tight_layout()
    plt.savefig(f"{save_path_base}_networks.png", dpi=300, bbox_inches='tight')
    plt.close()

def train_and_evaluate(tasks=['restingstate', 'workingmemory'], test_size=0.2, random_state=42, 
                     experiment_name=None, description=None):
    """
    Train and evaluate an XGBoost classifier on the connectome data.
    
    Parameters:
    -----------
    tasks : list of str
        List of tasks to compare
    test_size : float
        Proportion of subjects to use for testing
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    model : XGBClassifier
        Trained model
    metrics : dict
        Dictionary containing evaluation metrics
    """
    # Prepare data
    X_train, X_test, y_train, y_test, subject_train, subject_test = prepare_dataset(
        tasks=tasks,
        test_size=test_size,
        random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=random_state,
        n_estimators=100,
        learning_rate=0.1
    )
    
    model.fit(
        X_train_scaled, 
        y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=True
    )
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        'classification_report': classification_report(y_test, y_pred, target_names=tasks),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'feature_importance': model.feature_importances_
    }
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if experiment_name:
        exp_name = f"{timestamp}_{experiment_name}"
    else:
        exp_name = timestamp
    
    exp_dir = os.path.join(RESULTS_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save experiment metadata
    metadata = {
        'timestamp': timestamp,
        'tasks': tasks,
        'test_size': test_size,
        'random_state': random_state,
        'description': description,
        'model_params': model.get_params()
    }
    
    with open(os.path.join(exp_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save classification report
    with open(os.path.join(exp_dir, 'classification_report.txt'), 'w') as f:
        f.write(metrics['classification_report'])
    
    # Save subject splits
    np.save(os.path.join(exp_dir, 'subject_train.npy'), subject_train)
    np.save(os.path.join(exp_dir, 'subject_test.npy'), subject_test)
    
    # Save data splits info
    splits_info = {
        'n_train_subjects': len(np.unique(subject_train)),
        'n_test_subjects': len(np.unique(subject_test)),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'train_subjects': subject_train.tolist(),
        'test_subjects': subject_test.tolist()
    }
    with open(os.path.join(exp_dir, 'splits.json'), 'w') as f:
        json.dump(splits_info, f, indent=2)
    
    # Save confusion matrix
    save_confusion_matrix(
        metrics['confusion_matrix'],
        tasks,
        os.path.join(exp_dir, 'confusion_matrix')
    )
    
    # Save feature importance
    save_feature_importance(
        metrics['feature_importance'],
        os.path.join(exp_dir, 'feature_importance')
    )
    
    # Save model
    model.save_model(os.path.join(exp_dir, 'model.json'))
    
    return model, metrics, exp_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs=2, default=['restingstate', 'workingmemory'],
                      help="Two tasks to compare")
    parser.add_argument("--test-size", type=float, default=0.2,
                      help="Proportion of subjects to use for testing")
    parser.add_argument("--random-state", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--experiment-name", type=str,
                      help="Name for this experiment run")
    parser.add_argument("--description", type=str,
                      help="Description of this experiment")
    args = parser.parse_args()
    
    # Train and evaluate model
    model, metrics, exp_dir = train_and_evaluate(
        tasks=args.tasks,
        test_size=args.test_size,
        random_state=args.random_state,
        experiment_name=args.experiment_name,
        description=args.description
    )
    
    # Print results and save location
    print(f"\nExperiment results saved to: {exp_dir}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
