import os
import numpy as np
import pandas as pd
import json
from glob import glob

from tfmri_classifier.config import RESULTS_DIR, CONNECTOMES_DIR

def get_yeo_network_importance(task):
    """Get the importance of each Yeo network by training a model and computing feature importance."""
    from tfmri_classifier.modelling.prepare_data import prepare_dataset
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, _, _ = prepare_dataset(tasks=[task, 'restingstate'])
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Get feature importance
    feature_importance = model.feature_importances_
    
    # Network ROI ranges (approximate)
    network_ranges = [
        (0, 16),    # Visual
        (17, 31),   # Somatomotor
        (32, 46),   # Dorsal Attention
        (47, 59),   # Ventral Attention
        (60, 74),   # Limbic
        (75, 89),   # Frontoparietal
        (90, 99)    # Default
    ]
    
    # Initialize importance for each network
    network_importance = []
    
    # Get the size of the connectome from the feature vector length
    n = int((1 + np.sqrt(1 + 8 * len(feature_importance))) / 2)
    
    # Create an nxn matrix to map feature importances back to connectome
    connectome_importance = np.zeros((n, n))
    triu_indices = np.triu_indices(n, k=1)
    connectome_importance[triu_indices] = feature_importance
    connectome_importance = connectome_importance + connectome_importance.T  # Make symmetric
    
    # Calculate importance for each network
    network_names = ['Visual', 'Somatomotor', 'Dorsal Attention', 
                    'Ventral Attention', 'Limbic', 'Frontoparietal', 'Default']
    
    for i, (start, end) in enumerate(network_ranges):
        # Get all connections involving this network
        network_connections = connectome_importance[start:end+1, :]
        mean_importance = np.mean(network_connections)
            
        network_importance.append({
            'network': f"Yeo {i+1}",
            'importance': mean_importance
        })
    
    return network_importance

def plot_network_importance(task):
    """Plot the importance of each Yeo network."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get network importance
    network_importance = get_yeo_network_importance(task)
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    networks = [imp['network'] for imp in network_importance]
    importance = [imp['importance'] for imp in network_importance]
    
    sns.barplot(x=networks, y=importance)
    plt.title(f'Yeo Network Importance: {task} vs. restingstate')
    plt.xlabel('Network')
    plt.ylabel('Mean Feature Importance')
    plt.xticks(rotation=45)
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(RESULTS_DIR, f"{timestamp}_network_importance_{task}")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'network_importance.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
