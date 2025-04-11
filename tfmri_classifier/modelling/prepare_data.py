import os
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

def load_connectome(file_path):
    """Load a connectome from a .npy file and extract the upper triangle as features."""
    connectome = np.load(file_path)
    # Ensure the matrix is square
    assert connectome.shape[0] == connectome.shape[1], f"Expected square matrix, got shape {connectome.shape}"
    
    # Get upper triangle indices (excluding diagonal)
    triu_indices = np.triu_indices(connectome.shape[0], k=1)
    # Extract upper triangle values as feature vector
    features = connectome[triu_indices]
    return features.astype(np.float32)  # Ensure consistent dtype

def prepare_dataset(tasks=['restingstate', 'workingmemory'], test_size=0.2, random_state=42):
    """
    Prepare dataset for classification between two tasks.
    Returns train and test sets with subjects completely separated.
    
    Parameters:
    -----------
    tasks : list of str
        List of tasks to compare (default: ['restingstate', 'workingmemory'])
    test_size : float
        Proportion of subjects to use for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    X_train, X_test : arrays
        Feature matrices for training and testing
    y_train, y_test : arrays
        Labels for training and testing
    subject_train, subject_test : arrays
        Subject IDs for training and testing data
    """
    from tfmri_classifier.config import CONNECTOMES_DIR
    
    # Initialize lists to store data
    features_list = []
    labels = []
    subjects = []
    
    # Load data for each task
    for task_idx, task in enumerate(tasks):
        task_dir = os.path.join(CONNECTOMES_DIR, task)
        connectome_files = glob(os.path.join(task_dir, "*_connectome.npy"))
        
        for file_path in connectome_files:
            # Extract subject ID from filename
            subject_id = os.path.basename(file_path).split("_")[0]
            
            # Load and flatten connectome
            features = load_connectome(file_path)
            
            features_list.append(features)
            labels.append(task_idx)  # Use task index as label
            subjects.append(subject_id)
    
    # Convert to numpy arrays
    X = np.array(features_list)
    y = np.array(labels)
    subjects = np.array(subjects)
    
    # Get unique subjects
    unique_subjects = np.unique(subjects)
    
    # Split subjects into train/test
    subject_train, subject_test = train_test_split(
        unique_subjects, 
        test_size=test_size,
        random_state=random_state
    )
    
    # Create masks for train/test split
    train_mask = np.isin(subjects, subject_train)
    test_mask = np.isin(subjects, subject_test)
    
    # Split data
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    subject_train = subjects[train_mask]
    subject_test = subjects[test_mask]
    
    return X_train, X_test, y_train, y_test, subject_train, subject_test
