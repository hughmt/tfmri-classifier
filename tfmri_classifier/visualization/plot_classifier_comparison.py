import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from tfmri_classifier.config import RESULTS_DIR

def load_latest_results():
    """Load the most recent classifier comparison results."""
    # Find the most recent results directory
    result_dirs = [d for d in os.listdir(RESULTS_DIR) if 'classifier_comparison' in d]
    latest_dir = sorted(result_dirs)[-1]
    results_path = os.path.join(RESULTS_DIR, latest_dir, 'comparison_results.json')
    
    with open(results_path, 'r') as f:
        return json.load(f)

def plot_overall_performance(results, save_dir):
    """Plot overall accuracy and timing metrics."""
    metrics = results['average_metrics']
    classifiers = list(metrics.keys())
    
    # Prepare data
    accuracies = [metrics[clf]['accuracy'] for clf in classifiers]
    train_times = [metrics[clf]['train_time'] for clf in classifiers]
    predict_times = [metrics[clf]['predict_time'] for clf in classifiers]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot accuracies
    bars = ax1.bar(classifiers, accuracies)
    ax1.set_title('Classification Accuracy by Model')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0.7, 1.0)  # Assuming accuracies are above 0.7
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # Plot timing (log scale)
    width = 0.35
    x = np.arange(len(classifiers))
    ax2.bar(x - width/2, train_times, width, label='Training Time')
    ax2.bar(x + width/2, predict_times, width, label='Prediction Time')
    ax2.set_yscale('log')
    ax2.set_title('Model Timing (Log Scale)')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classifiers, rotation=45)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overall_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_task_performance(results, save_dir):
    """Plot per-task F1 scores for each classifier."""
    metrics = results['average_metrics']
    classifiers = list(metrics.keys())
    tasks = list(metrics[classifiers[0]]['per_task_metrics'].keys())
    
    # Prepare data
    data = []
    for clf in classifiers:
        for task in tasks:
            data.append({
                'Classifier': clf,
                'Task': task,
                'F1-Score': metrics[clf]['per_task_metrics'][task]['f1-score']
            })
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    data_matrix = np.zeros((len(classifiers), len(tasks)))
    for i, clf in enumerate(classifiers):
        for j, task in enumerate(tasks):
            data_matrix[i, j] = metrics[clf]['per_task_metrics'][task]['f1-score']
    
    sns.heatmap(data_matrix, 
                annot=True, 
                fmt='.3f',
                xticklabels=tasks,
                yticklabels=classifiers,
                cmap='YlOrRd',
                vmin=0.7,  # Assuming F1 scores are above 0.7
                vmax=1.0)
    plt.title('F1-Scores by Model and Task')
    plt.xlabel('Task')
    plt.ylabel('Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'task_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load results
    results = load_latest_results()
    
    # Create visualization directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    viz_dir = os.path.join(RESULTS_DIR, f"{timestamp}_classifier_comparison_viz")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create visualizations
    plot_overall_performance(results, viz_dir)
    plot_task_performance(results, viz_dir)
    print(f"Visualizations saved to: {viz_dir}")

if __name__ == "__main__":
    main()
