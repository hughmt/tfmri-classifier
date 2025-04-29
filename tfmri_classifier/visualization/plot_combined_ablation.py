import json
import matplotlib.pyplot as plt
import numpy as np
import os
from tfmri_classifier.config import RESULTS_DIR

def plot_combined_ablation_results():
    """Create a combined plot of ablation results for all tasks."""
    # Load the combined results
    results_file = os.path.join(RESULTS_DIR, '20250428_211652_all_ablation_results.json')
    with open(results_file, 'r') as f:
        all_results = json.load(f)
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Colors for each task
    colors = {
        'workingmemory': '#1f77b4',  # blue
        'faces': '#2ca02c',          # green
        'emomatching': '#d62728',    # red
        'gstroop': '#9467bd',        # purple
        'anticipation': '#ff7f0e'    # orange
    }
    
    # Plot each task
    for task, results in all_results.items():
        # Get accuracies for each ablation step
        accuracies = [r['metrics']['accuracy'] for r in results['ablation_results']]
        n_networks = range(len(accuracies))
        
        # Plot with task-specific color and label
        plt.plot(n_networks, accuracies, 'o-', 
                color=colors[task], 
                label=task.capitalize(),
                linewidth=2,
                markersize=8)
    
    # Customize the plot
    plt.xlabel('Number of Networks Removed', fontsize=12)
    plt.ylabel('Classification Accuracy', fontsize=12)
    plt.title('Impact of Network Ablation on Classification Performance', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Set axis limits
    plt.ylim(-0.05, 1.05)
    plt.xlim(-0.2, 7.2)
    
    # Add x-tick labels
    plt.xticks(range(8))
    
    # Save the plot
    output_file = os.path.join(RESULTS_DIR, 'combined_ablation_results.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined ablation plot saved to: {output_file}")

if __name__ == '__main__':
    plot_combined_ablation_results()
