"""
Main script for comparing GA and PSO for feature selection.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from utils.data_loader import load_and_preprocess_data
from metaheuristics.ga import GeneticAlgorithm
from metaheuristics.pso import ParticleSwarmOptimization


def plot_convergence(ga_history, pso_history, save_path='convergence_comparison.png'):
    """
    Plot convergence curves for GA and PSO.
    
    Args:
        ga_history: GA fitness history
        pso_history: PSO fitness history
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(ga_history, label='Genetic Algorithm', linewidth=2, marker='o', markersize=4)
    plt.plot(pso_history, label='Particle Swarm Optimization', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Generation/Iteration', fontsize=12)
    plt.ylabel('Best Fitness', fontsize=12)
    plt.title('Convergence Comparison: GA vs PSO', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Convergence plot saved to {save_path}")
    plt.close()


def plot_bar_comparison(ga_results, pso_results, save_path='results_comparison.png'):
    """
    Plot bar charts comparing GA and PSO results.
    
    Args:
        ga_results: Dictionary with GA results (accuracy, n_features)
        pso_results: Dictionary with PSO results (accuracy, n_features)
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    methods = ['GA', 'PSO']
    accuracies = [ga_results['accuracy'], pso_results['accuracy']]
    colors = ['#3498db', '#e74c3c']
    
    ax1.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, acc in enumerate(accuracies):
        ax1.text(i, acc + 0.01, f'{acc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Feature count comparison
    n_features = [ga_results['n_features'], pso_results['n_features']]
    
    ax2.bar(methods, n_features, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Number of Features Selected', fontsize=12)
    ax2.set_title('Feature Count Comparison', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, nf in enumerate(n_features):
        ax2.text(i, nf + 0.5, f'{int(nf)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Bar chart comparison saved to {save_path}")
    plt.close()


def print_comparison_summary(ga_results, pso_results, ga_time, pso_time):
    """
    Print a clear comparison summary.
    
    Args:
        ga_results: Dictionary with GA results
        pso_results: Dictionary with PSO results
        ga_time: GA computation time
        pso_time: PSO computation time
    """
    print("\n" + "="*70)
    print("FEATURE SELECTION COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Metric':<30} {'GA':<20} {'PSO':<20}")
    print("-"*70)
    print(f"{'Accuracy':<30} {ga_results['accuracy']:<20.4f} {pso_results['accuracy']:<20.4f}")
    print(f"{'Number of Features Selected':<30} {ga_results['n_features']:<20} {pso_results['n_features']:<20}")
    print(f"{'Computation Time (seconds)':<30} {ga_time:<20.2f} {pso_time:<20.2f}")
    print(f"{'Best Fitness':<30} {ga_results['fitness']:<20.4f} {pso_results['fitness']:<20.4f}")
    print("-"*70)
    
    # Determine winner
    if ga_results['accuracy'] > pso_results['accuracy']:
        print("\n✓ GA achieved higher accuracy")
    elif pso_results['accuracy'] > ga_results['accuracy']:
        print("\n✓ PSO achieved higher accuracy")
    else:
        print("\n✓ Both methods achieved similar accuracy")
    
    if ga_results['n_features'] < pso_results['n_features']:
        print("✓ GA selected fewer features")
    elif pso_results['n_features'] < ga_results['n_features']:
        print("✓ PSO selected fewer features")
    else:
        print("✓ Both methods selected the same number of features")
    
    if ga_time < pso_time:
        print("✓ GA was faster")
    elif pso_time < ga_time:
        print("✓ PSO was faster")
    else:
        print("✓ Both methods took similar time")
    
    print("="*70 + "\n")


def main():
    """
    Main function to run GA and PSO comparison.
    """
    print("="*70)
    print("FEATURE SELECTION USING GENETIC ALGORITHM AND PARTICLE SWARM OPTIMIZATION")
    print("="*70)
    
    # Load and preprocess data
    print("\n[1/5] Loading and preprocessing data...")
    try:
        X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess_data()
        print(f"✓ Data loaded successfully")
        print(f"  - Training samples: {X_train.shape[0]}")
        print(f"  - Test samples: {X_test.shape[0]}")
        print(f"  - Total features: {X_train.shape[1]}")
    except FileNotFoundError:
        print("\n✗ Error: Dataset not found!")
        print("Please download the Breast Cancer Wisconsin dataset from:")
        print("https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data")
        print("And place it as 'data/data.csv' in the project root.")
        return
    
    n_features = X_train.shape[1]
    
    # Configuration
    model_type = 'knn'  # or 'logistic'
    alpha = 0.1
    
    # Run Genetic Algorithm
    print("\n[2/5] Running Genetic Algorithm...")
    ga = GeneticAlgorithm(
        n_features=n_features,
        population_size=50,
        n_generations=50,
        crossover_rate=0.8,
        mutation_rate=0.1,
        tournament_size=3,
        elitism_rate=0.1,
        model_type=model_type,
        alpha=alpha
    )
    
    start_time = time.time()
    ga_best_subset, ga_accuracy, ga_history = ga.run(X_train, X_test, y_train, y_test)
    ga_time = time.time() - start_time
    
    ga_n_features = np.sum(ga_best_subset)
    ga_fitness = ga_history[-1]
    
    print(f"✓ GA completed in {ga_time:.2f} seconds")
    print(f"  - Best accuracy: {ga_accuracy:.4f}")
    print(f"  - Features selected: {ga_n_features}/{n_features}")
    print(f"  - Best fitness: {ga_fitness:.4f}")
    
    # Run Particle Swarm Optimization
    print("\n[3/5] Running Particle Swarm Optimization...")
    pso = ParticleSwarmOptimization(
        n_features=n_features,
        n_particles=30,
        n_iterations=50,
        w=0.7,
        c1=1.5,
        c2=1.5,
        model_type=model_type,
        alpha=alpha
    )
    
    start_time = time.time()
    pso_best_subset, pso_accuracy, pso_history = pso.run(X_train, X_test, y_train, y_test)
    pso_time = time.time() - start_time
    
    pso_n_features = np.sum(pso_best_subset)
    pso_fitness = pso_history[-1]
    
    print(f"✓ PSO completed in {pso_time:.2f} seconds")
    print(f"  - Best accuracy: {pso_accuracy:.4f}")
    print(f"  - Features selected: {pso_n_features}/{n_features}")
    print(f"  - Best fitness: {pso_fitness:.4f}")
    
    # Prepare results
    ga_results = {
        'accuracy': ga_accuracy,
        'n_features': ga_n_features,
        'fitness': ga_fitness,
        'subset': ga_best_subset
    }
    
    pso_results = {
        'accuracy': pso_accuracy,
        'n_features': pso_n_features,
        'fitness': pso_fitness,
        'subset': pso_best_subset
    }
    
    # Generate plots
    print("\n[4/5] Generating plots...")
    plot_convergence(ga_history, pso_history)
    plot_bar_comparison(ga_results, pso_results)
    print("✓ Plots generated successfully")
    
    # Print comparison summary
    print("\n[5/5] Generating comparison summary...")
    print_comparison_summary(ga_results, pso_results, ga_time, pso_time)
    
    # Print selected features
    print("\nSelected Features:")
    print("-"*70)
    print("GA Selected Features:")
    ga_selected = [feature_names[i] for i in np.where(ga_best_subset == 1)[0]]
    for i, feat in enumerate(ga_selected, 1):
        print(f"  {i}. {feat}")
    
    print("\nPSO Selected Features:")
    pso_selected = [feature_names[i] for i in np.where(pso_best_subset == 1)[0]]
    for i, feat in enumerate(pso_selected, 1):
        print(f"  {i}. {feat}")
    print("-"*70)
    
    print("\n✓ All tasks completed successfully!")


if __name__ == "__main__":
    main()


