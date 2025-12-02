"""
Fitness function for feature selection evaluation.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def evaluate_fitness(feature_subset, X_train, X_test, y_train, y_test, 
                     model_type='knn', alpha=0.1):
    """
    Evaluate fitness of a feature subset.
    
    Fitness = accuracy - alpha * (number_of_features_selected / total_features)
    
    Args:
        feature_subset: Binary vector (1 = selected, 0 = not selected)
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        model_type: 'knn' or 'logistic'
        alpha: Penalty coefficient for feature count
        
    Returns:
        fitness: Fitness score
        accuracy: Classification accuracy
    """
    # Ensure at least one feature is selected
    if np.sum(feature_subset) == 0:
        return -np.inf, 0.0
    
    # Get selected features
    selected_indices = np.where(feature_subset == 1)[0]
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    
    # Train classifier
    if model_type == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_type == 'logistic':
        model = LogisticRegression(random_state=42, max_iter=1000)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train_selected, y_train)
    
    # Predict and calculate accuracy
    y_pred = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate fitness
    total_features = len(feature_subset)
    num_selected = np.sum(feature_subset)
    fitness = accuracy - alpha * (num_selected / total_features)
    
    return fitness, accuracy


def ensure_at_least_one_feature(feature_subset):
    """
    Ensure at least one feature is selected by randomly selecting one if none selected.
    
    Args:
        feature_subset: Binary vector
        
    Returns:
        Modified binary vector with at least one feature selected
    """
    if np.sum(feature_subset) == 0:
        idx = np.random.randint(0, len(feature_subset))
        feature_subset[idx] = 1
    return feature_subset


