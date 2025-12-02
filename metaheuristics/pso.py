"""
Particle Swarm Optimization implementation for binary feature selection.
"""

import numpy as np
from utils.fitness import evaluate_fitness, ensure_at_least_one_feature


class ParticleSwarmOptimization:
    """
    Binary Particle Swarm Optimization for feature selection.
    """
    
    def __init__(self, n_features, n_particles=30, n_iterations=50,
                 w=0.7, c1=1.5, c2=1.5, model_type='knn', alpha=0.1):
        """
        Initialize Particle Swarm Optimization.
        
        Args:
            n_features: Number of features
            n_particles: Number of particles
            n_iterations: Number of iterations
            w: Inertia weight
            c1: Cognitive coefficient
            c2: Social coefficient
            model_type: 'knn' or 'logistic'
            alpha: Penalty coefficient for feature count
        """
        self.n_features = n_features
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.model_type = model_type
        self.alpha = alpha
        
        self.best_fitness_history = []
        self.global_best_position = None
        self.global_best_fitness = -np.inf
        
    def initialize_particles(self):
        """
        Initialize particles with random positions and velocities.
        
        Returns:
            positions: Binary position vectors
            velocities: Continuous velocity vectors
        """
        # Initialize positions (binary)
        positions = np.random.randint(0, 2, size=(self.n_particles, self.n_features))
        
        # Ensure at least one feature is selected in each particle
        for i in range(self.n_particles):
            positions[i] = ensure_at_least_one_feature(positions[i])
        
        # Initialize velocities (continuous, clamped to [-4, 4])
        velocities = np.random.uniform(-1, 1, size=(self.n_particles, self.n_features))
        
        return positions, velocities
    
    def sigmoid(self, x):
        """
        Sigmoid function for converting velocity to probability.
        
        Args:
            x: Input value
            
        Returns:
            Sigmoid output
        """
        # Clamp x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def update_position(self, position, velocity):
        """
        Update binary position using sigmoid velocity.
        
        Args:
            position: Current binary position
            velocity: Current velocity
            
        Returns:
            new_position: Updated binary position
        """
        # Convert velocity to probability using sigmoid
        prob = self.sigmoid(velocity)
        
        # Update position based on probability
        new_position = (np.random.rand(self.n_features) < prob).astype(int)
        
        # Ensure at least one feature is selected
        new_position = ensure_at_least_one_feature(new_position)
        
        return new_position
    
    def update_velocity(self, velocity, position, personal_best, global_best):
        """
        Update velocity using PSO update rule.
        
        Args:
            velocity: Current velocity
            position: Current position
            personal_best: Personal best position
            global_best: Global best position
            
        Returns:
            new_velocity: Updated velocity
        """
        r1 = np.random.rand(self.n_features)
        r2 = np.random.rand(self.n_features)
        
        # PSO velocity update
        new_velocity = (self.w * velocity +
                       self.c1 * r1 * (personal_best - position) +
                       self.c2 * r2 * (global_best - position))
        
        # Clamp velocity to prevent explosion
        new_velocity = np.clip(new_velocity, -4, 4)
        
        return new_velocity
    
    def evaluate_particles(self, positions, X_train, X_test, y_train, y_test):
        """
        Evaluate fitness of all particles.
        
        Args:
            positions: Current positions
            X_train, X_test, y_train, y_test: Data splits
            
        Returns:
            fitness_scores: Array of fitness scores
            accuracies: Array of accuracies
        """
        fitness_scores = np.zeros(self.n_particles)
        accuracies = np.zeros(self.n_particles)
        
        for i in range(self.n_particles):
            fitness, accuracy = evaluate_fitness(
                positions[i], X_train, X_test, y_train, y_test,
                self.model_type, self.alpha
            )
            fitness_scores[i] = fitness
            accuracies[i] = accuracy
        
        return fitness_scores, accuracies
    
    def run(self, X_train, X_test, y_train, y_test):
        """
        Run Particle Swarm Optimization.
        
        Args:
            X_train, X_test, y_train, y_test: Data splits
            
        Returns:
            best_position: Best feature subset found
            best_accuracy: Best accuracy achieved
            convergence_history: List of best fitness per iteration
        """
        # Initialize particles
        positions, velocities = self.initialize_particles()
        
        # Evaluate initial particles
        fitness_scores, accuracies = self.evaluate_particles(
            positions, X_train, X_test, y_train, y_test
        )
        
        # Initialize personal bests
        personal_best_positions = positions.copy()
        personal_best_fitness = fitness_scores.copy()
        
        # Initialize global best
        best_idx = np.argmax(fitness_scores)
        self.global_best_fitness = fitness_scores[best_idx]
        self.global_best_position = positions[best_idx].copy()
        self.best_fitness_history.append(self.global_best_fitness)
        
        # Optimization loop
        for iteration in range(self.n_iterations):
            # Update each particle
            for i in range(self.n_particles):
                # Update velocity
                velocities[i] = self.update_velocity(
                    velocities[i],
                    positions[i],
                    personal_best_positions[i],
                    self.global_best_position
                )
                
                # Update position
                positions[i] = self.update_position(positions[i], velocities[i])
            
            # Evaluate particles
            fitness_scores, accuracies = self.evaluate_particles(
                positions, X_train, X_test, y_train, y_test
            )
            
            # Update personal bests
            for i in range(self.n_particles):
                if fitness_scores[i] > personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness_scores[i]
                    personal_best_positions[i] = positions[i].copy()
            
            # Update global best
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.global_best_fitness:
                self.global_best_fitness = fitness_scores[best_idx]
                self.global_best_position = positions[best_idx].copy()
            
            self.best_fitness_history.append(self.global_best_fitness)
            
            if (iteration + 1) % 10 == 0:
                print(f"PSO Iteration {iteration + 1}/{self.n_iterations}: "
                      f"Best Fitness = {self.global_best_fitness:.4f}")
        
        # Get final accuracy for best position
        _, best_accuracy = evaluate_fitness(
            self.global_best_position, X_train, X_test, y_train, y_test,
            self.model_type, self.alpha
        )
        
        return self.global_best_position, best_accuracy, self.best_fitness_history


