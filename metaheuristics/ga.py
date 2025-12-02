"""
Genetic Algorithm implementation for feature selection.
"""

import numpy as np
from utils.fitness import evaluate_fitness, ensure_at_least_one_feature


class GeneticAlgorithm:
    """
    Genetic Algorithm for feature selection.
    """
    
    def __init__(self, n_features, population_size=50, n_generations=50,
                 crossover_rate=0.8, mutation_rate=0.1, tournament_size=3,
                 elitism_rate=0.1, model_type='knn', alpha=0.1):
        """
        Initialize Genetic Algorithm.
        
        Args:
            n_features: Number of features
            population_size: Size of population
            n_generations: Number of generations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            tournament_size: Size of tournament for selection
            elitism_rate: Fraction of best individuals to keep
            model_type: 'knn' or 'logistic'
            alpha: Penalty coefficient for feature count
        """
        self.n_features = n_features
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_count = int(elitism_rate * population_size)
        self.model_type = model_type
        self.alpha = alpha
        
        self.best_fitness_history = []
        self.best_individual = None
        self.best_fitness = -np.inf
        
    def initialize_population(self):
        """
        Initialize population with random binary vectors.
        Ensures at least one feature is selected in each individual.
        
        Returns:
            population: Array of binary vectors
        """
        population = np.random.randint(0, 2, size=(self.population_size, self.n_features))
        
        # Ensure at least one feature is selected in each individual
        for i in range(self.population_size):
            population[i] = ensure_at_least_one_feature(population[i])
        
        return population
    
    def tournament_selection(self, population, fitness_scores):
        """
        Tournament selection.
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for each individual
            
        Returns:
            selected: Selected individual
        """
        tournament_indices = np.random.choice(
            len(population), size=self.tournament_size, replace=False
        )
        tournament_fitness = fitness_scores[tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def one_point_crossover(self, parent1, parent2):
        """
        One-point crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            child1, child2: Two offspring
        """
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        point = np.random.randint(1, self.n_features)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        
        # Ensure at least one feature is selected
        child1 = ensure_at_least_one_feature(child1)
        child2 = ensure_at_least_one_feature(child2)
        
        return child1, child2
    
    def bit_flip_mutation(self, individual):
        """
        Bit flip mutation.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            mutated: Mutated individual
        """
        mutated = individual.copy()
        
        for i in range(self.n_features):
            if np.random.rand() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]
        
        # Ensure at least one feature is selected
        mutated = ensure_at_least_one_feature(mutated)
        
        return mutated
    
    def evaluate_population(self, population, X_train, X_test, y_train, y_test):
        """
        Evaluate fitness of entire population.
        
        Args:
            population: Current population
            X_train, X_test, y_train, y_test: Data splits
            
        Returns:
            fitness_scores: Array of fitness scores
            accuracies: Array of accuracies
        """
        fitness_scores = np.zeros(self.population_size)
        accuracies = np.zeros(self.population_size)
        
        for i in range(self.population_size):
            fitness, accuracy = evaluate_fitness(
                population[i], X_train, X_test, y_train, y_test,
                self.model_type, self.alpha
            )
            fitness_scores[i] = fitness
            accuracies[i] = accuracy
        
        return fitness_scores, accuracies
    
    def run(self, X_train, X_test, y_train, y_test):
        """
        Run Genetic Algorithm.
        
        Args:
            X_train, X_test, y_train, y_test: Data splits
            
        Returns:
            best_individual: Best feature subset found
            best_accuracy: Best accuracy achieved
            convergence_history: List of best fitness per generation
        """
        # Initialize population
        population = self.initialize_population()
        
        # Evaluate initial population
        fitness_scores, accuracies = self.evaluate_population(
            population, X_train, X_test, y_train, y_test
        )
        
        # Track best individual
        best_idx = np.argmax(fitness_scores)
        self.best_fitness = fitness_scores[best_idx]
        self.best_individual = population[best_idx].copy()
        self.best_fitness_history.append(self.best_fitness)
        
        # Evolution loop
        for generation in range(self.n_generations):
            # Create new population
            new_population = []
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitness_scores)[-self.elitism_count:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                # Crossover
                child1, child2 = self.one_point_crossover(parent1, parent2)
                
                # Mutation
                child1 = self.bit_flip_mutation(child1)
                child2 = self.bit_flip_mutation(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            # Update population
            population = np.array(new_population)
            
            # Evaluate new population
            fitness_scores, accuracies = self.evaluate_population(
                population, X_train, X_test, y_train, y_test
            )
            
            # Update best individual
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_individual = population[best_idx].copy()
            
            self.best_fitness_history.append(self.best_fitness)
            
            if (generation + 1) % 10 == 0:
                print(f"GA Generation {generation + 1}/{self.n_generations}: "
                      f"Best Fitness = {self.best_fitness:.4f}")
        
        # Get final accuracy for best individual
        best_accuracy = accuracies[best_idx]
        
        return self.best_individual, best_accuracy, self.best_fitness_history


