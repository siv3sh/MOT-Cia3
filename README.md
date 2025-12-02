# Feature Selection using Genetic Algorithm and Particle Swarm Optimization

This project implements and compares two metaheuristic optimization techniques (Genetic Algorithm and Particle Swarm Optimization) for feature selection on the Breast Cancer Wisconsin dataset.

## Project Structure

```
MOT/
├── data/
│   └── data.csv          # Breast Cancer Wisconsin dataset (download required)
├── utils/
│   ├── __init__.py
│   ├── data_loader.py    # Data loading and preprocessing
│   └── fitness.py        # Fitness function evaluation
├── metaheuristics/
│   ├── __init__.py
│   ├── ga.py            # Genetic Algorithm implementation
│   └── pso.py           # Particle Swarm Optimization implementation
├── main.py              # Main script for comparison
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Installation

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Breast Cancer Wisconsin dataset:
   - Visit: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
   - Download the dataset and place it as `data/data.csv` in the project root

## Usage

Run the main script:
```bash
python main.py
```

The script will:
1. Load and preprocess the dataset
2. Run Genetic Algorithm for feature selection
3. Run Particle Swarm Optimization for feature selection
4. Compare results and generate visualization plots
5. Print a detailed comparison summary

## Features

### Genetic Algorithm (GA)
- Population-based evolutionary algorithm
- Tournament selection
- One-point crossover
- Bit-flip mutation
- Elitism preservation
- Tracks convergence over generations

### Particle Swarm Optimization (PSO)
- Swarm-based optimization algorithm
- Binary PSO with sigmoid velocity update
- Personal best and global best tracking
- Velocity clamping to prevent explosion
- Tracks convergence over iterations

### Fitness Function
```
fitness = accuracy - alpha * (number_of_features_selected / total_features)
```
where `alpha = 0.1` (penalty coefficient)

### Machine Learning Model
- K-Nearest Neighbors (KNN) classifier with k=5
- Alternative: Logistic Regression (configurable in code)
- 80/20 train/test split

## Output

The script generates:
1. **Convergence plots**: Comparison of GA and PSO fitness over iterations
2. **Bar charts**: Comparison of accuracy and feature count
3. **Console output**: Detailed comparison summary and selected features

## Configuration

You can modify the following parameters in `main.py`:

- **Model type**: Change `model_type = 'knn'` to `'logistic'` for Logistic Regression
- **GA parameters**: Population size, generations, crossover/mutation rates
- **PSO parameters**: Number of particles, iterations, inertia weight, coefficients
- **Alpha**: Penalty coefficient for feature count (default: 0.1)

## Results

The comparison includes:
- Final accuracy achieved
- Number of features selected
- Computation time
- Convergence behavior
- Selected feature subsets

## Requirements

- Python 3.7+
- numpy
- pandas
- scikit-learn
- matplotlib

## License

This project is for educational purposes.


