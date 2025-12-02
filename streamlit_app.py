"""
Streamlit Application for Feature Selection using GA and PSO
Comprehensive assignment report with interactive features
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from PIL import Image
import os

# Import project modules
from utils.data_loader import load_and_preprocess_data
from metaheuristics.ga import GeneticAlgorithm
from metaheuristics.pso import ParticleSwarmOptimization

# Page configuration
st.set_page_config(
    page_title="Feature Selection: GA vs PSO",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2980b9;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #2980b9;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #3498db;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background-color: #ecf0f1;
        border-left: 5px solid #3498db;
    }
    .metric-box {
        background-color: #ecf0f1;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #3498db;
        margin: 0.5rem 0;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 2rem;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
</style>
""", unsafe_allow_html=True)

def load_images():
    """Load all visualization images"""
    images = {}
    image_files = {
        'ga_flowchart': 'ga_flowchart.png',
        'pso_flowchart': 'pso_flowchart.png',
        'convergence': 'convergence_comparison.png',
        'results': 'results_comparison.png'
    }
    
    for key, filename in image_files.items():
        if os.path.exists(filename):
            images[key] = Image.open(filename)
    return images

def plot_convergence(ga_history, pso_history):
    """Plot convergence curves"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ga_history, label='Genetic Algorithm', linewidth=2, marker='o', markersize=4, color='#3498db')
    ax.plot(pso_history, label='Particle Swarm Optimization', linewidth=2, marker='s', markersize=4, color='#e74c3c')
    ax.set_xlabel('Generation/Iteration', fontsize=12)
    ax.set_ylabel('Best Fitness', fontsize=12)
    ax.set_title('Convergence Comparison: GA vs PSO', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_bar_comparison(ga_results, pso_results):
    """Plot bar charts comparing results"""
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
    
    for i, acc in enumerate(accuracies):
        ax1.text(i, acc + 0.01, f'{acc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Feature count comparison
    n_features = [ga_results['n_features'], pso_results['n_features']]
    
    ax2.bar(methods, n_features, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Number of Features Selected', fontsize=12)
    ax2.set_title('Feature Count Comparison', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, nf in enumerate(n_features):
        ax2.text(i, nf + 0.5, f'{int(nf)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown('<div class="main-header">🧬 Feature Selection using Genetic Algorithm and Particle Swarm Optimization</div>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("📑 Navigation")
    page = st.sidebar.radio(
        "Select Section",
        ["🏠 Home", "📊 Executive Summary", "🔬 Implementation", "📈 Results & Analysis", "🎯 Run Algorithms", "📚 Full Report"]
    )
    
    # Load images
    images = load_images()
    
    if page == "🏠 Home":
        st.markdown("""
        ## Welcome to the Feature Selection Analysis Platform
        
        This interactive application presents a comprehensive analysis of feature selection 
        using two metaheuristic optimization algorithms:
        
        - **Genetic Algorithm (GA)**
        - **Particle Swarm Optimization (PSO)**
        
        ### Dataset
        **Breast Cancer Wisconsin (Diagnostic) Dataset**
        - Total Samples: 569
        - Total Features: 30
        - Target: Binary Classification (Malignant/Benign)
        
        ### Features
        - Interactive algorithm execution
        - Real-time convergence visualization
        - Comprehensive results comparison
        - Full assignment report
        - Algorithm flowcharts
        """)
        
        if 'convergence' in images:
            st.image(images['convergence'], caption="Convergence Comparison", use_container_width=True)
    
    elif page == "📊 Executive Summary":
        st.markdown('<div class="section-header">Executive Summary</div>', unsafe_allow_html=True)
        
        st.markdown("""
        This assignment implements and compares two metaheuristic optimization algorithms—Genetic Algorithm (GA) 
        and Particle Swarm Optimization (PSO)—for feature selection on the Breast Cancer Wisconsin dataset. 
        The project successfully demonstrates how these algorithms can reduce dimensionality while maintaining 
        high classification accuracy.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="metric-box">
            <h3>🎯 Key Results</h3>
            <ul>
            <li>Both algorithms achieved <b>99.12% accuracy</b></li>
            <li>GA selected <b>6 features</b> (80% reduction)</li>
            <li>PSO selected <b>10 features</b> (67% reduction)</li>
            <li>PSO was <b>30% faster</b> in execution</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-box">
            <h3>📊 Performance Metrics</h3>
            <ul>
            <li><b>GA Fitness:</b> 0.9712</li>
            <li><b>PSO Fitness:</b> 0.9579</li>
            <li><b>GA Time:</b> 2.97 seconds</li>
            <li><b>PSO Time:</b> 2.06 seconds</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        if 'results' in images:
            st.image(images['results'], caption="Results Comparison", use_container_width=True)
    
    elif page == "🔬 Implementation":
        st.markdown('<div class="section-header">Implementation Details</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Genetic Algorithm", "Particle Swarm Optimization", "Fitness Function"])
        
        with tab1:
            st.markdown("### Genetic Algorithm (GA)")
            st.markdown("""
            **Key Components:**
            - Population Size: 50 individuals
            - Generations: 50
            - Representation: Binary chromosome (1 = feature selected, 0 = not selected)
            - Selection: Tournament selection (tournament size = 3)
            - Crossover: One-point crossover (rate = 0.8)
            - Mutation: Bit-flip mutation (rate = 0.1)
            - Elitism: Top 10% of population preserved
            """)
            
            if 'ga_flowchart' in images:
                st.image(images['ga_flowchart'], caption="Genetic Algorithm Flowchart", use_container_width=True)
            
            st.markdown("""
            **Algorithm Flow:**
            1. Initialize random population
            2. Evaluate fitness for each individual
            3. For each generation:
               - Select parents using tournament selection
               - Perform crossover to create offspring
               - Apply mutation
               - Evaluate new population
               - Apply elitism
            4. Return best solution
            """)
        
        with tab2:
            st.markdown("### Particle Swarm Optimization (PSO)")
            st.markdown("""
            **Key Components:**
            - Swarm Size: 30 particles
            - Iterations: 50
            - Representation: Binary position vector (1 = feature selected, 0 = not selected)
            - Velocity Update: Sigmoid function for binary conversion
            - Parameters:
              - Inertia weight (w) = 0.7
              - Cognitive coefficient (c1) = 1.5
              - Social coefficient (c2) = 1.5
            """)
            
            if 'pso_flowchart' in images:
                st.image(images['pso_flowchart'], caption="Particle Swarm Optimization Flowchart", use_container_width=True)
            
            st.markdown("""
            **Algorithm Flow:**
            1. Initialize swarm with random positions and velocities
            2. Evaluate fitness for each particle
            3. Initialize personal best (pbest) and global best (gbest)
            4. For each iteration:
               - Update velocity
               - Update position using sigmoid function
               - Evaluate fitness
               - Update pbest and gbest
            5. Return best solution
            """)
        
        with tab3:
            st.markdown("### Fitness Function")
            st.latex(r'''
            \text{fitness} = \text{accuracy} - \alpha \times \frac{n_{\text{selected\_features}}}{n_{\text{total\_features}}}
            ''')
            st.markdown("""
            Where:
            - **accuracy**: Classification accuracy on test set
            - **α**: Penalty coefficient (0.1)
            - **n_selected_features**: Number of features in the selected subset
            - **n_total_features**: Total number of features (30)
            
            This formulation encourages:
            - High classification accuracy
            - Fewer selected features (parsimony)
            """)
    
    elif page == "📈 Results & Analysis":
        st.markdown('<div class="section-header">Results and Analysis</div>', unsafe_allow_html=True)
        
        # Results Table
        st.markdown("### Comparison Summary")
        comparison_data = {
            'Metric': ['Accuracy', 'Features Selected', 'Computation Time (seconds)', 'Best Fitness'],
            'GA': ['99.12%', '6', '2.97', '0.9712'],
            'PSO': ['99.12%', '10', '2.06', '0.9579']
        }
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            if 'convergence' in images:
                st.image(images['convergence'], caption="Convergence Comparison", use_container_width=True)
        
        with col2:
            if 'results' in images:
                st.image(images['results'], caption="Results Comparison", use_container_width=True)
        
        # Detailed Analysis
        st.markdown("### Detailed Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Accuracy Performance", "Feature Reduction", "Convergence Behavior"])
        
        with tab1:
            st.markdown("""
            Both algorithms achieved **identical accuracy of 99.12%**, which is exceptional for a medical diagnosis task. 
            This demonstrates that:
            - Feature selection can maintain high classification performance
            - The selected feature subsets contain sufficient discriminative information
            - Both algorithms successfully identified critical features for breast cancer classification
            """)
        
        with tab2:
            st.markdown("""
            **GA Performance**: Selected **6 features** out of 30, achieving an **80% reduction** in dimensionality.
            - This represents a highly parsimonious solution
            - Reduces computational complexity significantly
            - Improves model interpretability
            
            **PSO Performance**: Selected **10 features** out of 30, achieving a **67% reduction** in dimensionality.
            - Still a substantial reduction
            - More features than GA but still manageable
            
            **Comparison**: GA achieved a **40% better feature reduction** (4 fewer features) while maintaining identical accuracy.
            """)
        
        with tab3:
            st.markdown("""
            **GA Convergence Pattern:**
            - Generation 10: Fitness = 0.9591
            - Generation 20: Fitness = 0.9658
            - Generation 30: Fitness = 0.9658 (plateau)
            - Generation 40: Fitness = 0.9712 (improvement)
            - Generation 50: Fitness = 0.9712 (final)
            
            **PSO Convergence Pattern:**
            - Iteration 10: Fitness = 0.9346
            - Iteration 20: Fitness = 0.9579
            - Iteration 30-50: Fitness = 0.9579 (early convergence)
            
            **Key Observations:**
            1. GA shows sustained improvement with occasional plateaus
            2. PSO shows rapid early convergence but potential premature convergence
            3. GA's evolutionary operators allow it to escape local optima
            """)
        
        # Selected Features
        st.markdown("### Selected Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **GA Selected Features (6):**
            1. feature_2
            2. feature_4
            3. feature_23
            4. feature_25
            5. feature_26
            6. feature_27
            """)
        
        with col2:
            st.markdown("""
            **PSO Selected Features (10):**
            1. feature_1
            2. feature_2
            3. feature_4
            4. feature_7
            5. feature_9
            6. feature_18
            7. feature_20
            8. feature_25
            9. feature_27
            10. feature_28
            """)
        
        st.info("**Common Features (4)**: feature_2, feature_4, feature_25, feature_27 - These appear in both solutions, suggesting they are highly discriminative.")
    
    elif page == "🎯 Run Algorithms":
        st.markdown('<div class="section-header">Run Algorithms</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Run the Genetic Algorithm and Particle Swarm Optimization algorithms with customizable parameters.
        The results will be displayed in real-time with convergence plots and performance metrics.
        """)
        
        # Parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### GA Parameters")
            ga_pop_size = st.slider("Population Size", 20, 100, 50, key="ga_pop")
            ga_generations = st.slider("Generations", 20, 100, 50, key="ga_gen")
            ga_crossover = st.slider("Crossover Rate", 0.0, 1.0, 0.8, key="ga_cross")
            ga_mutation = st.slider("Mutation Rate", 0.0, 0.5, 0.1, key="ga_mut")
        
        with col2:
            st.markdown("### PSO Parameters")
            pso_particles = st.slider("Number of Particles", 10, 50, 30, key="pso_part")
            pso_iterations = st.slider("Iterations", 20, 100, 50, key="pso_iter")
            pso_w = st.slider("Inertia Weight (w)", 0.0, 1.0, 0.7, key="pso_w")
            pso_c1 = st.slider("Cognitive Coefficient (c1)", 0.0, 3.0, 1.5, key="pso_c1")
            pso_c2 = st.slider("Social Coefficient (c2)", 0.0, 3.0, 1.5, key="pso_c2")
        
        model_type = st.selectbox("Model Type", ["knn", "logistic"], index=0)
        alpha = st.slider("Alpha (Penalty Coefficient)", 0.0, 0.5, 0.1, key="alpha")
        
        if st.button("🚀 Run Algorithms", type="primary"):
            try:
                # Load data
                with st.spinner("Loading data..."):
                    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess_data()
                    n_features = X_train.shape[1]
                
                # Run GA
                st.markdown("### Running Genetic Algorithm...")
                ga_progress = st.progress(0)
                ga_status = st.empty()
                
                ga = GeneticAlgorithm(
                    n_features=n_features,
                    population_size=ga_pop_size,
                    n_generations=ga_generations,
                    crossover_rate=ga_crossover,
                    mutation_rate=ga_mutation,
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
                
                ga_progress.progress(100)
                ga_status.success(f"✓ GA completed in {ga_time:.2f} seconds")
                
                # Run PSO
                st.markdown("### Running Particle Swarm Optimization...")
                pso_progress = st.progress(0)
                pso_status = st.empty()
                
                pso = ParticleSwarmOptimization(
                    n_features=n_features,
                    n_particles=pso_particles,
                    n_iterations=pso_iterations,
                    w=pso_w,
                    c1=pso_c1,
                    c2=pso_c2,
                    model_type=model_type,
                    alpha=alpha
                )
                
                start_time = time.time()
                pso_best_subset, pso_accuracy, pso_history = pso.run(X_train, X_test, y_train, y_test)
                pso_time = time.time() - start_time
                pso_n_features = np.sum(pso_best_subset)
                pso_fitness = pso_history[-1]
                
                pso_progress.progress(100)
                pso_status.success(f"✓ PSO completed in {pso_time:.2f} seconds")
                
                # Store results in session state
                st.session_state['ga_results'] = {
                    'accuracy': ga_accuracy,
                    'n_features': ga_n_features,
                    'fitness': ga_fitness,
                    'subset': ga_best_subset,
                    'history': ga_history,
                    'time': ga_time
                }
                st.session_state['pso_results'] = {
                    'accuracy': pso_accuracy,
                    'n_features': pso_n_features,
                    'fitness': pso_fitness,
                    'subset': pso_best_subset,
                    'history': pso_history,
                    'time': pso_time
                }
                st.session_state['feature_names'] = feature_names
                
                st.success("✅ Algorithms completed successfully!")
                
            except FileNotFoundError:
                st.error("❌ Dataset not found! Please ensure data/data.csv exists.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        # Display results if available
        if 'ga_results' in st.session_state and 'pso_results' in st.session_state:
            st.markdown("---")
            st.markdown("### Results")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("GA Accuracy", f"{st.session_state['ga_results']['accuracy']:.4f}")
            with col2:
                st.metric("PSO Accuracy", f"{st.session_state['pso_results']['accuracy']:.4f}")
            with col3:
                st.metric("GA Features", int(st.session_state['ga_results']['n_features']))
            with col4:
                st.metric("PSO Features", int(st.session_state['pso_results']['n_features']))
            
            # Plots
            col1, col2 = st.columns(2)
            
            with col1:
                fig_conv = plot_convergence(
                    st.session_state['ga_results']['history'],
                    st.session_state['pso_results']['history']
                )
                st.pyplot(fig_conv)
            
            with col2:
                fig_bar = plot_bar_comparison(
                    st.session_state['ga_results'],
                    st.session_state['pso_results']
                )
                st.pyplot(fig_bar)
            
            # Selected features
            st.markdown("### Selected Features")
            col1, col2 = st.columns(2)
            
            with col1:
                ga_selected = [st.session_state['feature_names'][i] 
                              for i in np.where(st.session_state['ga_results']['subset'] == 1)[0]]
                st.markdown("**GA Selected Features:**")
                for i, feat in enumerate(ga_selected, 1):
                    st.write(f"{i}. {feat}")
            
            with col2:
                pso_selected = [st.session_state['feature_names'][i] 
                               for i in np.where(st.session_state['pso_results']['subset'] == 1)[0]]
                st.markdown("**PSO Selected Features:**")
                for i, feat in enumerate(pso_selected, 1):
                    st.write(f"{i}. {feat}")
    
    elif page == "📚 Full Report":
        st.markdown('<div class="section-header">Complete Assignment Report</div>', unsafe_allow_html=True)
        
        # Read and display the markdown report
        try:
            with open('ASSIGNMENT_REPORT.md', 'r', encoding='utf-8') as f:
                report_content = f.read()
                # Remove the YAML front matter
                if report_content.startswith('---'):
                    parts = report_content.split('---', 2)
                    if len(parts) > 2:
                        report_content = parts[2]
                st.markdown(report_content)
        except FileNotFoundError:
            st.error("Report file not found. Please ensure ASSIGNMENT_REPORT.md exists.")
            st.markdown("""
            ## Complete Assignment Report Content
            
            ### 1. Introduction
            This assignment implements and compares two metaheuristic optimization algorithms...
            
            [Full report content would be displayed here]
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
    <p>Feature Selection using Genetic Algorithm and Particle Swarm Optimization</p>
    <p>Assignment Report | December 2024</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

