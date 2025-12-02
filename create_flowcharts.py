"""
Script to create flowcharts for GA and PSO algorithms.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

def create_ga_flowchart():
    """Create flowchart for Genetic Algorithm."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Define box styles
    box_style = dict(boxstyle="round,pad=0.5", facecolor="lightblue", edgecolor="black", linewidth=1.5)
    decision_style = dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="black", linewidth=1.5)
    process_style = dict(boxstyle="round,pad=0.5", facecolor="lightgreen", edgecolor="black", linewidth=1.5)
    
    # Start
    start = FancyBboxPatch((3.5, 13), 3, 0.6, **box_style)
    ax.add_patch(start)
    ax.text(5, 13.3, "START", ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Initialize population
    init = FancyBboxPatch((2, 11.5), 6, 0.6, **process_style)
    ax.add_patch(init)
    ax.text(5, 11.8, "Initialize Population\n(Random binary vectors)", ha='center', va='center', fontsize=10)
    
    # Evaluate fitness
    eval1 = FancyBboxPatch((2, 10), 6, 0.6, **process_style)
    ax.add_patch(eval1)
    ax.text(5, 10.3, "Evaluate Fitness\nfor each individual", ha='center', va='center', fontsize=10)
    
    # Generation loop
    gen = FancyBboxPatch((2, 8.5), 6, 0.6, **decision_style)
    ax.add_patch(gen)
    ax.text(5, 8.8, "Generation < Max\nGenerations?", ha='center', va='center', fontsize=10)
    
    # Selection
    select = FancyBboxPatch((0.5, 6.5), 4, 0.6, **process_style)
    ax.add_patch(select)
    ax.text(2.5, 6.8, "Tournament Selection", ha='center', va='center', fontsize=10)
    
    # Crossover
    cross = FancyBboxPatch((0.5, 5), 4, 0.6, **process_style)
    ax.add_patch(cross)
    ax.text(2.5, 5.3, "One-Point Crossover", ha='center', va='center', fontsize=10)
    
    # Mutation
    mutate = FancyBboxPatch((0.5, 3.5), 4, 0.6, **process_style)
    ax.add_patch(mutate)
    ax.text(2.5, 3.8, "Bit-Flip Mutation", ha='center', va='center', fontsize=10)
    
    # Evaluate new population
    eval2 = FancyBboxPatch((0.5, 2), 4, 0.6, **process_style)
    ax.add_patch(eval2)
    ax.text(2.5, 2.3, "Evaluate New\nPopulation", ha='center', va='center', fontsize=10)
    
    # Elitism
    elite = FancyBboxPatch((5.5, 4.5), 4, 0.6, **process_style)
    ax.add_patch(elite)
    ax.text(7.5, 4.8, "Apply Elitism", ha='center', va='center', fontsize=10)
    
    # Increment generation
    inc = FancyBboxPatch((5.5, 3), 4, 0.6, **process_style)
    ax.add_patch(inc)
    ax.text(7.5, 3.3, "Increment\nGeneration", ha='center', va='center', fontsize=10)
    
    # Return best
    ret = FancyBboxPatch((3.5, 0.5), 3, 0.6, **box_style)
    ax.add_patch(ret)
    ax.text(5, 0.8, "RETURN\nBest Solution", ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Arrows
    arrows = [
        [(5, 13), (5, 12.1)],  # Start to Init
        [(5, 11.5), (5, 10.6)],  # Init to Eval1
        [(5, 10), (5, 9.1)],  # Eval1 to Gen
        [(2, 8.5), (2.5, 7.1)],  # Gen to Select (Yes)
        [(2.5, 6.5), (2.5, 5.6)],  # Select to Cross
        [(2.5, 5), (2.5, 4.1)],  # Cross to Mutate
        [(2.5, 3.5), (2.5, 2.6)],  # Mutate to Eval2
        [(4.5, 3.5), (5.5, 4.1)],  # Mutate to Elite
        [(7.5, 4.5), (7.5, 3.6)],  # Elite to Inc
        [(7.5, 3), (8, 8.8)],  # Inc back to Gen
        [(8, 8.5), (5, 1.1)],  # Gen to Return (No)
    ]
    
    for (x1, y1), (x2, y2) in arrows:
        if abs(y1 - y2) < 0.5:  # Horizontal
            arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                                  arrowstyle='->', lw=1.5, color='black',
                                  connectionstyle="arc3,rad=0.3")
        else:  # Vertical
            arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                                  arrowstyle='->', lw=1.5, color='black')
        ax.add_patch(arrow)
    
    # Labels
    ax.text(1.5, 7.5, "Yes", ha='center', va='center', fontsize=9, style='italic')
    ax.text(8.5, 5.5, "Loop", ha='center', va='center', fontsize=9, style='italic')
    ax.text(8.5, 8.5, "No", ha='center', va='center', fontsize=9, style='italic')
    
    plt.title('Genetic Algorithm Flowchart', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('ga_flowchart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("GA flowchart saved to ga_flowchart.png")


def create_pso_flowchart():
    """Create flowchart for Particle Swarm Optimization."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Define box styles
    box_style = dict(boxstyle="round,pad=0.5", facecolor="lightblue", edgecolor="black", linewidth=1.5)
    decision_style = dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="black", linewidth=1.5)
    process_style = dict(boxstyle="round,pad=0.5", facecolor="lightgreen", edgecolor="black", linewidth=1.5)
    
    # Start
    start = FancyBboxPatch((3.5, 13), 3, 0.6, **box_style)
    ax.add_patch(start)
    ax.text(5, 13.3, "START", ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Initialize swarm
    init = FancyBboxPatch((2, 11.5), 6, 0.6, **process_style)
    ax.add_patch(init)
    ax.text(5, 11.8, "Initialize Swarm\n(Positions & Velocities)", ha='center', va='center', fontsize=10)
    
    # Evaluate fitness
    eval1 = FancyBboxPatch((2, 10), 6, 0.6, **process_style)
    ax.add_patch(eval1)
    ax.text(5, 10.3, "Evaluate Fitness\nfor each particle", ha='center', va='center', fontsize=10)
    
    # Initialize pbest and gbest
    init_best = FancyBboxPatch((2, 8.5), 6, 0.6, **process_style)
    ax.add_patch(init_best)
    ax.text(5, 8.8, "Initialize pbest\nand gbest", ha='center', va='center', fontsize=10)
    
    # Iteration loop
    iter_loop = FancyBboxPatch((2, 7), 6, 0.6, **decision_style)
    ax.add_patch(iter_loop)
    ax.text(5, 7.3, "Iteration < Max\nIterations?", ha='center', va='center', fontsize=10)
    
    # Update velocity
    update_v = FancyBboxPatch((0.5, 5), 4, 0.6, **process_style)
    ax.add_patch(update_v)
    ax.text(2.5, 5.3, "Update Velocity\n(v = w*v + c1*r1*(pbest-x)\n+ c2*r2*(gbest-x))", ha='center', va='center', fontsize=9)
    
    # Update position
    update_p = FancyBboxPatch((0.5, 3.5), 4, 0.6, **process_style)
    ax.add_patch(update_p)
    ax.text(2.5, 3.8, "Update Position\n(Sigmoid function)", ha='center', va='center', fontsize=10)
    
    # Evaluate fitness
    eval2 = FancyBboxPatch((0.5, 2), 4, 0.6, **process_style)
    ax.add_patch(eval2)
    ax.text(2.5, 2.3, "Evaluate Fitness", ha='center', va='center', fontsize=10)
    
    # Update pbest and gbest
    update_best = FancyBboxPatch((5.5, 4.5), 4, 0.6, **process_style)
    ax.add_patch(update_best)
    ax.text(7.5, 4.8, "Update pbest\nand gbest", ha='center', va='center', fontsize=10)
    
    # Increment iteration
    inc = FancyBboxPatch((5.5, 3), 4, 0.6, **process_style)
    ax.add_patch(inc)
    ax.text(7.5, 3.3, "Increment\nIteration", ha='center', va='center', fontsize=10)
    
    # Return best
    ret = FancyBboxPatch((3.5, 0.5), 3, 0.6, **box_style)
    ax.add_patch(ret)
    ax.text(5, 0.8, "RETURN\nBest Solution", ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Arrows
    arrows = [
        [(5, 13), (5, 12.1)],  # Start to Init
        [(5, 11.5), (5, 10.6)],  # Init to Eval1
        [(5, 10), (5, 9.1)],  # Eval1 to Init_best
        [(5, 8.5), (5, 7.6)],  # Init_best to Iter
        [(2, 7), (2.5, 5.6)],  # Iter to Update_v (Yes)
        [(2.5, 5), (2.5, 4.1)],  # Update_v to Update_p
        [(2.5, 3.5), (2.5, 2.6)],  # Update_p to Eval2
        [(4.5, 3.5), (5.5, 4.1)],  # Update_p to Update_best
        [(7.5, 4.5), (7.5, 3.6)],  # Update_best to Inc
        [(7.5, 3), (8, 7.3)],  # Inc back to Iter
        [(8, 7), (5, 1.1)],  # Iter to Return (No)
    ]
    
    for (x1, y1), (x2, y2) in arrows:
        if abs(y1 - y2) < 0.5:  # Horizontal
            arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                                  arrowstyle='->', lw=1.5, color='black',
                                  connectionstyle="arc3,rad=0.3")
        else:  # Vertical
            arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                                  arrowstyle='->', lw=1.5, color='black')
        ax.add_patch(arrow)
    
    # Labels
    ax.text(1.5, 6, "Yes", ha='center', va='center', fontsize=9, style='italic')
    ax.text(8.5, 5, "Loop", ha='center', va='center', fontsize=9, style='italic')
    ax.text(8.5, 7, "No", ha='center', va='center', fontsize=9, style='italic')
    
    plt.title('Particle Swarm Optimization Flowchart', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('pso_flowchart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("PSO flowchart saved to pso_flowchart.png")


if __name__ == "__main__":
    create_ga_flowchart()
    create_pso_flowchart()
    print("\nAll flowcharts created successfully!")

