"""
Run experiments on both OR-Library and Taillard instances.
"""
import os
import sys
import time
import tracemalloc
from datetime import datetime
from typing import Dict, List
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import psutil
import json

# Add parent directory to path to find modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_scripts.algorithms.genetic_algorithm import GeneticAlgorithm
from test_scripts.algorithms.particle_swarm import ParticleSwarmOptimization
from test_scripts.algorithms.simulated_annealing import SimulatedAnnealing
from test_scripts.algorithms.ilp_solver import ILPSolver
from test_scripts.instance_parser import parse_instance
from test_scripts.scripts.analyze_results import plot_results, generate_summary_report

def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def get_instance_info(instance_file: str) -> Dict:
    """Get basic information about an instance."""
    with open(instance_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # For Taillard instances
        if any(line.startswith("Nb of jobs") for line in lines):
            # Find the line with dimensions
            for i, line in enumerate(lines):
                if line.startswith("Nb of jobs"):
                    dim_line = lines[i+1]
                    n_jobs, n_machines = map(int, dim_line.split()[:2])
                    break
        # For OR-Library instances
        else:
            # Get dimensions from second line
            n_jobs, n_machines = map(int, lines[1].split())
            
        return {
            'n_jobs': n_jobs,
            'n_machines': n_machines,
            'instance_type': 'taillard' if 'tai' in instance_file else 'orlib'
        }

def run_algorithm(name: str, instance_file: str, **kwargs) -> Dict:
    """Run a specific algorithm on an instance and collect metrics."""
    # Start memory tracking
    tracemalloc.start()
    
    # Parse instance and get info
    n_jobs, n_machines, proc_times, machine_order = parse_instance(instance_file)
    
    # Initialize metrics
    metrics = {
        'algorithm': name,  # Set algorithm name first
        'instance': os.path.basename(instance_file),
        'start_time': datetime.now().isoformat(),
        'n_jobs': n_jobs,
        'n_machines': n_machines,
        'instance_type': 'taillard' if 'tai' in instance_file else 'orlib',
        'convergence_history': []
    }
    
    start_time = time.time()
    start_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # Convert to MB
    
    # Run algorithm
    try:
        if name == "ILP":
            solver = ILPSolver(instance_file)
            makespan = solver.solve()  # Get just the makespan value
            current_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
            metrics['makespan'] = makespan
            metrics['convergence_history'] = [{
                'generation': 0,
                'makespan': makespan,
                'best_makespan': makespan,
                'memory': current_memory - start_memory
            }]
        else:
            # Extract max_generations from kwargs
            max_generations = kwargs.pop('max_generations', 100)
            
            # Initialize algorithm
            if name == "GA":
                algo = GeneticAlgorithm(instance_file)
            elif name == "PSO":
                algo = ParticleSwarmOptimization(instance_file)
            else:  # SA
                algo = SimulatedAnnealing(instance_file)
            
            # Run with progress tracking
            best_makespan = float('inf')
            for gen in range(max_generations):
                current_makespan = algo.step()  # Use step() for all algorithms
                best_makespan = min(best_makespan, current_makespan)
                
                # Print progress every 10 generations
                if gen % 10 == 0:
                    print(f"Generation {gen}: Best Makespan = {best_makespan}")
                
                # Get current memory usage
                current_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
                
                metrics['convergence_history'].append({
                    'generation': gen,
                    'makespan': current_makespan,
                    'best_makespan': best_makespan,
                    'memory': current_memory - start_memory
                })
            
            metrics['makespan'] = best_makespan
            
    except Exception as e:
        print(f"Error running {name} on {instance_file}: {str(e)}")
        metrics['error'] = str(e)
        metrics['makespan'] = float('inf')
    
    # Add final metrics
    current_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
    metrics.update({
        'execution_time': time.time() - start_time,
        'memory_usage': current_memory - start_memory,
        'end_time': datetime.now().isoformat()
    })
    
    # Stop memory tracking
    tracemalloc.stop()
    
    return metrics

def run_experiments(instances: List[str], output_dir: str, algorithms: List[str] = None):
    """Run comprehensive experiments on multiple instances.
    
    Args:
        instances: List of instance file paths
        output_dir: Directory to save results
        algorithms: List of algorithms to run (default: all)
    """
    # Create output directories
    results_dir = os.path.join(output_dir, 'results')
    figures_dir = os.path.join(output_dir, 'figures')
    
    # Clear old results
    if os.path.exists(results_dir):
        for f in os.listdir(results_dir):
            os.remove(os.path.join(results_dir, f))
    if os.path.exists(figures_dir):
        for f in os.listdir(figures_dir):
            os.remove(os.path.join(figures_dir, f))
            
    # Create fresh directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Default to all algorithms if none specified
    if not algorithms:
        algorithms = ["GA", "PSO", "SA", "ILP"]
    
    # Run experiments for each instance
    for instance in instances:
        instance_name = os.path.splitext(os.path.basename(instance))[0]
        instance_results = []
        
        # Run each algorithm
        for algo in algorithms:
            print(f"\nRunning {algo} on {instance_name}...")
            result = run_algorithm(algo, instance, max_generations=100)
            instance_results.append(result)
            
            # Save individual result
            result_file = os.path.join(results_dir, f"{instance_name}_{algo}.json")
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, cls=NumpyEncoder)
        
        # Save combined results for this instance
        df = pd.DataFrame([r for r in instance_results if 'makespan' in r])
        if not df.empty:
            print("\nResults for", instance_name)
            print(df[['algorithm', 'makespan', 'execution_time']].to_string())
            
            # Generate plots and report for this instance
            plot_results([r for r in instance_results if 'makespan' in r], 
                       os.path.join(figures_dir, instance_name))
            generate_summary_report(df, os.path.join(output_dir, instance_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run JSSP experiments')
    parser.add_argument('--instances', nargs='+', help='List of instances to test')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--algorithms', nargs='+', choices=['GA', 'PSO', 'SA', 'ILP'],
                      help='Algorithms to run (default: all)')
    
    args = parser.parse_args()
    
    if not args.instances:
        parser.error("At least one instance file must be specified")
    
    run_experiments(args.instances, args.output_dir, args.algorithms)
