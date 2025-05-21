"""
Simulated Annealing implementation for Job Shop Scheduling Problem.

This implementation uses a temperature-based acceptance criterion for
worse solutions and supports both OR-Library and Taillard instance formats.
"""
import numpy as np
import random
import math
import time
import os
import sys

# Add parent directory to path to find modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from test_scripts.instance_parser import parse_instance

class SimulatedAnnealing:
    """Simulated Annealing solver for Job Shop Scheduling Problem."""
    
    def __init__(self, instance_file, initial_temp=1000, cooling_rate=0.95, min_temp=1):
        """Initialize SA solver.
        
        Args:
            instance_file: Path to the JSSP instance file
            initial_temp: Initial temperature (default: 1000)
            cooling_rate: Temperature cooling rate (default: 0.95)
            min_temp: Minimum temperature to stop at (default: 1)
        """
        # Parse instance
        self.n_jobs, self.n_machines, self.proc_times, self.machine_order = parse_instance(instance_file)
        
        # SA parameters
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.temperature = initial_temp  # Initialize current temperature
        
        # Initialize solution
        self.current_solution = self.create_initial_solution()
        self.current_makespan = self.calculate_makespan(self.current_solution)
        
        # Best solution found
        self.best_solution = self.current_solution.copy()
        self.best_makespan = self.current_makespan
        
        # Store instance name for reference
        self.instance_file = instance_file

    def create_initial_solution(self):
        """Create a valid initial schedule."""
        schedule = []
        for job in range(self.n_jobs):
            schedule.extend([job] * self.n_machines)
        random.shuffle(schedule)
        return schedule
        
    def calculate_makespan(self, schedule):
        """Calculate makespan for a schedule."""
        machine_times = np.zeros(self.n_machines)
        job_op_count = np.zeros(self.n_jobs, dtype=int)
        job_completion = np.zeros((self.n_jobs, self.n_machines))
        
        for job in schedule:
            op = job_op_count[job]
            if op >= self.n_machines:
                return float('inf')
            
            machine = self.machine_order[job][op]
            if op == 0:
                earliest_start = 0
            else:
                earliest_start = job_completion[job][op-1]
            
            start_time = max(machine_times[machine], earliest_start)
            duration = self.proc_times[job][op]
            finish_time = start_time + duration
            
            machine_times[machine] = finish_time
            job_completion[job][op] = finish_time
            job_op_count[job] += 1
        
        if not all(count == self.n_machines for count in job_op_count):
            return float('inf')
        
        return max(machine_times)
        
    def get_neighbor(self, schedule):
        """Generate neighbor solution by swapping two random positions."""
        neighbor = schedule.copy()
        i, j = random.sample(range(len(schedule)), 2)
        while neighbor[i] == neighbor[j]:  # Ensure different jobs
            j = random.randint(0, len(schedule) - 1)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor
        
    def acceptance_probability(self, current_cost, new_cost, temperature):
        """Calculate probability of accepting worse solution."""
        if new_cost < current_cost:
            return 1.0
        return math.exp((current_cost - new_cost) / temperature)
        
    def solve(self, n_iterations=10000):
        """Run SA algorithm and return best makespan.
        
        Args:
            n_iterations: Maximum number of iterations (default: 10000)
        
        Returns:
            Best makespan found
        """
        random.seed(42)  # For reproducibility
        start_time = time.time()
        
        iteration = 0
        
        while self.temperature > self.min_temp and iteration < n_iterations:
            self.step()
            
            # Cool down and increment
            self.temperature *= self.cooling_rate
            iteration += 1
            
            if iteration % 1000 == 0:  # Print progress every 1000 iterations
                print(f"Iteration {iteration}: Best Makespan = {self.best_makespan}")
        
        execution_time = time.time() - start_time
        print(f"Execution time: {execution_time:.2f}s")
        return self.best_makespan

    def step(self):
        """Run one iteration of SA."""
        # Generate and evaluate neighbor
        neighbor = self.get_neighbor(self.current_solution)
        neighbor_makespan = self.calculate_makespan(neighbor)
        
        # Decide if we should accept the neighbor
        if self.acceptance_probability(self.current_makespan, neighbor_makespan, self.temperature) > random.random():
            self.current_solution = neighbor
            self.current_makespan = neighbor_makespan
            
            # Update best solution if needed
            if neighbor_makespan < self.best_makespan:
                self.best_solution = neighbor.copy()
                self.best_makespan = neighbor_makespan
        
        return self.best_makespan

if __name__ == "__main__":
    instance_file = sys.argv[1]
    initial_temp = float(sys.argv[2]) if len(sys.argv) > 2 else 1000
    cooling_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 0.95
    min_temp = float(sys.argv[4]) if len(sys.argv) > 4 else 1
    
    sa = SimulatedAnnealing(instance_file, initial_temp, cooling_rate, min_temp)
    makespan = sa.solve()
    print(f"Final Makespan: {makespan}")
