"""
Particle Swarm Optimization implementation for Job Shop Scheduling Problem.

This implementation uses rank-based mapping for position updates and includes
a repair mechanism to ensure valid schedules. It supports both OR-Library and
Taillard instance formats.
"""
import numpy as np
import random
import time
import os
import sys

# Add parent directory to path to find modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from test_scripts.instance_parser import parse_instance

class Particle:
    """Particle class for PSO algorithm.
    
    Each particle represents a potential solution to the JSSP,
    maintaining its current position, velocity, and best known position.
    """
    def __init__(self, n_jobs, n_machines):
        """Initialize particle with random position and velocity.
        
        Args:
            n_jobs: Number of jobs in the instance
            n_machines: Number of machines/operations per job
        """
        # Initialize position as a valid schedule
        self.position = []
        for job in range(n_jobs):
            self.position.extend([job] * n_machines)
        random.shuffle(self.position)
        
        # Initialize velocity as small random values
        self.velocity = np.random.uniform(-0.1, 0.1, len(self.position))
        
        # Initialize best position and score
        self.best_position = self.position.copy()
        self.best_score = float('inf')

class ParticleSwarmOptimization:
    """Particle Swarm Optimization for Job Shop Scheduling Problem."""
    
    def __init__(self, instance_file, n_particles=50, inertia=0.5, c1=1.5, c2=1.5):
        """Initialize PSO solver.
        
        Args:
            instance_file: Path to the JSSP instance file
            n_particles: Number of particles in the swarm (default: 50)
            inertia: Inertia weight for velocity update (default: 0.5)
            c1: Cognitive coefficient (default: 1.5)
            c2: Social coefficient (default: 1.5)
        """
        # Parse instance
        self.n_jobs, self.n_machines, self.proc_times, self.machine_order = parse_instance(instance_file)
        
        # PSO parameters
        self.n_particles = n_particles
        self.inertia = inertia
        self.c1 = c1  # cognitive coefficient
        self.c2 = c2  # social coefficient
        
        # Initialize particles
        self.particles = [Particle(self.n_jobs, self.n_machines) for _ in range(n_particles)]
        
        # Initialize global best with first particle's position
        self.global_best_position = self.particles[0].position.copy()
        self.global_best_score = self.evaluate(self.global_best_position)
        
        # Store instance name for reference
        self.instance_file = instance_file
        
    def evaluate(self, schedule):
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
        
    def repair_schedule(self, schedule):
        """Repair schedule to ensure job count constraints."""
        job_counts = np.bincount(schedule, minlength=self.n_jobs)
        target_count = self.n_machines
        
        # Find jobs that appear too many or too few times
        excess_jobs = [j for j, count in enumerate(job_counts) if count > target_count]
        missing_jobs = [j for j, count in enumerate(job_counts) if count < target_count]
        
        schedule = schedule.copy()
        for j_excess in excess_jobs:
            excess_positions = [i for i, job in enumerate(schedule) if job == j_excess]
            excess_count = job_counts[j_excess] - target_count
            positions_to_fix = excess_positions[-excess_count:]
            
            for pos, j_missing in zip(positions_to_fix, missing_jobs):
                schedule[pos] = j_missing
                job_counts[j_excess] -= 1
                job_counts[j_missing] += 1
                
                if job_counts[j_missing] == target_count:
                    missing_jobs.remove(j_missing)
        
        return schedule
        
    def update_particle(self, particle):
        """Update particle position and velocity."""
        # Update velocity with bounds
        r1, r2 = random.random(), random.random()
        cognitive = self.c1 * r1 * (np.array(particle.best_position) - np.array(particle.position))
        social = self.c2 * r2 * (np.array(self.global_best_position) - np.array(particle.position))
        
        new_velocity = self.inertia * particle.velocity + cognitive + social
        # Clip velocity to prevent too large jumps
        particle.velocity = np.clip(new_velocity, -1.0, 1.0)
        
        # Update position using rank-based mapping
        new_position = np.array(particle.position) + particle.velocity
        sorted_indices = np.argsort(new_position)
        
        # Create new schedule maintaining job repetition constraint
        new_schedule = []
        for job in range(self.n_jobs):
            new_schedule.extend([job] * self.n_machines)
        particle.position = [new_schedule[i] for i in sorted_indices]
        
        # Evaluate new position
        score = self.evaluate(particle.position)
        if score < particle.best_score:
            particle.best_position = particle.position.copy()
            particle.best_score = score
            if score < self.global_best_score:
                self.global_best_position = particle.position.copy()
                self.global_best_score = score

    def solve(self, n_iterations=100):
        """Run PSO algorithm.
        
        Args:
            n_iterations: Maximum number of iterations to run (default: 100)
            
        Returns:
            Best makespan found
        """
        random.seed(42)  # For reproducibility
        np.random.seed(42)
        
        start_time = time.time()
        
        # Evaluate initial positions
        for particle in self.particles:
            score = self.evaluate(particle.position)
            if score < particle.best_score:
                particle.best_position = particle.position.copy()
                particle.best_score = score
                if score < self.global_best_score:
                    self.global_best_position = particle.position.copy()
                    self.global_best_score = score
        
        # Main PSO loop
        for iteration in range(n_iterations):
            self.step()
            
            if iteration % 10 == 0:  # Print progress every 10 iterations
                print(f"Iteration {iteration}: Best Makespan = {self.global_best_score}")
        
        execution_time = time.time() - start_time
        print(f"Execution time: {execution_time:.2f}s")
        return self.global_best_score

    def step(self):
        """Run one iteration of PSO."""
        for particle in self.particles:
            self.update_particle(particle)
        return self.global_best_score

if __name__ == "__main__":
    instance_file = sys.argv[1]
    n_particles = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    inertia = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    c1 = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
    c2 = float(sys.argv[5]) if len(sys.argv) > 5 else 1.5
    
    pso = ParticleSwarmOptimization(instance_file, n_particles, inertia, c1, c2)
    makespan = pso.solve()
    print(f"Final Makespan: {makespan}")
