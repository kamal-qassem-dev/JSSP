"""
Genetic Algorithm for Job Shop Scheduling Problem.
"""
import os
import sys
import time
import random
import numpy as np
from deap import base, creator, tools, algorithms

# Clear any existing DEAP types
if 'FitnessMin' in creator.__dict__:
    del creator.FitnessMin
if 'Individual' in creator.__dict__:
    del creator.Individual

# Create DEAP types at module level
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

class GeneticAlgorithm:
    def __init__(self, instance_file, pop_size=100, cross_rate=0.8, mut_rate=0.1):
        """Initialize GA with instance file and parameters."""
        self.instance_file = instance_file
        self.pop_size = pop_size
        self.cross_rate = cross_rate
        self.mut_rate = mut_rate
        
        # Parse instance
        self.n_jobs, self.n_machines, self.proc_times, self.machine_order = self._parse_instance(instance_file)
        
        # Initialize toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("indices", self._create_schedule)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.indices)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register("mate", self._custom_mate)
        self.toolbox.register("mutate", self._custom_mutate, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Initialize population
        self.population = self.toolbox.population(n=self.pop_size)
        self.best_makespan = float('inf')
        self.generation = 0
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, self.population))
        for ind, fit in zip(self.population, fitnesses):
            ind.fitness.values = fit
            self.best_makespan = min(self.best_makespan, fit[0])
            
    def _parse_instance(self, filename):
        """Parse instance file format."""
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
            # For Taillard instances
            if any(line.startswith("Nb of jobs") for line in lines):
                # Find the line with dimensions
                for i, line in enumerate(lines):
                    if line.startswith("Nb of jobs"):
                        dim_line = lines[i+1]
                        n_jobs, n_machines = map(int, dim_line.split()[:2])
                        break
                
                # Find processing times and machine order
                proc_times = np.zeros((n_jobs, n_machines), dtype=int)
                machine_order = np.zeros((n_jobs, n_machines), dtype=int)
                
                # Find processing times
                for i, line in enumerate(lines):
                    if line == "Times":
                        data_lines = lines[i+1:i+1+n_jobs]
                        proc_times = np.array([list(map(int, line.split()))
                                           for line in data_lines])
                        break
                
                # Find machine order
                for i, line in enumerate(lines):
                    if line == "Machines":
                        data_lines = lines[i+1:i+1+n_jobs]
                        machine_order = np.array([list(map(lambda x: int(x)-1, line.split()))
                                              for line in data_lines])
                        break
                        
            # For OR-Library instances
            else:
                # Get dimensions from second line
                n_jobs, n_machines = map(int, lines[1].split())
                
                # Parse job data
                jobs_data = [list(map(int, line.split())) for line in lines[2:]]
                
                # Convert to processing times and machine order matrices
                proc_times = np.zeros((n_jobs, n_machines), dtype=int)
                machine_order = np.zeros((n_jobs, n_machines), dtype=int)
                
                for i, job in enumerate(jobs_data):
                    for j in range(n_machines):
                        machine_order[i][j] = job[j*2]  # Even indices are machine numbers
                        proc_times[i][j] = job[j*2 + 1]  # Odd indices are processing times
                
                # Convert machine numbers to 0-based indexing
                machine_order = machine_order - 1
            
            return n_jobs, n_machines, proc_times, machine_order
            
    def _create_schedule(self):
        """Create a valid schedule sequence."""
        schedule = []
        for job in range(self.n_jobs):
            schedule.extend([job] * self.n_machines)
        random.shuffle(schedule)
        return schedule
        
    def _evaluate(self, individual):
        """Calculate makespan for a schedule."""
        machine_times = np.zeros(self.n_machines)
        job_op_count = np.zeros(self.n_jobs, dtype=int)
        job_completion = np.zeros((self.n_jobs, self.n_machines))
        
        for job in individual:
            op = job_op_count[job]
            if op >= self.n_machines:
                return 999999,
            
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
            return 999999,
        
        return max(machine_times),
        
    def _custom_mutate(self, individual, indpb):
        """Swap mutation that maintains job count constraints."""
        for i in range(len(individual)):
            if random.random() < indpb:
                j = random.randint(0, len(individual)-1)
                individual[i], individual[j] = individual[j], individual[i]
        return individual,
        
    def _custom_mate(self, ind1, ind2):
        """Order crossover that maintains job count constraints."""
        size = min(len(ind1), len(ind2))
        point1, point2 = sorted(random.sample(range(size), 2))
        
        # Create offspring
        off1 = ind1[point1:point2]
        off2 = ind2[point1:point2]
        
        # Complete offspring1
        remaining1 = [x for x in ind2 if x not in off1]
        off1 = remaining1[:point1] + off1 + remaining1[point1:]
        
        # Complete offspring2
        remaining2 = [x for x in ind1 if x not in off2]
        off2 = remaining2[:point1] + off2 + remaining2[point1:]
        
        return off1, off2
        
    def step(self):
        """Run one generation of the genetic algorithm."""
        # Select next generation
        offspring = self.toolbox.select(self.population, len(self.population))
        offspring = list(map(creator.Individual, offspring))
        
        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < self.cross_rate:
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Apply mutation
        for mutant in offspring:
            if random.random() < self.mut_rate:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            self.best_makespan = min(self.best_makespan, fit[0])
        
        # Replace population
        self.population[:] = offspring
        self.generation += 1
        
        return self.best_makespan
        
    def solve(self, n_generations=100):
        """Run genetic algorithm until convergence or max generations.
        
        Args:
            n_generations: Maximum number of generations to run (default: 100)
            
        Returns:
            Best makespan found
        """
        random.seed(42)  # For reproducibility
        
        start_time = time.time()
        
        # Run evolution
        for gen in range(n_generations):
            makespan = self.step()
            if gen % 10 == 0:  # Print progress every 10 generations
                print(f"Generation {gen}: Best Makespan = {makespan}")
        
        execution_time = time.time() - start_time
        print(f"Execution time: {execution_time:.2f}s")
        return self.best_makespan

if __name__ == "__main__":
    instance_file = sys.argv[1]
    pop_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    mut_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    cross_rate = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8
    
    ga = GeneticAlgorithm(instance_file, pop_size, cross_rate, mut_rate)
    makespan = ga.solve()
    print(f"Final Makespan: {makespan}")
