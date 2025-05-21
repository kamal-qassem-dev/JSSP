"""
Integer Linear Programming solver for Job Shop Scheduling Problem.
"""
import pulp
import os
import sys
import time
import numpy as np

# Add parent directory to path to find modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from test_scripts.instance_parser import parse_instance

class ILPSolver:
    def __init__(self, instance_file):
        """Initialize ILP solver with instance file."""
        self.instance_file = instance_file
        self.n_jobs, self.n_machines, self.proc_times, self.machine_order = parse_instance(instance_file)
        
    def solve(self):
        """Solve JSSP using Integer Linear Programming.
        
        Returns:
            Optimal makespan
        """
        # Create ILP problem
        prob = pulp.LpProblem("JSSP", pulp.LpMinimize)
        
        # Calculate Big M as sum of all processing times + 1
        M = sum(self.proc_times.flatten()) + 1
        
        # Decision variables
        x = {}  # Start times
        y = {}  # Binary precedence variables
        C_max = pulp.LpVariable("C_max", 0, None)  # Makespan
        
        # Create start time variables for each operation
        for i in range(self.n_jobs):
            for j in range(self.n_machines):
                x[i,j] = pulp.LpVariable(f"x_{i}_{j}", 0, None)
        
        # Create precedence variables for operations on same machine
        for m in range(self.n_machines):
            for i in range(self.n_jobs):
                for k in range(i+1, self.n_jobs):
                    y[i,k,m] = pulp.LpVariable(f"y_{i}_{k}_{m}", 0, 1, pulp.LpBinary)
        
        # Objective: Minimize makespan
        prob += C_max
        
        # Constraints
        # 1. Operation sequence within each job
        for i in range(self.n_jobs):
            for j in range(self.n_machines-1):
                prob += x[i,j] + self.proc_times[i][j] <= x[i,j+1]
        
        # 2. No overlap on machines
        for m in range(self.n_machines):
            for i in range(self.n_jobs):
                for k in range(i+1, self.n_jobs):
                    # Find positions where machine m is used
                    pos_i = np.where(self.machine_order[i] == m)[0][0]
                    pos_k = np.where(self.machine_order[k] == m)[0][0]
                    
                    # Either job i before k or k before i on machine m
                    prob += x[i,pos_i] + self.proc_times[i][pos_i] <= x[k,pos_k] + M * (1 - y[i,k,m])
                    prob += x[k,pos_k] + self.proc_times[k][pos_k] <= x[i,pos_i] + M * y[i,k,m]
        
        # 3. Makespan constraint
        for i in range(self.n_jobs):
            prob += x[i,self.n_machines-1] + self.proc_times[i][self.n_machines-1] <= C_max
        
        # Start timing
        start_time = time.time()
        
        # Use CBC solver with default settings
        solver = pulp.PULP_CBC_CMD(msg=1)
        prob.solve(solver)
        
        if prob.status != 1:
            raise Exception("Failed to find optimal solution")
            
        # Get execution time
        execution_time = time.time() - start_time
        
        # Log results
        print(f"\nResults:")
        print(f"Makespan: {pulp.value(C_max)}")
        print(f"Execution time: {execution_time:.2f}s")
        print(f"Status: {pulp.LpStatus[prob.status]}")
        print(f"Big M value: {M}")
        
        return pulp.value(C_max)

if __name__ == "__main__":
    import sys
    instance_file = sys.argv[1]
    solver = ILPSolver(instance_file)
    makespan = solver.solve()
