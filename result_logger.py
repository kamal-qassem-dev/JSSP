import os
import json
from datetime import datetime
import pandas as pd

class ResultLogger:
    def __init__(self, algorithm_name):
        self.algorithm_name = algorithm_name
        self.results_dir = os.path.join('results', algorithm_name)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create or load results DataFrame
        self.results_file = os.path.join(self.results_dir, 'results.csv')
        if os.path.exists(self.results_file):
            self.results_df = pd.read_csv(self.results_file)
        else:
            self.results_df = pd.DataFrame(columns=[
                'timestamp', 'instance_name', 'makespan', 'execution_time',
                'n_jobs', 'n_machines', 'parameters'
            ])
    
    def log_result(self, instance_file, makespan, execution_time, parameters):
        """Log a single result."""
        # Extract instance name from file path
        instance_name = os.path.splitext(os.path.basename(instance_file))[0]
        
        # Get instance dimensions
        with open(instance_file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            n_jobs, n_machines = map(int, lines[1].split())
        
        # Create result entry
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'instance_name': instance_name,
            'makespan': makespan,
            'execution_time': execution_time,
            'n_jobs': n_jobs,
            'n_machines': n_machines,
            'parameters': json.dumps(parameters)
        }
        
        # Append to DataFrame
        self.results_df = pd.concat([self.results_df, pd.DataFrame([result])], ignore_index=True)
        
        # Save DataFrame
        self.results_df.to_csv(self.results_file, index=False)
        
        # Save detailed result
        detail_file = os.path.join(
            self.results_dir,
            f'{instance_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        with open(detail_file, 'w') as f:
            json.dump({**result, 'algorithm': self.algorithm_name}, f, indent=2)
        
        return result
