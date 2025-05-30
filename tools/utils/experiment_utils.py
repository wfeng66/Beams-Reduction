import os
import csv
import datetime
import pandas as pd
from pathlib import Path


class ExperimentLogger:
    def __init__(self, log_file):
        self.log_file = Path(log_file)
        self.headers = [
            'date_time',
            'epochs',
            'method',
            'dataset',
            'pretrained',
            'training_dataset_version',
            'validation_dataset_version',
            'fold_idx',
            'Car_3d_easy',
            'Car_3d_moderate',
            'Car_3d_hard',
            'Pedestrian_3d_easy',
            'Pedestrian_3d_moderate',
            'Pedestrian_3d_hard',
            'Cyclist_3d_easy',
            'Cyclist_3d_moderate',
            'Cyclist_3d_hard',
            'mean_performance'
        ]
        
        # Create file with headers if it doesn't exist
        if not self.log_file.exists():
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
    
    def log_experiment(self, args, cfg, metrics):
        """Log experiment results to CSV file"""
        row = {
            'date_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epochs': args.epochs,
            'method': cfg.MODEL.NAME,
            'dataset': cfg.DATA_CONFIG.DATASET,
            'pretrained': args.pretrained_model is not None,
            'training_dataset_version': args.dataset_version,
            'validation_dataset_version': args.dataset_version,
            'fold_idx': args.fold_idx,
        }
        
        # Add metrics
        for key, value in metrics.items():
            if key in self.headers:
                row[key] = value
        
        # Write to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f)
            writer.writerow(row)

