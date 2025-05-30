import numpy as np
from sklearn.model_selection import KFold

def create_fold_split(dataset_size, n_folds=5, fold_idx=0, shuffle=True, random_state=42):
    """Create train/val split for k-fold cross validation"""
    kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    
    # Get all splits
    splits = list(kf.split(range(dataset_size)))
    
    # Get current fold's split
    train_idx, val_idx = splits[fold_idx]
    
    return train_idx, val_idx

def split_dataset(dataset, train_idx, val_idx):
    """Split dataset into train and validation sets"""
    # This function will need to be implemented based on your dataset structure
    # You might need to modify this based on how your dataset class works
    train_dataset = dataset.select_samples(train_idx)
    val_dataset = dataset.select_samples(val_idx)
    
    return train_dataset, val_dataset