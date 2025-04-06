"""Sampler utilities for dataset batching.

This module provides samplers for efficient data loading and batching.
"""

import torch
from torch.utils.data import Sampler
from typing import List, Iterator, Dict, Any

class VideoSampler(Sampler):
    """Sampler for sequence dataset with epoch-based shuffling.
    
    For each instance, it will be of size batch_size * sequence_length,
    then the collate_fn will reshape it to batch_size, sequence_length, ...
    
    Args:
        index_to_seq: List of dictionaries containing sequence indices
        batch_size: Size of batches to return
        num_replicas: Number of distributed replicas
        rank: Rank of current process
        seed: Random seed for reproducibility
    """
    def __init__(
        self, 
        index_to_seq: List[Dict[str, Any]],
        batch_size: int, 
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42, 
    ) -> None:
        # Make length divisible by batch_size * num_replicas
        total_size = len(index_to_seq)
        total_size = total_size - (total_size % (batch_size * num_replicas))
        self.num_samples = total_size // num_replicas
        
        self.original_index_to_seq = index_to_seq[:total_size]
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.set_epoch(0)

    def __len__(self) -> int:
        """Return the number of batches."""
        return self.num_chunks
    
    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler.
        
        Args:
            epoch: Current epoch number
        """
        self.epoch = epoch
        self.index_to_seq = self.original_index_to_seq.copy()
        
        # Shuffle consistently across ranks
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # Use torch's RNG for consistent cross-rank shuffling
        rand_indices = torch.randperm(len(self.index_to_seq), generator=g)
        self.index_to_seq = [self.index_to_seq[i] for i in rand_indices]

        self.num_chunks = self.num_samples // self.batch_size
        self.start_idx = self.rank * self.num_samples
        self.end_idx = self.start_idx + self.num_samples
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate indices for each epoch with proper shuffling.
        
        Uses both epoch and seed for reproducible but different shuffling per epoch.
        
        Returns:
            Iterator yielding lists of indices for each batch
        """
        # Create a generator with seed that combines epoch and base seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # Generate indices for this process
        indices = torch.arange(start=self.start_idx, end=self.end_idx)
        
        # Shuffle indices while maintaining sequence grouping
        num_batches = len(indices) // self.batch_size
        indices = indices.view(num_batches, self.batch_size)
        
        # Shuffle the batches
        perm = torch.randperm(num_batches, generator=g)
        indices = indices[perm].view(-1)
        
        curr_batch = []
        for idx, seq_idx in enumerate(indices):
            seq = self.index_to_seq[seq_idx]["indices"]
            curr_batch.extend(seq)
            if idx % self.batch_size == self.batch_size - 1:
                yield curr_batch
                curr_batch = [] 