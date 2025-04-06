"""Collate function utilities for dataset batching.

This module provides collate functions for efficient batch processing.
"""

import torch
from typing import Dict, List, Any, Optional, Callable

def collate_fn_lambda(
    batch: List[Dict[str, Any]], 
    sequence_length: int,
    num_pred_steps: int,
) -> Dict[str, torch.Tensor]:
    """Collate a batch of data.
    
    Args:
        batch: List of dictionaries containing trajectory data
        sequence_length: Length of sequences to return
        
    Returns:
        Dictionary containing collated batch data
    """
    # Extract data from batch
    actions = torch.stack([item["action"] for item in batch])
    proprio = torch.stack([item["proprio"] for item in batch])
    
    # Reshape to (batch_size, sequence_length, ...)
    actions = actions.view(-1, sequence_length, num_pred_steps, actions.shape[-1])
    proprio = proprio.view(-1, sequence_length, proprio.shape[-1])
    
    # Create output dictionary
    output = {
        "action": actions,
        "proprio": proprio,
    }
    
    # Add optional fields if present
    if "observation" in batch[0]:
        observation = torch.stack([item["observation"] for item in batch])
        observation = observation.view(-1, sequence_length, *observation.shape[1:])
        output["observation"] = observation
        
    if "text" in batch[0]:
        text = [item["text"] for item in batch]
        output["text"] = text
        
    if "eos" in batch[0]:
        eos = torch.stack([item["eos"] for item in batch])
        eos = eos.view(-1, sequence_length)
        output["eos"] = eos
        
    return output

def collate_fn_lambda_batched(
    batch: List[Dict[str, Any]], 
    sequence_length: int,
    num_pred_steps: int,
) -> Dict[str, torch.Tensor]:
    """Collate a batch of data with batched processing.
    
    Args:
        batch: List of dictionaries containing trajectory data
        sequence_length: Length of sequences to return
        
    Returns:
        Dictionary containing collated batch data
    """
    # Extract data from batch
    actions = torch.stack([item["action"] for item in batch])
    proprio = torch.stack([item["proprio"] for item in batch])
    
    # Reshape to (batch_size, sequence_length, ...)
    actions = actions.view(-1, sequence_length, num_pred_steps, actions.shape[-1])
    proprio = proprio.view(-1, sequence_length, proprio.shape[-1])
    
    # Create output dictionary
    output = {
        "action": actions,
        "proprio": proprio,
    }
    
    # Add optional fields if present
    if "observation" in batch[0]:
        observation = torch.stack([item["observation"] for item in batch])
        observation = observation.view(-1, sequence_length, *observation.shape[1:])
        output["observation"] = observation
        
    if "text" in batch[0]:
        text = [item["text"] for item in batch]
        output["text"] = text
        
    if "eos" in batch[0]:
        eos = torch.stack([item["eos"] for item in batch])
        eos = eos.view(-1, sequence_length)
        output["eos"] = eos
        
    return output

class CollateFunction:
    """Collate function for efficient batch processing.
    
    Args:
        sequence_length: Length of sequences to return
        batched: Whether to use batched processing
    """
    def __init__(
        self, 
        sequence_length: int, 
        num_pred_steps: int,
        batched: bool = True
    ):
        self.sequence_length = sequence_length
        self.num_pred_steps = num_pred_steps
        self.batched = batched
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of data.
        
        Args:
            batch: List of dictionaries containing trajectory data
            
        Returns:
            Dictionary containing collated batch data
        """
        if self.batched:
            return collate_fn_lambda_batched(
                batch, 
                sequence_length=self.sequence_length,
                num_pred_steps=self.num_pred_steps,
            )
        else:
            return collate_fn_lambda(
                batch, 
                sequence_length=self.sequence_length,
                num_pred_steps=self.num_pred_steps,
            )