"""Data processing utilities for robot trajectory data.

This module provides functions for processing and transforming robot trajectory data,
including action conversion, multi-step prediction, and data scaling.
"""

import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Literal, Any, Optional, Union
from ..transforms.rotation import rot_6d_to_rot_mat, rot_mat_to_rot_6d
from scipy.spatial.transform import Rotation
from transformers import AutoProcessor

def combine_dicts(dicts: List[Dict[str, Any]], keys: List[str]) -> Dict[str, Any]:
    """Combines a list of dictionaries into a single dictionary based on specified keys.
    
    Args:
        dicts: List of dictionaries to combine
        keys: Keys to include in the combined dictionary
        
    Returns:
        Combined dictionary
    """
    result = {}
    for key in keys:
        result[key] = [d[key] for d in dicts]
    return result

def convert_multi_step(data: torch.Tensor, num_pred_steps: int) -> torch.Tensor:
    """Chunk data for predicting data `num_pred_steps` steps into the future.
    
    The resulting data have shape (batch, data.shape[-2] - (num_pred_steps - 1), num_pred_steps, action_dim)
    For example: chunk_data([a_1, a_2, a_3, a_4, a_5], 3) ->
        [
            [a_1, a_2, a_3],
            [a_2, a_3, a_4],
            [a_3, a_4, a_5],
            [a_4, a_5, a_5],
            [a_5, a_5, a_5],
        ]
    
    Args:
        data: Input tensor of shape (seq_length, action_dim)
        num_pred_steps: Number of steps to predict ahead
        
    Returns:
        Chunked tensor for multi-step prediction
    """
    assert data.ndim == 2, f"Expected data to have shape (seq length, action_dim), but got shape {data.shape}"
    window_size = data.shape[0]
    chunk_window_size = window_size

    curr_step = torch.arange(chunk_window_size, device=data.device)
    action_offset = torch.arange(num_pred_steps, device=data.device)
    chunk_indices = torch.minimum(curr_step[:, None] + action_offset[None, :], torch.tensor(chunk_window_size - 1))
    return data[chunk_indices]

def convert_multi_step_np(data: np.ndarray, num_steps: int) -> np.ndarray:
    """Chunks data for predicting multiple steps into the future.
    
    Args:
        data: Input data array
        num_steps: Number of steps to predict
        
    Returns:
        Tensor with shape (batch_size, num_steps, ...)
    """
    return np.stack([data[i:i+num_steps] for i in range(len(data)-num_steps+1)], axis=0)

def convert_delta_action(action: np.ndarray, proprio: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculates the delta action given the action and proprioception.
    
    Args:
        action: Action array
        proprio: Optional proprioception array
        
    Returns:
        Delta action array
    """
    if proprio is None:
        return np.diff(action, axis=0)
    else:
        return action - proprio

def convert_abs_action(delta_action: np.ndarray, proprio: np.ndarray) -> np.ndarray:
    """Computes the absolute action from the delta action and current proprioception.
    
    Args:
        delta_action: Delta action array
        proprio: Proprioception array
        
    Returns:
        Absolute action array
    """
    return delta_action + proprio

def find_increasing_subsequences(numbers: List[int]) -> List[List[int]]:
    """Identifies all increasing subsequences in a list of integers.
    
    Args:
        numbers: List of integers
        
    Returns:
        List of lists containing start and end values of increasing subsequences
    """
    if not numbers:
        return []
        
    result = []
    start = numbers[0]
    prev = numbers[0]
    
    for num in numbers[1:]:
        if num != prev + 1:
            result.append([start, prev])
            start = num
        prev = num
        
    result.append([start, prev])
    return result

def scale_action(action: Union[np.ndarray, torch.Tensor], stats: Dict[str, np.ndarray], key: str) -> Union[np.ndarray, torch.Tensor]:
    """Scales action values using provided statistics.
    
    Args:
        action: Action values to scale
        stats: Dictionary containing scaling statistics
        key: Key for accessing statistics
        
    Returns:
        Scaled action values
    """
    if isinstance(action, np.ndarray):
        return (action - stats[key]["mean"]) / stats[key]["std"]
    else:
        mean = torch.from_numpy(stats[key]["mean"]).to(action.device)
        std = torch.from_numpy(stats[key]["std"]).to(action.device)
        return (action - mean) / std

def unscale_action(action: Union[np.ndarray, torch.Tensor], stats: Dict[str, np.ndarray], key: str) -> Union[np.ndarray, torch.Tensor]:
    """Reverses the scaling of action values using the same statistics.
    
    Args:
        action: Scaled action values
        stats: Dictionary containing scaling statistics
        key: Key for accessing statistics
        
    Returns:
        Unscaled action values
    """
    if isinstance(action, np.ndarray):
        return action * stats[key]["std"] + stats[key]["mean"]
    else:
        mean = torch.from_numpy(stats[key]["mean"]).to(action.device)
        std = torch.from_numpy(stats[key]["std"]).to(action.device)
        return action * std + mean

def load_json(file_path: str) -> Dict:
    """Loads a JSON file and returns its contents as a dictionary.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the JSON data
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def normalize(data: Union[np.ndarray, torch.Tensor], mean: np.ndarray, std: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
    """Normalizes data using mean and standard deviation.
    
    Args:
        data: Data to normalize
        mean: Mean values
        std: Standard deviation values
        
    Returns:
        Normalized data
    """
    if isinstance(data, np.ndarray):
        return (data - mean) / std
    else:
        mean = torch.from_numpy(mean).to(data.device)
        std = torch.from_numpy(std).to(data.device)
        return (data - mean) / std

def discretize_action(actions: torch.Tensor, tokenizer: AutoProcessor, max_token_length: int) -> Dict[str, torch.Tensor]:
    """Discretizes continuous action values into tokens.
    
    Args:
        actions: Continuous action values
        tokenizer: Tokenizer for processing actions
        max_token_length: Maximum length of tokenized sequence
        
    Returns:
        Dictionary containing tokenized actions
    """
    # Flatten actions for tokenization
    batch_size = actions.shape[0]
    actions_flat = actions.reshape(-1, actions.shape[-1])
    
    # Convert to string representation
    action_strings = [f"{a.tolist()}" for a in actions_flat]
    
    # Tokenize actions
    tokenized = tokenizer(
        action_strings,
        padding="max_length",
        max_length=max_token_length,
        truncation=True,
        return_tensors="pt"
    )
    
    # Reshape back to batch dimensions
    for key in tokenized:
        if isinstance(tokenized[key], torch.Tensor):
            tokenized[key] = tokenized[key].view(batch_size, -1, tokenized[key].shape[-1])
            
    return tokenized 