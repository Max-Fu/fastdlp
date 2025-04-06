"""
TODO: 
1. weight different tasks 
2. calculation action statistics 
3. proprio data aug
4. repeat trajectory

the dataset has the following structure

- root "/home/mfu/dataset/new_icrt"
    - Primitive 1 
        - Task 1
            - Trajectory 1
                - images 
                    - image key 1
                        - 00000.jpg
                        - 00001.jpg
                        - ...
                        - n.jpg
                    - image key 2
                        - ...
                - proprio.zarr 
                - joints.zarr
                - action.zarr
                - description.txt
                - ... 
"""

from PIL import Image
from torch.utils.data import Sampler, Dataset
from typing import List, Iterator
from typing import Union, Optional
import numpy as np
import os
import random
import torch 
import torchvision.transforms as transforms
import zarr 
import json

from fastdlp.dataset import DataConfig
from fastdlp.utils.transforms import euler_to_rot_6d 
from fastdlp.utils.data import convert_delta_action

class VideoSampler(Sampler):
    """
    Sampler for the sequence dataset with epoch-based shuffling.
    For each instance, it will be of size batch_size * sequence_length,
    then the collate_fn will reshape it to batch_size, sequence_length, ...
    """
    def __init__(
        self, 
        index_to_seq : list,
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
        return self.num_chunks
    
    def set_epoch(self, epoch):
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
    
    def __iter__(self) -> Iterator[int]:
        """
        Generate indices for each epoch with proper shuffling.
        Uses both epoch and seed for reproducible but different shuffling per epoch.
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

def collate_fn_lambda(batch: List[dict], sequence_length: int):
    """
    Collate function for the sequence dataset that converts individual timesteps into batched sequences.
    
    Args:
        batch: List of dictionaries containing individual timesteps
        sequence_length: Length of sequences to create
        
    Returns:
        Dictionary containing batched sequences with prompt masks
    """
    # Get the keys from the first item in batch
    keys = batch[0].keys()
    collated = {}
    
    # Collate each key into proper batch/sequence format
    for key in keys:
        data = [sample[key] for sample in batch]
        # try:
        data = torch.stack(data)
        # except:
        #     import pdb; pdb.set_trace()
        #     print(key)
        #     print([i.shape for i in data])
        # Reshape into (batch_size, sequence_length, ...)
        data = data.view(-1, sequence_length, *data.shape[1:])
        collated[key] = data

    # Get eos tokens for mask generation 
    eos = collated["eos"]  # Shape: (batch_size, sequence_length)
    batch_size = eos.shape[0]
    
    # Initialize masks for each sequence in the batch
    prompt_masks = []
    weight_masks = []
    
    # Generate masks for each sequence in the batch
    for b in range(batch_size):
        # Find indices where EOS is True in this sequence
        eos_indices = torch.where(eos[b])[0]
        
        # Initialize masks for this sequence
        prompt_mask = torch.zeros(sequence_length)
        weight_mask = torch.zeros(sequence_length)
        
        if len(eos_indices) > 0:
            # Get the first EOS index
            eos_idx = eos_indices[0]
            
            # Set prompt mask: Everything before EOS (except first token) should be 1
            prompt_mask[1:eos_idx + 1] = 1
            
            # Set weight mask: Use first few steps for weighting
            # You might want to adjust this based on your needs
            weight_steps = min(5, eos_idx)  # Example: weight first 5 steps or up to EOS
            weight_mask[1:weight_steps + 1] = 1
        else:
            # If no EOS found, use the whole sequence except first token
            prompt_mask[1:] = 1
            weight_mask[1:6] = 1  # Example: weight first 5 steps
        
        prompt_masks.append(prompt_mask)
        weight_masks.append(weight_mask)
    
    # Stack masks into tensors
    collated["prompt_mask"] = torch.stack(prompt_masks)  # (batch_size, sequence_length)
    collated["weight_mask"] = torch.stack(weight_masks)  # (batch_size, sequence_length)
    
    return collated

def collate_fn_lambda_batched(batch: List[dict], sequence_length: int):
    """
    Vectorized collate function for the sequence dataset that converts individual timesteps into batched sequences.
    
    Args:
        batch: List of dictionaries containing individual timesteps
        sequence_length: Length of sequences to create
        
    Returns:
        Dictionary containing batched sequences with prompt masks
    """
    # Get the keys from the first item in batch
    keys = batch[0].keys()
    collated = {}
    
    # Collate each key into proper batch/sequence format
    for key in keys:
        data = [sample[key] for sample in batch]
        data = torch.stack(data)
        # Reshape into (batch_size, sequence_length, ...)
        data = data.view(-1, sequence_length, *data.shape[1:])
        collated[key] = data

    # Get eos tokens for mask generation 
    eos = collated["eos"]  # Shape: (batch_size, sequence_length)
    batch_size = eos.shape[0]
    
    # Create position indices tensor
    positions = torch.arange(sequence_length).unsqueeze(0).expand(batch_size, -1)
    
    # Find first EOS position for each sequence
    eos_indices = torch.argmax(eos.float(), dim=1)
    # Handle cases where no EOS token is found (argmax returns 0)
    no_eos_mask = ~torch.any(eos, dim=1)
    eos_indices[no_eos_mask] = sequence_length - 1
    
    # Expand eos_indices for broadcasting
    eos_indices = eos_indices.unsqueeze(1)
    
    # Generate prompt masks
    # Everything after position 0 and before/at EOS should be 1
    prompt_masks = (positions > 0) & (positions <= eos_indices)
    
    # Generate weight masks
    # Weight the first 5 steps after position 0, or up to EOS if earlier
    weight_steps = torch.minimum(
        torch.full_like(eos_indices, 5),  # First 5 steps
        eos_indices  # Or up to EOS if earlier
    )
    weight_masks = (positions > 0) & (positions <= weight_steps)
    
    # Add masks to collated dictionary
    collated["prompt_mask"] = prompt_masks
    collated["weight_mask"] = weight_masks
    
    return collated

class CollateFunction:
    def __init__(self, sequence_length, batched=True):
        self.sequence_length = sequence_length
        self.batched = batched
        
    def __call__(self, batch):
        if self.batched:
            return collate_fn_lambda_batched(batch, sequence_length=self.sequence_length)
        else:
            collate_fn_lambda(batch, sequence_length=self.sequence_length)

class SequenceDataset(Dataset):

    # set minimum trajectory length
    # we use 30 as the control frequency of the robot is 15 Hz
    minimum_length : int = 30 
    maximum_length : int = 450

    # remove long tail situations 
    min_demos : int = 4

    def __init__(
        self,
        data_config : DataConfig,
        vision_transform : transforms.Compose,
        no_aug_vision_transform : transforms.Compose = None, # this is for wrist camera in particular
        split : str = "train",
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
    ):
        self.dataset_root = data_config.dataset_root
        self.seq_length = data_config.seq_length

        # change prediction to be k steps 
        self.num_pred_steps = data_config.num_pred_steps
        assert self.num_pred_steps >= 1, "Number of prediction steps must be at least 1"
        print("Number of prediction steps: ", self.num_pred_steps)

        # adding vision transforms
        self.vision_transform = vision_transform    
        if no_aug_vision_transform is not None:
            self.no_aug_vision_transform = no_aug_vision_transform
        else:
            print("No augmentation vision transform is not provided, using the same as the vision transform")
            self.no_aug_vision_transform = vision_transform
        
        # decide the split of the dataset
        self.split = split 

        # repeat trajectory parameters
        self.num_repeat_traj = data_config.num_repeat_traj
        self.shuffle_repeat_traj = data_config.shuffle_repeat_traj

        # rebalance tasks or primitives
        # if rebalance primitives is true, we will balance the task at the primitive level
        # otherwise we will balance the tasks at the task level if rebalance tasks is true
        self.rebalance_primitives = data_config.rebalance_primitives
        self.rebalance_tasks = not self.rebalance_primitives and data_config.rebalance_tasks

        # set the seed and rank of the dataset
        self.seed = seed
        self.rank = rank
        self.num_replicas = num_replicas

        # parse the dataset folder to get metadata
        self._parse_folder()

        # flatten the dataset
        self._flatten_dataset()

        # save meta data 
        self._save_metadata()


    def get_sampler(self, batch_size) -> VideoSampler:
        """
        Create a sampler for the dataset.
        """
        return VideoSampler(
            self.index_to_seq, 
            batch_size, 
            num_replicas=self.num_replicas,
            rank=self.rank,
            seed=self.seed,
        )

    def _save_metadata(self):
        """
        save the metadata to the dataset folder
        """
        metadata_path = os.path.join(self.dataset_root, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f)

    def _parse_folder(self):
        """
        Parse dataset folder to extract meta data
        the meta data has the following format 
        {
            "primitives":
            {
                "primitive 1" : {
                    "task 1" : 
                        {
                        traj_1 : sequence length
                        ...
                        }
                    "task 2" : ... 
                }
            }
            "statistics": {
                TODO: "store action statistics here"
            }
            "camera keys" : []
        }
        """
        metadata = {
            "primitives" : {}, 
            "statistics" : {}, 
        }
        total_seq_length = 0
        primitives = [p for p in os.listdir(self.dataset_root) if os.path.isdir(os.path.join(self.dataset_root, p))]
        for p in primitives:
            primitive_dir = os.path.join(self.dataset_root, p)
            metadata["primitives"][p] = {}
            tasks = os.listdir(primitive_dir)
            # construct all task for the primitive
            for t in tasks:
                task_dir = os.path.join(primitive_dir, t)
                trajectories = os.listdir(task_dir)
                traj_stats = {}
                # list all the trajectories in the task
                for traj in trajectories:
                    traj_dir = os.path.join(task_dir, traj)
                    if "camera_keys" not in metadata:
                        metadata["camera_keys"] = os.listdir(os.path.join(traj_dir, "images"))
                    proprio_data = zarr.load(
                        os.path.join(traj_dir, "proprio.zarr")
                    )
                    traj_length = len(proprio_data)
                    traj_stats[traj] = traj_length
                    total_seq_length += traj_length
                metadata["primitives"][p][t] = traj_stats
        metadata["total_seq_length"] = total_seq_length
        self.metadata = metadata
    
    def total_seq_length(self):
        if "total_seq_length" not in self.metadata:
            raise ValueError("Total sequence length not found in metadata, needs to run _parse_folder() first")
        return self.metadata["total_seq_length"]

    def _get_task_trajectories(self, primitive: str, task: str) -> list:
        """Get all trajectories for a given task with their repeats.

        Args:
            primitive: Name of the primitive
            task: Name of the task

        Returns:
            List of dictionaries containing trajectory information
        """
        rng = np.random.default_rng(self.seed)
        task_trajectories = []
        
        for traj in self.metadata["primitives"][primitive][task]:
            traj_length = self.metadata["primitives"][primitive][task][traj]
            traj_dir = os.path.join(self.dataset_root, primitive, task, traj)
            num_repeats = rng.choice(np.arange(self.num_repeat_traj)) + 1
            
            task_trajectories.append({
                'dir': traj_dir,
                'length': traj_length,
                'repeats': num_repeats
            })
        
        return task_trajectories

    def _process_trajectory(self, trajectory: dict, idx_to_item: list, idx_to_eos: list) -> tuple:
        """Process a single trajectory, adding its timesteps to the dataset.

        Args:
            trajectory: Dictionary containing trajectory information
            idx_to_item: List mapping index to dataset items
            idx_to_eos: List of end-of-sequence flags

        Returns:
            tuple: (range_start, range_end) for this trajectory
        """
        range_start = len(idx_to_item)
        
        for i in range(trajectory['length']):
            idx_to_item.append((trajectory['dir'], i))
            idx_to_eos.append(i == trajectory['length'] - 1)
        
        return (range_start, len(idx_to_item))

    def _shuffle_consecutive_trajectories(self, traj_ranges: list, idx_to_item: list, idx_to_eos: list):
        """Shuffle consecutive repeated trajectories.

        Args:
            traj_ranges: List of (start, end) ranges for trajectories
            idx_to_item: List mapping index to dataset items
            idx_to_eos: List of end-of-sequence flags
        """
        for i in range(len(traj_ranges) - 1):
            if np.random.random() < 0.5:
                range1, range2 = traj_ranges[i], traj_ranges[i + 1]
                
                # Extract and swap items and eos flags
                items1 = idx_to_item[range1[0]:range1[1]]
                items2 = idx_to_item[range2[0]:range2[1]]
                eos1 = idx_to_eos[range1[0]:range1[1]]
                eos2 = idx_to_eos[range2[0]:range2[1]]
                
                idx_to_item[range1[0]:range1[1]] = items2
                idx_to_item[range2[0]:range2[1]] = items1
                idx_to_eos[range1[0]:range1[1]] = eos2
                idx_to_eos[range2[0]:range2[1]] = eos1

    def _generate_sequence_indices(self, task_start_end: list) -> list:
        """Generate sequence indices based on task boundaries.

        Args:
            task_start_end: List of (start, end, primitive_name, task_name) indices for each task

        Returns:
            List of sequence indices, each of length seq_length, containing
            the starting index of each sequence that should belong to a single task
        """
        index_to_seq = []
        for t_start, t_end, primitive_name, task_name in task_start_end:
            if t_end - t_start < (self.seq_length + self.num_pred_steps):
                t_end = t_start + self.seq_length
            for i in range(t_start, t_end - (self.seq_length + self.num_pred_steps - 1)):
                index_to_seq.append({
                    "indices" : list(range(i, i + self.seq_length)), 
                    "primitive" : primitive_name,
                    "task" : task_name
                })
        return index_to_seq

    def _flatten_dataset(self):
        """Flatten the dataset and create index mappings."""
        idx_to_item = []
        idx_to_eos = []
        task_start_end = []
        primitive_counts = {}
        task_counts = {}
        task_start, task_end = 0, 1

        # First pass: count samples
        for primitive in self.metadata["primitives"]:
            primitive_counts[primitive] = 0
            for task in self.metadata["primitives"][primitive]:
                task_key = f"{primitive}/{task}"
                task_counts[task_key] = 0
                task_trajectories = self._get_task_trajectories(primitive, task)
                for traj in task_trajectories:
                    primitive_counts[primitive] += traj['length']
                    task_counts[task_key] += traj['length']

        # Calculate minimum samples and repeat factors
        min_samples_primitive = max(primitive_counts.values()) // 2
        min_samples_task = max(task_counts.values()) // 2
        
        # Calculate minimum samples and repeat factors
        if self.rebalance_primitives:
            min_samples_primitive = max(primitive_counts.values()) // 2
            primitive_repeats = {
                p: max(1, min_samples_primitive // count)
                for p, count in primitive_counts.items()
            }
            task_repeats = {t: 1 for t in task_counts.keys()}  # No additional task balancing
        elif self.rebalance_tasks:
            min_samples_task = max(task_counts.values()) // 2
            task_repeats = {
                t: max(1, min_samples_task // count)
                for t, count in task_counts.items()
            }
            primitive_repeats = {p: 1 for p in primitive_counts.keys()}
        else:
            primitive_repeats = {p: 1 for p in primitive_counts.keys()}
            task_repeats = {t: 1 for t in task_counts.keys()}

        # Second pass: build dataset with repeats
        # TODO check if shuffling of trajectories within the same task 
        # is taken care of
        for primitive in self.metadata["primitives"]:
            for task in self.metadata["primitives"][primitive]:
                task_key = f"{primitive}/{task}"
                task_trajectories = self._get_task_trajectories(primitive, task)
                repeats = primitive_repeats[primitive] if self.rebalance_primitives else task_repeats[task_key]
                
                # Process trajectories and track ranges
                traj_ranges = []
                for traj in task_trajectories:
                    for _ in range(repeats * traj['repeats']):
                        traj_range = self._process_trajectory(traj, idx_to_item, idx_to_eos)
                        traj_ranges.append(traj_range)
                        task_end = len(idx_to_item)
                
                if self.shuffle_repeat_traj:
                    self._shuffle_consecutive_trajectories(traj_ranges, idx_to_item, idx_to_eos)
                
                task_start_end.append((task_start, task_end, primitive, task_key))
                task_start = task_end

        self.idx_to_item = idx_to_item
        self.idx_to_eos = idx_to_eos
        self.index_to_seq = self._generate_sequence_indices(task_start_end)

    def load_image(self, idx):
        """
        load images from the idx with the correct camera keys 
        """
        parent_dir, step_idx = self.idx_to_item[idx]
        image_dir = os.path.join(parent_dir, "images")
        images = []
        for key in self.metadata["camera_keys"]:
            image_path = os.path.join(image_dir, key, f"{step_idx:05d}.jpg")
            image = Image.open(image_path)
            if "wrist" in image_path:
                # use no_aug_vision_transform for wrist camera
                image = self.no_aug_vision_transform(image)
            else:
                image = self.vision_transform(image)
            images.append(image)
        return torch.stack(images) # shape: (num_cameras, C, H, W)

    def _load_proprio(self, idx):
        parent_dir, step_idx = self.idx_to_item[idx]
        proprio_path = os.path.join(parent_dir, "proprio.zarr") 
        proprio_data = zarr.load(proprio_path)
        proprio = proprio_data[step_idx]

        # convert euler rotation to 6d
        rot = euler_to_rot_6d(proprio[3:6][None, ...]).squeeze() # euler_to_rot_6d expects (N, 3) input
        proprio = np.concatenate([proprio[:3], rot, proprio[6:]])

        return proprio
    
    def _load_action(self, idx):
        # load idx:idx+num_pred_steps
        parent_dir, step_idx = self.idx_to_item[idx]
        action_path = os.path.join(parent_dir, "action.zarr")
        action_data = zarr.load(action_path)
        action = action_data[step_idx:step_idx+self.num_pred_steps]

        if len(action) < self.num_pred_steps:
            # post padding with the last action
            action = np.concatenate([action, np.repeat(action[-1][None, ...], self.num_pred_steps - len(action), axis=0)], axis=0)

        # convert euler rotation to 6d
        rot = euler_to_rot_6d(action[:, 3:6])
        action = np.concatenate([action[:, :3], rot, action[:, 6:]], axis=-1)
        return action
    
    def load_proprio_action(self, idx):
        proprio = self._load_proprio(idx)
        action = self._load_action(idx)
        # calculate delta actions 
        proprio = proprio[None, None, ...] # (1, 1, action_dim)
        action = action[None, ...] # (1, num_pred_steps, action_dim)
        delta_action = convert_delta_action(action, proprio)
        # proprio dim: (1, action_dim)
        # action dim: (num_pred_steps, action_dim)
        proprio, delta_action = torch.tensor(proprio).squeeze(0), torch.tensor(delta_action).squeeze()
        return proprio, delta_action

    def load_eos(self, idx):
        """
        return True if idx is the end of the sequence
        """
        return torch.tensor(self.idx_to_eos[idx])

    def __len__(self):
        return len(self.index_to_seq)

    def __getitem__(self, index):
        images = self.load_image(index)
        proprio, delta_action = self.load_proprio_action(index)
        eos = self.load_eos(index)
        return {
            "observation" : images.float(), 
            "proprio" : proprio.float(),
            "action" : delta_action.float(),
            "eos" : eos
        }