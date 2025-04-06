import dataclasses
from typing import Literal, Optional, Tuple, Union
import enum
import pathlib

import tyro

@dataclasses.dataclass
class DataConfig:
    """Configuration for dataset loading and processing.
    
    Args:
        dataset_root: Root path to the dataset
        seq_length: Number of frames in a sequence
        num_repeat_traj: Number of times to repeat each trajectory
        shuffle_repeat_traj: Whether to shuffle repeated trajectories
        rebalance_tasks: Whether to rebalance tasks in the dataset
        rebalance_primitives: Whether to rebalance primitives in the dataset
        seed: Random seed for reproducibility
        scale_action: Path to action statistics file for scaling
        use_delta_action: Whether to use delta actions
        proprio_noise: Variance of noise added to proprioception
        joint_or_pose: Whether to use joint or pose representation
        camera_keys: List of camera keys to use
        subsample_steps: Number of steps to subsample
        data_subsample_ratio: Ratio of data to subsample (0.0 to 1.0)
        train_split: Ratio of data to use for training (0.0 to 1.0)
        balance_data: Whether to balance left and right arm data
        action_statistics: Path to action statistics file
        num_workers: Number of workers for dataloader
    """
    # Dataset paths and basic config
    dataset_root: str
    seq_length: int = 1024
    num_pred_steps: int = 16
    
    # Data augmentation and processing
    num_repeat_traj: int = 1
    shuffle_repeat_traj: bool = True
    rebalance_tasks: bool = False
    rebalance_primitives: bool = False
    seed: int = 42
    scale_action: Optional[str] = None
    use_delta_action: bool = True
    proprio_noise: float = 0.005
    joint_or_pose: Literal["joint", "pose"] = "pose"
    
    # Camera and data loading
    camera_keys: Tuple[str, ...] = ("camera_1", "camera_2")
    subsample_steps: int = 1
    data_subsample_ratio: float = 0.0
    train_split: float = 0.8
    balance_data: bool = False
    action_statistics: Optional[str] = None
    num_workers: int = 12

if __name__ == "__main__":
    args = tyro.cli(DataConfig)
    dict_args = dataclasses.asdict(args)