"""
python test/test_image_dataset.py --dataset-cfg.dataset-root /home/mfu/dataset/new_icrt
"""
import os
import torch
import tyro 
import tqdm
import time
import mediapy as media
import matplotlib.pyplot as plt
import numpy as np 

from fastdlp import SequenceDataset, CollateFunction, MultiEpochsDataLoader, DataConfig
from fastdlp.utils.vision.transforms import aug_vision_transform, default_vision_transform

def undo_vision_transform(obs : torch.Tensor, mean : tuple, std : tuple):
    """
    Undo the vision transform applied to the observations.
    torch tensor has shape T, num_cam, 3, H, W
    return np.ndarray with shape T, num_cam * H, W, 3 at np.uint8
    """
    # undo normalization
    mean, std = torch.tensor(mean), torch.tensor(std)
    obs = obs.permute(0, 1, 3, 4, 2)
    obs = obs * std + mean
    obs = obs.numpy()
    obs = np.clip(obs * 255, 0, 255).astype(np.uint8)
    obs = np.concatenate([obs[:, i] for i in range(obs.shape[1])], axis=1)
    return obs

def main(args : DataConfig):
    batch_size = 2
    out_dir = "test_outputs/test_dataset_output"
    os.makedirs(out_dir, exist_ok=True)

    # making train val split
    dataset_train = SequenceDataset(
        dataset_config=args,
        vision_transform=aug_vision_transform(),
        no_aug_vision_transform=default_vision_transform(),
        split="train",
    )

    # dataloader 
    sampler = dataset_train.get_sampler(batch_size=batch_size)
    collate_fn = CollateFunction(
        sequence_length=args.seq_length,
        num_pred_steps=args.num_pred_steps,
    )
    dataloader_train = MultiEpochsDataLoader(
        dataset_train,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    dataset_total_seq_length = dataset_train.total_seq_length()
    if dataset_total_seq_length < args.seq_length:
        print("Warning: dataset sequence length is less than the model sequence length")
        dataset_train.update_seq_length(dataset_total_seq_length)
        dataset_train.shuffle_dataset()
        print("Updated sequence length: ", dataset_train.seq_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    retrieval_time = []
    for i, data in enumerate(tqdm.tqdm(dataloader_train)):
        if i != 0:
            retrieval_time.append(time.time() - start_time)

        proprio = data['proprio'].to(device, non_blocking=True) # (only plotting the first proprio step) T, K 
        action = data['action'].to(device, non_blocking=True) # (only plotting the first action step) T, M

        # plot observation
        obs = data['observation'].to(device, non_blocking=True) # T, num_cam, 3, H, W
        # obs = undo_vision_transform(obs, mean, std)
        # media.write_video(f"{out_dir}/{i}_obs.mp4", obs, fps=10)

        # # plot proprio and action 
        # proprio = data['proprio'].numpy()[0] # (only plotting the first proprio step) T, K 
        # action = data['action'].numpy()[0, :, 0] # (only plotting the first action step) T, M

        # # plot proprio with each dimension as subplot
        # T, K = proprio.shape
        # fig, axs = plt.subplots(K, 1, figsize=(10, 2 * K))
        # for k in range(K):
        #     axs[k].plot(proprio[:, k])
        #     axs[k].set_title(f"proprio_{k}")
        # plt.tight_layout()
        # plt.savefig(f"{out_dir}/{i}_proprio.png")   
        # plt.clf()

        # # plot action with each dimension as subplot
        # T, M = action.shape
        # fig, axs = plt.subplots(M, 1, figsize=(10, 2 * M))
        # for m in range(M):
        #     axs[m].plot(action[:, m])
        #     axs[m].set_title(f"action_{m}")
        # plt.tight_layout()
        # plt.savefig(f"{out_dir}/{i}_action.png")
        # plt.clf()
        start_time = time.time()
        if i == 100:
            break
    
    print(f"average retrieval time: {np.mean(retrieval_time):.4f} seconds with std {np.std(retrieval_time):.4f}")

if __name__ == "__main__":
    args = tyro.cli(DataConfig)
    main(args)