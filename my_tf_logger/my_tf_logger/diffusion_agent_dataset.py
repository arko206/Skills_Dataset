
#@markdown ### **Dataset**
#@markdown
#@markdown Defines `PushTStateDataset` and helper functions
#@markdown
#@markdown The dataset class
#@markdown - Load data (obs, action) from a zarr storage
#@markdown - Normalizes each dimension of obs and action to [-1,1]
#@markdown - Returns
#@markdown  - All possible segments with length `pred_horizon`
#@markdown  - Pads the beginning and the end of each episode with repetition
#@markdown  - key `obs`: shape (obs_horizon, obs_dim)
#@markdown  - key `action`: shape (pred_horizon, action_dim)
import numpy as np
import torch
import os
import zarr
from zarr import DirectoryStore

def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

# dataset
class PushTStateDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path,
                 pred_horizon, obs_horizon, action_horizon):

        # read from zarr dataset
    
       
        dataset_path = '/home/arka/Desktop/Go2_movement_collection/Front_Jump/Corrected_Jump_Diff_data/Corrected_Hundred_Jump_Go2.zarr'
        store = zarr.DirectoryStore(dataset_path)
        dataset_root = zarr.open_group(store=store, mode='r')
        # All demonstration episodes are concatinated in the first dimension N
        train_data = {
            # (N, action_dim)
            'action': dataset_root['data']['action'][:],
            # (N, obs_dim)
            'obs': dataset_root['data']['observation'][:]
        }
        # Marks one-past the last index for each episode
        episode_ends = dataset_root['meta']['timestep'][:]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        # all possible segments of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['obs'] = nsample['obs'][:self.obs_horizon,:]
        return nsample
    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default='/home/arka/Desktop/Go2_movement_collection/Front_Jump/Corrected_Jump_Diff_data/Corrected_Hundred_Jump_Go2.zarr')
    parser.add_argument('--pred_horizon', type=int, default=16)
    parser.add_argument('--obs_horizon', type=int, default=2)
    parser.add_argument('--action_horizon', type=int, default=8)
    args = parser.parse_args()



    idx = 0



    
    # if you kept hardcoded paths above, replace them with args.dataset_path
    # and args.* for horizons before constructing PushTStateDataset
    dataset = PushTStateDataset(
        dataset_path=args.dataset_path,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon
    )

    stats = dataset.stats

    # … your verification / dataloader code here …
    batch = next(iter(torch.utils.data.DataLoader(
        dataset, batch_size=256, num_workers=1, shuffle=True,
        pin_memory=True, persistent_workers=True
    )))
    print("batch['obs'].shape:", batch['obs'].shape)
    print("batch['action'].shape:", batch['action'].shape)

    buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = dataset.indices[idx]

    print("The buffer start index is: ", buffer_start_idx)
    print("The buffer end index is: ", buffer_end_idx)
    print("The sample start index is: ", sample_start_idx)
    print("The sample end index is: ", sample_end_idx)
    print("The length of the dataset is: ", len(dataset.indices[idx]))

    nsample_at_first_index = dataset.__getitem__(idx)

    print("The first sample of the dataset is: ", nsample_at_first_index)

    obs_norm = nsample_at_first_index['obs']

    print("The normalized observation at the first index is: ", obs_norm)
    print("The shape of the normalized observation at the first index is: ", obs_norm.shape)

    action_norm = nsample_at_first_index['action']
    print("The normalized action at the first index is: ", action_norm)
    print("The shape of the normalized action at the first index is: ", action_norm.shape)

    ###unormalizing the data
    obs = unnormalize_data(obs_norm, stats['obs'])
    print("The unnormalized observation at the first index is: ", obs)
    print("The shape of the unnormalized observation at the first index is: ", obs.shape)

    action = unnormalize_data(action_norm, stats['action'])
    print("The unnormalized action at the first index is: ", action)
    print("The shape of the unnormalized action at the first index is: ", action.shape)


 

if __name__ == '__main__':
    main()

