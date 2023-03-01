import os
import json

import math
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple

Sample = namedtuple("Sample", ["prompt", "image_path"])


class ImageDataset(Dataset):
    def __init__(self, dir_tmp, dir_name):
        self.dir_tmp = dir_tmp
        self.dir_name = dir_name
        self.full_path = os.path.join(self.dir_tmp, self.dir_name)
        self.images, self.prompts = self._get_items()

    def __getitem__(self, idx):
        chosen_image = self.images[idx]
        image_path = os.path.join(self.full_path, chosen_image)
        raw_image = np.asarray(Image.open(image_path).convert('RGB'))
        prompt = self.prompts[chosen_image]
        chosen_image = torch.from_numpy(raw_image)
        return chosen_image, prompt

    def _get_items(self):
        path_to_json = os.path.join(self.full_path, self.dir_name + '.json')
        with open(path_to_json) as f:
            dics = json.load(f)
            images = list(dics.keys())
            prompts = {i: dics[i]['p'] for i in images}
            return images, prompts

    def __len__(self):
        return len(self.images)


class ImageDataset_csv(Dataset):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data_pair = self._get_items()

    def __getitem__(self, index):
        chosen = self.data_pair[index]
        prompt = chosen.prompt
        image_path = chosen.image_path
        raw_image = Image.open(image_path).convert('RGB')
        return raw_image, prompt

    def _get_items(self):
        path_wrong = '/kaggle/input/gustavosta-stable-diffusion-prompts-SD2/'
        path_true = '/remote-home/bxy/data/Image2Text_6G/'
        samples = []
        ImgTxt = pd.read_csv(self.csv_path)
        ImgTxt = ImgTxt.replace(to_replace=path_wrong, value=path_true, regex=True)
        for i in range(len(ImgTxt)):
            sample = Sample(prompt=ImgTxt['Prompt'][i], image_path=ImgTxt['image_path'][i])
            samples.append(sample)
        return samples

    def __len__(self):
        return len(self.data_pair)

class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples

