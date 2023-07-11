import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist
import math
from typing import TypeVar, Optional, Iterator
from re import template


# Sampler for multi-domain zero-shot entity linking datasets
T_co = TypeVar('T_co', covariant=True)
class MultiDomainSampler(Sampler[T_co]):

    def __init__(self, dataset: Dataset, batch_size,srcs, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.srcs = {}
        self.worlds = ['american_football', 'doctor_who', 'fallout', 'final_fantasy', 'military', 'pro_wrestling', 'starwars', 'world_of_warcraft']
        for src in self.worlds:
            self.srcs[src] = []
        for i in range(len(srcs)):
            self.srcs[srcs[i]].append(i)

        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.epoch = 0
        self.drop_last = drop_last
        self.num_samples = 0
        self.total_size = 0
        for src in self.worlds:
            tmp = math.ceil(len(self.srcs[src]) / (self.num_replicas*self.batch_size))
            self.num_samples += tmp * self.batch_size
            self.total_size += tmp * (self.num_replicas*self.batch_size)
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = []
            subworld_idx = {}
            for src,src_list in self.srcs.items():
                world_value = {}
                world_value['idx'] = src_list
                world_value['perm_idx'] = torch.randperm(len(src_list), generator=g).tolist()
                world_value['pointer'] = 0
                subworld_idx[src] = world_value
            world_names = list(self.srcs.keys())
            while(len(world_names)>0):
                world_idx = torch.randint(len(world_names), size=(1, ), generator=g).tolist()[0]
                world_name = world_names[world_idx]
                world_value = subworld_idx[world_name]
                start_pointer = world_value['pointer']
                sample_perm_idx = world_value['perm_idx'][start_pointer : start_pointer + self.batch_size* self.num_replicas]
                world_value['pointer'] += self.batch_size* self.num_replicas
                if len(sample_perm_idx) == 0:
                    world_names.remove(world_name)
                    continue
                if len(sample_perm_idx) < self.batch_size* self.num_replicas:
                    world_names.remove(world_name)
                    sample_perm_idx = sample_perm_idx + world_value['perm_idx'][:self.batch_size* self.num_replicas - len(sample_perm_idx)]
                sample_idx = [world_value['idx'][idx] for idx in sample_perm_idx]
                indices.extend(sample_idx)
        else:
            indices = list(range(len(self.dataset)))  # type: ignore
        try:
            assert len(indices) == self.total_size
        except:
            print(len(indices),self.total_size)

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch