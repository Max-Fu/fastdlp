import torch

class _RepeatSampler:
    """
    A sampler wrapper that repeats and tracks epochs.
    
    Args:
        sampler (Sampler): Base sampler to repeat
    """
    def __init__(self, sampler):
        self.sampler = sampler
        self.epoch = 0

    def __iter__(self):
        while True:
            if hasattr(self.sampler, 'set_epoch'):
                self.sampler.set_epoch(self.epoch)
            yield from iter(self.sampler)
            self.epoch += 1

    def __len__(self):
        return len(self.sampler)

class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """
    based on # from timm.data.loader import MultiEpochsDataLoader
    DataLoader that handles multiple epochs with proper epoch tracking for samplers.
    Supports samplers with epoch-based shuffling through the set_epoch method.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        
        # Handle both regular sampler and batch sampler cases
        if self.batch_sampler is None:
            self.sampler = _RepeatSampler(self.sampler)
        else:
            self.batch_sampler = _RepeatSampler(self.batch_sampler)
            
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        if self.batch_sampler is not None:
            return len(self.batch_sampler.sampler)
        return len(self.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        
        # Handle both regular sampler and batch sampler cases
        if self.batch_sampler is None:
            self.sampler = _RepeatSampler(self.sampler)
        else:
            self.batch_sampler = _RepeatSampler(self.batch_sampler)
            
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        if self.batch_sampler is not None:
            return len(self.batch_sampler.sampler)
        return len(self.sampler)

    def __iter__(self):
        self.iterator = super().__iter__()
        return self 

    def __next__(self):
        try: 
            return next(self.iterator)
        except StopIteration:
            self.iterator = super().__iter__()
            return next(self.iterator)