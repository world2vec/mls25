import types

import os, sys, logging
import json
import resource
os.environ['TOKENIZERS_PARALLELISM'] = 'False'
from tqdm import tqdm
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
from glob import glob
from functools import partial
from itertools import islice
import re
import math
import util
import types
#import albumentations as A
from copy import deepcopy
import torch
import torch.distributed as dist
from torch.utils.data.dataloader import RandomSampler, default_collate
from torch.utils.data.distributed import DistributedSampler
import psutil
from collections import namedtuple
from PIL import Image
from scipy.special import softmax

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class Sampler(torch.utils.data.Sampler):
    def __init__(self, cfg, data_type, ds):
        self.cfg = cfg
        self.data_type = data_type
        self.ds = ds
        self.inds = np.arange(len(self.ds.data))
        ind_num = defaultdict(int)
        for rec in self.ds.data:
            ind_num[rec.ind] += 1
        weights = {k:1/v for k, v in ind_num.items()}
        self.weights = [weights[rec.ind] for rec in self.ds.data]
        self.weights = np.array(self.weights)/np.sum(self.weights)

        assert abs(1-sum(self.weights))<1e-10

    def __len__(self):
        return self.cfg.n_sample_epoch

    def __iter__(self):
        for ind in self.gen_inds():
            yield ind

    def gen_inds(self):
        if self.data_type=='train':
            inds = np.random.choice(self.inds, self.__len__(), p=self.weights)
        else:
            raise NotImplementedError(self.data_type)
        return inds



class DatasetMix():
    def __init__(self, cfg, data_type, data, tokenizer=None, model_config=None):
        self.cfg = cfg
        self.data_type = data_type
        self.tokenizer=tokenizer
        self.model_config = model_config

        with util.timer('preprocess'):
            self.data = self.preprocess_data(data)

    def __len__(self):
        num = len(self.data)
        if self.data_type=='train':
            num *= self.cfg.n_repeat
        return num

    def sample_item(self, index):
        for i in range(10):
            index2 = np.random.randint(self.__len__())
            if index2!=index:
                break
        item = self.getitem(index2)
        return item

    def __getitem__(self, index):
        item = self.getitem(index)
        return item

    def preprocess_data(self, data):
        data = data.to_records(index=False)
        logger.info('num of data %s is %s', self.data_type, len(data))
        return data

    def _getitem(self, index, rec=None):
        pass

    def getitem(self, index, rec=None):
        item = self._getitem(index, rec=rec)
        return item

    def collate(self, batch):
        new_batch = dict()
        batch = default_collate(batch)
        batch.update(new_batch)
        return batch


class IterMixBase(DatasetMix):
    def __len__(self):
        return len(self.data)

    def get_distribute_data(self, data, world_size=None, rank=None):
        rank = dist.get_rank() if rank is None else rank
        world_size = dist.get_world_size() if world_size is None else world_size
        per_rank = int(math.ceil(len(data) / world_size))
        return data[rank * per_rank:(rank + 1) * per_rank]

    def get_iter_items(self, index):
        rec = self.data[index]
        yield self.getitem(index, rec=rec)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        data = self.data
        if worker_info is not None:
            self.data = self.get_distribute_data(data, worker_info.num_workers, worker_info.id)
        for index in range(len(self.data)):
            items = self.get_iter_items(index)
            for item in items:
                if item is not None:
                    yield item


def gen_ds(args, data_type, data, **kwargs):
    drop_last, shuffle, num_workers, sampler, batch_size, collate_func = False, False, args.n_dl_worker, None, args.batch_size, None
    if data_type=='train':
        ds_cls = globals()[args.ds_cls]
        drop_last, shuffle = True, True
    elif data_type=='val':
        ds_cls = globals()[args.val_ds_cls]
        batch_size = args.val_batch_size
    else:
        ds_cls = globals()[args.test_ds_cls]
        batch_size = args.val_batch_size
    ds = ds_cls(args, data_type, data, **kwargs)
    collate_func = ds.collate

    if args.use_sampler and data_type == 'train':
        sampler = Sampler(args, data_type, ds)
        shuffle = False

    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers, shuffle=shuffle,
                                     drop_last=drop_last, collate_fn=collate_func, sampler=sampler)
    return dl

if __name__ == '__main__':
    import util
    args = util.parser.parse_args([])
