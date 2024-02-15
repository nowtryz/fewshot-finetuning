import contextlib
from datetime import datetime
from operator import itemgetter

import humanize
import torch
import numpy as np
import random
import monai


def set_seeds(seed_value):
    monai.utils.set_determinism(seed_value)


def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def print_monitoring():
    import psutil

    p = psutil.Process()
    peak_memory = int(p.memory_info().rss)
    peak_memory = humanize.naturalsize(peak_memory, binary=True)

    print(f'peak_memory={peak_memory}')


@contextlib.contextmanager
def capture_duration():
    start = datetime.now()
    try:
        yield start
    finally:
        print(f'duration={datetime.now() - start}')


@contextlib.contextmanager
def monitor_resources():
    try:
        yield
    finally:
        print_monitoring()


def keys_extractor(*keys):
    getter = itemgetter(*keys)
    return lambda output: dict(zip(keys, getter(output)))
