import os
from collections import deque
from typing import Callable
from uuid import uuid4
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult

import cv2
import numpy.typing as npt


from .images import Images, MAX_BUFFER_COUNT, MAX_BUFFER_SIZE


class ImagesFromAPI(Images):
    def __init__(self, ids: "list[str]", loader: "Callable[[str], npt.NDArray]"):
        self.uri = 'ImagesFromAPI'

        self.buffer_size = MAX_BUFFER_SIZE * MAX_BUFFER_COUNT
        self.buffer = {}
        self.buffer_queue = deque(maxlen=self.buffer_size)
        self.ids = ids
        self.loader = loader

    def _load(self, idx):
        return self.loader(idx)


def load_async(idx: int, identifier: "str", uri: str, loader):
    output = loader(identifier)
    cv2.imwrite(os.path.join(uri, f"{idx}.jpg"), output) 
    return output


IMAGE_FROM_API_CACHE_DIR = './.images_from_api'


class ImagesFromAPIAsync(Images):
    def __init__(self, ids: "list[str]", loader: "Callable[[str], npt.NDArray]", dirname: str | None = None):
        self.uri = dirname or str(uuid4())
        self.ids = ids
        self.ids2idx = {
            identifier: idx
            for idx, identifier
            in enumerate(self.ids)
        }
        self.finished = set()
        self.pool = Pool()

        os.makedirs(IMAGE_FROM_API_CACHE_DIR, exist_ok=True)
        os.makedirs(os.path.join(IMAGE_FROM_API_CACHE_DIR, self.uri), exist_ok=True)

        self.loads: "list[None | AsyncResult[npt.NDArray]]" = [
            self.pool.apply_async(load_async, (idx, identifier, os.path.join(IMAGE_FROM_API_CACHE_DIR, self.uri), loader))
            for idx, identifier
            in enumerate(ids)
        ]

        self.buffer_size = MAX_BUFFER_SIZE * MAX_BUFFER_COUNT
        self.buffer = {}
        self.buffer_queue = deque(maxlen=self.buffer_size)

    def _load(self, idx):
        idx = self.ids2idx[idx]
        load = self.loads[idx]
        if load is not None:
            img = load.get()
            self.finished.add(idx)
            if len(self.finished) == len(self.ids):
                print('cleanup')
                self.pool.close()
                self.pool.join()
            self.loads[idx] = None
            return img
        return cv2.imread(os.path.join(self.uri, f"{idx}.jpg"))
