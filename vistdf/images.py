from abc import ABC, abstractmethod
from threading import Thread
from typing import TYPE_CHECKING, Callable
import shutil
from uuid import uuid4

import cv2
import os
from collections import deque

if TYPE_CHECKING:
    from numpy import uint8
    from numpy.typing import NDArray


MAX_BUFFER_SIZE = 16
MAX_BUFFER_COUNT = 8


class BaseImage(ABC):
    @property
    @abstractmethod
    def image(self) -> "NDArray[uint8]":
        raise NotImplementedError()


class BaseImages(ABC):
    @abstractmethod
    def load(self, idx: int | str) -> "NDArray[uint8]":
        raise NotImplementedError()
    
    def _setup_buffer(self, size: int):
        self.buffer_size = size
        self.buffer = {}
        self.buffer_queue = deque(maxlen=self.buffer_size)
    
    def _evict(self):
        index = self.buffer_queue.popleft()
        self.buffer.pop(index)


class Frame(BaseImage):
    def __init__(self, images: "BaseImages", index: int, keyframe: bool = False):
        self.images = images
        self.index = index
        self.keyframe = keyframe

    def decode(self):
        return self.images.load(self.index)

    @property
    def image(self):
        # print(self.index, self.images.uri)
        return self.images.load(self.index)

    @property
    def metadata(self):
        return {"type": 'I' if self.keyframe else 'P'}

    @property
    def shape(self):
        return [0, 0]

    def __repr__(self):
        return f"{self.images} : {self.index}"


class ImageFiles(BaseImages):
    def __init__(self, filenames: list[str]):
        self.filenames: list[str] = filenames
        self.keys: dict[str, int] = {filename: idx for idx, filename in enumerate(filenames)}

        self._setup_buffer(MAX_BUFFER_SIZE * MAX_BUFFER_COUNT)

    def _load(self, idx: str | int):
        if isinstance(idx, int):
            idx = self.filenames[idx]
        return cv2.imread(idx)
    
    def load(self, idx: int | str):
        if isinstance(idx, str):
            idx = self.keys[idx]
        if idx not in self.buffer:
            while len(self.buffer) >= self.buffer_size:
                self._evict()
            self.buffer[idx] = self._load(idx)
            self.buffer_queue.append(idx)
        return self.buffer[idx]

    def __getitem__(self, key: str | int):
        if isinstance(key, str):
            key = self.keys[key]
        assert isinstance(key, int), f'key must be int, received {type(key)}'
        return Frame(self, key)

    def __len__(self):
        return len(self.filenames)

    def __iter__(self):
        return iter(self[i] for i in range(len(self)))
    
    def __repr__(self):
        return f"ImageFiles ({len(self)})"


class ImagesDir(ImageFiles):
    def __init__(self, dirname: str):
        filenames: "list[str]" = [f for f in os.listdir(dirname)]
        super().__init__(filenames)

        self.dirname = dirname
    
    def __repr__(self):
        return f"ImageDir ({self.dirname})"


def load_to_list(loader: "Callable[[str], NDArray[uint8]]", idx: str, filename: str):
    if not os.path.isfile(filename):
        result = loader(idx)
        cv2.imwrite(filename, result) 


class ImagesFromAPI(ImageFiles):
    CACHE_DIR = './.images_from_api'

    def __init__(
            self,
            ids: "list[str]",
            loader: "Callable[[str], NDArray[uint8]]",
            dirname: str | None = None,
            reload=False,
        ):
        os.makedirs(ImagesFromAPI.CACHE_DIR, exist_ok=True)
        self.dirname = dirname or str(uuid4())
        self.dirname = os.path.join(ImagesFromAPI.CACHE_DIR, self.dirname)
        if reload:
            shutil.rmtree(self.dirname)
        os.makedirs(self.dirname, exist_ok=True)
        super().__init__([os.path.join(self.dirname, f"{idx}.jpg") for idx in ids])

        self.ids = ids
        self.keys: dict[str, int] = {
            identifier: idx
            for idx, identifier
            in [*enumerate(self.ids)] + [*enumerate(os.path.join(self.dirname, f"{idx}.jpg") for idx in self.ids)]
        }

        self.results: "list[NDArray[uint8] | None]" = [None] * len(ids)
        self.threads: list[Thread | None] = [
            Thread(target=load_to_list, args=(loader, idx, filename))
            for idx, filename
            in zip(ids, self.filenames)
        ]
        for t in self.threads:
            if t is not None:
                t.start()
    
    def _load(self, idx: str | int):
        if isinstance(idx, int):
            idx = self.ids[idx]
        iidx = self.keys[idx]
        thread = self.threads[iidx]
        if thread is not None:
            thread.join()
            self.results[iidx]
        return cv2.imread(os.path.join(self.dirname, f"{idx}.jpg"))
    
    def __repr__(self):
        return f"ImageFromAPI ({self.dirname})"
