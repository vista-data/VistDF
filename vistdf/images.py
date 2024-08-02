import cv2
import os
from collections import deque


MAX_BUFFER_SIZE = 16
MAX_BUFFER_COUNT = 8


class Images:
    def __init__(self, dirname):
        self.uri = dirname
        self.buffer_size = MAX_BUFFER_SIZE * MAX_BUFFER_COUNT
        self.buffer = {}
        self.buffer_queue = deque(maxlen=self.buffer_size)

        self.ids: "list[str]" = [f for f in os.listdir(dirname)]

    def _evict(self):
        index = self.buffer_queue.popleft()
        self.buffer.pop(index)

    def _load(self, idx):
        return cv2.imread(os.path.join(self.uri, idx))

    def decode(self, idx: int | str):
        if isinstance(idx, int):
            idx = self.ids[idx]
        if idx not in self.buffer:
            while len(self.buffer) >= self.buffer_size:
                self._evict()
            self.buffer[idx] = self._load(idx)
            self.buffer_queue.append(idx)
        return self.buffer[idx]

    def __getitem__(self, key):
        if isinstance(key, int):
            return Frame(self, key)
        raise Exception('key must be int, received', type(key))

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        return iter(self[i] for i in range(len(self)))


class Frame:
    def __init__(self, images: "Images", index: int, keyframe: bool = False):
        self.images = images
        self.index = index
        self.keyframe = keyframe

    def decode(self):
        return self.images.decode(self.index)

    @property
    def image(self):
        print(self.index, self.images.uri)
        return self.images.decode(self.index)

    @property
    def metadata(self):
        return {"type": 'I' if self.keyframe else 'P'}

    @property
    def shape(self):
        return [0, 0]

    def __repr__(self):
        return f"{self.images.uri.split('/')[-1]} : {self.index}"
