import math
import subprocess
import cv2
from collections import deque

from .images import Images, Frame, MAX_BUFFER_COUNT, MAX_BUFFER_SIZE
    

def iinterval(filename):
    args = [
        'ffprobe',
        '-hide_banner', 
        '-loglevel', 'quiet',
        '-show_entries', 'frame=pict_type:side_data=',
        '-print_format', 'default=noprint_wrappers=1:nokey=1',
        filename,
    ]
    p = subprocess.Popen(args=args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    assert p.stdout is not None
    line = str(next(p.stdout), encoding='utf-8').strip()
    assert line == 'I', line

    for idx, line in enumerate(p.stdout):
        line = str(line, encoding='utf-8').strip()
        assert line == 'I' or line == 'P', line
        if line == 'I':
            p.terminate()
            return idx + 1


class Video(Images):
    def __init__(self, filename, buffer_size=None):
        self.uri = filename

        self.buffer_size = buffer_size or self._get_buffer_size()
        self.buffer = {}
        self.buffer_queue = deque(maxlen=MAX_BUFFER_COUNT)

        self.buffer_start = -1

        self.frame_count = int(self._get_frame_count())
        self.log = []

    def _get_buffer_size(self):
        self.iframe_interval = iinterval(self.uri)
        assert self.iframe_interval is not None
        if self.iframe_interval > MAX_BUFFER_SIZE:
            return MAX_BUFFER_SIZE
        return self.iframe_interval * (MAX_BUFFER_SIZE // self.iframe_interval)

    def _get_frame_count(self):
        cap = cv2.VideoCapture(self.uri)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        cv2.destroyAllWindows()
        return frame_count

    def _evict(self):
        index = self.buffer_queue.popleft()
        buffer = self.buffer.pop(index)
        buffer.clear()

    def decode(self, idx: int):
        buffer_window_index = idx // self.buffer_size
        log = [idx, buffer_window_index, False]
        if buffer_window_index not in self.buffer:
            while len(self.buffer) >= MAX_BUFFER_COUNT:
                self._evict()
            log[2] = True
            cap = cv2.VideoCapture(self.uri)
            if cap.get(cv2.CAP_PROP_POS_FRAMES) != buffer_window_index * self.buffer_size:
                cap.set(cv2.CAP_PROP_POS_FRAMES, buffer_window_index * self.buffer_size)
            buffer: list[None | cv2.typing.MatLike] = [None for _ in range(self.buffer_size)]
            self.buffer[buffer_window_index] = buffer
            self.buffer_queue.append(buffer_window_index)
            for i in range(self.buffer_size):
                ret, frame = cap.read()
                if not ret:
                    break
                buffer[i] = frame
            cap.release()
            cv2.destroyAllWindows()
        self.log.append(tuple(log))
        return self.buffer[buffer_window_index][idx % self.buffer_size]

    def __getitem__(self, key):
        if isinstance(key, int):
            return Frame(self, key)
        elif isinstance(key, slice):
            return SliceVideo(self, key)
        raise Exception('key must be int or slice, received', type(key))

    def __len__(self):
        return self.frame_count

    def __iter__(self):
        return iter(self[i] for i in range(len(self)))


class SliceVideo(Video):
    def __init__(self, video, frame_slice):
        self.video = video
        self.frame_slice = self._normalize_slice(frame_slice)

        self.frame_count = self._get_frame_count()

        self.buffer_size = MAX_BUFFER_SIZE
        self.buffer = {}
        self.buffer_queue = deque(maxlen=MAX_BUFFER_COUNT)

    def _normalize_slice(self, frame_slice):
        if frame_slice.start is None:
            frame_slice.start = 0
        if frame_slice.start < 0:
            frame_slice.start = len(self) - frame_slice.start
            assert frame_slice.start >= 0
        if frame_slice.stop is None:
            frame_slice.stop = len(self)
        if frame_slice.stop < 0:
            frame_slice.stop = len(self) - frame_slice.stop
            assert frame_slice.stop >= 0
        if frame_slice.step is None:
            frame_slice.step = 1
        assert frame_slice.step > 0, frame_slice.step
        return frame_slice

    def _get_frame_count(self):
        fs = self.frame_slice
        return int(math.ceil((fs.stop - fs.start) / fs.step))

    def _evict(self):
        index = self.buffer_queue.popleft()
        buffer = self.buffer.pop(index)
        buffer.clear()

    def decode(self, idx):
        buffer_window_index = idx // self.buffer_size
        log = [idx, buffer_window_index, False]
        if buffer_window_index not in self.buffer:
            while len(self.buffer) >= MAX_BUFFER_COUNT:
                self._evict()
            log[2] = True
            buffer = [None for _ in range(self.buffer_size)]
            self.buffer[buffer_window_index] = buffer
            self.buffer_queue.append(buffer_window_index)
            for i in range(self.buffer_size):
                frame = self.video.decode((buffer_window_index + i) * self.frame_slice.step + self.frame_slice.start)
                buffer[i] = frame
        self.log.append(tuple(log))
        return self.buffer[buffer_window_index][idx % self.buffer_size]
        pass

    def __getitem__(self, key):
        if isinstance(key, int):
            return Frame(self, key)
        elif isinstance(key, slice):
            key = self._normalize_slice(key)
            assert key.stop <= len(self), key.stop
            return SliceVideo(
                self.video,
                slice(
                    self.frame_slice.start + self.frame_slice.step * key.start,
                    self.frame_slice.start + self.frame_slice.step * key.stop,
                    key.step * self.frame_slice.step
                )
            )
        raise Exception('key must be int or slice, received', type(key))

    def __len__(self):
        return self.frame_count

    def __iter__(self):
        return iter(self[i] for i in range(len(self)))