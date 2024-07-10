import subprocess
import cv2
    

MAX_BUFFER_SIZE = 64


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

    l = str(next(p.stdout), encoding='utf-8').strip()
    assert l == 'I', l

    for idx, l in enumerate(p.stdout):
        l = str(l, encoding='utf-8').strip()
        assert l == 'I' or l == 'P', l
        if l == 'I':
            p.terminate()
            return idx + 1


class Video:
    def __init__(self, filename):
        self.filename = filename

        self.buffer_size = self._get_buffer_size()
        self.buffer = [None for _ in range(self.buffer_size)]
        self.buffer_start = -1

        self.frame_count = self._get_frame_count()

    def _get_buffer_size(self):
        self.iframe_interval = iinterval(self.filename)
        if self.iframe_interval > MAX_BUFFER_SIZE:
            return MAX_BUFFER_SIZE
        return self.iframe_interval * (MAX_BUFFER_SIZE // self.iframe_interval)

    def _get_frame_count(self):
        cap = cv2.VideoCapture(self.filename)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        cv2.destroyAllWindows()
        return frame_count
    
    def decode(self, idx):
        if idx // self.buffer_size != self.buffer_start:
            cap = cv2.VideoCapture(self.filename)
            cap.set(cv2.CAP_PROP_POS_FRAMES, (idx // self.buffer_size) * self.buffer_size)
            self.buffer = [None for _ in range(self.buffer_size)]
            for i in range(self.buffer_size):
                ret, frame = cap.read()
                if not ret:
                    break
                self.buffer[i] = frame
            cap.release()
            cv2.destroyAllWindows()
        return self.buffer[idx % self.buffer_size]
    
    def __getitem__(self, key):
        return Frame(self, key)
    
    def __len__(self):
        return int(self.frame_count)
    
    def __iter__(self):
        return iter(self[i] for i in range(len(self)))


class Frame:
    def __init__(self, video: "Video", index: int):
        self.video = video
        self.index = index
    
    def decode(self):
        return self.video.decode(self.index)
    
    def is_keyframe(self):
        return self.index % self.video.buffer_size == 0

    def __repr__(self):
        return f"{self.video.filename.split('/')[-1]} : {self.index}"