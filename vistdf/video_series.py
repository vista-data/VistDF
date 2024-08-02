from .lazy_series import LazySeries
from .video import Video


class VideoSeries:
    def __init__(self, filename):
        self.filename = filename
    
    def execute(self):
        video = Video(self.filename)
        for frame in video:
            yield frame.image
    
    def apply(self, fn):
        return LazySeries(fn, self)