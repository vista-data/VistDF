from typing import NamedTuple

from .video import Video, Frame

import pandas as pd


def from_file(filename: str):
    v = Video(filename=filename)
    series = pd.Series([*v], name="frame")
    return pd.DataFrame({
        "filename": [filename] * len(series),
        "frame": series,
    })


def play(df: "pd.DataFrame", column="frame"):
    from IPython.display import Video as JupyterVideo
    from IPython.display import HTML
    from base64 import b64encode


    frame: "Frame" = df[column].iloc[0]
    video = frame.images
    assert isinstance(video, Video)
    print(video.uri)
    filename = video.uri
    # return JupyterVideo(video.uri)
    html = ''
    video = open(filename,'rb').read()
    src = 'data:video/mp4;base64,' + b64encode(video).decode()
    html += '<video width=1000 controls autoplay loop><source src="%s" type="video/mp4"></video>' % src 
    return HTML(html)

class BBox:
    x1: int
    x2: int
    y1: int
    y2: int
    
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __repr__(self):
        return f"BB({self.x1},{self.x2},{self.y1},{self.y2})"

    def __str__(self):
        return f"BB({self.x1},{self.x2},{self.y1},{self.y2})"


def detect(df: "pd.Series"):
    import random
    def gen_bboxes():
        x, y = random.randint(50, 100), random.randint(50, 100)
        w, h = random.randint(10, 30), random.randint(10, 30)
        return BBox(x, x + w, y, y + h)
    return df.apply(lambda x: [gen_bboxes() for _ in range(random.randint(3, 5))])


class TBBox:
    def __init__(self, bbox: BBox, frame: int):
        self.bbox = bbox
        self.frame = frame
    
    def __repr__(self):
        return f"{self.bbox}@{self.frame}"


class Trajectory:
    def __init__(self, bboxes: list[BBox], frames: list[int]):
        self.bboxes = bboxes
        self.frames = frames
    def __repr__(self):
        return " ".join(str(TBBox(b, f)) for b, f in zip(self.bboxes, self.frames))

def track(df: "pd.Series"):
    import random
    df = df.sample(frac=0.5)
    def create_traj(b):
        start = random.randint(0, 50)
        return Trajectory(b, [*range(start, start + len(b))])
    df = df.apply(create_traj)
    return pd.DataFrame({"trajectory": df})

