import cv2


class Images:
    def __init__(self, filenames):
        self.filenames = filenames

    def decode(self, idx):
        return cv2.imread(self.filenames[idx])
    
    def __getitem__(self, key):
        return Image(self, key)
    
    def __len__(self):
        return int(self.filenames)
    
    def __iter__(self):
        return iter(self[i] for i in range(len(self)))


class Image:
    def __init__(self, images: "Images", index: int):
        self.images = images
        self.index = index
    
    def decode(self):
        return self.images.decode(self.index)

    def __repr__(self):
        return f"{self.images.filenames[self.idx].split('/')[-1]} : {self.index}"