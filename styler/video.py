import cv2

from styler.utils import resize
import numpy as np

class Video:

    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(self.path)
        self.frames = []

    def __enter__(self):
        if not self.cap.isOpened():
            raise Exception('Cannot open video: {}'.format(self.path))
        return self

    def __len__(self):
        return len(self.frames)

    def read_frames(self, image_h, image_w):
        '''
        5.
         - Read video frames from `self.cap` and collect frames into list
         - Apply `resize()` on each frame before add it to list
         - Also assign frames to "self" object
         - Return your results
        '''
        frames = []
        # 5-1 /5-2 Read video and collect them

        while (self.cap.isOpened()):
            ret, getframe=self.cap.read()
            if ret==True:
                #print (getframe.shape)
                resize_frame=resize(getframe,image_h,image_w)
                frames.append(resize_frame)
            else:
                break
            
        self.frames=frames  # 5-3 let object have the result
        return self.frames  # return your results

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()
