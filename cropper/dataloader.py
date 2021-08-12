import os
import random
import cv2
import json
from tensorflow import convert_to_tensor
from glob import glob
import numpy as np
'''
Data Structure
root---\---- 3727258001 --
        \--- 3727258002 --
         \-- 3727258003 --
'''
DATA_PATH = "/home/reidite/Dataset/PSA/Cropper/train"

class DataLoader:
    def __init__(self):
        self.batch_size = 8
        self.shape = (2698, 1620, 3)
        self.dim = (405, 506)
        self.data_path = os.path.join(DATA_PATH)
        self.image_paths = []

        self.train_paths = []
        self.train_pivot = 0
        self.num_batch4train = -1

        self.test_paths = []
        self.test_pivot = 0
        self.num_batch4test = -1

        
    def load(self):
        self.image_paths = [x[0] for x in os.walk(DATA_PATH)][1:]
        pass

    def next_train_batch(self):
        inputs_img = np.zeros([self.batch_size, 512, 512, 3], np.float32)
        inputs_gtr = np.zeros([self.batch_size, 512, 512, 1], np.float32)

        for i in range(0, self.batch_size, 2):
            img_path = self.train_paths[self.train_pivot]
            (inputs_img[i][:506, :405], inputs_img[i+1][:506, :405]),\
                 (inputs_gtr[i][:506, :405], inputs_gtr[i+1][:506, :405]) = self.dataparser(img_path)
            self.train_pivot += 1
        return convert_to_tensor(inputs_img), convert_to_tensor(inputs_gtr)

    def next_test_batch(self):
        inputs_img = np.zeros([self.batch_size, 512, 512, 3], np.float32)
        inputs_gtr = np.zeros([self.batch_size, 512, 512, 1], np.float32)

        for i in range(self.batch_size//2):
            img_path = self.test_paths[self.test_pivot]
            (inputs_img[i][:506, :405], inputs_img[i+1][:506, :405]),\
                 (inputs_gtr[i][:506, :405], inputs_gtr[i+1][:506, :405]) = self.dataparser(img_path)
            self.test_pivot += 1
        return convert_to_tensor(inputs_img), convert_to_tensor(inputs_gtr)
    
    def isEnoughData(self, isTrained=True):
        numOfRemainedData = len(self.train_paths) - (self.train_pivot) if isTrained else len(self.test_paths) - (self.test_pivot)
        return True if numOfRemainedData >= self.batch_size else False
    
    def slit(self, ratio):
        cls_lst = self.image_paths.copy()
        random.shuffle(cls_lst)
        split_idx = int(ratio*len(cls_lst))
        self.train_paths = cls_lst[:split_idx]
        self.num_batch4train = len(self.train_paths) // self.batch_size
        self.test_paths = cls_lst[split_idx:]
        self.num_batch4test = len(self.test_paths) // self.batch_size
        pass
    
    def shuffle(self):
        random.shuffle(self.train_paths)
        random.shuffle(self.test_paths)
        pass

    def reset(self):
        self.train_pivot = 0
        self.test_pivot = 0

    def dataparser(self, img_path):
        front_path = os.path.join(img_path, "front.jpg")
        back_path = os.path.join(img_path, "back.jpg")
        front_img = cv2.imread(front_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        back_img = cv2.imread(back_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        front_seg = np.zeros([front_img.shape[0], front_img.shape[1], 1],dtype=np.float32)
        back_seg = np.zeros([back_img.shape[0], back_img.shape[1], 1],dtype=np.float32)
        front_points = np.array([[round(pt[0]),round(pt[1])] \
             for pt in json.load(open(os.path.join(img_path, "front.json")))['shapes'][0]['points']])
        back_points = np.array([[round(pt[0]),round(pt[1])] \
             for pt in json.load(open(os.path.join(img_path, "back.json")))['shapes'][0]['points']])
        cv2.fillPoly(front_seg, pts =[front_points], color=(255,255,255))
        cv2.fillPoly(back_seg, pts =[back_points], color=(255,255,255))
        
        input_f = cv2.resize(front_img[self.shape[0]//4:,:], self.dim, interpolation = cv2.INTER_AREA)
        gtr_f = cv2.resize(front_seg[self.shape[0]//4:,:], self.dim, interpolation = cv2.INTER_AREA)
        input_b = cv2.resize(back_img[self.shape[0]//4:,:], self.dim, interpolation = cv2.INTER_AREA)
        gtr_b = cv2.resize(back_seg[self.shape[0]//4:,:], self.dim, interpolation = cv2.INTER_AREA)
        return [input_f, input_b], [np.expand_dims(gtr_f, -1), np.expand_dims(gtr_b, -1)]