import os
import cv2
import json
import numpy as np
from tqdm import tqdm
ROOT_PATH = "/home/reidite/Dataset/PSA/Cropper"


class PreProcessor:
    def __init__(self):
        self.read_path = os.path.join(ROOT_PATH, "labeled")
        self.save_path = os.path.join(ROOT_PATH, "train")
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        self.image_paths = []

        self.shape = (2698, 1620, 3)
        self.dim = (405, 506)

    def load(self):
        self.image_paths = [x[0] for x in os.walk(self.read_path)][1:]
        pass

    def operate(self):
        for img_path in tqdm(self.image_paths):
            img_name = os.path.split(img_path)[1]

            front_path = os.path.join(img_path, "front.jpg")
            back_path = os.path.join(img_path, "back.jpg")

            front_img = cv2.imread(front_path, cv2.IMREAD_COLOR)
            back_img = cv2.imread(back_path, cv2.IMREAD_COLOR)

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
            
            img_folder = os.path.join(self.save_path, img_name)
            if not os.path.isdir(img_folder):
                os.mkdir(img_folder)
            cv2.imwrite("{}/front.jpg".format(img_folder), input_f)
            cv2.imwrite("{}/back.jpg".format(img_folder), input_b)
            cv2.imwrite("{}/gtr_front.jpg".format(img_folder), gtr_f)
            cv2.imwrite("{}/gtr_back.jpg".format(img_folder), gtr_b)

if __name__ == '__main__':
    processor = PreProcessor()
    processor.load()
    processor.operate()