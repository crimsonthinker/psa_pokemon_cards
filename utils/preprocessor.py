import numpy as np
import cv2
import math
import os
from tqdm import tqdm
import json

from utils.utilities import ensure_dir
from models.unet import UNET
from tensorflow import convert_to_tensor
class UNETPreProcessor(object):
    def __init__(self,
        original_height,
        original_width,
        dim):
        self.train_read_path = os.path.join("unet_labeled", "train")
        self.test_read_path  = os.path.join("unet_labeled", "test")
        self.train_save_path = os.path.join("preprocessed_data", "UNET", "train")
        self.test_save_path  = os.path.join("preprocessed_data", "UNET", "test")

        ensure_dir(self.train_save_path)
        ensure_dir(self.test_save_path)

        self.train_image_paths = []
        self.test_image_paths = []

        self.shape = (original_height, original_width, dim)
        self.feed_size = (405, 506)

    def load(self):
        self.train_image_paths = [x[0] for x in os.walk(self.train_read_path)][1:]
        self.test_image_paths = [x[0] for x in os.walk(self.test_read_path)][1:]

    def operate(self):
        for img_path in tqdm(self.train_image_paths):
            self.process(img_path, isTrain=True)
        for img_path in tqdm(self.test_image_paths):
            self.process(img_path, isTrain=False)
    
    def process(self, img_path, isTrain=True):
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
        
        input_f = cv2.resize(front_img[self.shape[0]//4:,:], self.feed_size, interpolation = cv2.INTER_AREA)
        gtr_f = cv2.resize(front_seg[self.shape[0]//4:,:], self.feed_size, interpolation = cv2.INTER_AREA)
        input_b = cv2.resize(back_img[self.shape[0]//4:,:], self.feed_size, interpolation = cv2.INTER_AREA)
        gtr_b = cv2.resize(back_seg[self.shape[0]//4:,:], self.feed_size, interpolation = cv2.INTER_AREA)
            
        img_folder = os.path.join(self.train_save_path, img_name) if isTrain else \
                    os.path.join(self.test_save_path, img_name)
        if not os.path.isdir(img_folder):
            os.mkdir(img_folder)
        cv2.imwrite("{}/front.jpg".format(img_folder), input_f)
        cv2.imwrite("{}/back.jpg".format(img_folder), input_b)
        cv2.imwrite("{}/gtr_front.jpg".format(img_folder), gtr_f)
        cv2.imwrite("{}/gtr_back.jpg".format(img_folder), gtr_b)

class VGG16PreProcessor(object):
    def __init__(self):
        self.shape = (2698, 1620, 3)
        self.feed_size = (405, 506)
        self.model = UNET((512, 512, 3))
        self.pretrained_model_path = os.path.join('checkpoint/cropper/pretrained/checkpoint')
        self.model.load_weights(self.pretrained_model_path)

    def crop_image(self, image : np.ndarray):
        inputs_img = np.zeros([1, 512, 512, 3], np.float32)
        inputs_img[0][:506, :405] = cv2.resize(image[self.shape[0]//4:,:], self.feed_size, interpolation = cv2.INTER_AREA).astype(np.float32) / 255.0
        preds = self.model(convert_to_tensor(inputs_img), training=False).numpy()
        pred_mask = preds[0][:506, :405]
        pred_mask[pred_mask>=0.5] = 1.0
        pred_mask[pred_mask<0.5] = 0.0

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(16,27))
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
                                                 
        contours, hierarchy = cv2.findContours(image=pred_mask.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        c = np.concatenate(contours, axis=0)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = self.rearrange_box(box) * 4

        PSA_WIDTH = np.linalg.norm(box[1]-box[0])
        PSA_HEIGHT = np.linalg.norm(box[3]-box[0])
        aligned_box = np.float32([[0, 0], [PSA_WIDTH, 0], [PSA_WIDTH, PSA_HEIGHT], [0, PSA_HEIGHT]])
        M = cv2.getPerspectiveTransform(box, aligned_box)
        cropped_img = cv2.warpPerspective(image[self.shape[0]//4:,:], M, (PSA_WIDTH, PSA_HEIGHT))
        
        # Visualize results
        # cv2.drawContours(inputs_img[0][:506, :405],[np.uint0(box / 4)],0,(0,191,255),2)
        # cv2.imshow("img", inputs_img[0][:506, :405])
        # cv2.imshow("prd", pred_mask)
        # cv2.imshow("crop", cv2.resize(cropped_img, (cropped_img.shape[1] // 2, cropped_img.shape[0] // 2), interpolation = cv2.INTER_AREA))
        # cv2.waitKey(0)
        return cropped_img

    def rearrange_box(self, box):
        pts = np.zeros_like(box)
        cons = 300.0
        for p in box:
	        if p[0] <= cons and p[1] <= cons:
	        	pts[0] = p
	        elif p[1] <= cons:
	        	pts[1] = p
	        elif p[0] <= cons:
	        	pts[3] = p
	        else:
	        	pts[2] = p
        return pts

    def extract_contour_for_pop_image(self, image : np.ndarray):
        """Extract cards from images

        Args:
            image (np.ndarray): [description]

        Returns:
            [type]: [description]
        """
        # to grayscale image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # get edges
        _, otsu_grad = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # get contours
        contours, _ = cv2.findContours(otsu_grad, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # image size
        height, width = otsu_grad.shape
        image_area = height * width

        # sort contour index
        index_sort = sorted(range(len(contours)), key=lambda i : cv2.contourArea(contours[i]),reverse=True)
        contours_sort = [contours[i] for i in index_sort]

        # get area and perimeter
        contour_peri = [cv2.arcLength(contours_sort[i], True) for i in range(len(index_sort))]
        approx = [cv2.approxPolyDP(contours_sort[i], 0.001 * contour_peri[i], True) for i in range(len(index_sort))]
        bounding_box = [cv2.boundingRect(approx[i]) for i in range(len(index_sort))]
        contour_area = [bounding_box[i][2] * bounding_box[i][3]  for i in range(len(index_sort))]
        is_card = list(filter(lambda x : x >= 0, [i if contour_area[i] >= 0.48 * image_area and contour_area[i] <= 0.65 * image_area else -1 for i in range(len(index_sort))]))

        if len(is_card) > 0:
            is_card_index = is_card[-1]
            x,y,w,h = bounding_box[is_card_index]
            return image[int(y) : int(y + h), int(x) : int(x + w)]
        else:
            return None

    def extract_contour_for_dim_image(self, image : np.ndarray):
        """[summary]

        Args:
        image (np.ndarray): [description]

        Returns:
        [type]: [description]
        """
        # to grayscale image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # get edges
        blur = cv2.GaussianBlur(gray, (3, 3), -10)
        adaptive_binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7,3)
        edges = cv2.Canny(adaptive_binary,100,200)
        binarized_grad = 255 - edges

        # denoises again
        open_binarized_grad = cv2.morphologyEx(
        binarized_grad, 
        cv2.MORPH_OPEN, 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))

        # get contours
        contours, _ = cv2.findContours(open_binarized_grad, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # image size
        height, width = binarized_grad.shape
        image_area = height * width

        # sort contour index
        index_sort = sorted(range(len(contours)), key=lambda i : cv2.contourArea(contours[i]),reverse=True)
        contours_sort = [contours[i] for i in index_sort]

        # get area and perimeter
        contour_area = [cv2.contourArea(contours_sort[i]) for i in range(len(index_sort))]
        contour_peri = [cv2.arcLength(contours_sort[i], True) for i in range(len(index_sort))]
        approx = [cv2.approxPolyDP(contours_sort[i], 0.001 * contour_peri[i], True) for i in range(len(index_sort))]
        bounding_box = [cv2.boundingRect(approx[i]) for i in range(len(index_sort))]
        is_card = list(filter(lambda x : x >= 0, [i if contour_area[i] >= 0.48 * image_area and contour_area[i] <= 0.65 * image_area else -1 for i in range(len(index_sort))]))

        if len(is_card) > 0:
            is_card_index = is_card[-1]
            x,y,w,h = bounding_box[is_card_index]
            return image[int(y) : int(y + h), int(x) : int(x + w)]
        else:
            return None
