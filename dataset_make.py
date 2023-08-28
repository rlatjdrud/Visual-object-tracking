from __future__ import unicode_literals
from subprocess import check_call
from concurrent import futures
import os
from os.path import join
from utils.coordinate import Center, center2corner
import sys
import cv2
import pandas as pd
import numpy as np


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    print(bbox)
    c = -a * bbox[0]
    d = -b * bbox[1]
    
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
   
    return crop


def pos_s_2_bbox(pos, s):
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
    target_size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad
   
    bbox_z=pos_s_2_bbox(target_pos, s_z)
    bbox_x=pos_s_2_bbox(target_pos, s_x)
    bbox_z=np.array(bbox_z).astype(int)
    bbox_x=np.array(bbox_x).astype(int)
    
    image_x=image[bbox_x[1]:bbox_x[3],bbox_x[0]:bbox_x[2]]
    image_z=image[bbox_z[1]:bbox_z[3],bbox_z[0]:bbox_z[2]]
  
    image_x=cv2.resize(image_x,(255,255))
    cv2.imshow("hi_x",image_x)
    cv2.imshow("hi_z",image_z)
    
    z = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
    return z, x

def _get_bbox( image, shape):
    imh, imw = image.shape[:2]
    if len(shape) == 4:
        w, h = shape[2]-shape[0], shape[3]-shape[1]
    else:
        w, h = shape
    context_amount = 0.5
    exemplar_size = 127
    wc_z = w + context_amount * (w+h)
    hc_z = h + context_amount * (w+h)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    w = w*scale_z
    h = h*scale_z
    cx, cy = imw//2, imh//2
    bbox = center2corner(Center(cx, cy, w, h))
    # 这里的bbox是cropped_x上的坐标了，不再是raw image上的原本坐标了
    return bbox



if __name__=="__main__":
    img=cv2.imread("/media/ksk/T7/VOT_DATASET/lasot/airplane/airplane-1/img/00000001.jpg")
    mapping = np.array([[2, 0,0],
                        [0, 2, 0]]).astype(np.float)
    
    bbox = np.array([367,101,408,117])
    z,x=crop_like_SiamFC(img,bbox)
    crop = cv2.warpAffine(img, mapping, (255, 255), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    changed_bbox = _get_bbox(x,bbox)
    changed_bbox1 = _get_bbox(z,bbox)
    print(changed_bbox)

    cv2.rectangle(x,(int(changed_bbox[0]),int(changed_bbox[1])),(int(changed_bbox[2]),int(changed_bbox[3])),(255,0,0),3)
    cv2.rectangle(z,(int(changed_bbox1[0]),int(changed_bbox1[1])),(int(changed_bbox1[2]),int(changed_bbox1[3])),(255,0,0),3)
    cv2.imshow("Z",z)
    cv2.imshow("x",x)
    cv2.imshow("crop",crop)
    cv2.waitKey(0) 

