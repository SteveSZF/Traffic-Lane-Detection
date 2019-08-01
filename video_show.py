#!/usr/bin/env python
#import rospy
#from sensor_msgs.msg import Image
#from ld_lsi.msg import CnnOutput
#import rospkg
#from rospy.numpy_msg import numpy_msg
#from cv_bridge import CvBridge

import torch
import importlib
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import cv2

import sys
import os
import time

from modeling.erfnet_road import *
### Colors for visualization
# Ego: red, other: blue
COLORS_DEBUG = [(255,0,0), (0,0,255)]

# Road name map
ROAD_MAP = ['Residential', 'Highway', 'City Street', 'Other']

def video_show(weights_path, video_path, with_road=False, num_classes=3, num_scenes = 4):
    
    # Assuming the main constructor is method Net()
    #cnn = importlib.import_module(model_name).Net()
    cnn = ERFNet(num_classes_pixel = num_classes, num_classes_scene = num_scenes, multitask = with_road)
    # GPU only mode, setting up
    cnn = torch.nn.DataParallel(cnn).cuda()
    cnn.module.load_state_dict(torch.load(weights_path)['state_dict'])
    cnn.eval()
    index = 0
    cap = cv2.VideoCapture(video_path)
    while(1):
        # get a frame
        ret, image = cap.read()
        if not ret:
            break
        # show a frame
        #cv2.imshow("capture", frame)
        input_tensor = torch.from_numpy(image)
        input_tensor = torch.div(input_tensor.float(), 255)
        input_tensor = input_tensor.permute(2,0,1).unsqueeze(0)
        with torch.no_grad():
            input_tensor = Variable(input_tensor).cuda()
            output = cnn(input_tensor)

        output, output_road = output
        if output_road is not None:
            road_type = output_road.max(dim=1)[1][0]
        else:
            road_type = 0

        ### Classification
        output = output.max(dim=1)[1]
        output = output.float().unsqueeze(0)

        ### Resize to desired scale for easier clustering
        #output = F.interpolate(output, size=(output.size(2) / self.resize_factor, output.size(3) / self.resize_factor) , mode='nearest')
        # Convert the image and substitute the colors for egolane and other lane
        output = output.squeeze().unsqueeze(2).data.cpu().numpy()
        output = output.astype(np.uint8)

        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        output[np.where((output == [1, 1, 1]).all(axis=2))] = COLORS_DEBUG[0]
        output[np.where((output == [2, 2, 2]).all(axis=2))] = COLORS_DEBUG[1]

        # Blend the original image and the output of the CNN
        output = cv2.resize(output, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        image = cv2.addWeighted(image, 1, output, 0.4, 0)
        if with_road:
            cv2.putText(image, ROAD_MAP[road_type], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Visualization
        #cv2.imshow("CNN Output", cv2.resize(image, (320, 240), cv2.INTER_NEAREST))
        cv2.imwrite('/workspace/Videos/video_lane_test/seg_result/%d.jpg'%(index), image)
        index += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Node initialization
if __name__ == '__main__':
    weights_path = "run/bdd100k/erfnet/model_best.pth.tar"
    video_path = "/workspace/Videos/video_lane_test/pandora.avi"
    video_show(weights_path, video_path)
