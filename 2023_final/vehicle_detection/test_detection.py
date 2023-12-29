
# import some common libraries
import numpy as np
import cv2
import random
import os
from ultralytics import YOLO
# radiate sdk
import sys
sys.path.insert(0, '..')
import radiate

# path to the sequence
root_path = '../data//mini_test/'
sequence_name = 'city_7_0' # just for example

# time (s) to retrieve next frame
dt = 0.25

# load sequence
seq = radiate.Sequence(os.path.join(root_path, sequence_name), config_file='../config/config.yaml')

model = YOLO('./runs/detect/yolov8s_480_train/weights/best.pt')
for t in np.arange(seq.init_timestamp, seq.end_timestamp, dt):
    output = seq.get_from_timestamp(t)
    if output != {}:
        radar = output['sensors']['radar_cartesian']
        camera = output['sensors']['camera_right_rect']
        

        predictions = predictions["instances"].to("cpu")
        boxes = predictions.pred_boxes 

        objects = []
        results = model.predict(radar)
        #Note: only one result object in result list
        for result in results:
            boxes = result.boxes.cpu().numpy()
            boxes_xyxy = boxes.xyxy.tolist()
            boxes_conf = boxes.conf.tolist()
            for box in boxes_xyxy:
                min_x = box[0]
                min_y = box[1]
                max_x = box[2]
                max_y = box[3]
                width = max_x - min_x
                height = max_y - min_y
                bb = [min_x, min_y, width, height]
                objects.append({'bbox': {'position': bb, 'rotation': 0}, 'class_name': 'car'})

        radar = seq.vis(radar, objects, color=(255,0,0))

        bboxes_cam = seq.project_bboxes_to_camera(objects,
                                                seq.calib.right_cam_mat,
                                                seq.calib.RadarToRight)
        # camera = seq.vis_3d_bbox_cam(camera, bboxes_cam)
        camera = seq.vis_bbox_cam(camera, bboxes_cam)

        cv2.imshow('radar', radar)
        cv2.imshow('camera_right_rect', camera)
        # You can also add other sensors to visualize
        cv2.waitKey(1)
