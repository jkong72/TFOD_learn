##### 라이브러리 불러오기
import tensorflow as tf
import numpy as np
import pathlib
import zipfile
import os

import matplotlib.pyplot as plt
from PIL import Image
import cv2

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import time

##### 내부 패키지 불러오기
from object_detector import object_detector
from video_taker import video_taker
########################################


PATH_TO_LABELS = 'C:\\Users\\5-6\\Documents\\TensorFlow\\models\\research\\object_detection\\data\\mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    return str(model_dir)

MODEL_DATE = '20200711'
MODEL_NAME = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)


def load_model(model_dir):
    model_full_dir = model_dir + "/saved_model"

    detection_model = tf.saved_model.load(model_full_dir)
    return detection_model

detection_model = load_model(PATH_TO_MODEL_DIR)


# 비디오를 실행하는 코드
cap = cv2.VideoCapture('data/video.mp4') # 비디오가 위치한 경로
# cap = cv2.VideoCapture(0) # on cam

if cap.isOpened() == False:
    print('비디오 실행 에러')

else:
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('data/out.avi', # 출력 경로
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        20, # FPS
        (frame_width, frame_height))
    
    # 비디오 캡쳐에서, 이미지를 1장씩 가져온다.
    # 1장씩 가져온 이미지에서 사물탐지를 수행한다.
    while cap.isOpened() :
        ret, frame = cap.read()

        if ret == True:
            # 각 프레임이 곧 이미지이며 넘파이 어레이 이므로
            # 각 프레임을 사물 탐지 한다.

            # 현재 학습 환경에서는 이미지 행렬 연산을 CPU로 처리하기 때문에 너무 느리다.
            # 따라서 각 프레임에 대한 연산을 모두 끝낸 후, 동영상으로 다시 저장하고자 함.
            
            start_time = time.time()
            object_detector(detection_model, frame, category_index)   # 프레임별 작업이 끝날 때마다 직접 창을 띄워 확인하는 코드
            # video_taker (detection_model, frame, category_index, out)   # 프레임별 작업이 끝날 떄마다 동영상 파일로 저장하는 코드

            end_time = time.time()
            print(f'연산에 걸린 시간 {(end_time-start_time)}')
            if cv2.waitKey(20) & 0xFF == 27:
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
