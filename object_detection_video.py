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

from object_detector import object_detector
from video_taker import video_taker


# 로컬에 설치된 레이블 파일을 인덱스와 연결
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
MODEL_NAME = 'centernet_hg104_1024x1024_coco17_tpu-32'
PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)


# 모델 불러오기
def load_model(model_dir):
    model_full_dir = model_dir + "/saved_model"

    # print('Loading model...', end='')
    # start_time = time.time()

    # Load saved model and build the detection function
    detection_model = tf.saved_model.load(model_full_dir)
    return detection_model

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print('Done! Took {} seconds'.format(elapsed_time))
    # 시간 기록이 꼭 필요하지 않아서, 라이브러리를 import하지 않았고, 코드를 실행시키지도 않기로 함.

detection_model = load_model(PATH_TO_MODEL_DIR)

# 비디오를 실행하는 코드
cap = cv2.VideoCapture('data/video.mp4')

if cap.isOpened() == False:
    print('비디오 실행 에러')
else : 
    # 비디오 캡쳐에서, 이미지를 1장씩 가져온다.
    # 1장씩 가져온 이미지에서 사물탐지를 수행한다.

    while cap.isOpened() :
        ret, frame = cap.read()

        if ret == True:
            # 각 프레임이 곧 이미지이며 넘파이 어레이 이므로
            # 각 프레임을 사물 탐지 한다.

            # 현재 학습 환경에서는 이미지 행렬 연산을 CPU로 처리하기 때문에 너무 느리다.
            # 따라서 각 프레임에 대한 연산을 모두 끝낸 후, 동영상으로 다시 저장하고자 함.
            
            # object_detector(detection_model, frame, category_index)
            video_taker (detection_model, frame, category_index)

            if cv2.waitKey(0) & 0xFF == 27:
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
