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
##############################


# 로컬환경의 자료를 통한 object detection을 수행하는 코드입니다.

# 로컬에 설치된 레이블 파일을 인덱스와 연결
PATH_TO_LABELS = 'C:\\Users\\5-6\\Documents\\TensorFlow\\models\\research\\object_detection\\data\\mscoco_label_map.pbtxt' # 생성한 TensorFlow 경로 내부에 있는 .pbtxt파일
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True) # 자동완성에 유사한 함수가 있으니 틀리지 않도록 확인
# print (category_index) # 경로 확인


def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    return str(model_dir)

MODEL_DATE = '20200711'                                     # 새로운 모델 사용시 변경
MODEL_NAME = 'centernet_hg104_1024x1024_coco17_tpu-32'      # 새로운 모델 사용시 변경
PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)


def load_model(model_dir):
    model_full_dir = model_dir + "/saved_model"

    detection_model = tf.saved_model.load(model_full_dir)
    return detection_model

detection_model = load_model(PATH_TO_MODEL_DIR)

# 각 자료를 확인해보는 코드
# print ()
# print (detection_model.signatures['serving_default'].inputs)
# print ()
# print (detection_model.signatures['serving_default'].output_dtypes)
# print ()
# print (detection_model.signatures['serving_default'].output_shapes)


# 로컬 이미지 경로에서 이미지를 가져오는 코드
PATH_TO_IMAGE_DIR = pathlib.Path('data\\images') # 가져올 경로 설정
IMAGE_PATHS = list (PATH_TO_IMAGE_DIR.glob('*.jpg')) # 인자로 주어진 문자열이 포함된 파일들만 가져옴 (*은 모든 문자열을 입력받겠다는 뜻.)
# print (IMAGE_PATHS) # 경로 확인


# 가져온 이미지를 넘파이 행렬로 변경하는 함수.
def load_image_into_numpy_array(path):
    return cv2.imread(str(path)) # opencv이용.

for image_path in IMAGE_PATHS:

    print('Running inference for {}... '.format(image_path), end='')

    image_np = load_image_into_numpy_array(image_path)

    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detection_model(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    cv2.imshow(str(image_path), image_np_with_detections)

cv2.waitKey(0)
cv2.destroyAllWindows()