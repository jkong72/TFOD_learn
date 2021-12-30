#!/usr/bin/env python
# coding: utf-8
"""
Object Detection From TF2 Saved Model
=====================================
"""

# %%
# 이 데모(시험판)는 텐서플로2와 호환되는 무설치 사물 탐지 모델을 사용하는 단계를 안내합니다.
# 이 데모에서 사용하는 모델에 대해서는 다음 사이트에서 더 자세하게 알아볼 수 있습니다.
#`Saved Model Format <https://www.tensorflow.org/guide/saved_model>`


# %%
# 테스트 이미지 다운로드
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 우선 이 안내서(튜토리얼)에서 사용할 이미지들을 다운로드 받겠습니다.
# 하단의 코드는 다음 사이트로부터 이미지를 가져와 ``data/images`` 폴더에 저장합니다.
# `TensorFlow Model Garden <https://github.com/tensorflow/models/tree/master/research/object_detection/test_images>`_
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # TensorFlow logging을 무시하는 코드 (1)
import pathlib
import tensorflow as tf

tf.get_logger().setLevel('ERROR')           # TensorFlow logging을 무시하는 코드 (2)

# GPU 다이나믹 메모리 할당 허용
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def download_images():
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/'
    filenames = ['image1.jpg', 'image2.jpg']
    image_paths = []
    for filename in filenames:
        image_path = tf.keras.utils.get_file(fname=filename,
                                            origin=base_url + filename,
                                            untar=False)
        image_path = pathlib.Path(image_path)
        image_paths.append(str(image_path))
    return image_paths

IMAGE_PATHS = download_images()


# %%
# 모델 다운로드 함수
# ~~~~~~~~~~~~~~~~~~
# 아래의 코드들은 학습을 끝낸 사물 탐지 모델을 다운로드 하는데 사용됩니다.
# 예시로 사용된 모델은 다음과 같습니다.
# `CenterNet HourGlass104 1024x1024`.
# 이 외에도 더 많은 모델을 다음 사이트에서 찾아볼 수 있습니다. (ctrl+클릭으로 바로 이동 가능)
# `TensorFlow 2 Detection Model Zoo <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md>`_.
# URL 경로를 통해 다른 모델을 사용할 수도 있습니다.
# 방법:
#
# 1. `Model name`에 마우스 우클릭합니다.
# 2. `링크 주소 복사`를 클릭해 다운로드 링크를 복사합니다.
# 3. text 파일을 붙여 넣을 수 있는 곳에서 다음과 같은 링크의 형식을 확인할 수 있습니다.
#    ``download.tensorflow.org/models/object_detection/tf2/YYYYYYYY/XXXXXXXXX.tar.gz``;
# 4. ``XXXXXXXXX``부분을 아래 ``MODEL_NAME``변수에 새로 저장합니다.
# 5. ``YYYYYYYY`` 부분을 아래 ``MODEL_DATE`` 변수에 새로 저장합니다.
#
# 예시로 사용한 모델의 링크주소: ``download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_1024x1024_coco17_tpu-32.tar.gz``

# 모델 다운로드 및 설치
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

# %%
# 레이블 다운로드
# ~~~~~~~~~~~~~~~~~~~
# 아래 코드는 탐지된 각 사물을 표현하는 문자열(예: 사람) 리스트에 담겨있는 .pbtxt 파일로 구성된 레이블 파일을 다운로드 합니다.
# 학습된 모델들은 COCO 데이터셋을 바탕으로 훈련을 진행했습니다. 따라서 우리도 훈련 데이터셋과 같은 레이블을 사용할 필요가 있습니다.
# ``mscoco_label_map.pbtxt``
# TensroFlow models Garden의 전체 레이블 파일 리스트는 다음 사이트에서 찾아볼 수 있습니다.
# <https://github.com/tensorflow/models/tree/master/research/object_detection/data>`__.

# 레이블 파일 다운로드
def download_labels(filename):
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    label_dir = tf.keras.utils.get_file(fname=filename,
                                        origin=base_url + filename,
                                        untar=False)
    label_dir = pathlib.Path(label_dir)
    return str(label_dir)

LABEL_FILENAME = 'mscoco_label_map.pbtxt'
PATH_TO_LABELS = download_labels(LABEL_FILENAME)

# %%
# 모델 불러오기
# ~~~~~~~~~~~~~~
# 이제 다운로드한 모델을 불러옵니다.
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# %%
# 레이블 맵 데이터 불러오기 (화면 표시용)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 레이블 맵 데이터는 각 카테고리의 이름과 일치되어 있습니다. 따라서 우리는`5`라는 예측 결과를 통해 이것이 `비행기`임을 알 수 있습니다.
# 여기서는 내부 유틸리티 함수를 사용하지만, 정수와 문자열을 맞춘 사전(예: {1:'사람'})을 직접 건네주어도 됩니다.

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

# %%
# 전체 기능 구성
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 아래의 코드는 탐지 모델을 실행하고, 키포인트를 포함한 해당 결과를 시각화합니다.
#
# 코드를 처음 실행할 때는 
# tf.function's trace-compilation --- 부분에서 몇 분 정도 시간이 소요될 수 있습니다. 하지만 
# 이후의 실행에서는 (새로운 이미지를 사용하더라도) 더 빠르게 실행 될 것입니다.
#
# 이런 의문이 들었나요?:
#
# * 입력한 이미지를 뒤집거나 그레이 스케일 이미지로 바꾸는 등의 변형을 가해도 탐지는 똑같이 작동 합니다. 아래쪽에 실험해볼만한 관련 코드가 있습니다. (단, 입력한 이미지의 채널 수는 항상 3이어야 합니다.
# * `detection_boxes` 부분을 print 해보고 이미지의 실제 박스 위치와 일치하는지 확인해보세요. 이 값이 0~1사이의 값으로 정규화된 값이라는 것만 기억해 두시면 됩니다.
# * `min_score_thresh`ections` 부분을 0과 1사이의 다른 값으로 설정해보세요. 탐지된 물체들의 예측 확률(퍼센트)에 따라 걸러낼 수 있습니다.
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Matplotlib warnings을 무시하는 코드

def load_image_into_numpy_array(path):
    """이미지 파일을 numpy 배열로 변환합니다.

    tensorflow 그래프 활용을 위해 이미지를 numpy 배열로 바꿔야 합니다.
    (높이, 너비, 채널)순서의 형태를 띄는 배열로 반환하며,
    색상값(RGB)에 따라 채널은 3입니다.

    Args: #인자
      path: 이미지 파일의 경로

    Returns: #반환
      형태가 명시된 uint8 numpy 배열 (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


for image_path in IMAGE_PATHS:

    print('Running inference for {}... '.format(image_path), end='')

    image_np = load_image_into_numpy_array(image_path)

    # 추가 실험:
    # 세로로 뒤집기
    # image_np = np.fliplr(image_np).copy()

    # 그레이 스케일(회색조) 이미지로 변환
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)


    # 입력값은 tensor로 건네줘야 하므로 `tf.convert_to_tensor`.함수를 사용합니다.
    input_tensor = tf.convert_to_tensor(image_np)
    # 모델은 이미지들을 한번에 가져오기를 기대하므로 `tf.newaxis`.라는 축을 추가해줍니다.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # "모든" 출력값은 텐서로 묶여있습니다.
    # 이것을 numpy 배열로 변환한 후, 필요 없는 파일은 제외하고,
    # 첫번째에 위치한 num_detections 만 가져오기 위해
    # [0]번 인덱스만 가져올 필요가 있습니다.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes 는 정수형으로 주어져야 합니다.
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

    plt.figure()
    plt.imshow(image_np_with_detections)
    print('Done')
plt.show()

# sphinx_gallery_thumbnail_number = 2