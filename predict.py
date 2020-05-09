"""
SvmPredict PROJECT
Tool for predict rails crosses

Usage:
    predict.py (--models-path=PATH) (--image-path=PATH) [--output-path=PATH] [--check-points-path=PATH]
    predict.py --help

Options:
    -m, --models-path=PATH              Path to SVM models.
    -i, --image-path=PATH               Path to image to process.
    -o, --output-path=PATH              Path to output image.
    -d, --check-points-path=PATH        Path to file with points to check.
    -h, --help                          Show this message.

Mention:
    Result image will be saved to `--output-path`.
    Example:
        Command line: predict.py --models-path=models --image-path=images/example.jpg --output-path res.jpg
"""

import os
import re
import sys
from typing import List
from params import *
import cv2
import pickle
from skimage.feature import hog
import docopt

from utils import get_size_by_y

global need_to_plot

class Model:

    def __init__(self, model, scaler, pca=None):
        self.model = model
        self.pca = pca
        self.scaler = scaler


def predict_in_point(img, x, y, win_w, win_h, pix_per_cell, models):
    real_win_h, real_win_w = get_size_by_y(img.shape[0], min_win_h, max_win_h, y) if need_to_resize else (win_h, win_w)
    half_w, half_h = int(real_win_w / 2), int(real_win_h / 2)

    if y + half_h > img.shape[0] or x + half_w > img.shape[1] or y - half_h < 0 or x - half_w < 0:
        return False

    real_window = img[y - half_h: y + half_h, x - half_w: x + half_w]
    resize_window = cv2.resize(real_window, (win_w, win_h)) if need_to_resize else real_window

    feature = [hog(resize_window, pixels_per_cell=pix_per_cell, cells_per_block=cells_per_bloch)]
    for model_idx, cur_model in enumerate(models):

        if cur_model.scaler is not None:
            feature = cur_model.scaler.transform(feature)

        if cur_model.pca is not None:
            feature = cur_model.pca.transform(feature)

        predict_res = cur_model.model.predict(feature)

        if predict_res[0] == 1:
            color = 0
            if model_idx % 2 == 0:
                color = 255

            print(f'({x}, {y}): {model_idx % 2}')
            cv2.rectangle(img, (x - half_w, y - half_h), (x + half_w, y + half_h), color)
            return True

    return False


def predict_in_heigh(img, models: List[Model], min_h: int, max_h: int, win_w: int, win_h: int, win_step: int, pix_per_cell) -> List:
    x = 0
    while x < img.shape[1] - win_w:
        y = min_h
        while y < max_h - win_h:
            predict_in_point(img, x, y, win_w, win_h, pix_per_cell, models)
            y += win_step
        x += win_step


def prepare_image(image_path: str):
    img = cv2.imread(image_path)

    # crop 1/5 of image
    img = img[int(img.shape[0] / 5): img.shape[0], 0: img.shape[1]]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def predict_window(image_path: str, big_models: List[Model], small_models: List[Model]) -> List:
    res = []
    img = prepare_image(image_path)
    img_h = img.shape[0]

    predict_in_heigh(img, small_models, 0, int(small_heigh_part * img_h), small_w,
                     small_h, 2, pixels_per_cell_small)

    predict_in_heigh(img, big_models, int(small_heigh_part * img_h - small_h), img_h, big_w,
                     big_h, 10, pixels_per_cell_big)

    if need_to_plot:
        cv2.imshow("crop image", img)
        if '--output-path' in opts:
            cv2.imwrite(opts['--output-path'], img)
        cv2.waitKey()

    return res


def run_predictions(path_to_image_dir: str, big_models: List, small_models: List):
    image_names = os.listdir(path_to_image_dir)

    for image_name in image_names:
        res = predict_window(image_name, big_models, small_models)


def load_single_model(file_path):
    try:
        model_fid = open(file_path, 'rb')
        model = pickle.load(model_fid)
    except Exception as e:
        return None

    return model


def get_models(models_dir_path: str) -> (List[Model], List[Model]):
    file_names = os.listdir(models_dir_path)

    small_models: List[Model] = []
    big_models: List[Model] = []
    for model_file_name in file_names:
        if 'model' not in model_file_name:
            continue

        if model_file_name == 'model_big.pkl' and model_file_name == 'model_small.pkl':# \
                #or model_file_name == 'model_1_2_small.pkl' or model_file_name == 'model_1_2_big.pkl':
            continue

        svm_model = load_single_model(os.path.join(models_dir_path, model_file_name))
        scaler = load_single_model(os.path.join(models_dir_path, model_file_name.replace('model', 'scaler')))
        pca = load_single_model(os.path.join(models_dir_path, model_file_name.replace('model', 'pca')))

        model = Model(svm_model, scaler, pca)
        if 'big' in model_file_name:
            big_models.append(model)
        else:
            small_models.append(model)

    return big_models, small_models


def predict_single(image_path, big_models, small_models, path_to_points):
    img = prepare_image(image_path)

    points = open(path_to_points, 'r')
    lines = points.readlines()
    for line in lines:
        xy = re.findall('\d+', line)
        x, y = int(xy[0]), int(xy[1])

        if y > 0.2 * img.shape[0]:
            win_w, win_h, pix_per_cell, models = big_w, big_h, pixels_per_cell_big, big_models
        else:
            win_w, win_h, pix_per_cell, models = small_w, small_h, pixels_per_cell_small, small_models

        predict_in_point(img, x, y, win_w, win_h, pix_per_cell, models)

    if need_to_plot:
        cv2.imshow("image", img)

        if opts['--output-path'] is not None:
            cv2.imwrite(opts['--output-path'], img)
        cv2.waitKey()


def main():

    # parse program options
    models_dir = opts['--models-path']
    image_path = opts['--image-path']

    global need_to_plot

    need_to_plot = True

    # load SVM models
    big_models, small_models = get_models(models_dir)

    if opts['--check-points-path'] is not None:
        predict_single(image_path, big_models, small_models, opts['--check-points-path'])
    else:
        # predict results
        predict_window(image_path, big_models, small_models)


if __name__ == '__main__':
    opts = docopt.docopt(__doc__)
    main()

