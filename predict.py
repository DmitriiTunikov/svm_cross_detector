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
        Command line: predict.py --models-path=models --image-path=images/img0002.jpg --output-path res.jpg
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
global need_to_plot

class Model:

    def __init__(self, model, scaler, pca=None):
        self.model = model
        self.pca = pca
        self.scaler = scaler


def predict_in_point(img, x, y, win_w, win_h, pix_per_cell, models, is_pca=False):
    half_w, half_h = int(win_w / 2), int(win_h / 2)

    if y + win_h > img.shape[0] or x + win_w > img.shape[1] or y - half_h < 0 or x - half_w < 0:
        return False

    window = img[y - half_h: y + half_h, x - half_w: x + half_w]
    feature = hog(window, pixels_per_cell=pix_per_cell, cells_per_block=cells_per_bloch)

    for model_idx, cur_model in enumerate(models):

        scaled_feature = cur_model.scaler.transform([feature])

        feature_reduced = scaled_feature
        # is pca reduced
        if is_pca:
            feature_reduced = cur_model.pca.transform(scaled_feature)

        predict_res = cur_model.model.predict(feature_reduced)

        if predict_res[0] == 1:
            color = 0
            if model_idx % 2 == 0:
                color = 255

            print(f'({x}, {y}): {model_idx % 2}')
            cv2.rectangle(img, (x - half_w, y - half_h), (x + half_w, y + half_h), color)
            return True

    return False


def predict_in_heigh(img, models: List[Model], min_h: int, max_h: int, win_w: int, win_h: int, win_step: int, pix_per_cell,
                     is_pca: bool) -> List:
    x = 0
    while x < img.shape[1] - win_w:
        y = min_h
        while y < max_h - win_h:
            predict_in_point(img, x, y, win_w, win_h, pix_per_cell, models, is_pca)
            y += win_step
        x += win_step


def prepare_image(image_path: str):
    img = cv2.imread(image_path)

    # crop 1/5 of image
    img = img[int(img.shape[0] / 5): img.shape[0], 0: img.shape[1]]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def predict_window(image_path: str, big_models: List[Model], small_models: List[Model], is_pca: bool) -> List:
    res = []
    img = prepare_image(image_path)
    img_h = img.shape[0]

    predict_in_heigh(img, small_models, int(small_h / 2), int(small_threshhold * img_h), small_w,
                     small_h, 2, pixels_per_cell_small, is_pca)

    predict_in_heigh(img, big_models, int(small_threshhold * img_h - small_h / 2), img_h, big_w,
                     big_h, 10, pixels_per_cell_big, is_pca)

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


def get_models(models_dir_path: str, is_pca: bool) -> (List[Model], List[Model]):
    file_names = os.listdir(models_dir_path)

    small_models: List[Model] = []
    big_models: List[Model] = []
    for model_file_name in file_names:
        if 'svm_model' not in model_file_name:
            continue

        if model_file_name == 'svm_model_big.pkl' or model_file_name == 'svm_model_small.pkl':#\
                # or model_file_name == 'svm_model_1_2_small.pkl' or model_file_name == 'svm_model_1_2_big.pkl':
            continue

        # if model_file_name != 'svm_model_big.pkl' and model_file_name != 'svm_model_small.pkl':
        #     continue

        pca_file_name = model_file_name.replace('svm_model', 'pca')
        scaler_file_name = model_file_name.replace('svm_model', 'scaler')

        svm_model_fid = open(os.path.join(models_dir_path, model_file_name), 'rb')
        scaler_fid = open(os.path.join(models_dir_path, scaler_file_name), 'rb')

        cur_pca = None
        if is_pca:
            pca_fid = open(os.path.join(models_dir_path, pca_file_name), 'rb')
            cur_pca = pickle.load(pca_fid)

        cur_scaler = pickle.load(scaler_fid)
        cur_svm_model = pickle.load(svm_model_fid)

        cur_model = Model(cur_svm_model, cur_scaler, cur_pca)
        if 'big' in model_file_name:
            big_models.append(cur_model)
        else:
            small_models.append(cur_model)

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
        if '--output-path' in opts:
            cv2.imwrite(opts['--output-path'], img)
        cv2.waitKey()


def main():

    # parse program options
    models_dir = opts['--models-path']
    image_path = opts['--image-path']

    global need_to_plot

    need_to_plot = True
    is_pca = False

    # load SVM models
    big_models, small_models = get_models(models_dir, is_pca)

    if opts['--check-points-path'] is not None:
        predict_single(image_path, big_models, small_models, opts['--check-points-path'])
    else:
        # predict results
        predict_window(image_path, big_models, small_models, is_pca)


if __name__ == '__main__':
    opts = docopt.docopt(__doc__)
    main()

