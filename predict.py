"""
SvmPredict PROJECT
Tool for predict rails crosses

Usage:
    predict.py (--models-path=PATH) (--image-path=PATH) (--resize-window=INT) [--output-path=PATH] [--check-points-path=PATH]
    predict.py --help

Options:
    -m, --models-path=PATH              Path to SVM models.
    -i, --image-path=PATH               Path to image to process.
    -o, --output-path=PATH              Path to output image.
    -d, --check-points-path=PATH        Path to file with points to check.
    -r, --resize-window=INT             Need to resize window.
    -h, --help                          Show this message.

Mention:
    Result image will be saved to `--output-path`.
    Example:
        Command line: predict.py --models-path=models --image-path=images/example.jpg --output-path=res.jpg --resize-window=1
"""
import multiprocessing
import os
import re
import sys
import time
from typing import List
from params import *
import cv2
import pickle
from skimage.feature import hog
import docopt

from train import Point, Rect
from utils import get_size_by_y

global need_to_plot


class Model:

    def __init__(self, model, scaler, pca=None):
        self.model = model
        self.pca = pca
        self.scaler = scaler


def predict_in_point(img, x, y, win_w, win_h, pix_per_cell, models) -> (Point, Rect):
    real_win_h, real_win_w = get_size_by_y(img.shape[0], min_win_h, max_win_h, y) if need_to_resize else (win_h, win_w)
    half_w, half_h = int(real_win_w / 2), int(real_win_h / 2)

    if y + half_h > img.shape[0] or x + half_w > img.shape[1] or y - half_h < 0 or x - half_w < 0:
        return None

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
            print(f"({x}, {y})")
            return Point(x, y), Rect(Point(x - half_w, y - half_h), Point(x + half_w, y + half_h))

    return None


def predict_in_heigh(img, models: List[Model], min_h: int, max_h: int, win_w: int, win_h: int, win_step: int, pix_per_cell) -> List:
    x = 0
    while x < img.shape[1]:
        y = min_h
        while y < max_h:
            predict_in_point(img, x, y, win_w, win_h, pix_per_cell, models)
            y += win_step
        x += win_step


def prepare_image(image_path: str):
    img = cv2.imread(image_path)

    # crop 1/5 of image
    img = img[int(img.shape[0] / 5): img.shape[0], 0: img.shape[1]]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img_gray, img


def horizontal_window(img, y, step, width, win_w, win_h, pix_per_cell, models, proc_num, return_dict) -> List[Point]:
    x = 0

    res: List[Point] = []
    while x < width:
        predicted = predict_in_point(img, x, y, win_w, win_h, pix_per_cell, models)
        if predicted is not None:
            res.append(predicted)

        x += step

    return_dict[proc_num] = res
    return res


def draw_result(res_points: List, img):
    for result_point_and_rect in res_points:
        point = result_point_and_rect[0]
        rect = result_point_and_rect[1]

        cv2.circle(img, (point.x, point.y), 1, (0, 255, 0), 2)
        cv2.rectangle(img, (rect.p1.x, rect.p1.y), (rect.p2.x, rect.p2.y), (255, 0, 0))


def predict_window(image_path: str, big_models: List[Model], small_models: List[Model]) -> List:
    res = []
    img, original_img = prepare_image(image_path)

    y = 0
    heigh, width = img.shape
    jobs = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    proc_num = 0
    while y < heigh:
        step = get_size_by_y(heigh, min_window_step, max_window_step, y)[0]

        win_w, win_h, pix_per_cell, models = (small_w, small_h, pixels_per_cell_small, small_models) \
            if y < int(small_heigh_part * heigh) else (big_w, big_h, pixels_per_cell_big, big_models)

        p = multiprocessing.Process(target=horizontal_window, args=(img, y, step, width, win_w, win_h, pix_per_cell, models, proc_num, return_dict))
        p.start()
        jobs.append(p)

        y += step
        proc_num += 1

    for j in jobs:
        j.join()

    for job_return in return_dict.values():
        proc_result = job_return

        draw_result(proc_result, original_img)

    if need_to_plot:
        cv2.imshow("original", original_img)
        # cv2.imshow("crop image", img)
        if '--output-path' in opts:
            cv2.imwrite(opts['--output-path'], original_img)
        # cv2.waitKey()

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

        # if model_file_name != 'model_1_2_small.pkl':
        #     continue
        if model_file_name == 'model_big.pkl' and model_file_name == 'model_small.pkl':
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
    img, original = prepare_image(image_path)

    points = open(path_to_points, 'r')
    lines = points.readlines()
    res_points = []
    for line in lines:
        xy = re.findall('\d+', line)
        x, y = int(xy[0]), int(xy[1])

        cur_range = int(get_size_by_y(img.shape[0], 10, 20, y)[0] / 2)

        for cur_x in range(x - cur_range, x + cur_range):
            for cur_y in range(y - cur_range, y + cur_range):
                if cur_y > 0.2 * img.shape[0]:
                    win_w, win_h, pix_per_cell, models = big_w, big_h, pixels_per_cell_big, big_models
                else:
                    win_w, win_h, pix_per_cell, models = small_w, small_h, pixels_per_cell_small, small_models

                predict_res = predict_in_point(img, cur_x, cur_y, win_w, win_h, pix_per_cell, models)
                if predict_res is not None:
                    res_points.append(predict_res)

    if need_to_plot:
        draw_result(res_points, original)

        cv2.imshow("image", original)

        if opts['--output-path'] is not None:
            cv2.imwrite(opts['--output-path'], original)
        cv2.waitKey()


def main():

    # parse program options
    models_dir = opts['--models-path']
    image_path = opts['--image-path']

    global need_to_resize
    need_to_resize = int(opts['--resize-window']) == 1

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
    start = time.time()
    opts = docopt.docopt(__doc__)
    main()
    print(f"time: {time.time() - start}")

