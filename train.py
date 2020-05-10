import json
import multiprocessing
import os
import shutil
import sys
from typing import List
from skimage.feature import hog
import cv2
import time
from params import *
from pca_train import svm_grid
import logging
from utils import get_size_by_y


class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y


class Rect:

    def __init__(self, p1: Point, p2: Point):
        self.p1 = p1
        self.p2 = p2


class FeaturePoint:

    def __init__(self, feature, is_small_size, win_w, win_h, x, y):
        self.feature = feature
        self.is_small_size = is_small_size
        self.win_w = win_w
        self.win_h = win_h
        self.x = x
        self.y = y


def get_hog_in_point(x: int, y: int, img) -> FeaturePoint:
    if y > small_heigh_part * img.shape[0]:
        is_small_size = False
    else:
        is_small_size = True

    if need_to_resize:
        win_h, win_w = get_size_by_y(img.shape[0], min_win_h, max_win_h, y)
    else:
        win_h, win_w = (small_h, small_w) if is_small_size else (big_h, big_w)

    half_width = int(win_w / 2)
    half_heigh = int(win_h / 2)
    if y + half_heigh > img.shape[0] or x + half_width > img.shape[1] \
            or y - half_heigh < 0 or x - half_width < 0:
        return None

    crop_img = img[y - half_heigh: y + half_heigh, x - half_width: x + half_width]
    if not is_small_size:
        crop_img = cv2.resize(crop_img, (big_w, big_h)) if need_to_resize else crop_img
        pix_per_cell = pixels_per_cell_big
    else:
        crop_img = cv2.resize(crop_img, (small_w, small_h)) if need_to_resize else crop_img
        pix_per_cell = pixels_per_cell_small

    return FeaturePoint(hog(crop_img, pixels_per_cell=pix_per_cell, cells_per_block=cells_per_bloch), is_small_size,
                        win_w, win_h, x, y)


def get_cross_type(obj: dict):
    cross_type = ''
    if 'tags' in obj and len(obj['tags']) > 0 and obj['tags'][0]['name'] == 'cross_type':
        cross_type = obj['tags'][0]['value']
        if cross_type == '2_1_small':
            cross_type = '1_2_small'
        elif cross_type == '2_1_big':
            cross_type = '1_2_big'

    return cross_type


def add_positive_point_to_result(point: FeaturePoint, cross_type: str, result_array_x: List, result_array_y: List,
                                 ds_count: dict):
    if point is None:
        return

    prefix = '2_2_'
    if '1_2_' in cross_type:
        prefix = '1_2_'

    if point.is_small_size:
        postfix = 'small'
    else:
        postfix = 'big'

    cross_type = prefix + postfix

    # add to full cross_type data
    cross_types = [cross_type, postfix]
    for cr_type in cross_types:
        ds_count[cr_type] += 1
        result_array_x[cr_type].append(point.feature)
        result_array_y[cr_type].append(1)


def add_negative_point_to_result(point: FeaturePoint, ds_count_neg: dict, result_array_x, result_array_y):
    if point is None:
        return

    str_size = 'big'
    if point.is_small_size:
        str_size = 'small'

    cross_types = [str_size, f'1_2_{str_size}', f'2_2_{str_size}']
    for cr_type in cross_types:
        ds_count_neg[str_size] += 1
        result_array_x[cr_type].append(point.feature)
        result_array_y[cr_type].append(0)


def multiply_positive_points(x: int, y: int, img, min_range, max_range) -> (List[FeaturePoint], int):
    res = [get_hog_in_point(x, y, img)]

    positive_range = get_size_by_y(img.shape[0], min_range, max_range, y)[0]
    half_range = int(positive_range / 2)

    for cur_x in range(x - half_range, x + half_range):
        for cur_y in range(y - half_range, y + half_range):
            if cur_x == x and cur_y == y:
                continue

            feature = get_hog_in_point(cur_x, cur_y, img)
            if res is not None:
                res.append(feature)

    return res, half_range


def contains_in_positive_points(x, y, positive_points: List[Rect]):
    for p in positive_points:
        if p.p1.x <= x <= p.p2.x and p.p1.y <= y <= p.p2.y:
            return True

    return False


def generate_negative_points(img, positive_points: List[Rect]) -> List[FeaturePoint]:
    res: List[FeaturePoint] = []
    img_heigh, img_width = img.shape

    # y = 0
    # while y < img_heigh:
    #     step = get_size_by_y(img_heigh, min_negative_points_step, max_negative_points_step, y)[0]
    #
    #     x = 0
    #     while x < img_width:
    #         if not contains_in_positive_points(x, y, positive_points):
    #             feature = get_hog_in_point(x, y, img)
    #             if feature is not None:
    #                 res.append(feature)
    #         x += step
    #     y += step
    x = 0

    while x < img_width:
        y = 0
        while y < img_heigh:
            step = get_size_by_y(img_heigh, min_negative_points_step, max_negative_points_step, y)[0]
            if not contains_in_positive_points(x, y, positive_points):
                feature = get_hog_in_point(x, y, img)
                if feature is not None:
                    res.append(feature)
            y += step
        x += x_step

    return res


def train_on_image(proc_num: int, path_to_images_dir, image_name, path_to_res_dir, return_dict):
    ds_count, ds_count_neg, result_array_x, result_array_y = get_result_arrays()
    img = cv2.imread(os.path.join(path_to_images_dir, image_name))

    old_heigh = img.shape[0]
    img = img[int(img.shape[0] / 5): img.shape[0], 0: img.shape[1]]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    res_file_name = f'{image_name}.json'
    global_positive_points: List[Rect] = []
    has_objects = False
    with open(os.path.join(path_to_res_dir, res_file_name)) as json_file:
        data = json.load(json_file)
        objects = data['objects']
        for j, obj in enumerate(objects):

            is_positive_point = obj['classTitle'] == 'point'
            is_negative_point = obj['classTitle'] == 'negative_point'

            if not is_positive_point and not is_negative_point:
                continue

            has_objects = True
            x, y = obj['points']['exterior'][0]
            y -= (old_heigh - img.shape[0])
            if y < 0:
                continue

            if is_positive_point:
                cross_type = get_cross_type(obj)
                if cross_type == '':
                    continue

                positive_points, pos_points_half_range = multiply_positive_points(x, y, img, min_positive_range, max_positive_range)
                global_positive_points.append(Rect(Point(x - pos_points_half_range, y - pos_points_half_range),
                                                   Point(x + pos_points_half_range, y + pos_points_half_range)))
                for pos_p in positive_points:
                    add_positive_point_to_result(pos_p, cross_type, result_array_x, result_array_y, ds_count)
            else:
                feature = get_hog_in_point(x, y, img)
                add_negative_point_to_result(feature, ds_count_neg, result_array_x, result_array_y)

    if has_objects:
        negative_features = generate_negative_points(img, global_positive_points)
        for feature in negative_features:
            add_negative_point_to_result(feature, ds_count_neg, result_array_x, result_array_y)

    return_dict[proc_num] = ds_count, ds_count_neg, result_array_x, result_array_y


def get_result_arrays():
    ds_count = {'1_2_small': 0, '1_2_big': 0, '2_2_small': 0, '2_2_big': 0, 'big': 0, 'small': 0}
    ds_count_neg = {'small': 0, 'big': 0}
    result_array_x = {}
    result_array_y = {}
    for key, val in ds_count.items():
        result_array_x[key] = []
        result_array_y[key] = []

    return ds_count, ds_count_neg, result_array_x, result_array_y


def train(path_to_images_dir: str, path_to_res_dir: str):
    global need_to_resize

    if need_to_resize:
        log_file = 'resize.log'
    else:
        log_file = 'not_resize.log'

    logging.basicConfig(filename=log_file, filemode='w', level=logging.DEBUG, format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    ds_count, ds_count_neg, result_array_x, result_array_y = get_result_arrays()

    image_names = os.listdir(path_to_images_dir)
    start = time.time()
    jobs = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    for i, image_name in enumerate(image_names):
        # train_on_image(i, path_to_images_dir, image_name, path_to_res_dir, return_dict)
        p = multiprocessing.Process(target=train_on_image, args=(i, path_to_images_dir, image_name, path_to_res_dir, return_dict))
        p.start()
        jobs.append(p)

    for j in jobs:
        j.join()

    for job_return in return_dict.values():
        cur_ds_count = job_return[0]
        cur_ds_count_neg = job_return[1]
        cur_res_x = job_return[2]
        cur_res_y = job_return[3]
        for key, value in cur_ds_count.items():
            ds_count[key] += value
        for key, value in cur_ds_count_neg.items():
            ds_count_neg[key] += value
        for key, value in cur_res_x.items():
            result_array_x[key].extend(value)
        for key, value in cur_res_y.items():
            result_array_y[key].extend(value)

    print(f"time: {time.time() - start}")
    sum = 0
    for key, val in ds_count.items():
        sum += val
        logging.info(f'|{key}|{val}|')

    sum = 0
    for key, val in ds_count_neg.items():
        sum += val
        logging.info(f'|negative_{key}|{val}|')

    model_name = 'svm_not_resize' if not need_to_resize else 'svm_resize'
    if os.path.isdir(model_name):
        shutil.rmtree(model_name)
    os.mkdir(model_name)

    start = time.time()
    for key, clf in result_array_x.items():
        start_key = time.time()
        if key == 'big' or key == 'small':
            continue
        svm_grid(result_array_x[key], result_array_y[key], key, model_name)
        print(f'{key} time: {time.time() - start_key}')

    logging.info(f"svm model building time: {time.time() - start}")


def main():
    if len(sys.argv) < 3:
        logging.error('usage: collect_train.py path_to_images_dir path_to_res_dir')
        exit(-1)

    train(sys.argv[1], sys.argv[2])


# if __name__ == '__main__':
#     main()

if __name__ == '__main__':
    global need_to_resize

    # need_to_resize = False
    # train(sys.argv[1], sys.argv[2])
    need_to_resize = True
    train(sys.argv[1], sys.argv[2])

    # need_to_resize = True
    # train(sys.argv[1], sys.argv[2])
