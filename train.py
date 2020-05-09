import json
import os
import pickle
import re
import shutil
import sys
from typing import List
from skimage.feature import hog
import cv2
import time
import logging
from sklearn import svm
from sklearn.decomposition import PCA
from ada_boost import train_ada_boost, decision_tree_grid, desiction_tree_ensemble, lda_classifier
from params import *
from pca_train import train_svm, svm_grid
import numpy as np

from utils import get_size_by_y

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def get_weights(X, y) -> List[float]:
    arr_x = np.array(X)
    arr_y = np.array(y)

    positive_count = np.where(1, arr_y).shape[0]
    negative_count = np.where(0, arr_y).shape[0]

    positive_weight = 1
    res: List[float] = []

    pass


def main():
    if len(sys.argv) < 3:
        logging.error('usage: collect_train.py path_to_images_dir path_to_res_dir')
        exit(-1)

    path_to_images_dir = sys.argv[1]
    path_to_res_dir = sys.argv[2]

    ds_count = {'1_2_small': 0, '1_2_big': 0, '2_2_small': 0, '2_2_big': 0, 'big': 0, 'small': 0}
    ds_count_neg = {'small': 0, 'big': 0}
    result_files = {}
    result_array_x = {}
    result_array_y = {}

    for key, val in ds_count.items():
        result_array_x[key] = []
        result_array_y[key] = []

    image_names = os.listdir(path_to_images_dir)
    start = time.time()
    for i, image_name in enumerate(image_names):
        img = cv2.imread(os.path.join(path_to_images_dir, image_name))

        old_heigh = img.shape[0]
        img = img[int(img.shape[0] / 5): img.shape[0], 0: img.shape[1]]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        res_file_name = f'{image_name}.json'

        with open(os.path.join(path_to_res_dir, res_file_name)) as json_file:
            data = json.load(json_file)
            objects = data['objects']
            for j, obj in enumerate(objects):
                is_positive_point = obj['classTitle'] == 'point'
                is_negative_point = obj['classTitle'] == 'negative_point'
                if is_positive_point or is_negative_point:
                    x, y = obj['points']['exterior'][0]
                    y -= (old_heigh - img.shape[0])
                    if y < 0:
                        continue

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
                        continue

                    crop_img = img[y - half_heigh: y + half_heigh, x - half_width: x + half_width]
                    if not is_small_size:
                        crop_img = cv2.resize(crop_img, (big_w, big_h)) if need_to_resize else crop_img
                        pix_per_cell = pixels_per_cell_big
                    else:
                        crop_img = cv2.resize(crop_img, (small_w, small_h)) if need_to_resize else crop_img
                        pix_per_cell = pixels_per_cell_small

                    res = hog(crop_img, pixels_per_cell=pix_per_cell, cells_per_block=cells_per_bloch)

                    has_tag = False
                    if 'tags' in obj and len(obj['tags']) > 0 and obj['tags'][0]['name'] == 'cross_type':
                        cross_type = obj['tags'][0]['value']
                        has_tag = True
                        if cross_type == '2_1_small':
                            cross_type = '1_2_small'
                        elif cross_type == '2_1_big':
                            cross_type = '1_2_big'

                    if is_positive_point and has_tag:
                        # img_flip_lr = cv2.flip(crop_img, 1)
                        # res_flip = hog(img_flip_lr, pixels_per_cell=pix_per_cell, cells_per_block=cells_per_bloch)

                        prefix = '2_2_'
                        if '1_2_' in cross_type:
                            prefix = '1_2_'

                        if is_small_size:
                            postfix = 'small'
                        else:
                            postfix = 'big'

                        cross_type = prefix + postfix

                        # add to full cross_type data
                        cross_types = [cross_type, postfix]
                        for cr_type in cross_types:
                            ds_count[cr_type] += 1
                            result_array_x[cr_type].append(res)
                            result_array_y[cr_type].append(1)
                    elif is_negative_point:
                        if is_small_size:
                            ds_count_neg['small'] += 1
                            cur_cross_types = ['1_2_small', '2_2_small', 'small']
                        else:
                            ds_count_neg['big'] += 1
                            cur_cross_types = ['1_2_big', '2_2_big', 'big']

                        for cross_type in cur_cross_types:
                            result_array_x[cross_type].append(res)
                            result_array_y[cross_type].append(0)

    print(f"time: {time.time() - start}")

    sum = 0
    for key, val in ds_count.items():
        sum += val
        print(f'|{key}|{val}|')

    sum = 0
    for key, val in ds_count_neg.items():
        sum += val
        print(f'|negative_{key}|{val}|')

    model_name = 'svm'
    if os.path.isdir(model_name):
        shutil.rmtree(model_name)
    os.mkdir(model_name)

    for key, clf in result_array_x.items():
        # if key == 'big' or key == 'small':
        #     continue

        svm_grid(result_array_x[key], result_array_y[key], key)


if __name__ == '__main__':
    main()
