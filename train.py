import json
import os
import pickle
import re
import sys
from typing import List
from skimage.feature import hog
import cv2
import time
import logging
from sklearn import svm
from sklearn.decomposition import PCA
from params import *
from pca_train import train_svm, train_and_test


def main():
    if len(sys.argv) < 3:
        logging.error('usage: collect_train.py path_to_images_dir path_to_res_dir')
        exit(-1)

    path_to_images_dir = sys.argv[1]
    path_to_res_dir = sys.argv[2]

    ds_count = {'1_2_small': 0, '1_2_big': 0, '2_2_small': 0, '2_2_big': 0}
    ds_count_neg = {'small': 0, 'big': 0}
    result_files = {}
    result_array_x = {}
    result_array_y = {}
    clfrs = {}

    for key, val in ds_count.items():
        result_array_x[key] = []
        result_array_y[key] = []
        clfrs[key] = svm.SVC()

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

                    if y > small_threshhold * img.shape[1]:
                        is_small_size = False
                    else:
                        is_small_size = True

                    if is_small_size:
                        half_width = int(small_w / 2)
                        half_heigh = int(small_h / 2)
                        pix_per_cell = pixels_per_cell_small
                    else:
                        half_width = int(big_w / 2)
                        half_heigh = int(big_h / 2)
                        pix_per_cell = pixels_per_cell_big

                    if y + half_heigh > img.shape[0] or x + half_width > img.shape[1]:
                        continue

                    crop_img = img[y - half_heigh: y + half_heigh, x - half_width: x + half_width]
                    res = hog(crop_img, pixels_per_cell=pix_per_cell, cells_per_block=cells_per_bloch)

                    if is_positive_point:
                        res_as_str = '+1 '
                    else:
                        res_as_str = '-1 '

                    for feature_idx, feature_val in enumerate(res):
                        res_as_str += f'{feature_idx + 1}:{"%.3f" % feature_val} '

                    has_tag = False
                    if 'tags' in obj and len(obj['tags']) > 0 and obj['tags'][0]['name'] == 'cross_type':
                        cross_type = obj['tags'][0]['value']
                        has_tag = True
                        if cross_type == '2_1_small':
                            cross_type = '1_2_small'
                        elif cross_type == '2_1_big':
                            cross_type = '1_2_big'
                        ds_count[cross_type] += 1

                    if is_positive_point and has_tag:

                        prefix = '2_2_'
                        if '1_2_' in cross_type:
                            prefix = '1_2_'

                        postfix = 'big'
                        if is_small_size:
                            postfix = 'small'
                        cross_type = prefix + postfix
                        result_array_x[cross_type].append(res)
                        result_array_y[cross_type].append(1)
                    elif is_negative_point:
                        if is_small_size:
                            ds_count_neg['small'] += 1
                            cur_cross_types = ['1_2_small', '2_2_small']
                        else:
                            # if ds_count_neg['small'] > 86:
                            #     continue
                            ds_count_neg['big'] += 1
                            cur_cross_types = ['1_2_big', '2_2_big']

                        for cross_type in cur_cross_types:
                            result_array_x[cross_type].append(res)
                            result_array_y[cross_type].append(0)

    print(f"time: {time.time() - start}")
    sum = 0
    for key, val in ds_count.items():
        sum += val
        print(f'{key}: {val}')
    print('positive sum: ' + str(sum))

    sum = 0
    for key, val in ds_count_neg.items():
        sum += val
        print(f'negative_{key}: {val}')
    print('nagative sum: ' + str(sum))

    for key, clf in clfrs.items():
        weights = []

        for i, res_elem in enumerate(result_array_x[key]):
            if result_array_y[key][i] == 1:
                weights.append(1)
            else:
                weights.append(1)

        train_and_test(result_array_x[key], result_array_y[key], weights, key)


if __name__ == '__main__':
    main()
