import os
import sys
from typing import List

from params import *
import cv2
import pickle
from skimage.feature import hog


class Model:

    def __init__(self, model, scaler, pca=None, lda=None):
        self.model = model
        self.pca = pca
        self.lda = lda
        self.scaler = scaler


def test_classifier(model_file_name: str):
    if 'big' in model_file_name:
        width = big_w
        heigh = big_h
        pix_per_cell = pixels_per_cell_big
    else:
        width = small_w
        heigh = small_h
        pix_per_cell = pixels_per_cell_small

    with open(model_file_name, 'rb') as fid:
        clf = pickle.load(fid)

    img = cv2.imread('dataset1/ds/img/img0002.jpg')
    img = img[int(img.shape[0] / 5): img.shape[0], 0: img.shape[1]]

    half_width = int(width / 2)
    half_heigh = int(heigh / 2)
    # x_center = 324
    # y_center = 59
    # crop_img = img[y_center - half_heigh: y_center + half_heigh, x_center - half_width: x_center + half_width]
    # feature = hog(crop_img, pixels_per_cell=pix_per_cell, cells_per_block=cells_per_bloch)
    # predict_res = clf.predict([feature])
    # ds = clf.decision_function([feature])
    # print(ds)
    #
    for x in range(0, img.shape[1] - width, 2):
        for y in range(0, int(small_threshhold * img.shape[0]), 2):
            crop_img = img[y: y + heigh, x: x + width]
            feature = hog(crop_img, pixels_per_cell=pix_per_cell, cells_per_block=cells_per_bloch)
            predict_res = clf.predict([feature])
            if predict_res[0] == 1:
                ds = clf.decision_function([feature])
                print(f'({x},{y}) : {ds[0]}')
                cv2.rectangle(img, (x, y), (x + width, y + heigh), (0, 255, 0))

    cv2.imshow('original', img)

    cv2.waitKey()


def predict_in_heigh(img, models: List[Model], min_h: int, max_h: int, win_w: int, win_h: int, win_step: int, pix_per_cell,
                     is_pca: bool, is_lda: bool) -> List:
    img_w = img.shape[1]

    x = 0
    while x < img_w - win_w:
        y = min_h
        while y < max_h - win_h:
            window = img[y: y + win_h, x: x + win_w]
            feature = hog(window, pixels_per_cell=pix_per_cell, cells_per_block=cells_per_bloch)

            for model_idx, cur_model in enumerate(models):

                scaled_feature = cur_model.scaler.transform([feature])

                feature_reduced = scaled_feature
                # is pca reduced
                if is_pca:
                    feature_reduced = cur_model.pca.transform(scaled_feature)
                # is lda reduced
                elif is_lda:
                    feature_reduced = cur_model.pca.transform(scaled_feature)

                predict_res = cur_model.model.predict(feature_reduced)

                if predict_res[0] == 1:
                    color = 0
                    if model_idx % 2 == 0:
                        color = 255

                    print(f'({x}, {y}): {model_idx % 2}')
                    cv2.rectangle(img, (x, y), (x + win_w, y + win_h), color)
                    break
            y += win_step
        x += win_step


def predict(image_path: str, big_models: List[Model], small_models: List[Model], is_pca: bool, is_lda: bool) -> List:
    res = []
    img = cv2.imread(image_path)

    # crop 1/5 of image
    img = img[int(img.shape[0] / 5): img.shape[0], 0: img.shape[1]]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h = img.shape[0]

    predict_in_heigh(img, small_models, 0, int(small_threshhold * img_h), small_w, small_h, 2, pixels_per_cell_small, is_pca, is_lda)
    predict_in_heigh(img, big_models, int(small_threshhold * img_h) - small_h, img_h, big_w, big_h, 10, pixels_per_cell_big, is_pca, is_lda)

    cv2.imshow("crop image", img)
    cv2.waitKey()
    return res


def run_predictions(path_to_image_dir: str, big_models: List, small_models: List):
    image_names = os.listdir(path_to_image_dir)

    for image_name in image_names:
        res = predict(image_name, big_models, small_models)


def main():
    # test_classifier('svm_model_1_2_small.pkl')
    if len(sys.argv) < 5:
        print("usage: path_to_models_dir path_to_image_dir is_pca is_lda")
        exit(-1)

    models_dir = sys.argv[1]
    is_pca, is_lda = int(sys.argv[3]) == 1, int(sys.argv[4]) == 1
    file_names = os.listdir(models_dir)

    small_models: List[Model] = []
    big_models: List[Model] = []
    for model_file_name in file_names:
        if 'svm_model' not in model_file_name:
            continue

        if model_file_name == 'svm_model_1_2_big.pkl' or model_file_name == 'svm_model_1_2_small.pkl':
            continue

        pca_file_name = model_file_name.replace('svm_model', 'pca')
        lda_file_name = model_file_name.replace('svm_model', 'lda')
        scaler_file_name = model_file_name.replace('svm_model', 'scaler')

        svm_model_fid = open(os.path.join(models_dir, model_file_name), 'rb')
        scaler_fid = open(os.path.join(models_dir, scaler_file_name), 'rb')
        pca_fid = open(os.path.join(models_dir, pca_file_name), 'rb')
        cur_lda = None
        if is_lda:
            lda_fid = open(os.path.join(models_dir, lda_file_name), 'rb')
            cur_lda = pickle.load(lda_fid)

        cur_pca = pickle.load(pca_fid)
        cur_scaler = pickle.load(scaler_fid)
        cur_svm_model = pickle.load(svm_model_fid)

        cur_model = Model(cur_svm_model, cur_scaler, cur_pca, cur_lda)
        if 'big' in model_file_name:
            big_models.append(cur_model)
        else:
            small_models.append(cur_model)

    # run_predictions(path_to_image_dir=sys.argv[5])
    predict('dataset1/ds/img/img0002.jpg', big_models, small_models, is_pca, is_lda)


if __name__ == '__main__':
    main()
