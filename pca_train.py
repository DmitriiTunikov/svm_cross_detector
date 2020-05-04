import pickle

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score


def plot_pca(svm_model, X_test_scaled_reduced, Y_test, classify, svm_name):
    def plot_contours(ax, clf, xx, yy, **params):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    def make_meshgrid(x, y, h=.1):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))  # ,
        # np.arange(z_min, z_max, h))
        return xx, yy

    X0, X1 = X_test_scaled_reduced[:, 0], X_test_scaled_reduced[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor('white')
    cdict1 = {0: 'lime', 1: 'deeppink'}

    yl1 = [int(target1) for target1 in Y_test]
    labels1 = yl1

    labl1 = {0: 'NOT_INTERSECTION', 1: 'INTERSECTION'}
    marker1 = {0: '*', 1: 'd'}
    alpha1 = {0: .8, 1: 0.5}

    for l1 in np.unique(labels1):
        ix1 = np.where(labels1 == l1)
        ax.scatter(X0[ix1], X1[ix1], c=cdict1[l1], label=labl1[l1], s=70, marker=marker1[l1], alpha=alpha1[l1])

    ax.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], s=40, facecolors='none',
               edgecolors='navy', label='Support Vectors')

    plot_contours(ax, classify, xx, yy, cmap='seismic', alpha=0.4)
    plt.legend(fontsize=15)
    plt.title(svm_name)
    plt.xlabel("1st Principal Component", fontsize=14)
    plt.ylabel("2nd Principal Component", fontsize=14)

    plt.savefig(f'models/{svm_name}.png', dpi=300)
    plt.show()


def cross_valid(X, Y):
    clf = SVC(kernel='rbf', C=1)
    scores = cross_val_score(clf, X, Y, cv=5)
    print(f'cross validation scored: {scores}')
    return clf


def save_results(svm_name, clf, scaler, pca=None):

    with open(f'models/svm_model_{svm_name}.pkl', 'wb') as fid:
        pickle.dump(clf, fid)

    if pca is not None:
        with open(f'models/pca_{svm_name}.pkl', 'wb') as fid:
            pickle.dump(pca1, fid)

    with open(f'models/scaler_{svm_name}.pkl', 'wb') as fid:
        pickle.dump(scaler, fid)


def find_best_params(X_train, y_train):
    C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    # Gamma = [1, 0.1, 0.01, 0.001, 0.00001, 10]
    class_weight = [{1: 0.5, 0: 0.5}, {1: 0.4, 0: 0.6}, {1: 0.6, 0: 0.4}, {1: 0.7, 0: 0.3}]
    # class_weight = [{2: 0.25, 1: 0.25, 0: 0.5}, {2: 0.2, 1: 0.2, 0: 0.6}, {2: 0.3, 1: 0.3, 0: 0.4}, {2: 0.35, 1: 0.35, 0: 0.3}]

    param_grid = dict(C=C,
                      class_weight=class_weight)
                      # gamma=Gamma)

    grid = GridSearchCV(estimator=SVC(),
                        param_grid=param_grid,
                        scoring='roc_auc',
                        verbose=1,
                        n_jobs=-1)

    grid_result = grid.fit(X_train, y_train)

    print('Best Score: ', grid_result.best_score_)
    print('Best Params: ', grid_result.best_params_)

    return grid


def train_and_test(X, Y, svm_name, multi_class: bool):
    if multi_class:
        from sklearn import preprocessing
        Y = preprocessing.label_binarize(Y, classes=[0, 1, 2])

    print(svm_name + ":")

    # scale data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    clf = find_best_params(X_train, Y_train)
    # clf = cross_valid(X, Y)

    save_results(svm_name, clf, scaler, None)
    return

    is_pca = True

    # train svm
    classify, pca = train_svm(X_train, Y_train, svm_name, is_pca)

    # prepare test data
    scaled_feature = scaler.transform(X_test)
    feature_reduced = scaled_feature
    if is_pca:
        feature_reduced = pca.transform(scaled_feature)
    predicted = classify.predict(feature_reduced)

    # test and output results
    print(f"score: {classify.score(feature_reduced, Y_test)}")
    print(f"precision_score(not to label as positive a sample that is negative): {precision_score(Y_test, predicted)}")

    save_results(svm_name, classify, scaler, pca)
    return


def train_svm(X_train, Y_train, svm_name, is_pca: bool):
    global components_count, pca1

    components_count, pca1 = None, None
    X_train_scaled_reduced = X_train
    if is_pca:
        components_count = 2
        pca1 = PCA(n_components=components_count)
        X_train_scaled_reduced = pca1.fit_transform(X_train)

    svm_model = SVC(kernel='rbf')

    classify = svm_model.fit(X_train_scaled_reduced, Y_train)

    if is_pca and components_count == 2:
        plot_pca(svm_model, X_train_scaled_reduced, Y_train, classify, svm_name)

    return classify, pca1
