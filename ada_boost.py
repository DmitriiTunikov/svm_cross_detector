import os
import shutil

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import graphviz
import numpy as np
from pca_train import save_results


def lda_classifier(X, y, model_name):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = LDA()
    clf.fit(X_train, Y_train)
    save_results(model_name, clf, model_dir='lda', score=str(clf.score(X_test, Y_test)))


def desiction_tree_ensemble(X, y, model_name):
    clf = RandomForestClassifier(n_estimators=50, max_depth=None,
                                 min_samples_split=2, random_state=0)

    scores = cross_val_score(clf, X, y, cv=5)

    print(f'{model_name}: {scores.mean()}')


def decision_tree_grid(X, Y, model_name):
    param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(3, 15),
                  'class_weight': [{1: 0.5, 0: 0.5}, {1: 0.4, 0: 0.6}, {1: 0.6, 0: 0.4}, {1: 0.7, 0: 0.3}, {1: 0.3, 0: 0.7}]}

    clf = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                       param_grid=param_grid,
                       # scoring='roc_auc',
                       verbose=1,
                       n_jobs=-1,
                       cv=5)

    clf = clf.fit(X, Y)
    save_results(model_name, clf, model_dir='decision_tree', score=clf.best_score_, params=clf.best_params_)
    # draw tree
    # dot_data = tree.export_graphviz(clf, out_file=None,
    #                                 class_names=['not_intersection', 'intersection'],
    #                                 filled=True, rounded=True,
    #                                 special_characters=True)
    #
    # graph = graphviz.Source(dot_data)
    # graph.render(os.path.join('tree', model_name))


def train_ada_boost(X, Y, svm_name):
    # scale data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, Y_train)

    res_file = open(f'train_res/{svm_name}', 'w')
    res_str = f"AdaBoost {svm_name}: " + str(clf.score(X_test, Y_test))
    res_file.write(res_str)
    print(res_str)

    save_results(svm_name, clf, scaler)
