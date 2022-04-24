from sklearn.feature_selection import f_regression
from sklearn import datasets
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, RFECV
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPClassifier

global X_train, X_test, y_train, y_test


def splitlist(list):
    list1 = []
    list2 = []
    for i in list:
        list1.append(i[0])
        list2.append(i[1])
    return list1, list2


def merge_shuffle(list1, list2):
    list = []
    for i in range(len(list1)):
        list.append(list1[i])
        list.append(list2[i])
    # random.shuffle(list)
    return list


def readfile(file):
    f = open(file, "r")
    result = []
    c = 0
    for i in f:
        if c <= 10000:
            # print(c)
            c += 1
            # print(c)
            list = []
            line = i[1:-3].split(", [")
            # print(line)
            list.append(int(line[0]))
            temp = []
            # print(list)
            for j in line[1].split(", "):
                if "." not in j:
                    temp.append(int(j))
                else:
                    temp.append(float(j))
            # print(temp)
            list.append(temp)
            # print("test")
            result.append(list)
    f.close()
    return result


def selectfeatures(features, indexs):
    result = []
    length = len(features[0])
    for w in features:
        temp = []
        for i in range(1, len(w)+1):
            if i in indexs:
                temp.append(w[i-1])
        result.append(temp)
    return result


def Scale():
    global X_train, X_test
    X_train_svm, X_test_svm = X_train, X_test
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_svm = scaler.transform(X_train)
    X_test_svm = scaler.transform(X_test)
    return X_train_svm, X_test_svm


def Kbest(X_train_svm, X_test_svm, n):
    sel = SelectKBest(chi2, k=n)
    sel.fit(X_train_svm, y_train)
    # print(sel.scores_)
    X_train_svm = sel.transform(X_train_svm)
    X_test_svm = sel.transform(X_test_svm)
    return X_train_svm, X_test_svm


idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
       28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
mal = splitlist(readfile("malfeatures.txt"))
beg = splitlist(readfile("begfeatures.txt"))
features = merge_shuffle(mal[1], beg[1])
labels = merge_shuffle(mal[0], beg[0])
features = np.array(features)
var_thr = VarianceThreshold(threshold=0.0)
features = var_thr.fit_transform(features)
rfa = 0
svma = 0
rfmrmr = []
svmmrmr = []

X = pd.DataFrame(features)
y = pd.Series(labels)


F = pd.Series(f_regression(X, y)[0], index=X.columns)
corr = pd.DataFrame(.00001, index=X.columns, columns=X.columns)
for i in tqdm(range(25, len(X.columns)+1)):
    selected = []
    not_selected = X.columns.to_list()
    for i in range(i):
        if i > 0:
            last_selected = selected[-1]
            corr.loc[not_selected, last_selected] = X[not_selected].corrwith(
                X[last_selected]).abs().clip(.00001)
        score = F.loc[not_selected] / corr.loc[not_selected,
                                               selected].mean(axis=1).fillna(.00001)
        best = score.index[score.argmax()]
        selected.append(best)
        not_selected.remove(best)

    idx = [i+1 for i in selected]
    f = selectfeatures(features, idx)

    X_train, X_test, y_train, y_test = train_test_split(
        f, labels, test_size=0.3, random_state=109)
    X_train = np.array(X_train)
    print(X_train.shape)

    X1 = Scale()
    X_train_svm = X1[0]
    X_test_svm = X1[1]

    classifier = MLPClassifier(hidden_layer_sizes=(
        150, 100, 50), max_iter=300, activation='relu', solver='adam', random_state=1)
    print('Training..')
    classifier.fit(X_train_svm, y_train)
    y_pred = classifier.predict(X_test_svm)
    print("Accuracy MLP: ", metrics.accuracy_score(y_test, y_pred))
    temp = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy MLP with ", i+1, "features : ",
          temp)
    if temp > rfa:
        rfa = temp
        rfmrmr = i+1

    """
    print("traning...")
    rf_w = RandomForestClassifier(random_state=100, n_estimators=100)
    rf_w.fit(X_train, y_train)
    y_pred_rf_w = rf_w.predict(X_test)
    temp = metrics.accuracy_score(y_test, y_pred_rf_w)
    print("Accuracy Random Forest with ", i+1, "features : ",
          temp)
    if temp > rfa:
        rfa = temp
        rfmrmr = i+1

    X1 = Scale()
    X_train_svm = X1[0]
    X_test_svm = X1[1]

    clf = svm.SVC(kernel='rbf', gamma=0.7, C=200)
    print("traning SVM...")
    clf.fit(X_train_svm, y_train)
    y_pred = clf.predict(X_test_svm)
    temp = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy linear SVM with ", i+1, "features : ",
          temp)
    if temp > svma:
        svma = temp
        svmmrmr = i+1
    """
print()
print()
print("Best MLP acc with ", rfmrmr, "features : ", rfa)
#print("Random Forest with ", rfmrmr, "features : ", rfa)
#print("SVM with ", svmmrmr, "features : ", svma)
