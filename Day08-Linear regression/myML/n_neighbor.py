import numpy as np
from .metrics import accuracy #同包

class KNeighborClassifier(object):
    def __init__(self, n_neighbors=5):
        self.n_neighbors=n_neighbors
        self.X_train=None
        self.y_train = None
    def fit(self, X_train, y_train):
        self.X_train=X_train
        self.y_train=y_train
        return self
    # print激活
    def __str__(self): # 类似java重写toString, C重载运算符
        return "KNeighborClassifier"
    #交互式（返回值）
    def __repr__(self):
        return "KNeighborClassifier"
    def __predict(self, x_single):
        distances = [np.sum((x-x_single)**2)**0.5 for x in self.X_train]
        votes = [self.y_train[i] for i in np.argsort(distances)[:self.n_neighbors]]
        from collections import Counter
        return Counter(votes).most_common(1)[0][0]
    def predict(self, X_test):
        return np.array([self.__predict(v) for v in X_test])
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuracy(y_test,y_pred)