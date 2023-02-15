import numpy as np
class StandardScaler(object):
    # contruct: Top down approach
    def __init__(self):
        self.mean_ = None #sklearn风格
        self.scale_ = None
    def fit(self, X):
        self.mean_ = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:, i]) for i in range(X.shape[1])])
        # 计算mean, std
    def transform(self, X):
        """return the transformed results"""
        res = np.empty(X.shape, dtype=float) #创建空矩阵
        for i in range(X.shape[1]):
            res[:, i] = (X[:, i]-self.mean_[i])/self.scale_[i]
        return res