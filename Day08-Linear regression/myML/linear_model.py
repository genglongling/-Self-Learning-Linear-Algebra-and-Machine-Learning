#简单线性回归
import numpy as np
from .metrics import r2

class SimpleLinearRegression(object):
    def __init__(self):#无超参数
        #模型参数
        #_表示导出
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        # compute a and b
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        num = 0
        den = 0
        # 2.求a
        for x_i, y_i in zip(x_train, y_train):
            num += (x_i - x_mean) * (y_i - y_mean)
            den += (x_i - x_mean) * (x_i - x_mean)
        a = num / den # divided 0 error
        b = y_mean - a * x_mean
        self.a_ = a
        self.b_ = b
        return self # repr/str

    # print激活
    def __str__(self):  # 类似java重写toString, C重载运算符
        return "SimpleLinearRegression"

    # 交互式（返回值）
    def __repr__(self):
        return "SimpleLinearRegression"

    def __predict(self, x_single):
        # predict y, given a, b, x
        return self.a_*x_single+self.b_
        # what if no a and b

    def predict(self, X_test):
        return np.array([self.__predict(v) for v in X_test])

    def score(self, X_test, y_test):
        # 使用R2衡量
        y_pred = self.predict(X_test)
        return r2(y_test,y_pred)

class LinearRegression(object):
    # theta, Xb
    def __init__(self):#无超参数
        #模型参数
        #私有变量
        self.__theta = None
        self.coef_ = None #对外提供系数 theta1,2,...
        self.interception_=None#提供截距 theta0

    # 公式
    def fit_normal(self, X_train, y_train):
        # Xb
        Xb=np.hstack([np.ones((X_train.shape[0],1)),X_train])
        # 求theta
        self.__theta=np.linalg.inv(Xb.T.dot(Xb)).dot(Xb.T).dot(y_train)
        self.coef_=self.__theta[1:]
        self.interception_=self.__theta[0]
        return self # repr/str

    # print激活
    def __str__(self):  # 类似java重写toString, C重载运算符
        return "LinearRegression"

    # 交互式（返回值）
    def __repr__(self):
        return "LinearRegression"

    def __predict(self, x_single):
        pass

    def predict(self, X_test):
        #测试
        Xb=np.hstack([np.ones((X_test.shape[0],1)),X_test])
        return Xb.dot(self.__theta)

    def score(self, X_test, y_test):
        # 使用R2衡量
        y_pred = self.predict(X_test)
        return r2(y_test, y_pred)

