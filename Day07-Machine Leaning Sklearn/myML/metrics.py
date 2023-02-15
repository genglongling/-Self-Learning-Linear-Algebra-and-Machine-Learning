def accuracy(y_true, y_pred):
    return sum(y_pred == y_true) / len(y_pred)

