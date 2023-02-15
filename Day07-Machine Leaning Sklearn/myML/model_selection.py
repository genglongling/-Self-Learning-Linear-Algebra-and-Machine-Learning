import numpy as np
def train_test_split(X, y, test_radio=0.2,random_state=None):
    if random_state:
        np.random.seed(random_state)
    num_total=len(X)
    num_train=int((1-test_radio)*num_total)
    ids = np.random.permutation(num_total)
    train_ids = ids[0:num_train]
    test_ids = ids[num_train:num_total]
    # randomize, split
    X_train=X[train_ids]
    X_test=X[test_ids]
    y_train=y[train_ids]
    y_test=y[test_ids]
    return X_train, X_test, y_train, y_test # (120,4) (30,4) (120,) (30,)
#起始?