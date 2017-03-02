# Run some recommendation experiments using MovieLens 100K
import numpy as np
import scipy as sp
import pandas as pd
from math import sqrt
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def RMSE(y_pred, y_test):
    return sqrt(mean_squared_error(y_test, y_pred))

data_dir = "../ml-100k/"
data_shape = (943, 1682)

df = pd.read_csv(data_dir + "u1.base", sep="\t", header=-1)
values = df.values
values[:, 0:2] -= 1

X_train = sp.sparse.csr_matrix((values[:, 2], (values[:, 0], values[:, 1])), dtype=np.float, shape=data_shape)

df = pd.read_csv(data_dir + "u1.test", sep="\t", header=-1)
values = df.values
values[:, 0:2] -= 1
X_test = sp.sparse.csr_matrix((values[:, 2], (values[:, 0], values[:, 1])), dtype=np.float, shape=data_shape)

# Compute means of nonzero elements
X_row_mean = np.zeros(data_shape[0])
X_row_sum = np.zeros(data_shape[0])

train_rows, train_cols = X_train.nonzero()

# Iterate through nonzero elements to compute sums and counts of rows elements
for i in range(train_rows.shape[0]):
    X_row_mean[train_rows[i]] += X_train[train_rows[i], train_cols[i]]
    X_row_sum[train_rows[i]] += 1


# Note that (X_row_sum == 0) is required to prevent divide by zero
X_row_mean /= X_row_sum + (X_row_sum == 0)

# Subtract mean rating for each user
for i in range(train_rows.shape[0]):
    X_train[train_rows[i], train_cols[i]] -= X_row_mean[train_rows[i]]

test_rows, test_cols = X_test.nonzero()
for i in range(test_rows.shape[0]):
    X_test[test_rows[i], test_cols[i]] -= X_row_mean[test_rows[i]]

X_train = np.array(X_train.toarray())
X_test = np.array(X_test.toarray())

ks = np.arange(2, 49)

# MAE
train_mae = np.zeros(ks.shape[0])
test_mae = np.zeros(ks.shape[0])

# RMSE
train_rmse = np.zeros(ks.shape[0])
test_rmse = np.zeros(ks.shape[0])

train_scores = X_train[(train_rows, train_cols)]
test_scores = X_test[(test_rows, test_cols)]

# Now take SVD of X_train
U, s, Vt = np.linalg.svd(X_train, full_matrices=False)

for j, k in enumerate(ks):
    X_pred = U[:, 0:k].dot(np.diag(s[0:k])).dot(Vt[0:k, :])

    pred_train_scores = X_pred[(train_rows, train_cols)]
    pred_test_scores = X_pred[(test_rows, test_cols)]


    train_mae[j] = mean_absolute_error(train_scores, pred_train_scores)
    test_mae[j] = mean_absolute_error(test_scores, pred_test_scores)

    train_rmse[j] = RMSE(train_scores, pred_train_scores)
    test_rmse[j] = RMSE(test_scores, pred_test_scores)
    print k

    if k==48:
        for i in range(pred_test_scores.shape[0]):
            print(test_scores[i])

    # print(k,  train_rmse[j], test_rmse[j])

plt.plot(ks, train_rmse, 'k', label="Train")
plt.plot(ks, test_rmse, 'r', label="Test")
plt.xlabel("k")
plt.ylabel("RMSE")
plt.legend()
plt.show()