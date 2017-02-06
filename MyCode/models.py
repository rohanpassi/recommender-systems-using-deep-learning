from math import sqrt

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import tree
# from xgboost import XGBRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model, neural_network, neighbors
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


from loadData import loadMovieLens

def RMSE(y_pred, y_test):
	return sqrt(mean_squared_error(y_test, y_pred))

def compute_rating(y_pred):
	for i in range(len(y_pred)):
		val = y_pred[i]
		intPart = int(val)
		fracPart = val - intPart
		if fracPart >= 0.5:
			y_pred[i] = intPart + 1
		else:
			y_pred[i] = intPart
	return y_pred

def mean_abs_error(y_pred, y_test):
	n = len(y_pred)
	diff = 0.0
	for i in range(n):
		diff += abs(y_pred[i] - y_test[i])
	diff = float(diff)
	n = float(n)
	return (diff/n)


x_train, y_train, x_test, y_test =  loadMovieLens()

# MODELS

# MODEL 1 : LINEAR REGRESSION
linear_regression = linear_model.LinearRegression()
linear_regression.fit(x_train, y_train)
y_pred = linear_regression.predict(x_test)
rmse_linear_regression = RMSE(y_pred, y_test)
mae_linear_regression = mean_abs_error(y_pred, y_test)
print("RMSE Linear Regression: %.2f" % rmse_linear_regression)
print("MAE Linear Regression: %.2f" % mae_linear_regression)



# MODEL 2 : LOGISTIC REGRESSION
logistic_regression = linear_model.LogisticRegression()
logistic_regression.fit(x_train, y_train)
y_pred = logistic_regression.predict(x_test)
rmse_logistic_regression = RMSE(y_pred, y_test)
mae_logistic_regression = mean_abs_error(y_pred, y_test)
print("RMSE Logistic Regression: %.2f" % rmse_logistic_regression)
print("MAE Logistic Regression: %.2f" % mae_logistic_regression)



# MODEL 3 : RIDGE REGRESSION
ridge_regression = linear_model.Ridge(alpha=0.5)
ridge_regression.fit(x_test, y_test)
y_pred = ridge_regression.predict(x_test)
rmse_ridge_regression = RMSE(y_pred, y_test)
mae_ridge_regression = mean_abs_error(y_pred, y_test)
print("RMSE Ridge Regression: %.2f" % rmse_ridge_regression)
print("MAE Ridge Regression: %.2f" % mae_ridge_regression)


# MODEL 4 : RIDGE REGRESSION CV
ridge_regression_cv = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
ridge_regression_cv.fit(x_test, y_test)
y_pred = ridge_regression_cv.predict(x_test)
rmse_ridge_regression_cv = RMSE(y_pred, y_test)
mae_ridge_regression_cv = mean_abs_error(y_pred, y_test)
print("RMSE Ridge Regression CV: %.2f" % rmse_ridge_regression_cv)
print("MAE Ridge Regression CV: %.2f" % mae_ridge_regression_cv)


# MODEL 5 : LASSO
lasso = linear_model.Lasso(alpha = 0.1)
lasso.fit(x_train, y_train)
y_pred = lasso.predict(x_test)
rmse_lasso = RMSE(y_pred, y_test)
mae_lasso = mean_abs_error(y_pred, y_test)
print("RMSE Lasso: %.2f" % rmse_lasso)
print("MAE Lasso: %.2f" % mae_lasso)


# MODEL 6 : BAYESIAN RIDGE REGRESSION
bayesian_ridge = linear_model.BayesianRidge()
bayesian_ridge.fit(x_train, y_train)
y_pred = bayesian_ridge.predict(x_test)
rmse_bayesian_ridge = RMSE(y_pred, y_test)
mae_bayesian_ridge = mean_abs_error(y_pred, y_test)
print("RMSE Bayesian Ridge: %.2f" % rmse_bayesian_ridge)
print("MAE Bayesian Ridge: %.2f" % mae_bayesian_ridge)


# MODEL 7 : DECISION TREE REGRESSION
decision_tree_regression = tree.DecisionTreeRegressor()
decision_tree_regression.fit(x_train, y_train)
y_pred = decision_tree_regression.predict(x_test)
rmse_decision_tree = RMSE(y_pred, y_test)
mae_decision_tree = mean_abs_error(y_pred, y_test)
print("RMSE Decision Tree: %.2f" % rmse_decision_tree)
print("MAE Decision Tree: %.2f" % mae_decision_tree)


# MODEL 8 : RANDOM FOREST REGRESSOR
rf_regression = RandomForestRegressor(random_state=0, n_estimators=500)
rf_regression.fit(x_train, y_train)
y_pred_rf = rf_regression.predict(x_test)
rmse_rf = RMSE(y_pred_rf, y_test)
mae_rf = mean_abs_error(y_pred_rf, y_test)
print("RMSE Random Forest: %.2f" % rmse_rf)
print("MAE Random Forest: %.2f" % mae_rf)


# MODEL 9 : XG BOOST REGRESSOR
xgb_regression = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=1000, silent=True, objective='reg:linear', nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, base_score=0.5, seed=0, missing=None)
xgb_regression.fit(x_train, y_train)
y_pred_xgb = xgb_regression.predict(x_test)
rmse_xgb = RMSE(y_pred_xgb, y_test)
mae_xgb = mean_abs_error(y_pred_xgb, y_test)
print("RMSE XGB: %.2f" % rmse_xgb)
print("MAE XGB: %.2f" % mae_xgb)


# MODEL 10 : SVM
svm_regression = svm.SVR()
svm_regression.fit(x_train, y_train)
y_pred = svm_regression.predict(x_test)
rmse_svm = RMSE(y_pred, y_test)
mae_svm = mean_abs_error(y_pred, y_test)
print("RMSE SVM: %.2f" % rmse_svm)
print("MAE SVM: %.2f" % mae_svm)


# MODEL 11 : GRADIENT BOOSTING TREES
gbt_regression = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=3, random_state=0, loss='ls')
gbt_regression.fit(x_train, y_train)
y_pred = gbt_regression.predict(x_test)
rmse_gbt = RMSE(y_pred, y_test)
mae_gbt = mean_abs_error(y_pred, y_test)
print("RMSE GBT: %.2f" % rmse_gbt)
print("MAE GBT: %.2f" % mae_gbt)


# MODEL 12 : SINGLE PERCEPTRON
perceptron_regression = linear_model.Perceptron(penalty='l2', alpha=0.001)
perceptron_regression.fit(x_train, y_train)
y_pred = perceptron_regression.predict(x_test)
rmse_perceptron = RMSE(y_pred, y_test)
mae_perceptron = mean_abs_error(y_pred, y_test)
print("RMSE PERCEPTRON: %.2f" % rmse_perceptron)
print("MAE PERCEPTRON: %.2f" % mae_perceptron)


# MODEL 13 : MULTI_LAYER PERCEPTRON
mlp_regression = neural_network.MLPRegressor(hidden_layer_sizes=(512,256,128,64),max_iter=500, activation='relu')
mlp_regression.fit(x_train, y_train)
y_pred = mlp_regression.predict(x_test)
rmse_mlp = RMSE(y_pred, y_test)
mae_mlp = mean_abs_error(y_pred, y_test)
print("RMSE MLP: %.2f" % rmse_mlp)
print("MAE MLP: %.2f" % mae_mlp)


# MODEL 14 : KNN REGRESSOR
knn_regression = neighbors.KNeighborsRegressor()
knn_regression.fit(x_train, y_train)
y_pred = knn_regression.predict(x_test)
rmse_knn = RMSE(y_pred, y_test)
mae_knn = mean_abs_error(y_pred, y_test)
print("RMSE KNN: %.2f" % rmse_knn)
print("MAE KNN: %.2f" % mae_knn)
