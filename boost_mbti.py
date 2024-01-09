import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import validation_curve, LearningCurveDisplay, GridSearchCV
from sklearn import ensemble, tree, metrics
from mbti import load_and_preprocess_data
import time

X_train, X_test, y_train, y_test = load_and_preprocess_data(norm=False)
# print(X_train, y_train)

# tuning
param_range = range(10, 400, 10)
train_scores, test_scores = validation_curve(
    ensemble.AdaBoostClassifier(),
    X_train,
    y_train,
    param_name="n_estimators",
    param_range=param_range,
    scoring="balanced_accuracy",
    cv=10,
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve for n_estimators with AdaBoost - Posture")
plt.xlabel("n_estimators")
plt.ylabel("Balanced Accuracy")
lw = 2
plt.plot(
    param_range, train_scores_mean, label="Training accuracy (balanced)", color="darkorange", lw=lw
)
plt.fill_between(
    param_range,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2,
    color="darkorange",
    lw=lw,
)
plt.plot(
    param_range, test_scores_mean, label="Cross-validation accuracy (balanced)", color="navy", lw=lw
)
plt.fill_between(
    param_range,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.2,
    color="navy",
    lw=lw,
)
plt.legend(loc="best")
plt.savefig('./images/boost_n_estimators_mbti.png')
plt.clf()

estimator_max_depth = range(1, 15)
estimators = [tree.DecisionTreeClassifier(max_depth=d) for d in estimator_max_depth]
train_scores, test_scores = validation_curve(
    ensemble.AdaBoostClassifier(),
    X_train,
    y_train,
    param_name="estimator",
    param_range=estimators,
    scoring="balanced_accuracy",
    cv=10,
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve for estimator max_depth with AdaBoost - Posture")
plt.xlabel("estimator max_depth")
plt.ylabel("Balanced Accuracy")
lw = 2
plt.plot(
    estimator_max_depth, train_scores_mean, label="Training accuracy (balanced)", color="darkorange", lw=lw
)
plt.fill_between(
    estimator_max_depth,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2,
    color="darkorange",
    lw=lw,
)
plt.plot(
    estimator_max_depth, test_scores_mean, label="Cross-validation accuracy (balanced)", color="navy", lw=lw
)
plt.fill_between(
    estimator_max_depth,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.2,
    color="navy",
    lw=lw,
)
plt.legend(loc="best")
plt.savefig('./images/boost_estimator_max_depth_mbti.png')
plt.clf()

# learning curve
boost = ensemble.AdaBoostClassifier()
param_dict = {
    "n_estimators": param_range,
    "estimator": estimators
}
grid = GridSearchCV(boost, param_dict, cv=10, scoring="accuracy")
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_estimator_)
print(grid.best_score_)

LearningCurveDisplay.from_estimator(grid.best_estimator_,
                                    X=X_train,
                                    y=y_train,
                                    cv=10,
                                    score_type='both',
                                    score_name='balanced_accuracy',
                                    train_sizes=np.linspace(0.1, 1.0, 10))
plt.legend(["Training Accuracy (balanced)", "Validation Accuracy (balanced)"])
plt.title('Learning curve for AdaBoost - Posture Prediction')
plt.ylabel("Accuracy")
plt.savefig('./images/boost_learning_curve_mbti.png')
plt.clf()

# final boost
boost_fin = ensemble.AdaBoostClassifier(estimator=tree.DecisionTreeClassifier(max_depth=1),
                   n_estimators=350)
train_start = time.time()
boost_fin.fit(X_train, y_train)
train_stop = time.time()
train_time = train_stop - train_start
print(f"Training time: {train_time}s")
train_pred = boost_fin.predict(X_train)
train_accuracy = metrics.balanced_accuracy_score(y_train, train_pred)
print(f"Training accuracy: {train_accuracy}s")
query_start = time.time()
test_pred = boost_fin.predict(X_test)
test_accuracy = metrics.balanced_accuracy_score(y_test, test_pred)
query_stop = time.time()
query_time = query_stop - query_start
print(f"Query time: {query_time}s")
print(f"Test accuracy: {test_accuracy}")
