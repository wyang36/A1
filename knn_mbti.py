def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import matplotlib.pyplot as plt
import graphviz
import numpy as np
from sklearn.model_selection import validation_curve, LearningCurveDisplay, GridSearchCV
from sklearn import neighbors, metrics
from mbti import load_and_preprocess_data
import time

X_train, X_test, y_train, y_test = load_and_preprocess_data()

# tuning
k_range = range(1, 25)
train_scores, test_scores = validation_curve(
    neighbors.KNeighborsClassifier(),
    X_train,
    y_train,
    param_name="n_neighbors",
    param_range=k_range,
    scoring="balanced_accuracy",
    cv=10,
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve for n_neighbors with KNN - Posture")
plt.xlabel("n_neighbors")
plt.ylabel("Balanced Accuracy")
lw = 2
plt.plot(
    k_range, train_scores_mean, label="Training accuracy (balanced)", color="darkorange", lw=lw
)
plt.fill_between(
    k_range,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2,
    color="darkorange",
    lw=lw,
)
plt.plot(
    k_range, test_scores_mean, label="Cross-validation accuracy (balanced)", color="navy", lw=lw
)
plt.fill_between(
    k_range,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.2,
    color="navy",
    lw=lw,
)
plt.legend(loc="best")
plt.savefig('./images/knn_n_neighbors_mbti.png')
plt.clf()

p_range = range(1, 10)
train_scores, test_scores = validation_curve(
    neighbors.KNeighborsClassifier(),
    X_train,
    y_train,
    param_name="p",
    param_range=p_range,
    scoring="balanced_accuracy",
    cv=10,
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve for p with KNN - Posture")
plt.xlabel("p")
plt.ylabel("Balanced Accuracy")
lw = 2
plt.plot(
    p_range, train_scores_mean, label="Training accuracy (balanced)", color="darkorange", lw=lw
)
plt.fill_between(
    p_range,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2,
    color="darkorange",
    lw=lw,
)
plt.plot(
    p_range, test_scores_mean, label="Cross-validation accuracy (balanced)", color="navy", lw=lw
)
plt.fill_between(
    p_range,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.2,
    color="navy",
    lw=lw,
)
plt.legend(loc="best")
plt.savefig('./images/knn_p_mbti.png')
plt.clf()

# learning curve
knn = neighbors.KNeighborsClassifier()
param_dict = {
    "n_neighbors": range(1, 25),
    "p": range(1, 10),
    "weights": ["uniform", "distance"]
}
grid = GridSearchCV(knn, param_dict, cv=10, scoring="balanced_accuracy")
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
plt.title('Learning curve for KNN - Posture Prediction')
plt.ylabel("Balanced Accuracy")
plt.savefig('./images/knn_learning_curve_mbti.png')
plt.clf()

# final knn
knn_fin = neighbors.KNeighborsClassifier(n_neighbors=13, weights='distance')
X_train, X_test, y_train, y_test = load_and_preprocess_data()
train_start = time.time()
knn_fin.fit(X_train, y_train)
train_stop = time.time()
train_time = train_stop - train_start
print(f"Training time: {train_time}s")
train_pred = knn_fin.predict(X_train)
train_accuracy = metrics.balanced_accuracy_score(y_train, train_pred)
print(f"Training accuracy: {train_accuracy}")

query_start = time.time()
test_pred = knn_fin.predict(X_test)
test_accuracy = metrics.balanced_accuracy_score(y_test, test_pred)
query_stop = time.time()
query_time = query_stop - query_start
print(f"Query time: {query_time}s")
print(f"Test accuracy: {test_accuracy}")
