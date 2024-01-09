import matplotlib.pyplot as plt
import graphviz
import numpy as np
from sklearn.model_selection import validation_curve, LearningCurveDisplay, GridSearchCV
from sklearn import svm, metrics
from heart import load_and_preprocess_data
import time

X_train, X_test, y_train, y_test = load_and_preprocess_data()

# tuning
kernel_types = ['linear', 'poly', 'rbf', 'sigmoid']
train_scores, test_scores = validation_curve(
    svm.SVC(),
    X_train,
    y_train,
    param_name="kernel",
    param_range=kernel_types,
    scoring="accuracy",
    cv=10,
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
x = range(4)
plt.title("Validation Curve for kernel_types with SVM - Heart Disease")
plt.xlabel("kernel_types")
plt.ylabel("Accuracy")
lw = 2
plt.plot(x, train_scores_mean, label="Training accuracy", color="darkorange", lw=lw)
plt.xticks(x, kernel_types)
plt.fill_between(
    x,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2,
    color="darkorange",
    lw=lw,
)
plt.plot(
    x, test_scores_mean, label="Cross-validation accuracy", color="navy", lw=lw
)
plt.fill_between(
    x,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.2,
    color="navy",
    lw=lw,
)
plt.legend(loc="best")
plt.savefig('./images/svm_kernel_types_heart.png')
plt.clf()

c = np.logspace(-1, 2)
train_scores, test_scores = validation_curve(
    svm.SVC(),
    X_train,
    y_train,
    param_name="C",
    param_range=c,
    scoring="accuracy",
    cv=10,
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.title("Validation Curve for C with SVM - Heart Disease")
plt.xlabel("C")
plt.ylabel("Accuracy")
lw = 2
plt.semilogx(c, train_scores_mean, label="Training accuracy", color="darkorange", lw=lw)
plt.fill_between(
    c,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2,
    color="darkorange",
    lw=lw,
)
plt.semilogx(
    c, test_scores_mean, label="Cross-validation accuracy", color="navy", lw=lw
)
plt.fill_between(
    c,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.2,
    color="navy",
    lw=lw,
)
plt.legend(loc="best")
plt.savefig('./images/svm_C_heart.png')
plt.clf()

# learning curve
svc = svm.SVC()
param_dict = {
    "kernel": kernel_types,
    "C": c
}
grid = GridSearchCV(svc, param_dict, cv=10, scoring="accuracy")
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_estimator_)
print(grid.best_score_)

LearningCurveDisplay.from_estimator(grid.best_estimator_,
                                    X=X_train,
                                    y=y_train,
                                    cv=10,
                                    score_type='both',
                                    score_name='accuracy',
                                    train_sizes=np.linspace(0.1, 1.0, 10))
plt.legend(["Training Accuracy", "Validation Accuracy"])
plt.title('Learning curve for SVM - Heart Disease Prediction')
plt.ylabel("Accuracy")
plt.savefig('./images/svm_learning_curve_heart.png')
plt.clf()

# final model
svc_fin = svm.SVC(C=0.8286427728546845, kernel='linear')
train_start = time.time()
svc_fin.fit(X_train, y_train)
train_stop = time.time()
train_time = train_stop - train_start
print(f"Training time: {train_time}s")
train_pred = svc_fin.predict(X_train)
train_accuracy = metrics.accuracy_score(y_train, train_pred)
print(f"Training accuracy: {train_accuracy}s")
query_start = time.time()
test_pred = svc_fin.predict(X_test)
test_accuracy = metrics.accuracy_score(y_test, test_pred)
query_stop = time.time()
query_time = query_stop - query_start
print(f"Query time: {query_time}s")
print(f"Test accuracy: {test_accuracy}")