import matplotlib.pyplot as plt
import graphviz
import numpy as np
from sklearn.model_selection import validation_curve, LearningCurveDisplay
from sklearn import tree
from heart import load_and_preprocess_data
import time

X_train, X_test, y_train, y_test = load_and_preprocess_data(norm=False)
# print(X_train, y_train)

# tuning
param_range = range(1, 10)
train_scores, test_scores = validation_curve(
    tree.DecisionTreeClassifier(),
    X_train,
    y_train,
    param_name="max_depth",
    param_range=param_range,
    scoring="accuracy",
    cv=10,
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve for max_depth with Decision Tree - Heart Disease")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
lw = 2
plt.plot(
    param_range, train_scores_mean, label="Training accuracy", color="darkorange", lw=lw
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
    param_range, test_scores_mean, label="Cross-validation accuracy", color="navy", lw=lw
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
plt.savefig('./images/dt_maxdepth_heart.png')
plt.clf()

param_range = range(10, 70, 4)
train_scores, test_scores = validation_curve(
    tree.DecisionTreeClassifier(),
    X_train,
    y_train,
    param_name="min_samples_split",
    param_range=param_range,
    scoring="accuracy",
    cv=10,
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve for min_samples_split with Decision Tree - Heart Disease")
plt.xlabel("min_samples_split")
plt.ylabel("Accuracy")
lw = 2
plt.plot(
    param_range, train_scores_mean, label="Training accuracy", color="darkorange", lw=lw
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
    param_range, test_scores_mean, label="Cross-validation accuracy", color="navy", lw=lw
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
plt.savefig('./images/dt_min_samples_split_heart.png')
plt.clf()

# learning curve
dt = tree.DecisionTreeClassifier(max_depth=4, min_samples_split=50)
LearningCurveDisplay.from_estimator(dt,
                                    X=X_train,
                                    y=y_train,
                                    cv=10,
                                    score_type='both',
                                    score_name='Accuracy',
                                    train_sizes=np.linspace(0.1, 1.0, 10))
plt.legend(["Training Accuracy", "Validation Accuracy"])
plt.title('Learning curve for Decision Tree - Heart Disease Prediction')
plt.ylabel("Accuracy")
plt.savefig('./images/dt_learning_curve_heart.png')
plt.clf()

# final tree
dt_fin = tree.DecisionTreeClassifier(max_depth=4, min_samples_split=50)
train_start = time.time()
dt_fin.fit(X_train, y_train)
train_stop = time.time()
train_time = train_stop - train_start
print(f"Training time: {train_time}s")
train_accuracy = dt_fin.score(X_train, y_train)
print(f"Training accuracy: {train_accuracy}s")

query_start = time.time()
test_accuracy = dt_fin.score(X_test, y_test)
query_stop = time.time()
query_time = query_stop - query_start
print(f"Query time: {query_time}s")
print(f"Test accuracy: {test_accuracy}s")

# visual
graph = graphviz.Source( tree.export_graphviz(dt_fin, out_file=None, feature_names=X_test.columns,
                                              class_names=True))
graph.format = 'png'
graph.render('./images/dtree_render')
