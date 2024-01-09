import matplotlib.pyplot as plt
import graphviz
import numpy as np
from sklearn.model_selection import validation_curve, LearningCurveDisplay, GridSearchCV
from sklearn import tree, metrics
from mbti import load_and_preprocess_data
import time

X_train, X_test, y_train, y_test = load_and_preprocess_data(norm=False)

# tuning
param_range = range(1, 15)
train_scores, test_scores = validation_curve(
    tree.DecisionTreeClassifier(),
    X_train,
    y_train,
    param_name="max_depth",
    param_range=param_range,
    scoring="balanced_accuracy",
    cv=10,
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve for max_depth with Decision Tree - Posture")
plt.xlabel("max_depth")
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
plt.savefig('./images/dt_maxdepth_mbti.png')
plt.clf()


param_range = range(2, 20, 2)
train_scores, test_scores = validation_curve(
    tree.DecisionTreeClassifier(),
    X_train,
    y_train,
    param_name="min_samples_split",
    param_range=param_range,
    scoring="balanced_accuracy",
    cv=10,
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve for min_samples_split with Decision Tree - Posture")
plt.xlabel("min_samples_split")
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
plt.savefig('./images/dt_min_samples_split_mbti.png')
plt.clf()

# learning curve
dt = tree.DecisionTreeClassifier()
param_dict = {
    "max_depth": range(1, 15),
    "min_samples_split": range(2, 10, 2),
    "criterion": ["gini", "entropy"]
}
grid = GridSearchCV(dt, param_dict, cv=10, scoring="balanced_accuracy")
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
plt.title('Learning curve for Decision Tree - Posture Prediction')
plt.ylabel("Balanced Accuracy")
plt.savefig('./images/dt_learning_curve_mbti.png')
plt.clf()

# final tree
dt_fin = tree.DecisionTreeClassifier(max_depth=7, min_samples_split=6)
train_start = time.time()
dt_fin.fit(X_train, y_train)
train_stop = time.time()
train_time = train_stop - train_start
print(f"Training time: {train_time}s")
train_pred = dt_fin.predict(X_train)
train_accuracy = metrics.balanced_accuracy_score(y_train, train_pred)
print(f"Training accuracy: {train_accuracy}s")
query_start = time.time()
test_pred = dt_fin.predict(X_test)
test_accuracy = metrics.balanced_accuracy_score(y_test, test_pred)
query_stop = time.time()
query_time = query_stop - query_start
print(f"Query time: {query_time}s")
print(f"Test accuracy: {test_accuracy}")

# visual
graph = graphviz.Source( tree.export_graphviz(dt_fin, out_file=None, feature_names=X_test.columns,
                                              class_names=True))
graph.format = 'png'
graph.render('./images/dtree_posture')
