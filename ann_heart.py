def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import matplotlib.pyplot as plt
import graphviz
import numpy as np
from sklearn.model_selection import validation_curve, LearningCurveDisplay, GridSearchCV
from sklearn import neural_network, metrics
from heart import load_and_preprocess_data
import time

X_train, X_test, y_train, y_test = load_and_preprocess_data()

# tuning
hidden_layer_sizes = [(5,), (10,), (20,), (10, 5), (20, 10), (10, 10, 10)]
train_scores, test_scores = validation_curve(
    neural_network.MLPClassifier(max_iter=1000),
    X_train,
    y_train,
    param_name="hidden_layer_sizes",
    param_range=hidden_layer_sizes,
    scoring="accuracy",
    cv=10,
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
xticks = ['(5,)', '(10,)', '(20,)', '(10, 5)', '(20, 10)', '(10, 10, 10)']
x = range(6)
plt.title("Validation Curve for hidden_layer_sizes with ANN - Heart Disease")
plt.xlabel("hidden_layer_sizes")
plt.ylabel("Accuracy")
lw = 2
plt.plot(x, train_scores_mean, label="Training accuracy", color="darkorange", lw=lw)
plt.xticks(x,xticks)
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
plt.savefig('./images/ann_hidden_layer_sizes_heart.png')
plt.clf()

x = range(4)
activations = ['identity', 'logistic', 'tanh', 'relu']
train_scores, test_scores = validation_curve(
    neural_network.MLPClassifier(max_iter=1000),
    X_train,
    y_train,
    param_name="activation",
    param_range=activations,
    scoring="accuracy",
    cv=10,
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve for activation with ANN - Heart Disease")
plt.xlabel("activation")
plt.ylabel("Accuracy")
lw = 2

plt.plot(
    x, train_scores_mean, label="Training accuracy", color="darkorange", lw=lw
)
plt.xticks(x,activations)
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
plt.savefig('./images/ann_activation_heart.png')
plt.clf()

# learning curve
dt = neural_network.MLPClassifier(max_iter=1000)
param_dict = {
    "hidden_layer_sizes": [(5,), (10,), (20,), (10, 5), (20, 10), (10, 10, 10)],
    "activation": ["identity", "logistic", "tanh", "relu"]
}
grid = GridSearchCV(dt, param_dict, cv=10, scoring="accuracy")
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
plt.title('Learning curve for ANN - Heart Disease Prediction')
plt.ylabel("Accuracy")
plt.savefig('./images/ann_learning_curve_heart.png')
plt.clf()

# final model
ann_fin = neural_network.MLPClassifier(activation='logistic', hidden_layer_sizes=(10,), max_iter=1000)
train_start = time.time()
ann_fin.fit(X_train, y_train)
train_stop = time.time()
plt.plot(ann_fin.loss_curve_)
plt.title('Loss curve for ANN - Heart Disease Prediction')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig('./images/ann_loss_curve_heart.png')
plt.clf()
train_time = train_stop - train_start
print(f"Training time: {train_time}s")
train_pred = ann_fin.predict(X_train)
train_accuracy = metrics.accuracy_score(y_train, train_pred)
print(f"Training accuracy: {train_accuracy}s")
query_start = time.time()
test_pred = ann_fin.predict(X_test)
test_accuracy = metrics.balanced_accuracy_score(y_test, test_pred)
query_stop = time.time()
query_time = query_stop - query_start
print(f"Query time: {query_time}s")
print(f"Test accuracy: {test_accuracy}")

