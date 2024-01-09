def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


import matplotlib.pyplot as plt
import graphviz
import numpy as np
from sklearn.model_selection import validation_curve, LearningCurveDisplay, GridSearchCV
from sklearn import neural_network, metrics
from mbti import load_and_preprocess_data
import time

X_train, X_test, y_train, y_test = load_and_preprocess_data()

# tuning
hidden_layer_sizes = [(10,), (20,), (20, 10), (20, 20, 10), (30, 20, 10), (30, 20, 20, 10)]
train_scores, test_scores = validation_curve(
    neural_network.MLPClassifier(),
    X_train,
    y_train,
    param_name="hidden_layer_sizes",
    param_range=hidden_layer_sizes,
    scoring="balanced_accuracy",
    cv=10,
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
xticks = ['(10,)', '(20,)', '(20,10)', '(20,20,10)', '(30,20,10)', '(30,20,20,10)']
x = range(6)
plt.title("Validation Curve for hidden_layer_sizes with ANN - Posture")
plt.xlabel("hidden_layer_sizes")
plt.ylabel("Balanced Accuracy")
lw = 2
plt.plot(x, train_scores_mean, label="Training accuracy (balanced)", color="darkorange", lw=lw)
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
    x, test_scores_mean, label="Cross-validation accuracy (balanced)", color="navy", lw=lw
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
plt.savefig('./images/ann_hidden_layer_sizes_mbti.png')
plt.clf()

x = range(4)
activations = ['identity', 'logistic', 'tanh', 'relu']
train_scores, test_scores = validation_curve(
    neural_network.MLPClassifier(),
    X_train,
    y_train,
    param_name="activation",
    param_range=activations,
    scoring="balanced_accuracy",
    cv=10,
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve for activation with ANN - Posture")
plt.xlabel("activation")
plt.ylabel("Balanced Accuracy")
lw = 2

plt.plot(
    x, train_scores_mean, label="Training accuracy (balanced)", color="darkorange", lw=lw
)
plt.xticks(x, activations)
plt.fill_between(
    x,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2,
    color="darkorange",
    lw=lw,
)
plt.plot(
    x, test_scores_mean, label="Cross-validation accuracy (balanced)", color="navy", lw=lw
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
plt.savefig('./images/ann_activation_mbti.png')
plt.clf()

# learning curve
dt = neural_network.MLPClassifier()
param_dict = {
    "hidden_layer_sizes": hidden_layer_sizes,
    "activation": ["identity", "logistic", "tanh", "relu"]
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
plt.legend(["Training Accuracy (Balanced)", "Validation Accuracy (Balanced)"])
plt.title('Learning curve for ANN - Posture Prediction')
plt.ylabel("Accuracy")
plt.savefig('./images/ann_learning_curve_mbti.png')
plt.clf()

# final model
ann_fin = neural_network.MLPClassifier(hidden_layer_sizes=(20, 20, 10), max_iter=1000)
train_start = time.time()
ann_fin.fit(X_train, y_train)
train_stop = time.time()
plt.plot(ann_fin.loss_curve_)
plt.title('Loss curve for ANN - Posture Prediction')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig('./images/ann_loss_curve_mbti.png')
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
#
# # visual
# graph = graphviz.Source( tree.export_graphviz(dt_fin, out_file=None, feature_names=X_test.columns,
#                                               class_names=True))
# graph.format = 'png'
# graph.render('./images/dtree_posture')
