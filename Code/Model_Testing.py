import keras
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from Neural_net_model import TextClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import plot_tree
from joblib import load

lr_model = load('../Models/lr_fake_news_classifier.joblib')
nn_model = keras.models.load_model('../Models/nn_fake_news_classifier.keras')
dt_model = load('../Models/dt_fake_news_classifier.joblib')

data = TextClassifier('../data/df_final.csv')
data.vectorize_text()

num_bootstraps = 20

print('------------------------Logistic Regression------------------------')

y_pred_lr = lr_model.predict(data.X_test_tfidf)

#Metrics
accuracy_lr = accuracy_score(data.y_test, y_pred_lr)
print(f'Logistic Regression Accuracy (without bootstrapping): {accuracy_lr}')
accuracy_lr_avg_bootstrap = data.bootstrap(model_name="Logistic Regression", model=lr_model, num_bootstraps=num_bootstraps, num_samples=1000)
print(f'Logistic Regression Average Accuracy over {num_bootstraps} Bootstraps: {accuracy_lr_avg_bootstrap:.2f}')
scores_lr = cross_val_score(lr_model, data.X_test_tfidf, data.y_test, cv=5)
print(f'Logistic Regression Cross Validation Score: {scores_lr}')
print(f'Logistic Regression Cross Validation Score: {scores_lr.mean():.2f}')
classification_report_lr = classification_report(data.y_test, y_pred_lr)
roc_auc_lr = roc_auc_score(data.y_test, y_pred_lr)
print(f'Logistic Regression ROC AUC: {roc_auc_lr:.2f}')
print(classification_report_lr)

y_prob_lr = lr_model.predict_proba(data.X_test_tfidf)[:,1]

# Plots

data.plot_roc_curve(model_name="Logistic Regression", y_pred_prob=y_prob_lr)
data.plot_confusion_matrix(model_name="Logistic Regression", y_pred=y_pred_lr)



print('------------------------Neural Network------------------------')

y_pred_nn = nn_model.predict(data.X_test_tfidf)
y_pred_nn_binary = np.argmax(y_pred_nn, axis=1)

#Metrics
accuracy_nn = accuracy_score(data.y_test, y_pred_nn_binary)
print(f'Neural Network Accuracy (without bootstrapping): {accuracy_nn}')
accuracy_nn_avg_bootstrap = data.bootstrap(model_name="Neural Network", model=nn_model, num_bootstraps=num_bootstraps, num_samples=1000)
print(f'Neural Network Average Accuracy over {num_bootstraps} Bootstraps: {accuracy_nn_avg_bootstrap:.2f}')
classification_report_nn = classification_report(data.y_test, y_pred_nn_binary)
roc_auc_nn = roc_auc_score(data.y_test, y_pred_nn_binary)
print(f'Neural Network ROC AUC: {roc_auc_nn:.2f}')
print(classification_report_nn)

# Plots
data.plot_roc_curve(model_name="Neural Network", y_pred_prob=y_pred_nn[:,1])
data.plot_confusion_matrix(model_name="Neural Network", y_pred=y_pred_nn_binary)


print('------------------------Decision Tree------------------------')

y_pred_dt = dt_model.predict(data.X_test_tfidf)

#Metrics
accuracy_dt = accuracy_score(data.y_test, y_pred_dt)
print(f'Decision Tree Accuracy (without bootstrapping): {accuracy_dt}')
accuracy_dt_avg_bootstrap = data.bootstrap(model_name="Decision Tree", model=dt_model, num_bootstraps=num_bootstraps, num_samples=1000)
print(f'Decision Tree Average Accuracy over {num_bootstraps} Bootstraps: {accuracy_dt_avg_bootstrap:.2f}')
scores_dt = cross_val_score(dt_model, data.X_test_tfidf, data.y_test, cv=5)
print(f'Decision Tree Cross Validation Score: {scores_dt.mean():.2f}')
classification_report_dt = classification_report(data.y_test, y_pred_dt)
roc_auc_dt = roc_auc_score(data.y_test, y_pred_dt)
print(f'Decision Tree ROC AUC: {roc_auc_dt:.2f}')
print(classification_report_dt)

# Plots
y_prob_dt = dt_model.predict_proba(data.X_test_tfidf)[:,1]
data.plot_roc_curve(model_name="Decision Tree", y_pred_prob=y_prob_dt)
data.plot_confusion_matrix(model_name="Decision Tree", y_pred=y_pred_dt)

# Tree Plot
plot_tree(dt_model, feature_names=data.vectorizer.get_feature_names_out(), class_names=['Class 0', 'Class 1'], filled=True)
plt.show()

# Top 10 Important Features
importances = dt_model.feature_importances_
feature_names = data.vectorizer.get_feature_names_out()
feature_importance = list(zip(feature_names, importances))
sorted_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)

print("Descision Tree: Top 10 Important Features:")
for feature, score in sorted_importance[:10]:
    print(f"{feature}: {score:.4f}")



