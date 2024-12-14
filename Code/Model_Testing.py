import keras
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from Neural_net_model import TextClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import plot_tree
from joblib import load
import shap


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


#feature importance
feature_importance = lr_model.coef_[0]
feature_names = data.vectorizer.get_feature_names_out()
feature_importance = list(zip(feature_names, feature_importance))

positive_features = [item for item in feature_importance if item[1] > 0]
negative_features = [item for item in feature_importance if item[1] < 0]
positive_features.sort(key=lambda x: x[1], reverse=True)
negative_features.sort(key=lambda x: x[1])

print('Top 10 Positive Features (drive prediction towards class 1):')
for feature, importance in positive_features[:10]:
    print(f'{feature}: {importance:.6f}')

print('\nTop 10 Negative Features (drive prediction towards class 0):')
for feature, importance in negative_features[:10]:
    print(f'{feature}: {importance:.6f}')

# Plots
data.plot_roc_curve(model_name="Logistic Regression", y_pred_prob=y_prob_lr)
data.plot_confusion_matrix(model_name="Logistic Regression", y_pred=y_pred_lr)
data.plot_feature_importance(feature_importance, title="Logistic Regression")

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

# feature importance
background_data = shap.sample(data.X_test_tfidf, 15)
test_subset = shap.sample(data.X_test_tfidf, 100)
background_data_dense = background_data.toarray()
test_subset_dense = shap.sample(data.X_test_tfidf, 100).toarray()
explainer = shap.DeepExplainer(nn_model, background_data_dense)
shap_values = explainer.shap_values(test_subset_dense)
feature_names = data.vectorizer.get_feature_names_out()
shap_values_mean = shap_values.mean(axis=(0, 2))
abs_shap_values_mean = np.abs(shap_values).mean(axis=(0, 2))
feature_importance = list(zip(feature_names, shap_values_mean))
feature_importance_abs = list(zip(feature_names, abs_shap_values_mean))
feature_importance.sort(key=lambda x: x[1], reverse=True)
feature_importance_abs.sort(key=lambda x: x[1], reverse=True)

print("Top 10 Important Features - Towards class True:")
for feature, importance in feature_importance[:10]:
    print(f"{feature}: {importance:.6f}")

print("Bottom 10 Important Features - Towards class False:")
for feature, importance in feature_importance[-10:]:
    print(f"{feature}: {importance:.6f}")

print("Top 10 Important Features - Towards any class (Absolute Values):")
for feature, importance in feature_importance_abs[:10]:
    print(f"{feature}: {importance:.6f}")

# Plots
data.plot_roc_curve(model_name="Neural Network", y_pred_prob=y_pred_nn[:,1])
data.plot_confusion_matrix(model_name="Neural Network", y_pred=y_pred_nn_binary)
data.plot_feature_importance(feature_importance, title="Neural Network")
data.plot_feature_importance(feature_importance_abs, title="Neural Network absolute")

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

#Feature importance
importances = dt_model.feature_importances_
feature_names = data.vectorizer.get_feature_names_out()
feature_importance = list(zip(feature_names, importances))
sorted_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)

print("Descision Tree: Top 10 Important Features:")
for feature, score in sorted_importance[:10]:
    print(f"{feature}: {score:.4f}")

# Plots
y_prob_dt = dt_model.predict_proba(data.X_test_tfidf)[:,1]
data.plot_roc_curve(model_name="Decision Tree", y_pred_prob=y_prob_dt)
data.plot_confusion_matrix(model_name="Decision Tree", y_pred=y_pred_dt)
plot_tree(dt_model, feature_names=feature_names, class_names=['Class 0', 'Class 1'], filled=True)
plt.show()
data.plot_feature_importance(feature_importance, title="Decision Tree")



