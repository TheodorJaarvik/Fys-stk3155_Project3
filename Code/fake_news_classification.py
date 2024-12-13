import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from keras.src.models import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import SGD, Adam
from keras.src.utils import to_categorical

# Load fake news data
fake = pd.read_csv('../data/fake.csv')

# Load true news data
true = pd.read_csv('../data/true.csv')

fake['label'] = 0
true['label'] = 1

data = pd.concat([fake, true], ignore_index=True)
data = data.sample(frac=1).reset_index(drop=True)

X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the testing data
X_test_tfidf = vectorizer.transform(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tfidf, y_train)

y_pred_lr = lr.predict(X_test_tfidf)

# Accuracy
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f'Logistic Regression Accuracy: {accuracy_lr}')

# Confusion Matrix
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
print('Confusion Matrix:')
print(conf_matrix_lr)

# Classification Report
print('Classification Report:')
print(classification_report(y_test, y_pred_lr))

# ROC Curve
y_prob_lr = lr.predict_proba(X_test_tfidf)[:,1]
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure()
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression (area = %0.2f)' % roc_auc_lr)
plt.plot([0,1],[0,1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.show()

# Convert labels to categorical one-hot encoding
y_train_nn = to_categorical(y_train)
y_test_nn = to_categorical(y_test)

# Build the model
model = Sequential()
model.add(Dense(64, input_dim=X_train_tfidf.shape[1], activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train_tfidf, y_train_nn, epochs=5, batch_size=64, verbose=1)

y_pred_nn_prob = model.predict(X_test_tfidf)
y_pred_nn = np.argmax(y_pred_nn_prob, axis=1)

# Accuracy
accuracy_nn = accuracy_score(y_test, y_pred_nn)
print(f'Neural Network Accuracy: {accuracy_nn}')

# Confusion Matrix
conf_matrix_nn = confusion_matrix(y_test, y_pred_nn)
print('Confusion Matrix:')
print(conf_matrix_nn)

# Classification Report
print('Classification Report:')
print(classification_report(y_test, y_pred_nn))

# ROC Curve
fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_test, y_pred_nn_prob[:,1])
roc_auc_nn = auc(fpr_nn, tpr_nn)

plt.figure()
plt.plot(fpr_nn, tpr_nn, label='Neural Network (area = %0.2f)' % roc_auc_nn)
plt.plot([0,1],[0,1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Neural Network')
plt.legend(loc='lower right')
plt.show()

# Rebuild the model with different activation function and optimizer
model2 = Sequential()
model2.add(Dense(64, input_dim=X_train_tfidf.shape[1], activation='tanh'))
model2.add(Dense(2, activation='softmax'))

# Compile the model with SGD optimizer
sgd = SGD(learning_rate=0.01)
model2.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit the model
model2.fit(X_train_tfidf, y_train_nn, epochs=5, batch_size=64, verbose=1)

# Evaluate the model
y_pred_nn2_prob = model2.predict(X_test_tfidf)
y_pred_nn2 = np.argmax(y_pred_nn2_prob, axis=1)

# Accuracy
accuracy_nn2 = accuracy_score(y_test, y_pred_nn2)
print(f'Neural Network with Tanh Activation and SGD Optimizer Accuracy: {accuracy_nn2}')

# Confusion Matrix
conf_matrix_nn2 = confusion_matrix(y_test, y_pred_nn2)
print('Confusion Matrix:')
print(conf_matrix_nn2)

# ROC Curve
fpr_nn2, tpr_nn2, thresholds_nn2 = roc_curve(y_test, y_pred_nn2_prob[:,1])
roc_auc_nn2 = auc(fpr_nn2, tpr_nn2)

plt.figure()
plt.plot(fpr_nn2, tpr_nn2, label='NN with Tanh and SGD (area = %0.2f)' % roc_auc_nn2)
plt.plot([0,1],[0,1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Neural Network with Tanh and SGD')
plt.legend(loc='lower right')
plt.show()

dt = DecisionTreeClassifier()
dt.fit(X_train_tfidf, y_train)
y_pred_dt = dt.predict(X_test_tfidf)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f'Decision Tree Accuracy: {accuracy_dt}')

conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
print('Confusion Matrix:')
print(conf_matrix_dt)


y_prob_dt = dt.predict_proba(X_test_tfidf)[:,1]
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, y_prob_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)

plt.figure()
plt.plot(fpr_dt, tpr_dt, label='Decision Tree (area = %0.2f)' % roc_auc_dt)
plt.plot([0,1],[0,1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.legend(loc='lower right')
plt.show()

models = ['Logistic Regression', 'Neural Network', 'Neural Network (Tanh + SGD)', 'Decision Tree']
accuracies = [accuracy_lr, accuracy_nn, accuracy_nn2, accuracy_dt]

for model_name, acc in zip(models, accuracies):
    print(f'{model_name} Accuracy: {acc:.4f}')

plt.figure(figsize=(8,6))
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression (area = %0.2f)' % roc_auc_lr)
plt.plot(fpr_nn, tpr_nn, label='Neural Network (area = %0.2f)' % roc_auc_nn)
plt.plot(fpr_nn2, tpr_nn2, label='NN Tanh + SGD (area = %0.2f)' % roc_auc_nn2)
plt.plot(fpr_dt, tpr_dt, label='Decision Tree (area = %0.2f)' % roc_auc_dt)
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc='lower right')
plt.show()

cv_scores = cross_val_score(lr, vectorizer.transform(X), y, cv=5, scoring='accuracy')

print('Cross-validation scores:', cv_scores)
print('Mean cross-validation score:', cv_scores.mean())
