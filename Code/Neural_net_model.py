import numpy as np
import pandas as pd
from keras.src.models import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import SGD, Adam
from keras.src.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from joblib import dump
import seaborn as sns



class TextClassifier:
    def __init__(self, data_path, test_size=0.2, random_state=42):

        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state

        try:
            self.df = pd.read_csv(self.data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File at {self.data_path} not found.")
        except pd.errors.EmptyDataError:
            raise ValueError(f"The file at {self.data_path} is empty.")

        # Check if required columns are present
        if 'text' not in self.df.columns or 'label' not in self.df.columns:
            raise ValueError("Dataset must contain 'text' and 'label' columns.")

        self.X = self.df['text']
        self.y = self.df['label']

        # Splitting the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, stratify=self.y, random_state=self.random_state
        )

    def logistic_regression(self):

        # Training a Logistic Regression model

        self.lr = LogisticRegression(max_iter=1000)
        self.lr.fit(self.X_train_tfidf, self.y_train)
        self.y_pred_lr = self.lr.predict(self.X_test_tfidf)



    def vectorize_text(self):

        # vectorizing the text

        self.vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        self.X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        self.X_test_tfidf = self.vectorizer.transform(self.X_test)

        # removing the 'reuters' feature

        '''feature_names = self.vectorizer.get_feature_names_out()
        indices_to_keep = [i for i, feature in enumerate(feature_names) if feature != 'reuters' and feature != 'trump' and feature != 'image' and feature != 'donald']
        self.X_train_tfidf = self.X_train_tfidf[:, indices_to_keep]
        self.X_test_tfidf = self.X_test_tfidf[:, indices_to_keep]'''

    def build_neural_network(self, activation='relu', optimizer='adam'):

        # building the neural network

        input_dim = self.X_train_tfidf.shape[1]
        self.model = Sequential([
            Dense(64, input_dim=input_dim, activation=activation),
            Dense(2, activation='softmax')
        ])

        if optimizer == 'adam':
            opt = Adam()
        elif optimizer == 'sgd':
            opt = SGD(learning_rate=0.01)
        else:
            raise ValueError("Unsupported optimizer. Choose 'adam' or 'sgd'.")

        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    def train_neural_network(self, epochs=5, batch_size=64):

        # training the neural network

        self.y_train_nn = to_categorical(self.y_train)
        self.model.fit(self.X_train_tfidf, self.y_train_nn, epochs=epochs, batch_size=batch_size, verbose=1)


    def train_decision_tree(self):

       # Training the decision tree

        self.dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=5)
        self.dt.fit(self.X_train_tfidf, self.y_train)
        self.y_pred_dt = self.dt.predict(self.X_test_tfidf)

    def save_model(self, file_path):
        if file_path == '../Models/lr_fake_news_classifier.joblib':
            dump(self.lr, file_path)
        elif file_path == '../Models/dt_fake_news_classifier.joblib':
            dump(self.dt, file_path)
        else:
            self.model.save(file_path)

    def plot_roc_curve(self, model_name, y_pred_prob):

        # Plotting the ROC curve for the neural network

        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.show()

    def plot_confusion_matrix(self, model_name, y_pred):

        conf_matrix = confusion_matrix(self.y_test, y_pred)
        plt.figure()
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.show()

    def bootstrap(self, model_name, model, num_bootstraps, num_samples):
        # Bootstrap resampling
        accuracy_bootstraps = []

        for i in range(num_bootstraps):
            X_test_bootstrap, y_test_bootstrap = resample(self.X_test_tfidf, self.y_test, replace=True, n_samples=num_samples, random_state=i)

            # Evaluate models on bootstrap sample
            if model_name == 'Logistic Regression':
                y_pred_lr_bootstrap = model.predict(X_test_bootstrap)
                accuracy_lr_bootstrap = accuracy_score(y_test_bootstrap, y_pred_lr_bootstrap)
                accuracy_bootstraps.append(accuracy_lr_bootstrap)

            elif model_name == 'Neural Network':
                y_pred_nn_bootstrap = model.predict(X_test_bootstrap)
                y_pred_nn_bootstrap_binary = np.argmax(y_pred_nn_bootstrap, axis=1)
                accuracy_nn_bootstrap = accuracy_score(y_test_bootstrap, y_pred_nn_bootstrap_binary)
                accuracy_bootstraps.append(accuracy_nn_bootstrap)

            elif model_name == 'Decision Tree':
                y_pred_dt_bootstrap = model.predict(X_test_bootstrap)
                accuracy_dt_bootstrap = accuracy_score(y_test_bootstrap, y_pred_dt_bootstrap)
                accuracy_bootstraps.append(accuracy_dt_bootstrap)

        # Calculate average accuracy over bootstraps
        accuracy_avg = np.mean(accuracy_bootstraps)
        return accuracy_avg

    def plot_feature_importance(self,feature_importance, title="Feature Importance"):

        if title == 'Logistic Regression':

            positive_features = [item for item in feature_importance if item[1] > 0]
            negative_features = [item for item in feature_importance if item[1] < 0]

            positive_features.sort(key=lambda x: x[1], reverse=True)
            negative_features.sort(key=lambda x: x[1])

            top_positive = positive_features[:10]
            top_negative = negative_features[:10]
            features = [item[0] for item in top_positive + top_negative]
            importances = [item[1] for item in top_positive + top_negative]

            plt.figure(figsize=(12, 6))
            plt.barh(features, importances, color=['green' if imp > 0 else 'red' for imp in importances])
            plt.gca().invert_yaxis()
            plt.xlabel('Coefficient Value')
            plt.title('Top Positive and Negative Features (Logistic Regression)')
            plt.show()

        elif title == 'Neural Network':
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            top_positive_features = [item for item in feature_importance if item[1] > 0][:10]
            top_negative_features = [item for item in feature_importance if item[1] < 0][-10:]
            combined_features = top_positive_features + top_negative_features
            combined_feature_names = [f[0] for f in combined_features]
            combined_shap_values = [f[1] for f in combined_features]

            plt.figure(figsize=(12, 8))
            colors = ['green' if value > 0 else 'red' for value in combined_shap_values]
            plt.barh(combined_feature_names, combined_shap_values, color=colors, align='center')
            plt.gca().invert_yaxis()
            plt.xlabel('SHAP Value (Feature Importance)')
            plt.title('Top 10 Positive and Negative Features (Neural Network)')
            plt.show()

        elif title == 'Neural Network absolute':
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            top_features_abs = feature_importance[:10]
            abs_feature_names = [f[0] for f in top_features_abs]
            abs_shap_values = [f[1] for f in top_features_abs]

            plt.figure(figsize=(12, 8))
            plt.barh(abs_feature_names, abs_shap_values, align='center', color='blue')
            plt.gca().invert_yaxis()
            plt.xlabel('Mean Absolute SHAP Value (Feature Importance)')
            plt.title('Top 10 Important Features by Absolute SHAP Value (Neural Network)')
            plt.show()

        elif title == 'Decision Tree':

            sorted_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)[:10]
            features = [item[0] for item in sorted_importance]
            values = [item[1] for item in sorted_importance]

            plt.figure(figsize=(12, 8))
            plt.barh(features, values, color='blue')
            plt.gca().invert_yaxis()
            plt.xlabel('Feature Importance (Impurity Decrease)')
            plt.title('Decision Tree: Top Features by Absolute Importance')
            plt.show()

        else:
            print("Invalid title. Please choose 'Logistic Regression', 'Neural Network', 'Decision Tree', or 'Neural Network absolute'.")








