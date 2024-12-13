import numpy as np
import pandas as pd
from keras.src.models import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import SGD, Adam
from keras.src.utils import to_categorical
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, matthews_corrcoef, \
    precision_recall_curve, log_loss
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

        self.accuracy_lr = accuracy_score(self.y_test, self.y_pred_lr)
        print(f'Logistic Regression Accuracy: {self.accuracy_lr:.2f}')



    def vectorize_text(self):

        # vectorizing the text

        self.vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        self.X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        self.X_test_tfidf = self.vectorizer.transform(self.X_test)

        '''feature_names = self.vectorizer.get_feature_names_out()
        indices_to_keep = [i for i, feature in enumerate(feature_names) if feature != 'reuters']
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

        self.dt = DecisionTreeClassifier(max_depth=8)
        self.dt.fit(self.X_train_tfidf, self.y_train)
        self.y_pred_dt = self.dt.predict(self.X_test_tfidf)

        self._print_metrics(self.y_test, self.y_pred_dt, model_name="Decision Tree")

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





