import numpy as np
import pandas as pd
from keras.src.models import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import SGD, Adam
from keras.src.utils import to_categorical
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, matthews_corrcoef, \
    precision_recall_curve, log_loss
import matplotlib.pyplot as plt


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

    def vectorize_text(self):

        # vectorizing the text

        self.vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        self.X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        self.X_test_tfidf = self.vectorizer.transform(self.X_test)

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

    def evaluate_neural_network(self):

        # evaluating the neural network using the test data

        self.y_pred_nn_prob = self.model.predict(self.X_test_tfidf)
        self.y_pred_nn = np.argmax(self.y_pred_nn_prob, axis=1)

        self._print_metrics(self.y_test, self.y_pred_nn, model_name="Neural Network")

    def plot_roc_curve(self, model_name="Neural Network"):

        # Plotting the ROC curve for the neural network

        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_nn_prob[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.show()

    def train_decision_tree(self):

       # Training the decision tree

        self.dt = DecisionTreeClassifier()
        self.dt.fit(self.X_train_tfidf, self.y_train)
        self.y_pred_dt = self.dt.predict(self.X_test_tfidf)

        self._print_metrics(self.y_test, self.y_pred_dt, model_name="Decision Tree")

    def plot_roc_curve_dt(self):

        # Plotting the ROC curve for the desision tree

        if hasattr(self.dt, "predict_proba"):
            y_prob_dt = self.dt.predict_proba(self.X_test_tfidf)[:, 1]
        else:
            raise AttributeError("Decision Tree does not support probability predictions.")

        fpr, tpr, thresholds = roc_curve(self.y_test, y_prob_dt)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f'Decision Tree (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Decision Tree')
        plt.legend(loc='lower right')
        plt.show()

    def _print_metrics(self, y_true, y_pred, model_name="Model"):

        # Accuracy, confusion matrix and classification report

        accuracy = accuracy_score(y_true, y_pred)
        print(f'{model_name} Accuracy: {accuracy:.2f}')
        print('Confusion Matrix:')
        print(confusion_matrix(y_true, y_pred))
        print('Classification Report:')
        print(classification_report(y_true, y_pred))

    def evaluate_model_comprehensively(self, y_true, y_pred, y_probs, model_name="Model"):

        # Evaluating the model, by accuracy, precision, recall, f1, specificity, mcc, logloss, precision-recall auc and calibration curve

        print(f"Evaluating {model_name}...")

        # Accuracy, Precision, Recall, F1
        self._print_metrics(y_true, y_pred, model_name)

        # Specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
        print(f"Specificity: {specificity:.2f}")

        # MCC
        mcc = matthews_corrcoef(y_true, y_pred)
        print(f"Matthews Correlation Coefficient: {mcc:.2f}")

        # Log Loss
        logloss = log_loss(y_true, y_probs)
        print(f"Log Loss: {logloss:.2f}")

        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
        auc_pr = auc(recall, precision)
        print(f"Precision-Recall AUC: {auc_pr:.2f}")

        # Calibration Curve
        prob_true, prob_pred = calibration_curve(y_true, y_probs[:, 1], n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'Calibration Curve - {model_name}')
        plt.show()



# Usage
if __name__ == "__main__":
    classifier = TextClassifier('../data/df_final.csv')

    try:
        classifier.vectorize_text()

        # Neural Network
        classifier.build_neural_network()
        classifier.train_neural_network()
        classifier.evaluate_neural_network()
        classifier.evaluate_model_comprehensively(classifier.y_test, classifier.y_pred_nn, classifier.y_pred_nn_prob, model_name="Neural Network")
        classifier.plot_roc_curve()

        # Decision Tree
        classifier.train_decision_tree()
        classifier.plot_roc_curve_dt()

    except Exception as e:
        print(f"An error occurred: {e}")
