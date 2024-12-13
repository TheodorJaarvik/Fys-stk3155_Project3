from Neural_net_model import TextClassifier
from Pre_Processing import Processing


if __name__ == "__main__":
    processing = Processing(data=None)
    processing.createDataset('../data/fake.csv', '../data/true.csv')

    classifier = TextClassifier('../data/df_final.csv')

    try:
        classifier.vectorize_text()

        # Logistic Regression
        classifier.logistic_regression()
        classifier.save_model('../Models/lr_fake_news_classifier.joblib')

        # Neural Network
        classifier.build_neural_network()
        classifier.train_neural_network()
        classifier.save_model('../Models/nn_fake_news_classifier.keras')

        # Decision Tree
        classifier.train_decision_tree()
        classifier.save_model('../Models/dt_fake_news_classifier.joblib')
    except Exception as e:
        print(f"An error occurred: {e}")