import keras
import pandas as pd
from sklearn.metrics import accuracy_score

from Neural_net_model import TextClassifier

model = keras.models.load_model('../Models/nn_fake_news_classifier.keras')
model.summary()

data = TextClassifier('../data/df_final.csv')
data.vectorize_text()

y_pred = model.predict(data.X_test_tfidf)
y_pred = y_pred.argmax(axis=1)

accuracy = accuracy_score(data.y_test, y_pred)
print(f'Accuracy: {accuracy}')

confusion_matrix = pd.crosstab(data.y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
print(confusion_matrix)


