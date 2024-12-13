# Fys-stk3155_Project3
A classification model using "Fake news classification" dataset by
Bhavik Jikadara for classifying fake news. Contributors: Elaha Ahmadi, Herman Scheele & Theodor Jaarvik


# Instructions 

If there is any problems with file pathing, remove "../" as there has been issues between contributors

1. Start by running the file "Run_All.py", this will create the final dataset, and build & save all the models
2. Then run the file "Model_Testing.py" to see our testing work.

# setup and imports

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.src.models import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import SGD, Adam
from keras.src.utils import to_categorical
from joblib import dump
from Neural_net_model import TextClassifier
from Pre_Processing import Processing
