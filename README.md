# Fys-stk3155_Project3
A classification model using "Fake news classification" dataset by
Bhavik Jikadara for classifying fake news.

We perform classification experiments on detecting fake versus credible news articles using three distinct models:
- Logistic Regression
- Neural Network classifier
- Decision Tree.

By comparing their performances and examining the most influential features,
we gain insight into how textual patterns inform classification decisions.


# Contributors:

Elaha Ahmadi, Herman Scheele & Theodor Jaarvik


# Instructions

If there is any problems with file pathing, remove "../" as there has been issues between contributors, use branch 'absolute_path_configured', you can do this
by writing "git checkout absolute_path_configured" in your terminal

1. Start by running the file "Run_All.py", this will create the final dataset, and build & save all the models, as well as do one run of our model testing file
2. Then run the file "Model_Testing.py" if you want to test the models more than once, no retraining is needed as the models are saved.

# setup and imports
The packages and frameworks needed to run this project is:

- Language Python 3.9
- IPython

Python Libraries:

- pandas (imported as pd)
- numpy (imported as np)
- matplotlib.pyplot (imported as plt)
- seaborn (imported as sns)
- scikit-learn (imported as sklearn)
- keras (imported as keras)
- joblib (imported as joblib)
- shap (imported as shap)

Scikit-learn Modules:

- model_selection (for train_test_split, cross_val_score)
- metrics (for accuracy_score, classification_report, roc_auc_score, roc_curve)
- linear_model (for LogisticRegression)
- tree (for DecisionTreeClassifier, plot_tree)
- ensemble (for RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier)
- feature_extraction.text (for TfidfVectorizer)

Keras Modules:

- models (for Sequential)
- layers (for Dense)
- optimizers (for SGD, Adam)
- utils (for to_categorical)

Other:

- os (for file path manipulation)
- re (for regular expressions)

# Other Notes and Known Issues

- Running the file "Run_All.py" will take a while to run, this is because we are training the Neural Network model(approximately 5 minutes).
- Running the file 'Model_Testing.py' will cause warnings, altough it should not cause any issues on the results as far as i am aware.
- (This might not affect you) I have had issues with file pathing, so i am using relative paths, other contributors have used absolute paths and have not had any issues. Use branch 'absolute_path_configured' if you have issues, you can do this
  by writing "git checkout absolute_path_configured" in your terminal

# Aknowledgements

- Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
- Chollet, F. (2015). Keras. https://keras.io
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. In Advances in Neural Information Processing Systems (pp. 4765-4774).
- Waskom, M. L. (2021). seaborn: Statistical data visualization. Journal of Open Source Software, 6(60), 3021.
- Joblib: Python Parallel Computing, https://joblib.readthedocs.io/, 2023.
