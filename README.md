# Binary-Classification-XGB

Assessment solution for Arya-ai Binary classification problem

hl

Project File Struture:

Root /fldr
* Alpha&Omega.ipynb <---- *Main Jupyter notebook (Forgot to change name)*
* Assignement - Data Scientist (1).docx <---- *Assessment problem document*
* Testing_predicitons.csv <---- *Target class outputs of the Test Data*
* README.md <---- *this very file*
* XGBFTW.sav <---- *XGBoost model export done over using pickle*
* requirements.txt <---- *Environment Screenshot*
* essentials_only_req.txt <---- *ipynb specific requirements*
* Data /fldr
  * Training_set.csv <---- *Training Dataset*
  * Test_set.csv <---- *Testing Dataset*

hl

Data Stats:
* Train Dataset Shape -> (3909,58)
* Train Dataset Shape -> (690,58)
* Dataset is Sparse and High Dimensional
* Features are highly skewed

hl

Key Decisions:
* Used RandomForest Classifier for feature selection.
* Selected top 30 features with respect to their feature importance.
* For metric considered Binary CrossEntropy | LogLoss and ROC-AUC score.
* Model of choice is Xgboost.

hl

Process Flow - Main.ipynb (Alpha&Omega.ipynb)
1. EDA
2. Splitting the data
3. Feature Selection
4. Data Scaling - Normalization
5. Model Training
6. Prediction Metrics
7. Processing and Predicting on Test Data
8. Saving Model for Future Usage
9. Exporting Y_test Predicted scores
10. Generating requirements. #Has an important Note. Must Read!

hl

Process Flow - Performance_print.py
1. Splitting the Data
2. Feature Selection
3. Importing Presaved model
4. Using presaved model to generate scores
5. Using Prettytable to print output table

hl

Had fun making this!!
