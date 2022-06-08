# Binary-Classification-XGB
Assessment solution for Arya-ai Binary classification problem

Project File Struture:
Root
* Alpha&Omega.ipynb <---------------------- Main Jupyter notebook (Forgot to change name)
* Assignement - Data Scientist (1).docx <-- Assessment problem document
* Testing_predicitons.csv <---------------- Target class outputs of the Test Data
* README.md <------------------------------ this very file
* XGBFTW.sav <----------------------------- XGBoost model export done over using pickle
* requirements.txt <----------------------- Environment Screenshot
* essentials_only_req.txt <---------------- ipynb specific requirements
* Data <----------------------------------- Folder
  * Training_set.csv <--------------------- Training Dataset
  * Test_set.csv <------------------------- Testing Dataset 

Data Stats:
* Train Dataset Shape -> (3909,58)
* Train Dataset Shape -> (690,58)
* Dataset is Sparse and High Dimensional
* Features are highly skewed

Key Decisions:
* Used RandomForest Classifier for feature selection.
* Selected top 30 features with respect to their feature importance.
* For metric considered Binary CrossEntropy | LogLoss and ROC-AUC score.
* Model of choice is Xgboost.
