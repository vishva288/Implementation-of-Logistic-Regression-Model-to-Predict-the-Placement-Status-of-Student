# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Data Collection and Preprocessing Load the placement dataset and remove unnecessary columns. Check for missing and duplicate
values, and convert all categorical variables into numerical form using Label Encoding.
Step 2: Feature Selection and Data Splitting Separate the dataset into independent variables (features) and the dependent variable (placement
status). Split the data into training and testing sets.
Step 3: Model Training Apply the Logistic Regression algorithm on the training data to build the prediction model.
Step 4: Prediction and Evaluation Use the trained model to predict placement status on test data and evaluate the performance using accuracy
score, confusion matrix, and classification report.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: vishva S
RegisterNumber:  
*/
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data = pd.read_csv(r"C:/Users/91908/Downloads/Placement_Data.csv")
data.head()
datal = data.copy()
datal = datal.drop(["sl_no", "salary"], axis=1)
datal.head()
print("Missing values:\n", datal.isnull().sum())
print("Duplicate rows:", datal.duplicated().sum())
le = LabelEncoder()
datal["gender"] = le.fit_transform(datal["gender"])
datal["ssc_b"] = le.fit_transform(datal["ssc_b"])
datal["hsc_b"] = le.fit_transform(datal["hsc_b"])
datal["hsc_s"] = le.fit_transform(datal["hsc_s"])
datal["degree_t"] = le.fit_transform(datal["degree_t"])
datal["workex"] = le.fit_transform(datal["workex"])
datal["specialisation"] = le.fit_transform(datal["specialisation"])
datal["status"] = le.fit_transform(datal["status"])
datal
x = datal.iloc[:, :-1]
y = datal["status"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)<img width="564" height="208" alt="Screenshot 2026-02-06 111331" src="https://github.com/user-attachments/assets/2661fd6f-26b4-4b48-886d-edacaa9aea9d" />

y_pred
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)
classification_report_output = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_output)
print("Prediction of LR:")
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
HEAD
<img width="1116" height="258" alt="Screenshot 2026-02-06 111258" src="https://github.com/user-attachments/assets/eb352322-5eb6-4031-9178-f9d36af801ba" />
COPY
<img width="405" height="63" alt="Screenshot 2026-02-06 111307" src="https://github.com/user-attachments/assets/7de891c7-dc06-4dc6-9f03-5d0be9b1f252" />
<img width="956" height="195" alt="Screenshot 2026-02-06 111313" src="https://github.com/user-attachments/assets/358432a6-0eea-457e-867a-a1912ebbcd6f" />
FIT TRANSFORM
<img width="564" height="208" alt="Screenshot 2026-02-06 111331" src="https://github.com/user-attachments/assets/1c196c29-4d71-40d9-b8c5-835fb8242fde" />
<img width="878" height="439" alt="Screenshot 2026-02-06 111343" src="https://github.com/user-attachments/assets/6abdd956-f40e-40ed-91b1-f0e1b3725765" />
LOGISTIC REGRESSION
<img width="646" height="102" alt="Screenshot 2026-02-06 111352" src="https://github.com/user-attachments/assets/11b501a1-8b61-4ed3-b5aa-956d787bc732" />
ACCURACY SCORE
<img width="370" height="69" alt="Screenshot 2026-02-06 111402" src="https://github.com/user-attachments/assets/c4f26337-c089-4356-82f7-55a9ca440d7c" />
CONFUSION MATRIX
<img width="366" height="108" alt="Screenshot 2026-02-06 111410" src="https://github.com/user-attachments/assets/131fae56-33ab-4243-be83-2692c74f9844" />
<img width="715" height="471" alt="Screenshot 2026-02-06 111421" src="https://github.com/user-attachments/assets/9a56b0c3-e3b2-4801-bf84-5867b58a5a2c" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
