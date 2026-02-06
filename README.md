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
Developed by: Vishva S
RegisterNumber:  25012833
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
y_pred = lr.predict(x_test)
y_pred
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)
classification_report_output = classification_report(y_test, y_pred
print("Classification Report:\n", classification_report_output)
print("Prediction of LR:")
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
HEAD

<img width="1116" height="258" alt="Screenshot 2026-02-06 111258" src="https://github.com/user-attachments/assets/64d05749-c854-4890-b567-54e3db38d0d6" />
COPY

<img width="405" height="63" alt="Screenshot 2026-02-06 111307" src="https://github.com/user-attachments/assets/ae9d7bfd-d27f-4e5e-9457-c3a9cd2dafb5" />
<img width="956" height="195" alt="Screenshot 2026-02-06 111313" src="https://github.com/user-attachments/assets/9cd1562d-d623-486f-ba4e-751c723be787" />
FIT TRANSFORM

<img width="564" height="208" alt="Screenshot 2026-02-06 111331" src="https://github.com/user-attachments/assets/b0bea2d0-739e-4596-91b2-fe2b0799b56f" />
<img width="878" height="439" alt="Screenshot 2026-02-06 111343" src="https://github.com/user-attachments/assets/991f1cae-af98-482f-bc21-de4155ff8ce5" />
LOGISTIC REGRESSION

<img width="646" height="102" alt="Screenshot 2026-02-06 111352" src="https://github.com/user-attachments/assets/78d489be-da17-4cf8-a5b5-e328153ccee7" />
ACCURACY SCORE

<img width="370" height="69" alt="Screenshot 2026-02-06 111402" src="https://github.com/user-attachments/assets/9723cc64-3f12-4c12-bd24-74a23495d004" />
CONFUSION MATRIX

<img width="366" height="108" alt="Screenshot 2026-02-06 111410" src="https://github.com/user-attachments/assets/0e7cf389-3885-4697-ab61-7dd86c3e949e" />

CLASSIFICATION REPORT & PREDICTION

<img width="715" height="471" alt="Screenshot 2026-02-06 111421" src="https://github.com/user-attachments/assets/32a15ccf-3abc-4dbc-89fd-6a77e804e813" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
