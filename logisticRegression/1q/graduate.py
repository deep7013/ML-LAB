import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "Admission_Predict.csv")

df = pd.read_csv(csv_path)

df.columns = df.columns.str.strip()

df["Admit"] = (df["Chance of Admit"] >= 0.75).astype(int)

X = df.drop(["Serial No.", "Chance of Admit", "Admit"], axis=1)
y = df["Admit"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("========== Logistic Regression ==========")
print("Accuracy :", accuracy_score(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Admit", "Admit"]
)

disp.plot(cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.show()