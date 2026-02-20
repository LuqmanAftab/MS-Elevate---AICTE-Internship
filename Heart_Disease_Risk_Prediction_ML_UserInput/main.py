import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("data/heart.csv")

X = data.drop("target", axis=1)
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("=== Heart Disease Risk Prediction System ===")

# User input
age = int(input("Enter age: "))
sex = int(input("Enter sex (1=Male, 0=Female): "))
cp = int(input("Enter chest pain type (0-3): "))
trestbps = int(input("Enter resting blood pressure: "))
chol = int(input("Enter cholesterol level: "))
fbs = int(input("Enter fasting blood sugar (1=True, 0=False): "))
restecg = int(input("Enter ECG result (0-2): "))
thalach = int(input("Enter max heart rate achieved: "))
exang = int(input("Exercise induced angina (1=Yes, 0=No): "))
oldpeak = float(input("Enter ST depression value: "))
slope = int(input("Enter slope (0-2): "))
ca = int(input("Enter number of major vessels (0-4): "))
thal = int(input("Enter thalassemia value (1-3): "))

user_data = [[age, sex, cp, trestbps, chol, fbs, restecg,
              thalach, exang, oldpeak, slope, ca, thal]]

prediction = model.predict(user_data)

if prediction[0] == 1:
    print("\nPrediction: HIGH risk of Heart Disease")
else:
    print("\nPrediction: LOW risk of Heart Disease")