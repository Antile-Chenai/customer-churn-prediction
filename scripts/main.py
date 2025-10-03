import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Starting Customer Churn Prediction Project...")

# Sample dataset
data = pd.DataFrame({
    "Feature1": [1,0,1,1,0,0,1],
    "Feature2": [5,3,4,2,4,3,5],
    "Churn": [0,1,0,0,1,1,0]
})

X = data[["Feature1","Feature2"]]
y = data["Churn"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

print("\nCustomer Churn Prediction Project Completed!")
