import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Student dataset
data = {
    "attendance": [90, 80, 75, 60, 95, 85, 70, 65, 88, 92,
                   78, 82, 55, 68, 96, 72, 89, 77, 84, 62],
    "internal_marks": [85, 75, 70, 55, 95, 80, 65, 60, 88, 90,
                       72, 78, 50, 62, 96, 68, 87, 74, 82, 58],
    "assignment_marks": [90, 80, 75, 60, 95, 85, 70, 65, 88, 92,
                         78, 82, 55, 68, 96, 72, 89, 77, 84, 62],
    "previous_score": [88, 78, 72, 58, 94, 82, 68, 62, 86, 91,
                       75, 80, 52, 65, 95, 70, 89, 76, 83, 60],
    "final_score": [88, 78, 73, 57, 95, 83, 69, 63, 87, 92,
                    76, 81, 53, 66, 96, 71, 90, 77, 84, 61]
}

df = pd.DataFrame(data)

X = df[["attendance", "internal_marks",
        "assignment_marks", "previous_score"]]

y = df["final_score"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Student Result Prediction Model Training")
print("----------------------------------------")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Save model
with open("student_result_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")