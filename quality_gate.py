import pandas as pd
import pickle
from sklearn.metrics import r2_score

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

# Load trained model
with open("student_result_model.pkl", "rb") as file:
    model = pickle.load(file)

# Make predictions
predictions = model.predict(X)

# Calculate R2 score
score = r2_score(y, predictions)

print("QUALITY GATE RESULTS")
print("--------------------")
print("R2 Score:", score)

# Quality threshold
if score >= 0.70:
    print("QUALITY GATE PASSED")
else:
    print("QUALITY GATE FAILED")
    raise SystemExit(1)