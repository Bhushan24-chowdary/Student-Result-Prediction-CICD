from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained ML model
with open("student_result_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return "Student Result Prediction Application is Running!"


# Health check API
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "message": "Application is working successfully"
    })


# Student result prediction API
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data:
        return jsonify({"error": "Please provide student details"}), 400

    try:
        attendance = float(data["attendance"])
        internal_marks = float(data["internal_marks"])
        assignment_marks = float(data["assignment_marks"])
        previous_score = float(data["previous_score"])

        features = np.array([[
            attendance,
            internal_marks,
            assignment_marks,
            previous_score
        ]])

        predicted_score = float(model.predict(features)[0])

        result = "PASS" if predicted_score >= 40 else "FAIL"

        return jsonify({
            "predicted_score": round(predicted_score, 2),
            "result": result
        })

    except (KeyError, ValueError, TypeError):
        return jsonify({
            "error": "Invalid or missing student details"
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)