from flask import Flask, request, render_template, jsonify
import joblib

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("service_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET"])
def index():
    # Show the HTML page
    return render_template("index.html")

@app.route("/", methods=["POST"])
def classify():
    try:
        user_input = request.form["text"]  # matches your <textarea name="text">
        X_input = vectorizer.transform([user_input])

        # Prediction
        pred = model.predict(X_input)[0]

        # Confidence (if available)
        try:
            prob = model.decision_function(X_input)[0]
            confidence = round(max(prob) / sum(abs(prob)), 2)
        except Exception:
            confidence = "N/A"

        return jsonify({
            "prediction": pred,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
