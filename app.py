from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    mail = request.form["mail"]

    mail_vector = vectorizer.transform([mail])

    prediction = model.predict(mail_vector)

    if prediction[0] == 1:
        result = "Spam Mail"
    else:
        result = "Not Spam"

    return render_template(
        "index.html",
        prediction=result
    )


if __name__ == "__main__":
    app.run(port=4000,debug=True)