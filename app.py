from flask import Flask, render_template, request
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from joblib import load

model = load(r"D:\Experiment\A-Z ML\Nlp\Restrant review\Model\SVC.joblib")
vectorizer = load(r"D:\Experiment\A-Z ML\Nlp\Restrant review\Model\CountVectorizer.joblib")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/result")
def result():
    flag = request.args.get('flag')
    return render_template("result.html", flag=flag)

@app.route("/submit", methods=['POST'])
def predict():
    input_value = request.form.get('Check', "")
    if not input_value:
        return render_template("index.html", error="Input cannot be empty")
    else:
        vectorized_input = vectorizer.transform([input_value]).toarray()
        predicted = model.predict(vectorized_input)
        flag = '1' if predicted == 1 else '0'
        return render_template("result.html", flag=flag)

if __name__ == "__main__":
    app.run(debug=True)
