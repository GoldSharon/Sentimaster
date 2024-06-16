from flask import Flask, render_template, request
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from joblib import load
import re 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer  
from textblob import TextBlob

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
    review = request.form.get('Check', "")
    if not review:
        return render_template("index.html", error="Input cannot be empty")
    else:
        review = re.sub('[^a-zA-Z]'," ",review)
        review = review.lower()
        review = review.split()
        
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ ps.stem(word) for word in review if word not in set(all_stopwords) ]
        review = " ".join(review)
        review = TextBlob(review).correct() # Correct the spelling mistakes
        review = str(review)
        vectorized_input = vectorizer.transform([review]).toarray()
        predicted = model.predict(vectorized_input)
        flag = '1' if predicted == 1 else '0'
        return render_template("result.html", flag=flag)

if __name__ == "__main__":
    app.run(debug=True)
