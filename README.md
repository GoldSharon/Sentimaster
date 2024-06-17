# Sentimaster

Welcome to **Sentimaster**, your go-to sentiment analysis tool. Sentimaster leverages powerful machine learning algorithms to analyze and interpret the sentiment behind textual data, helping you make informed decisions based on customer feedback, social media comments, and more.

## Features

- **Insightful Analysis**: Gain deep insights into sentiment trends with precision.
- **User-Friendly Interface**: Easily analyze sentiment with an intuitive and simple interface.
- **Real-Time Feedback**: Get instant sentiment analysis results.
- **Customizable Solutions**: Tailor the tool to fit your unique needs and goals.
- **Secure and Reliable**: Built on robust machine learning models ensuring accurate predictions.

## Directory Structure
│
├───Dataset<br>
├───Model<br>
│ ├── SVC.joblib<br>
│ └── CountVectorizer.joblib<br>
├───static<br>
│ └── images<br>
│     └── banner.jpg<br>
└───templates<br>
│ ├── index.html<br>
│ └── result.html<br>
├── app.py<br>
├── model.ipynb<br>
└── requirements.txt<br>

## Installation
1. **Clone the repository**:
    ```sh
    git clone https://github.com/GoldSharon/Sentimaster.git
    cd Sentimaste
    ```

2. **Set up a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Download the pre-trained model and vectorizer**:
    - Place `SVC.joblib` and `Vectorizer.joblib` in the `models/` directory.

## Usage
1. **Run the application**:
    ```sh
    python app.py
    ```

2. **Access the web interface**:
    Open your web browser and navigate to `http://127.0.0.1:5000/`.

## Usage
Home Page: Enter a restaurant review in the text box and click the submit button.

Result Page: Displays whether the review is positive or negative based on the model's prediction.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.


This updated `README.md` file includes the necessary instructions for setting up and running the Flask application, reflecting the given directory structure and the specific requirements of the project. Adjust paths, repository URLs, and other details as needed to match your specific setup and requirements.


   
