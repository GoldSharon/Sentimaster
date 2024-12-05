# Sentimaster: Sentiment Analysis Tool

Sentimaster is a web-based sentiment analysis tool developed to analyze restaurant reviews using AI-powered sentiment classification. It is built using a GPT-2 model, fine-tuned for sentiment analysis in the restaurant domain. This tool provides businesses with real-time actionable insights from customer feedback, improving customer service and decision-making.

## Tech Stack

- **GPT-2** (124M parameters) for sentiment analysis model
- **Flask** for backend web framework
- **AWS EC2** for deployment
- **HTML**, **CSS**, **JavaScript** for frontend development
- **PyTorch** for model training and inference
- **Tiktoken** for tokenization

## Features

- **Sentiment Analysis**: Classifies restaurant reviews as either "Positive" or "Negative".
- **Real-Time Feedback**: Users can submit reviews through the web interface, and receive real-time sentiment analysis.
- **Model**: The sentiment analysis model is based on GPT-2 (124M parameters) and is fine-tuned on restaurant-specific data.
- **Deployment**: Deployed using Flask, making the tool accessible via a web browser.

## Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/GoldSharon/Sentimaster.git
Navigate to the project directory:

bash
Copy code
cd Sentimaster
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Ensure that you have the pretrained model weights (model_and_optimizer.pth) placed in the project directory. If not, you can train the model by following the training guidelines (refer to the model's documentation or script).

Usage
Run the Flask app:

bash
Copy code
python app.py
The application will be available at http://127.0.0.1:5000/.

Open your browser and go to the URL. You can enter a restaurant review in the text field, and the model will classify the sentiment as "Positive" or "Negative".

Model Architecture
GPT-2 (124M parameters) is used for sentiment analysis. The model has been fine-tuned on restaurant-related reviews to improve accuracy in analyzing customer feedback.
The model consists of 12 layers, 768 embedding dimensions, and 12 attention heads.
The model was trained and optimized using Adam optimizer with a learning rate of 0.0004.
API Endpoints
/: Home route, renders the input form for the review.
/submit: POST method, accepts the review and returns sentiment classification.
/result: Displays the result of the sentiment analysis.
Example Workflow
The user submits a restaurant review via the form.
The model classifies the sentiment of the review (Positive/Negative).
The result is displayed on a new page, showing whether the review is positive or negative.
Impact
By using Sentimaster, businesses can gain insights from customer feedback, which can be used to:

Improve customer service.
Understand customer satisfaction.
Make data-driven decisions for business improvement.
Contributions
Feel free to fork the repository, create issues, and submit pull requests. Contributions are welcome!

License
This project is licensed under the MIT License.


This README file provides an overview of the project, how to set it up, and usage instructions, along with
