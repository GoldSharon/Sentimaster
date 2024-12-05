# **Sentimaster: Sentiment Analysis Tool**

Sentimaster is a web-based sentiment analysis tool developed to analyze restaurant reviews using AI-powered sentiment classification. Built using a fine-tuned GPT-2 model, Sentimaster empowers businesses with real-time actionable insights from customer feedback to improve customer service and decision-making.

---

## **Tech Stack**

- **GPT-2** (124M parameters) for sentiment analysis model  
- **Flask** for backend web framework  
- **AWS EC2** for deployment  
- **HTML**, **CSS**, **JavaScript** for frontend development  
- **PyTorch** for model training and inference  
- **Tiktoken** for tokenization  

---

## **Features**

- **Sentiment Analysis**: Classifies restaurant reviews as either "Positive" or "Negative".  
- **Real-Time Feedback**: Users can submit reviews through the web interface and receive real-time sentiment analysis.  
- **Model**: Fine-tuned GPT-2 (124M parameters) on restaurant-specific data for improved accuracy.  
- **Deployment**: Deployed using Flask, accessible via a web browser.  

---

## **Installation**

1. Clone this repository to your local machine:  
   ```bash
   git clone https://github.com/GoldSharon/Sentimaster.git
   ```

2. Navigate to the project directory:  
   ```bash
   cd Sentimaster
   ```

3. Install the required dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure that you have the pretrained model weights (`model_and_optimizer.pth`) placed in the project directory.  
   - If the model weights are not available, follow the training guidelines provided in the documentation or scripts to train the model.  

---

## **Usage**

1. Run the Flask app:  
   ```bash
   python app.py
   ```

2. The application will be available at:  
   [http://127.0.0.1:5000/](http://127.0.0.1:5000/)  

3. Open your browser and go to the URL. Enter a restaurant review in the text field, and the model will classify the sentiment as "Positive" or "Negative".  

---

## **Model Architecture**

- **GPT-2 (124M parameters)** is used for sentiment analysis, fine-tuned on restaurant-related reviews to improve domain-specific accuracy.  
- **Architecture Details**:  
  - 12 layers, 768 embedding dimensions, and 12 attention heads.  
  - Trained and optimized using the Adam optimizer with a learning rate of 0.0004.  

---

## **API Endpoints**

- `/`: Home route, renders the input form for the review.  
- `/submit`: POST method, accepts the review and returns sentiment classification.  
- `/result`: Displays the result of the sentiment analysis.  

---

## **Example Workflow**

1. The user submits a restaurant review via the form.  
2. The model classifies the sentiment of the review (Positive/Negative).  
3. The result is displayed on a new page, showing whether the review is positive or negative.  

---

## **Impact**

By using Sentimaster, businesses can:  
- Gain insights from customer feedback.  
- Improve customer service and satisfaction.  
- Make data-driven decisions for business growth and improvement.  

---

## **Contributions**

Feel free to fork the repository, create issues, and submit pull requests. Contributions are always welcome!  

---

## **License**

This project is licensed under the **MIT License**.  

---
