Multiple Disease Prediction System

A web-based application that leverages Machine Learning to predict the risk of chronic diseases (Heart, Liver, and Diabetes) based on user-provided health metrics. The system provides instant risk assessment, a helpful chatbot, and a feature to locate nearby hospitals.

🚀 Key Features

Disease Prediction: Provides real-time risk predictions for Heart Disease, Diabetes, and Liver Disease.

Intuitive UI: A user-friendly front-end for easy data input.

Chatbot Assistant: A simple, rule-based chatbot helps users understand the required input fields.

Hospital Locator: Integrates with the Google Maps Places API to find nearby hospitals.

🧠 Machine Learning Algorithms Explained

This project uses three different machine learning algorithms, each chosen for its effectiveness in specific prediction tasks.

K-Nearest Neighbors (KNN):
KNN is a simple, supervised learning algorithm used for classification. It works by classifying a new data point based on the majority class of its "k" nearest neighbors in the training data. For example, to predict if a patient has liver disease, the algorithm looks at the 'k' most similar patients in the dataset and makes a prediction based on what most of those neighbors were diagnosed with.

Random Forest:
Random Forest is an ensemble learning method that builds multiple decision trees during training and merges their predictions to produce a single, more accurate prediction.  It is highly effective because it reduces the risk of overfitting by averaging out the biases of individual decision trees. It is particularly well-suited for complex datasets like those used for heart disease prediction.

XGBoost (Extreme Gradient Boosting):
XGBoost is a powerful and highly efficient implementation of gradient boosting. The core idea is to build a series of decision trees, where each new tree corrects the errors of the previous ones. It is known for its speed and performance, making it an excellent choice for tasks that require high accuracy, such as predicting diabetes from a large dataset.

💻 Tech Stack

Back-End: Django, Python, requests, python-dotenv

Front-End: HTML, CSS, JavaScript

Models: Random Forest, KNN, XGBoost

Services: Google Maps Places API

🛠️ Getting Started

Follow these steps to set up and run the project locally.

Clone the repository:


git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

Set up a virtual environment and install dependencies:
python -m venv venv
source venv/bin/activate
pip install Django pandas scikit-learn numpy xgboost joblib requests python-dotenv
Prepare datasets:

Download the required .csv files for each disease.

Create a data folder in the project root.

Place the files in the data folder.

Run python train_models.py to train and save the models.

Configure API Key:

Create a Google Cloud account and enable the Places API (New).

Create a .env file in the project's root directory and add your key:

GOOGLE_MAPS_API_KEY=your_api_key_here
Run the application:
python manage.py runserver
