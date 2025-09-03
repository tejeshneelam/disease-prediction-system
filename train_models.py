# disease_prediction_project/train_models.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import joblib
import os

# The generic function to train a model for a specific disease.
def train_and_save_model(model, data_path, model_name, target_column):
    """
    Trains a machine learning model on a given dataset and saves it to a file.
    
    Args:
        model: The machine learning model (e.g., RandomForestClassifier).
        data_path: The path to the CSV file containing the dataset.
        model_name: The name for the saved model file (e.g., 'heart_model').
        target_column: The name of the column in the dataset you want to predict.
    """
    try:
        df = pd.read_csv(data_path)
        
        # --- NEW CODE FOR DATA PREPROCESSING ---
        # Handle the specific categorical columns in the diabetes dataset
        if model_name == 'diabetes_model':
            # Identify columns with object data type
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            # Perform One-Hot Encoding
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            
        # Separate features (X) from the target variable (y)
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Split the data into training and testing sets (80% for training)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Create the directory if it doesn't exist
        os.makedirs('prediction/models', exist_ok=True)
        
        # Save the trained model to a file
        joblib.dump(model, f'prediction/models/{model_name}.pkl')
        print(f"✅ {model_name} trained and saved successfully.")
    except Exception as e:
        print(f"❌ Error training {model_name}: {e}")

# ... (the if __name__ == '__main__': block remains the same) ...
if __name__ == '__main__':
    heart_data_path = 'data/heart.csv'
    diabetes_data_path = 'data/diabetes.csv'
    liver_data_path = 'data/Liver.csv'

    train_and_save_model(RandomForestClassifier(), heart_data_path, 'heart_model', 'target')
    
    train_and_save_model(KNeighborsClassifier(), liver_data_path, 'liver_model', 'Diagnosis')
    
    train_and_save_model(XGBClassifier(), diabetes_data_path, 'diabetes_model', 'diabetes')