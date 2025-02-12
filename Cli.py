import pickle
import numpy as np
import pandas as pd
import argparse
import torch  
import torch.nn as nn  
import torch.optim as optim  
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_model(filename="best_model.pkl"):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"]

def predict(model, scaler, filename):
    
    df = pd.read_csv(filename, encoding='latin-1') 
    
    # ----> Select only the numerical and encoded categorical features used during training
    numerical_features = ['Age', 'Billing Amount']
    categorical_features = ['Gender', 'Blood Type', 'Medical Condition', 'Doctor', 'Hospital', 
                            'Insurance Provider', 'Admission Type', 'Medication', 'Room Number']
    
    X = df[numerical_features + categorical_features].values # <---- Create X using these features
    
    # ----> Scale only the numerical features
    X_scaled = scaler.transform(X[:, :len(numerical_features)])
    
    # ----> Combine scaled numerical and original categorical features for prediction
    X_final = np.concatenate([X_scaled, X[:, len(numerical_features):]], axis=1)
    
    # ----> Convert all columns of X_final to numerical representation using LabelEncoder if necessary
    for i in range(len(numerical_features), X_final.shape[1]): # Loop through categorical columns
        le = LabelEncoder()
        X_final[:, i] = le.fit_transform(X_final[:, i].astype(str)) # Convert to numerical labels

    # ----> Convert all columns of X_final to float32 for compatibility (was initially intended for PyTorch)
    X_final = X_final.astype(np.float32)

    # ----> Use model.predict instead of model(X_tensor)
    predictions = model.predict(X_final)  
    
    df['Test Results'] = predictions  # Add predictions to DataFrame
    
    # Return the predictions
    return df['Test Results'] 

def main():
    # Provide the correct file path for your dataset
    filename = "healthcare_dataset.csv" 
    
    model, scaler = load_model()
    
    # Get the predictions
    predictions = predict(model, scaler, filename)
    
    # Print or use the predictions 
    print(predictions) 
    print(predictions.value_counts())
    # or
    # print(predictions.head())  # To view the first few predictions

if __name__ == "__main__":
    main()
