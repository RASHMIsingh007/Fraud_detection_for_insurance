Insurance Fraud Detection System

Overview

This project focuses on detecting potentially fraudulent insurance claims using Machine Learning. The goal is to help insurance companies identify suspicious claims at an early stage, reducing financial losses and improving claim verification efficiency.

The system is built using Python and Scikit-learn, with a trained Random Forest Classifier used to classify claims as fraudulent or non-fraudulent. A simple web interface is also included, allowing users to interact with the model and make predictions without writing any code.

---

Features

- Data preprocessing and cleaning
- Handling of missing values
- Encoding of categorical variables
- Machine Learning model training using Random Forest
- Hyperparameter tuning and model evaluation
- Fraud prediction on new insurance claims
- Streamlit-based web application for easy interaction
- Model persistence using Joblib

---

Project Structure

Fraud_detection_project/
│
├── data/
│   ├── X_train_cleaned.csv
│   ├── X_test_cleaned.csv
│   ├── X_val_cleaned.csv
│   └── Auto_Insurance_Fraud_Claims_Results_Submission.csv
│
├── output/
│   ├── best_fraud_detection_model.pkl
│   ├── model_columns.pkl
│   └── fraud_submission_predictions.csv
│
├── scripts/
│   └── run_fraud_detection.py
│
├── web_app/
│   ├── fraud_detection_app.py
│   └── templates/
│
└── requirements.txt

---

Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib
- Seaborn
- Plotly

---

Dataset

The project uses an insurance claims dataset containing information related to policyholders, claim details, and fraud indicators.

Typical attributes include:

- Customer information
- Policy details
- Claim amount
- Incident-related information
- Vehicle information
- Fraud indicator (target variable)

The target column used for prediction is:

Fraud_Ind

---

Model Development Process

1. Data Preparation

The dataset is loaded and validated before training. Missing values are handled appropriately, and categorical features are encoded for model compatibility.

2. Feature Engineering

- Label Encoding for categorical columns
- Data standardization where required
- Consistent feature mapping between training and prediction stages

3. Model Training

A Random Forest Classifier is trained on the processed dataset. Different parameter combinations are evaluated to improve model performance.

4. Model Evaluation

The model is evaluated using:

- Accuracy Score
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve Analysis

These metrics help assess how effectively the model identifies fraudulent claims.

---

Installation

Clone the Repository

git clone <repository-url>
cd Fraud_detection_project

Install Dependencies

pip install -r requirements.txt

---

Running the Training Pipeline

Execute the training script:

python scripts/run_fraud_detection.py

The script will:

1. Load and preprocess the data.
2. Train the fraud detection model.
3. Evaluate performance.
4. Save the trained model files in the "output" directory.

---

Running the Web Application

Launch the Streamlit application:

streamlit run web_app/fraud_detection_app.py

After starting the server, open the provided local URL in your browser.

The application allows users to:

- Enter claim-related information
- Generate fraud predictions
- View fraud probability scores
- Analyze prediction outcomes through visualizations

---

Output Files

File| Description
best_fraud_detection_model.pkl| Trained fraud detection model
model_columns.pkl| Feature metadata and encoding information
fraud_submission_predictions.csv| Generated prediction results

---

Future Improvements

Some possible enhancements for future versions include:

- Integration with real-time insurance claim systems
- Deployment on cloud platforms
- Deep Learning-based fraud detection models
- Automated feature selection
- Explainable AI dashboards for claim investigation
- API support for enterprise integration

---

Conclusion

This project demonstrates how Machine Learning can be applied to insurance fraud detection by analyzing claim-related data and identifying suspicious patterns. The combination of a trained Random Forest model and a user-friendly web interface makes the solution practical for both technical and non-technical users.

The project serves as a strong foundation for developing intelligent fraud detection systems that can assist insurance companies in reducing risk and improving operational efficiency.

---

Author

Rashmi Chouhan

Developed as a Machine Learning project for insurance claim fraud detection and prediction.
