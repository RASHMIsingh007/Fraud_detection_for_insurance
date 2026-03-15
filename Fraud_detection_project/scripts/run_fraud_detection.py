import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix, 
                           precision_recall_curve, roc_curve, auc, f1_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

class EnhancedRandomForestClassifier:
    def __init__(self, target_column='Fraud_Ind'):
        self.target_column = target_column
        self.label_encoders = {}
        self.y_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_names = None
        self.feature_dtypes = None
        self.categorical_columns = None
        self.target_classes = None
        self.n_features = None
        
    def load_data(self, train_path, test_path, val_path):
        """Load and validate datasets"""
        required_files = [train_path, test_path, val_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            print("\n💡 SOLUTIONS:")
            print("1. Ensure your CSV files are in the correct path")
            print("2. Check file names match exactly (case-sensitive)")
            print("3. Update file paths in the code")
            print(f"\n🔍 Current directory: {os.getcwd()}")
            print("📂 CSV files found in data/ directory:")
            data_dir = "data"
            if os.path.exists(data_dir):
                csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
                if csv_files:
                    for csv_file in csv_files:
                        print(f"  ✅ {csv_file}")
                else:
                    print("  ❌ No CSV files found in data/ directory")
            return False
        
        try:
            self.train_df = pd.read_csv(train_path)
            self.test_df = pd.read_csv(test_path)
            self.val_df = pd.read_csv(val_path)
            print("✅ Data loaded successfully")
            print(f"Train shape: {self.train_df.shape}")
            print(f"Test shape: {self.test_df.shape}")
            print(f"Validation shape: {self.val_df.shape}")
            
            # Display basic info about the dataset
            print(f"\n📊 Dataset Overview:")
            print(f"  🎯 Target column: '{self.target_column}'")
            if self.target_column in self.train_df.columns:
                if self.train_df[self.target_column].dtype == 'object':
                    fraud_rate = (self.train_df[self.target_column] == 'Y').mean()
                else:
                    fraud_rate = self.train_df[self.target_column].mean()
                print(f"  📈 Fraud rate in training: {fraud_rate:.2%}")
            print(f"  🔢 Total features: {self.train_df.shape[1] - 1}")
            
        except Exception as e:
            print(f"❌ Error reading CSV files: {e}")
            return False
        return True
    
    def preprocess_data(self):
        """Enhanced preprocessing with error handling"""
        # Check if target column exists
        if self.target_column not in self.train_df.columns:
            print(f"❌ Target column '{self.target_column}' not found in data")
            print(f"Available columns: {list(self.train_df.columns)}")
            return False
        
        # Handle validation set missing target column
        if self.target_column not in self.val_df.columns:
            print(f"⚠️  Target column '{self.target_column}' not found in validation set")
            print("This is normal for submission datasets. Creating dummy target for validation set.")
            # Create dummy target using training data's most frequent class
            most_frequent_class = self.train_df[self.target_column].mode()[0]
            self.val_df[self.target_column] = most_frequent_class
            
        # Separate features and target
        self.X_train = self.train_df.drop(columns=[self.target_column])
        self.y_train = self.train_df[self.target_column]
        
        self.X_test = self.test_df.drop(columns=[self.target_column])
        self.y_test = self.test_df[self.target_column]
        
        self.X_val = self.val_df.drop(columns=[self.target_column])
        self.y_val = self.val_df[self.target_column]
        
        # Store feature names and data types
        self.feature_names = self.X_train.columns.tolist()
        self.feature_dtypes = {col: str(self.X_train[col].dtype) for col in self.feature_names}
        
        # Handle missing values
        self._handle_missing_values()
        
        # Encode features
        self.X_train = self._encode_features(self.X_train, fit=True)
        self.X_test = self._encode_features(self.X_test, fit=False)
        self.X_val = self._encode_features(self.X_val, fit=False)
        
        # Store categorical columns info
        self.categorical_columns = list(self.label_encoders.keys())
        
        # Encode target - FIT ONLY ON TRAINING DATA
        self.y_train = self.y_encoder.fit_transform(self.y_train)
        self.y_test = self.y_encoder.transform(self.y_test)
        
        # Handle validation target encoding with unseen labels
        try:
            self.y_val = self.y_encoder.transform(self.y_val)
        except ValueError as e:
            if "previously unseen labels" in str(e):
                print(f"⚠️  Validation set contains unseen target values. Using most frequent class.")
                # Replace unseen values with the most frequent training class
                most_frequent_encoded = self.y_encoder.transform([self.train_df[self.target_column].mode()[0]])[0]
                self.y_val = np.full(len(self.y_val), most_frequent_encoded)
            else:
                raise e
        
        # Store target classes and feature count
        self.target_classes = self.y_encoder.classes_
        self.n_features = len(self.feature_names)
        
        print("✅ Data preprocessing completed")
        print(f"  🔤 Categorical features encoded: {len(self.categorical_columns)}")
        print(f"  🔢 Numerical features: {self.n_features - len(self.categorical_columns)}")
        print(f"  🎯 Target classes: {list(self.target_classes)}")
        
        return True
        
    def _handle_missing_values(self):
        """Handle missing values in datasets"""
        for df_name, df in [('train', self.X_train), ('test', self.X_test), ('val', self.X_val)]:
            missing_cols = df.isnull().sum()
            if missing_cols.sum() > 0:
                print(f"⚠️  Missing values in {df_name} set:")
                print(missing_cols[missing_cols > 0])
                
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col].fillna('Unknown', inplace=True)
                    else:
                        df[col].fillna(df[col].median(), inplace=True)
    
    def _encode_features(self, df, fit=True):
        """Enhanced feature encoding with error handling"""
        df_encoded = df.copy()
        
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    if fit:
                        le = LabelEncoder()
                        df_encoded[col] = le.fit_transform(df[col].astype(str))
                        self.label_encoders[col] = le
                    else:
                        le = self.label_encoders.get(col)
                        if le:
                            unique_vals = set(df[col].astype(str).unique())
                            known_vals = set(le.classes_)
                            unseen_vals = unique_vals - known_vals
                            
                            if unseen_vals:
                                print(f"⚠️  Unseen categories in {col}: {unseen_vals}")
                                most_frequent = le.classes_[0]
                                df_encoded[col] = df[col].astype(str).replace(list(unseen_vals), most_frequent)
                            
                            df_encoded[col] = le.transform(df_encoded[col].astype(str))
                except Exception as e:
                    print(f"❌ Error encoding column {col}: {e}")
                    
        return df_encoded
    
    def fast_optimize_hyperparameters(self):
        """Fast hyperparameter optimization - completes in 10-15 minutes"""
        print("🔧 Starting FAST hyperparameter optimization...")
        
        # Much smaller parameter grid - only 4 combinations
        param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [15, 20],
            'min_samples_split': [5],
            'min_samples_leaf': [2],
            'max_features': ['sqrt'],
            'class_weight': ['balanced']
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid,
            cv=3,  # Only 3 folds instead of 5
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        self.best_model = grid_search.best_estimator_
        
        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def train_model(self, use_optimization='fast'):
        """Train the model with different optimization levels"""
        if use_optimization == 'fast':
            print("⚡ Using fast optimization (10-15 minutes)...")
            self.fast_optimize_hyperparameters()
        elif use_optimization == 'none':
            print("🚀 Using optimized default parameters (2-3 minutes)...")
            # Use proven good parameters without any search
            self.best_model = RandomForestClassifier(
                n_estimators=200,           # Good balance of performance and speed
                max_depth=20,              # Prevents overfitting
                min_samples_split=5,       # Robust splits
                min_samples_leaf=2,        # Prevents overfitting
                max_features='sqrt',       # Good for fraud detection
                bootstrap=True,            # Standard for RF
                class_weight='balanced',   # Critical for imbalanced fraud data
                random_state=42,
                n_jobs=-1                  # Use all CPU cores
            )
            self.best_model.fit(self.X_train, self.y_train)
        else:
            # Full optimization (the slow one you experienced)
            print("🐌 Using full optimization (may take hours)...")
            self.optimize_hyperparameters()
        
        print("✅ Model training completed")
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        results = {}
        
        y_pred_test = self.best_model.predict(self.X_test)
        y_pred_val = self.best_model.predict(self.X_val)
        
        y_pred_proba_test = self.best_model.predict_proba(self.X_test)[:, 1]
        y_pred_proba_val = self.best_model.predict_proba(self.X_val)[:, 1]
        
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        test_f1 = f1_score(self.y_test, y_pred_test, average='weighted')
        test_report = classification_report(self.y_test, y_pred_test, 
                                          target_names=self.y_encoder.classes_, 
                                          output_dict=True)
        
        val_accuracy = accuracy_score(self.y_val, y_pred_val)
        val_f1 = f1_score(self.y_val, y_pred_val, average='weighted')
        val_report = classification_report(self.y_val, y_pred_val, 
                                         target_names=self.y_encoder.classes_, 
                                         output_dict=True)
        
        results = {
            'test': {'accuracy': test_accuracy, 'f1': test_f1, 'report': test_report},
            'validation': {'accuracy': val_accuracy, 'f1': val_f1, 'report': val_report}
        }
        
        print("🔍 TEST SET EVALUATION:")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"F1-Score: {test_f1:.4f}")
        print(classification_report(self.y_test, y_pred_test, target_names=self.y_encoder.classes_))
        
        print("\n🧪 VALIDATION SET EVALUATION:")
        print(f"Accuracy: {val_accuracy:.4f}")
        print(f"F1-Score: {val_f1:.4f}")
        print(classification_report(self.y_val, y_pred_val, target_names=self.y_encoder.classes_))
        
        return results
    
    def get_feature_importance(self, top_n=10):
        """Get and display feature importance"""
        if self.best_model is None:
            print("❌ Model not trained yet")
            return None
            
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 TOP {top_n} MOST IMPORTANT FEATURES:")
        print(importance_df.head(top_n))
        
        return importance_df
    
    def cross_validate_model(self, cv_folds=3):
        """Perform cross-validation"""
        if self.best_model is None:
            print("❌ Model not trained yet")
            return None
            
        cv_scores = cross_val_score(self.best_model, self.X_train, self.y_train, 
                                   cv=cv_folds, scoring='f1_weighted', n_jobs=-1)
        
        print(f"\n🔄 {cv_folds}-FOLD CROSS-VALIDATION RESULTS:")
        print(f"Mean F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Individual scores: {cv_scores}")
        
        return cv_scores
    
    def save_model_components(self, model_filepath='output/best_fraud_detection_model.pkl', 
                             columns_filepath='output/model_columns.pkl'):
        """Save the trained model and column information separately"""
        if self.best_model is None:
            print("❌ No model to save")
            return
            
        os.makedirs('output', exist_ok=True)
            
        model_package = {
            'model': self.best_model,
            'label_encoders': self.label_encoders,
            'y_encoder': self.y_encoder,
            'target_column': self.target_column
        }
        
        columns_package = {
            'feature_names': self.feature_names,
            'feature_dtypes': self.feature_dtypes,
            'categorical_columns': self.categorical_columns,
            'target_column': self.target_column,
            'target_classes': self.target_classes,
            'n_features': self.n_features
        }
        
        joblib.dump(model_package, model_filepath)
        joblib.dump(columns_package, columns_filepath)
        
        print(f"✅ Model saved to {model_filepath}")
        print(f"✅ Model columns saved to {columns_filepath}")
        
        return model_filepath, columns_filepath
    
    def load_model_components(self, model_filepath='output/best_fraud_detection_model.pkl', 
                             columns_filepath='output/model_columns.pkl'):
        """Load both model and column information"""
        try:
            model_package = joblib.load(model_filepath)
            self.best_model = model_package['model']
            self.label_encoders = model_package['label_encoders']
            self.y_encoder = model_package['y_encoder']
            self.target_column = model_package['target_column']
            
            columns_package = joblib.load(columns_filepath)
            self.feature_names = columns_package['feature_names']
            self.feature_dtypes = columns_package['feature_dtypes']
            self.categorical_columns = columns_package['categorical_columns']
            self.target_classes = columns_package['target_classes']
            self.n_features = columns_package['n_features']
            
            print(f"✅ Model loaded from {model_filepath}")
            print(f"✅ Model columns loaded from {columns_filepath}")
            print(f"📊 Expected features: {self.n_features}")
            print(f"🔤 Categorical columns: {len(self.categorical_columns)}")
            
        except FileNotFoundError as e:
            print(f"❌ File not found: {e}")
    
    def predict_new_data(self, new_data_df):
        """Make predictions on new data with validation"""
        if self.best_model is None:
            print("❌ No trained model available")
            return None
        
        # Select only the required features in correct order
        new_data_processed = new_data_df[self.feature_names].copy()
        
        # Handle missing values
        for col in new_data_processed.columns:
            if new_data_processed[col].dtype == 'object':
                new_data_processed[col].fillna('Unknown', inplace=True)
            else:
                new_data_processed[col].fillna(new_data_processed[col].median(), inplace=True)
        
        # Encode categorical features
        new_data_encoded = self._encode_features(new_data_processed, fit=False)
        
        # Make predictions
        predictions = self.best_model.predict(new_data_encoded)
        prediction_probabilities = self.best_model.predict_proba(new_data_encoded)
        
        # Convert back to original labels
        predicted_labels = self.y_encoder.inverse_transform(predictions)
        
        results_df = pd.DataFrame({
            'prediction': predicted_labels,
            'confidence': np.max(prediction_probabilities, axis=1)
        })
        
        # Add probability columns for each class
        for i, class_name in enumerate(self.y_encoder.classes_):
            results_df[f'prob_{class_name}'] = prediction_probabilities[:, i]
        
        return results_df

def main(speed_mode='fast'):
    """Main function with speed options
    
    Args:
        speed_mode (str): 'fast' (10-15 min), 'instant' (2-3 min), or 'full' (hours)
    """
    print("🚀 Starting Enhanced Random Forest Classification Pipeline")
    print("=" * 60)
    
    classifier = EnhancedRandomForestClassifier(target_column='Fraud_Ind')
    
    print(f"\n⚡ Speed Mode: {speed_mode.upper()}")
    if speed_mode == 'fast':
        print("📝 Will complete in 10-15 minutes with some optimization")
    elif speed_mode == 'instant':
        print("📝 Will complete in 2-3 minutes with optimized defaults")
    else:
        print("📝 Full optimization - may take hours")
    
    print("\n📁 LOADING DATA...")
    if not classifier.load_data(
        train_path="data/X_train_cleaned.csv",
        test_path="data/X_test_cleaned.csv", 
        val_path="data/X_val_cleaned.csv"
    ):
        return None
    
    print("\n🔧 PREPROCESSING DATA...")
    result = classifier.preprocess_data()
    if result == False:
        return None
    
    print("\n🤖 TRAINING MODEL...")
    if speed_mode == 'fast':
        classifier.train_model(use_optimization='fast')
    elif speed_mode == 'instant':
        classifier.train_model(use_optimization='none')
    else:
        classifier.train_model(use_optimization=True)
    
    print("\n📊 EVALUATING MODEL...")
    results = classifier.evaluate_model()
    
    print("\n🎯 ANALYZING FEATURE IMPORTANCE...")
    importance_df = classifier.get_feature_importance(top_n=15)
    
    print("\n🔄 CROSS-VALIDATION...")
    cv_scores = classifier.cross_validate_model()
    
    print("\n💾 SAVING MODEL...")
    classifier.save_model_components()
    
    print("\n🎯 SAVED FILES:")
    print("✅ output/best_fraud_detection_model.pkl")
    print("✅ output/model_columns.pkl")
    
    print(f"\n🏆 PIPELINE COMPLETED SUCCESSFULLY IN {speed_mode.upper()} MODE!")
    print("=" * 60)
    
    return classifier

def make_predictions_on_submission():
    """Make predictions on the submission dataset"""
    print("\n🔮 MAKING PREDICTIONS ON SUBMISSION DATA")
    print("=" * 50)
    
    classifier = EnhancedRandomForestClassifier()
    classifier.load_model_components()
    
    try:
        submission_data = pd.read_csv('data/Auto_insurance_fraud_claims_results_submission.csv')
        print(f"📄 Submission data loaded: {submission_data.shape}")
        
        predictions = classifier.predict_new_data(submission_data)
        
        submission_df = pd.DataFrame({
            'Claim_ID': submission_data['Claim_ID'],
            'Fraud_Prediction': predictions['prediction'],
            'Fraud_Probability': predictions['prob_Y'] if 'prob_Y' in predictions.columns else predictions['confidence']
        })
        
        submission_df.to_csv('output/fraud_submission_predictions.csv', index=False)
        print("✅ Predictions saved to output/fraud_submission_predictions.csv")
        
        fraud_count = (submission_df['Fraud_Prediction'] == 'Y').sum()
        total_count = len(submission_df)
        print(f"\n📊 PREDICTION SUMMARY:")
        print(f"  Total claims: {total_count}")
        print(f"  Predicted fraud: {fraud_count}")
        print(f"  Fraud rate: {fraud_count/total_count:.2%}")
        
        return submission_df
        
    except FileNotFoundError:
        print("❌ Submission file not found")
        return None

if __name__ == "__main__":
    # Choose your speed mode:
    
    # RECOMMENDED: Fast mode (10-15 minutes with some optimization)
    classifier = main(speed_mode='fast')
    
    # ALTERNATIVE: Instant mode (2-3 minutes with good defaults)
    # classifier = main(speed_mode='instant')
    
    # NOT RECOMMENDED: Full mode (hours of optimization)
    # classifier = main(speed_mode='full')
    
    # Make predictions on submission data
    if classifier is not None:
        predictions = make_predictions_on_submission()
