import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="🛡️ Insurance Fraud Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .safe-alert {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

class FraudDetectionApp:
    def __init__(self):
        self.classifier = None
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load the trained fraud detection model"""
        try:
            # Try to load model components
            model_package = joblib.load('C:\\Users\\KIIT\\Downloads\\Fraud_detection_project\\Fraud_detection_project\\output\\best_fraud_detection_model.pkl')
            columns_package = joblib.load('C:\\Users\\KIIT\\Downloads\\Fraud_detection_project\\Fraud_detection_project\\output\\model_columns.pkl')
            
            # Create model wrapper
            class ModelWrapper:
                def __init__(self, model_pkg, col_pkg):
                    self.best_model = model_pkg['model']
                    self.label_encoders = model_pkg['label_encoders']
                    self.y_encoder = model_pkg['y_encoder']
                    self.target_column = model_pkg['target_column']
                    self.feature_names = col_pkg['feature_names']
                    self.categorical_columns = col_pkg['categorical_columns']
                    self.target_classes = col_pkg['target_classes']
                    self.feature_dtypes = col_pkg['feature_dtypes']
                    self.n_features = col_pkg['n_features']
            
            self.classifier = ModelWrapper(model_package, columns_package)
            self.model_loaded = True
            
            # Cache common values for UI
            self._cache_feature_info()
            return True
            
        except FileNotFoundError as e:
            st.error(f"❌ Model files not found: {e}")
            st.info("💡 Please ensure model files exist in the 'output' directory")
            return False
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return False
    
    def _cache_feature_info(self):
        """Cache feature information for better performance"""
        if not self.classifier:
            return
            
        self.numerical_features = [
            f for f in self.classifier.feature_names 
            if f not in self.classifier.categorical_columns
        ]
        
        # Get unique values for categorical features (sample from encoders)
        self.categorical_options = {}
        for col in self.classifier.categorical_columns:
            if col in self.classifier.label_encoders:
                # Get first 10 classes for dropdown options
                classes = self.classifier.label_encoders[col].classes_[:10]
                self.categorical_options[col] = list(classes) + ['Unknown']
    
    def predict_fraud(self, input_data):
        """Make fraud prediction on input data"""
        if not self.model_loaded:
            return None
            
        try:
            # Create DataFrame from input
            df = pd.DataFrame([input_data])
            
            # Ensure all required features are present
            for feature in self.classifier.feature_names:
                if feature not in df.columns:
                    if feature in self.classifier.categorical_columns:
                        df[feature] = 'Unknown'
                    else:
                        df[feature] = 0
            
            # Select only model features in correct order
            df = df[self.classifier.feature_names]
            
            # Handle missing values
            for col in df.columns:
                if df[col].isna().any():
                    if col in self.classifier.categorical_columns:
                        df[col].fillna('Unknown', inplace=True)
                    else:
                        df[col].fillna(0, inplace=True)
            
            # Encode categorical features
            df_encoded = df.copy()
            for col in self.classifier.categorical_columns:
                if col in df_encoded.columns:
                    le = self.classifier.label_encoders.get(col)
                    if le:
                        try:
                            # Handle unseen categories
                            values = df_encoded[col].astype(str)
                            unknown_mask = ~values.isin(le.classes_)
                            if unknown_mask.any():
                                values[unknown_mask] = le.classes_[0]  # Use first class as default
                            df_encoded[col] = le.transform(values)
                        except Exception as e:
                            # Fallback: use first class
                            df_encoded[col] = le.transform([le.classes_[0]])[0]
            
            # Make prediction
            prediction = self.classifier.best_model.predict(df_encoded)[0]
            probabilities = self.classifier.best_model.predict_proba(df_encoded)[0]
            
            # Convert prediction back to original label
            predicted_label = self.classifier.y_encoder.inverse_transform([prediction])[0]
            
            return {
                'prediction': predicted_label,
                'is_fraud': predicted_label == 'Y',
                'confidence': float(max(probabilities)),
                'fraud_probability': float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0]),
                'no_fraud_probability': float(probabilities[0]) if len(probabilities) > 1 else float(1 - probabilities[0])
            }
            
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            return None

def main():
    # Initialize app
    app = FraudDetectionApp()
    
    # Main header
    st.markdown('<h1 class="main-header">🛡️ Insurance Fraud Detection System</h1>', unsafe_allow_html=True)
    
    if not app.model_loaded:
        st.error("❌ Model not loaded. Please train your model first and ensure files exist in 'output/' directory.")
        return
    
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type:",
        ["🔍 Single Claim Analysis", "📊 Batch Processing", "📈 Model Information", "🎯 Feature Explorer"]
    )
    
    # Display selected page
    if page == "🔍 Single Claim Analysis":
        single_claim_page(app)
    elif page == "📊 Batch Processing":
        batch_processing_page(app)
    elif page == "📈 Model Information":
        model_info_page(app)
    elif page == "🎯 Feature Explorer":
        feature_explorer_page(app)

def single_claim_page(app):
    st.header("🔍 Single Claim Fraud Analysis")
    st.markdown("---")
    
    # Create two modes: Simple and Advanced
    mode = st.radio("Select Input Mode:", ["🚀 Quick Analysis", "🔧 Advanced Input"], horizontal=True)
    
    if mode == "🚀 Quick Analysis":
        quick_analysis_form(app)
    else:
        advanced_analysis_form(app)

def quick_analysis_form(app):
    """Simplified form with most important features"""
    st.subheader("Quick Fraud Check")
    st.info("📝 Enter key claim details for instant fraud assessment")
    
    with st.form("quick_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**💰 Financial Details**")
            total_claim = st.number_input("Total Claim Amount ($)", min_value=0, max_value=200000, value=10000, step=500)
            policy_premium = st.number_input("Policy Premium ($)", min_value=0, max_value=10000, value=1500, step=100)
            vehicle_cost = st.number_input("Vehicle Value ($)", min_value=0, max_value=150000, value=25000, step=1000)
        
        with col2:
            st.markdown("**👤 Personal Info**")
            age = st.slider("Age", 18, 80, 35)
            gender = st.selectbox("Gender", ['MALE', 'FEMALE'])
            education = st.selectbox("Education Level", ['High School', 'College', 'Masters', 'PhD'])
        
        with col3:
            st.markdown("**🚗 Incident Details**")
            accident_severity = st.selectbox("Accident Severity", ['Minor Damage', 'Major Damage', 'Total Loss'])
            police_report = st.selectbox("Police Report Filed?", ['YES', 'NO'])
            witnesses = st.slider("Number of Witnesses", 0, 10, 2)
        
        submitted = st.form_submit_button("🔍 Analyze Claim", use_container_width=True)
        
        if submitted:
            # Create prediction data with defaults for missing features
            prediction_data = {}
            
            # Fill all required features with defaults
            for feature in app.classifier.feature_names:
                if feature in app.classifier.categorical_columns:
                    prediction_data[feature] = 'Unknown'
                else:
                    prediction_data[feature] = 0
            
            # Override with user inputs (map to likely column names)
            user_mappings = {
                'Total_Claim': total_claim,
                'Policy_Premium': policy_premium,
                'Vehicle_Cost': vehicle_cost,
                'Age_Insured': age,
                'Gender': gender,
                'Education': education,
                'Accident_Severity': accident_severity,
                'Police_Report': police_report,
                'Witnesses': witnesses
            }
            
            # Update only if column exists in model
            for col, value in user_mappings.items():
                if col in app.classifier.feature_names:
                    prediction_data[col] = value
            
            # Make prediction
            result = app.predict_fraud(prediction_data)
            
            if result:
                display_prediction_results(result, total_claim)

def advanced_analysis_form(app):
    """Advanced form showing actual model features"""
    st.subheader("Advanced Analysis - Real Model Features")
    
    # Feature selection
    st.info("🔧 Customize which features to input (others will use default values)")
    
    # Let user select which features to input
    categorical_to_show = st.multiselect(
        "Select Categorical Features to Input:",
        app.classifier.categorical_columns[:20],  # Limit to first 20
        default=app.classifier.categorical_columns[:5]  # Default first 5
    )
    
    numerical_to_show = st.multiselect(
        "Select Numerical Features to Input:",
        app.numerical_features[:20],  # Limit to first 20
        default=app.numerical_features[:5]  # Default first 5
    )
    
    with st.form("advanced_form"):
        prediction_data = {}
        
        # Initialize all features with defaults
        for feature in app.classifier.feature_names:
            if feature in app.classifier.categorical_columns:
                prediction_data[feature] = 'Unknown'
            else:
                prediction_data[feature] = 0
        
        if categorical_to_show:
            st.markdown("**📝 Categorical Features**")
            col1, col2 = st.columns(2)
            
            for i, feature in enumerate(categorical_to_show):
                column = col1 if i % 2 == 0 else col2
                with column:
                    options = app.categorical_options.get(feature, ['Unknown'])
                    value = st.selectbox(f"{feature}:", options, key=f"cat_{feature}")
                    prediction_data[feature] = value
        
        if numerical_to_show:
            st.markdown("**🔢 Numerical Features**")
            col1, col2 = st.columns(2)
            
            for i, feature in enumerate(numerical_to_show):
                column = col1 if i % 2 == 0 else col2
                with column:
                    value = st.number_input(f"{feature}:", value=0.0, key=f"num_{feature}")
                    prediction_data[feature] = value
        
        submitted = st.form_submit_button("🔍 Run Advanced Analysis", use_container_width=True)
        
        if submitted:
            result = app.predict_fraud(prediction_data)
            if result:
                display_prediction_results(result)

def display_prediction_results(result, claim_amount=None):
    """Display prediction results with visual indicators"""
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    # Main result
    if result['is_fraud']:
        st.markdown('<div class="fraud-alert"><h2>🚨 FRAUD DETECTED</h2><p>This claim shows high risk indicators</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="safe-alert"><h2>✅ LEGITIMATE CLAIM</h2><p>This claim appears to be genuine</p></div>', unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fraud_prob = result['fraud_probability'] * 100
        st.metric(
            "Fraud Probability", 
            f"{fraud_prob:.1f}%",
            delta=f"{'High' if fraud_prob > 50 else 'Low'} Risk"
        )
    
    with col2:
        confidence = result['confidence'] * 100
        st.metric(
            "Model Confidence", 
            f"{confidence:.1f}%",
            delta=f"{'High' if confidence > 80 else 'Medium' if confidence > 60 else 'Low'} Confidence"
        )
    
    with col3:
        st.metric(
            "Prediction", 
            "FRAUD" if result['is_fraud'] else "LEGITIMATE",
            delta="⚠️" if result['is_fraud'] else "✅"
        )
    
    with col4:
        if claim_amount:
            risk_level = "High" if fraud_prob > 70 else "Medium" if fraud_prob > 30 else "Low"
            st.metric("Risk Assessment", risk_level, delta=f"${claim_amount:,.0f}")
    
    # Probability visualization
    st.subheader("📈 Probability Breakdown")
    
    # Create probability chart
    prob_data = pd.DataFrame({
        'Category': ['Fraud', 'Legitimate'],
        'Probability': [result['fraud_probability'], result['no_fraud_probability']],
        'Color': ['#ff4444', '#44ff44']
    })
    
    fig = px.bar(
        prob_data, 
        x='Category', 
        y='Probability',
        color='Category',
        color_discrete_map={'Fraud': '#ff4444', 'Legitimate': '#44ff44'},
        title="Fraud vs Legitimate Probability"
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk gauge
    st.subheader("🎯 Risk Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = fraud_prob,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fraud Risk Level (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig_gauge.update_layout(height=400)
    st.plotly_chart(fig_gauge, use_container_width=True)

def batch_processing_page(app):
    st.header("📊 Batch Processing")
    st.markdown("Upload a CSV file with multiple claims for bulk fraud analysis")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded! Found {len(df)} claims")
            
            # Preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10))
            
            # Check columns
            missing_features = set(app.classifier.feature_names) - set(df.columns)
            if missing_features:
                st.warning(f"⚠️ Missing {len(missing_features)} features. Will use defaults for: {list(missing_features)[:5]}...")
            
            if st.button("🚀 Process All Claims", type="primary"):
                progress_bar = st.progress(0)
                results = []
                
                for i, (idx, row) in enumerate(df.iterrows()):
                    # Prepare data
                    claim_data = {}
                    for feature in app.classifier.feature_names:
                        if feature in row:
                            claim_data[feature] = row[feature]
                        elif feature in app.classifier.categorical_columns:
                            claim_data[feature] = 'Unknown'
                        else:
                            claim_data[feature] = 0
                    
                    # Predict
                    result = app.predict_fraud(claim_data)
                    if result:
                        results.append({
                            'Claim_ID': idx,
                            'Prediction': 'FRAUD' if result['is_fraud'] else 'LEGITIMATE',
                            'Fraud_Probability': f"{result['fraud_probability']*100:.1f}%",
                            'Confidence': f"{result['confidence']*100:.1f}%",
                            'Risk_Level': 'High' if result['fraud_probability'] > 0.7 else 'Medium' if result['fraud_probability'] > 0.3 else 'Low'
                        })
                    
                    progress_bar.progress((i + 1) / len(df))
                
                # Display results
                results_df = pd.DataFrame(results)
                st.subheader("📊 Processing Results")
                st.dataframe(results_df)
                
                # Summary
                fraud_count = (results_df['Prediction'] == 'FRAUD').sum()
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Processed", len(results_df))
                with col2:
                    st.metric("Fraud Detected", fraud_count)
                with col3:
                    st.metric("Fraud Rate", f"{fraud_count/len(results_df)*100:.1f}%")
                with col4:
                    high_risk = (results_df['Risk_Level'] == 'High').sum()
                    st.metric("High Risk", high_risk)
                
                # Download
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

def model_info_page(app):
    st.header("📈 Model Information & Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Model Details")
        
        info_data = {
            'Property': ['Algorithm', 'Target Column', 'Total Features', 'Categorical Features', 'Numerical Features', 'Target Classes'],
            'Value': [
                'Random Forest Classifier',
                app.classifier.target_column,
                len(app.classifier.feature_names),
                len(app.classifier.categorical_columns),
                len(app.numerical_features),
                ', '.join(app.classifier.target_classes)
            ]
        }
        
        info_df = pd.DataFrame(info_data)
        st.table(info_df)
    
    with col2:
        st.subheader("📊 Feature Distribution")
        
        # Feature type chart
        chart_data = pd.DataFrame({
            'Type': ['Categorical', 'Numerical'],
            'Count': [len(app.classifier.categorical_columns), len(app.numerical_features)]
        })
        
        fig = px.pie(chart_data, values='Count', names='Type', title="Feature Types")
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (if available)
    st.subheader("🎯 Feature Importance Analysis")
    try:
        if hasattr(app.classifier.best_model, 'feature_importances_'):
            importance_data = pd.DataFrame({
                'Feature': app.classifier.feature_names,
                'Importance': app.classifier.best_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(15)
            
            fig = px.bar(
                importance_data, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title="Top 15 Most Important Features"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type")
    except Exception as e:
        st.error(f"Error displaying feature importance: {e}")

def feature_explorer_page(app):
    st.header("🎯 Feature Explorer")
    st.markdown("Explore all model features and their characteristics")
    
    # Feature search
    search_term = st.text_input("🔍 Search features:", placeholder="Enter feature name...")
    
    # Filter features
    if search_term:
        filtered_features = [f for f in app.classifier.feature_names if search_term.lower() in f.lower()]
    else:
        filtered_features = app.classifier.feature_names
    
    st.write(f"📋 Showing {len(filtered_features)} of {len(app.classifier.feature_names)} features")
    
    # Display features in a nice format
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Categorical Features")
        cat_features = [f for f in filtered_features if f in app.classifier.categorical_columns]
        
        for i, feature in enumerate(cat_features[:20], 1):
            with st.expander(f"{i}. {feature}"):
                if feature in app.classifier.label_encoders:
                    classes = app.classifier.label_encoders[feature].classes_
                    st.write(f"**Unique values:** {len(classes)}")
                    if len(classes) <= 10:
                        st.write(f"**Values:** {', '.join(classes)}")
                    else:
                        st.write(f"**Sample values:** {', '.join(classes[:10])}...")
    
    with col2:
        st.subheader("🔢 Numerical Features")
        num_features = [f for f in filtered_features if f not in app.classifier.categorical_columns]
        
        for i, feature in enumerate(num_features[:20], 1):
            with st.expander(f"{i}. {feature}"):
                st.write(f"**Type:** Numerical")
                st.write(f"**Data type:** {app.classifier.feature_dtypes.get(feature, 'Unknown')}")
    
    # Feature statistics
    if len(filtered_features) < len(app.classifier.feature_names):
        st.info(f"💡 Found {len(filtered_features)} features matching '{search_term}'")

if __name__ == "__main__":
    main()
