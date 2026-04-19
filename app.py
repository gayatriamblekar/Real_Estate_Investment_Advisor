import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")

# Helper function to load models
@st.cache_resource
def load_models():
    try:
        classifier = joblib.load('models/best_classifier.pkl')
        regressor = joblib.load('models/best_regressor.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        feature_cols = joblib.load('models/feature_columns.pkl')
        return classifier, regressor, scaler, label_encoders, feature_cols
    except FileNotFoundError:
        st.error("Model files not found! Please run the training pipeline first.")
        return None, None, None, None, None

classifier, regressor, scaler, label_encoders, feature_cols = load_models()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Investment Predictor", "EDA Dashboard", "Model Performance"])

if page == "Investment Predictor":
    st.title("🏡 Property Investment Predictor")
    st.subheader("Enter Property Details to Get Estimates")
    
    if feature_cols is None:
        st.stop()
        
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            state = st.selectbox("State", label_encoders['State'].classes_)
            city = st.selectbox("City", label_encoders['City'].classes_)
            locality = st.selectbox("Locality", label_encoders['Locality'].classes_)
            property_type = st.selectbox("Property Type", label_encoders['Property_Type'].classes_)
            furnished_status = st.selectbox("Furnished Status", label_encoders['Furnished_Status'].classes_)
            security = st.selectbox("Security", label_encoders['Security'].classes_)
            
        with col2:
            amenities = st.selectbox("Amenities", label_encoders['Amenities'].classes_)
            facing = st.selectbox("Facing", label_encoders['Facing'].classes_)
            owner_type = st.selectbox("Owner Type", label_encoders['Owner_Type'].classes_)
            availability = st.selectbox("Availability Status", label_encoders['Availability_Status'].classes_)
            bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
            size = st.number_input("Size (SqFt)", min_value=100, max_value=10000, value=1000)
            
        with col3:
            price = st.number_input("Price (Lakhs)", min_value=1.0, max_value=10000.0, value=50.0)
            floor_no = st.number_input("Floor No", min_value=0, max_value=100, value=2)
            total_floors = st.number_input("Total Floors", min_value=1, max_value=150, value=5)
            schools = st.number_input("Nearby Schools", min_value=0, max_value=50, value=2)
            hospitals = st.number_input("Nearby Hospitals", min_value=0, max_value=50, value=1)
            transport = st.slider("Public Transport Accessibility (1-10)", 1, 10, 5)
            parking = st.number_input("Parking Space", min_value=0, max_value=10, value=1)
            year_built = st.number_input("Year Built", min_value=1950, max_value=2025, value=2015)
            
        submit = st.form_submit_button("Predict Investment Suitability")

    if submit:
        # Preprocess form input
        try:
            # 1. Start with raw dict
            input_data = {
                'State': state, 'City': city, 'Locality': locality, 'Property_Type': property_type,
                'Furnished_Status': furnished_status, 'Security': security, 'Amenities': amenities,
                'Facing': facing, 'Owner_Type': owner_type, 'Availability_Status': availability,
                'BHK': bhk, 'Size_in_SqFt': size, 'Floor_No': floor_no, 'Total_Floors': total_floors,
                'Nearby_Schools': schools, 'Nearby_Hospitals': hospitals, 
                'Public_Transport_Accessibility': transport, 'Parking_Space': parking
            }
            
            df_input = pd.DataFrame([input_data])
            
            # 2. Encode categoricals
            for col, le in label_encoders.items():
                if col in df_input.columns:
                    # Handle unseen if somehow bypassed
                    df_input[col] = le.transform(df_input[col].astype(str))
            
            # 3. Engineer features (need exactly what was built in preprocessing.py)
            price_per_sqft = (price * 100000) / size
            age = 2025 - year_built
            # Max schools normalization needs knowledge of max_schools, we'll just mock 10 as per generate_dataset
            school_density = schools / 10.0 
            
            df_input['Price_per_SqFt'] = price_per_sqft
            df_input['Age_of_Property'] = age
            df_input['School_Density_Score'] = school_density
            
            # 4. Filter columns
            df_input = df_input.reindex(columns=feature_cols, fill_value=0)
            
            # 5. Scale Numerical (we must identify which subset was scaled during training!)
            # Looking at preprocessing.py logic: features_to_scale = [c for c in numerical_cols if c not in ['Price_in_Lakhs', 'Year_Built']] + new_num_features
            # But the scaler was fit on all of them in order. Let's just scale the same way.
            # Actually, scaler expects exactly the columns it got during fit.
            # Let's see feature_in_ of scaler
            if hasattr(scaler, "feature_names_in_"):
                scaled_cols = scaler.feature_names_in_
                df_input[scaled_cols] = scaler.transform(df_input[scaled_cols])
            
            # Predictions
            is_good_inv = classifier.predict(df_input)[0]
            confidence = classifier.predict_proba(df_input)[0][is_good_inv] * 100
            future_price = regressor.predict(df_input)[0]
            
            st.markdown("---")
            cc1, cc2 = st.columns(2)
            
            with cc1:
                # Classification Result
                if is_good_inv == 1:
                    st.success(f"✅ **Good Investment!** (Confidence: {confidence:.1f}%)")
                else:
                    st.error(f"❌ **Not a Good Investment** (Confidence: {confidence:.1f}%)")
                    
            with cc2:
                # Regression Result
                roi = ((future_price - price) / price) * 100
                st.info(f"💰 **Estimated Price after 5 Years:** ₹{future_price:.2f} Lakhs")
                st.write(f"**Expected ROI:** {roi:.1f}%")
                
            # Feature Importance
            if hasattr(classifier, "feature_importances_"):
                st.markdown("---")
                st.subheader("Feature Importance")
                importances = classifier.feature_importances_
                indices = np.argsort(importances)[-10:] # Top 10
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(range(len(indices)), importances[indices], color='teal', align='center')
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([feature_cols[i] for i in indices])
                ax.set_xlabel('Relative Importance')
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")

elif page == "EDA Dashboard":
    st.title("📊 EDA Dashboard")
    st.write("Explore the visualized data from the Indian Housing Prices Dataset.")
    
    chart_dir = "eda_charts"
    if not os.path.exists(chart_dir):
        st.warning("EDA charts not found! Please run `eda.py` first.")
    else:
        # Load images
        charts = glob.glob(os.path.join(chart_dir, "*.png"))
        charts.sort() # Ensure some order
        
        if not charts:
            st.info("No charts found in the directory.")
        else:
            for i in range(0, len(charts), 2):
                col1, col2 = st.columns(2)
                with col1:
                    if i < len(charts):
                        st.image(Image.open(charts[i]), use_container_width=True)
                with col2:
                    if i + 1 < len(charts):
                        st.image(Image.open(charts[i+1]), use_container_width=True)

elif page == "Model Performance":
    st.title("📈 Model Performance Metrics")
    
    try:
        with open("models/classification_metrics.json", "r") as f:
            class_metrics = json.load(f)
            
        with open("models/regression_metrics.json", "r") as f:
            reg_metrics = json.load(f)
            
        st.subheader("Classification Models (Target: Good Investment)")
        st.dataframe(pd.DataFrame(class_metrics).T)
        
        st.subheader("Regression Models (Target: Future Price 5Y)")
        st.dataframe(pd.DataFrame(reg_metrics).T)
        
        st.subheader("Confusion Matrix (Best Classifier)")
        cm_path = "models/best_classifier_cm.png"
        if os.path.exists(cm_path):
            st.image(Image.open(cm_path), width=600)
            
    except FileNotFoundError:
        st.warning("Model metric files not found! Please run `train_models.py` to generate them.")
