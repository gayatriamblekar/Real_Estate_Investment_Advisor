import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def preprocess_data(file_path):
    # Load dataset
    print("Loading data...")
    df = pd.read_csv(file_path)

    # Drop duplicates
    print("Dropping duplicates...")
    df = df.drop_duplicates()

    # Identify numerical and categorical columns
    categorical_cols = [
        'State', 'City', 'Locality', 'Property_Type', 
        'Furnished_Status', 'Security', 'Amenities', 
        'Facing', 'Owner_Type', 'Availability_Status'
    ]
    
    numerical_cols = [
        col for col in df.columns 
        if col not in categorical_cols and df[col].dtype in ['int64', 'float64']
    ]

    # Impute missing values
    print("Imputing missing values...")
    for col in numerical_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df.loc[:, col] = df[col].fillna(median_val)
    
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df.loc[:, col] = df[col].fillna(mode_val)

    # Encode categorical columns
    print("Encoding categorical columns...")
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle unseen labels in the future by treating them as a new category or using a fallback if necessary
        # For simplicity here, we fit and transform directly on the dataset
        df.loc[:, col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    joblib.dump(label_encoders, 'models/label_encoders.pkl')

    # Engineer new features
    print("Engineering new features...")
    if 'Price_per_SqFt' not in df.columns:
        df['Price_per_SqFt'] = (df['Price_in_Lakhs'] * 100000) / df['Size_in_SqFt']
    
    df['Age_of_Property'] = 2025 - df['Year_Built']
    
    # Normalize Nearby_Schools to 0-1 range for a Density Score
    max_schools = df['Nearby_Schools'].max()
    df['School_Density_Score'] = df['Nearby_Schools'] / max_schools if max_schools > 0 else 0
    
    # 8% annual appreciation for 5 years
    df['Future_Price_5Y'] = df['Price_in_Lakhs'] * (1.08 ** 5)
    
    # Good Investment logic
    median_price_sqft = df['Price_per_SqFt'].median()
    df['Good_Investment'] = np.where(
        (df['Price_per_SqFt'] <= median_price_sqft) & (df['BHK'] >= 2), 
        1, 0
    )

    # Need to update numerical_cols based on new features so we scale them correctly
    new_num_features = ['Price_per_SqFt', 'Age_of_Property', 'School_Density_Score', 'Future_Price_5Y']
    features_to_scale = [c for c in numerical_cols if c not in ['Price_in_Lakhs', 'Year_Built']] + new_num_features

    # Separate target variables from scaling logic so we can predict them later natively
    features_to_scale = [c for c in features_to_scale if c not in ['Future_Price_5Y', 'Good_Investment']]

    # Scale numerical features
    print("Scaling numerical features...")
    scaler = StandardScaler()
    df.loc[:, features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Save feature columns (independent variables used for prediction)
    # We will exclude targets and intermediate calculation logic like Year_Built (captured by Age_of_Property)
    target_cols = ['Good_Investment', 'Future_Price_5Y']
    exclude_cols = ['Year_Built', 'Price_in_Lakhs', 'Price_per_SqFt', 'Age_of_Property', 'School_Density_Score'] # Remove some that are derived but might leak or not be inputs
    
    # Actually, user provides:
    # State, City, Property_Type, BHK, Size_in_SqFt, Price_in_Lakhs, Furnished_Status, Floor_No, Total_Floors, 
    # Nearby_Schools, Nearby_Hospitals, Public_Transport_Accessibility, Parking_Space, Security, Amenities, 
    # Facing, Owner_Type, Availability_Status, Year_Built
    # So the model feature columns should be those, transformed. Let's save all inputs required for models
    
    feature_cols = [col for col in df.columns if col not in target_cols]
    joblib.dump(feature_cols, 'models/feature_columns.pkl')

    # Save cleaned dataset
    print("Saving cleaned dataset...")
    df.to_csv('cleaned_data.csv', index=False)
    print(f"Dataset successfully preprocessed! Total remaining rows: {len(df)}")

if __name__ == "__main__":
    if not os.path.exists("india_housing_prices.csv"):
        print("Error: india_housing_prices.csv not found.")
    else:
        preprocess_data("india_housing_prices.csv")
