import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 2000

states = ['Maharashtra', 'Karnataka', 'Delhi', 'Haryana', 'Tamil Nadu', 'Telangana']
cities = ['Mumbai', 'Pune', 'Bangalore', 'New Delhi', 'Gurgaon', 'Chennai', 'Hyderabad']
localities = [f'Locality_{i}' for i in range(1, 51)]
property_types = ['Apartment', 'Independent House', 'Villa', 'Builder Floor']
furnished_statuses = ['Unfurnished', 'Semi-Furnished', 'Fully Furnished']
securities = ['Yes', 'No']
amenities = ['Basic', 'Premium', 'Luxury']
facings = ['North', 'South', 'East', 'West', 'North-East']
owner_types = ['Builder', 'Agent', 'Owner']
availability_statuses = ['Ready to Move', 'Under Construction']

data = {
    'State': np.random.choice(states, n_samples),
    'City': np.random.choice(cities, n_samples),
    'Locality': np.random.choice(localities, n_samples),
    'Property_Type': np.random.choice(property_types, n_samples),
    'BHK': np.random.randint(1, 6, n_samples),
    'Size_in_SqFt': np.random.randint(500, 5000, n_samples),
    'Furnished_Status': np.random.choice(furnished_statuses, n_samples),
    'Floor_No': np.random.randint(1, 40, n_samples),
    'Total_Floors': np.random.randint(5, 50, n_samples),
    'Security': np.random.choice(securities, n_samples),
    'Amenities': np.random.choice(amenities, n_samples),
    'Facing': np.random.choice(facings, n_samples),
    'Owner_Type': np.random.choice(owner_types, n_samples),
    'Availability_Status': np.random.choice(availability_statuses, n_samples),
    'Year_Built': np.random.randint(1990, 2025, n_samples),
    'Nearby_Schools': np.random.randint(0, 10, n_samples),
    'Nearby_Hospitals': np.random.randint(0, 5, n_samples),
    'Public_Transport_Accessibility': np.random.randint(1, 10, n_samples),
    'Parking_Space': np.random.randint(0, 4, n_samples),
}

df = pd.DataFrame(data)

# Ensure Total_Floors >= Floor_No
df['Total_Floors'] = np.where(df['Total_Floors'] < df['Floor_No'], df['Floor_No'] + np.random.randint(0, 10, n_samples), df['Total_Floors'])

# Construct synthetic pricing
base_price = (df['Size_in_SqFt'] * 10000) / 100000  # Base price in lakhs roughly 1 lakh per 10 sqft
bhk_multiplier = 1 + (df['BHK'] * 0.1)
amenity_multiplier = df['Amenities'].map({'Basic': 1.0, 'Premium': 1.2, 'Luxury': 1.5})
city_multiplier = df['City'].map({
    'Mumbai': 1.8, 'New Delhi': 1.6, 'Bangalore': 1.4, 'Gurgaon': 1.5,
    'Pune': 1.2, 'Chennai': 1.1, 'Hyderabad': 1.1
})

df['Price_in_Lakhs'] = base_price * bhk_multiplier * amenity_multiplier * city_multiplier

# Add some noise
df['Price_in_Lakhs'] = df['Price_in_Lakhs'] * np.random.uniform(0.8, 1.2, n_samples)

# Introduce a few missing values
for col in ['Security', 'Amenities', 'Size_in_SqFt', 'Floor_No']:
    df.loc[df.sample(frac=0.05).index, col] = np.nan

# Introduce a few duplicates
df = pd.concat([df, df.sample(frac=0.05)]).reset_index(drop=True)

df.to_csv('india_housing_prices.csv', index=False)
print("Generated india_housing_prices.csv with", len(df), "rows.")
