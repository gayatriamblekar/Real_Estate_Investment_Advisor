import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('eda_charts', exist_ok=True)

def generate_eda(file_path):
    print("Loading cleaned dataset for EDA...")
    df = pd.read_csv(file_path)
    
    # Reload the raw dataset since cleaned_data has scaled values which aren't great for EDA interpretations
    # Or we use cleaned but keep in mind that they are scaled. Actually, for EDA we want unscaled interpretation if possible.
    # The instructions say: "eda.py reads cleaned_data.csv". The user's specification implies EDA uses cleaned. 
    # But wait, scaled values mean prices are between -2 and 2. The charts might be confusing. Is there a way to unscale?
    # Or just use the cleaned_data but we'll deal with it. Actually, I only scaled features_to_scale. 
    # Let me make sure `eda.py` handles them nicely. I'll just plot them as they are in cleaned_data.csv.
    # For visualizations, using seaborn aesthetic defaults
    sns.set_theme(style="whitegrid")
    
    # 1. Distribution of property prices
    plt.figure(figsize=(10,6))
    sns.histplot(df['Price_in_Lakhs'], kde=True, color='blue')
    plt.title('Distribution of Property Prices')
    plt.xlabel('Price (Lakhs) - Note: Might be scaled')
    plt.tight_layout()
    plt.savefig('eda_charts/1_price_distribution.png')
    plt.close()
    
    # 2. Distribution of property sizes
    plt.figure(figsize=(10,6))
    sns.histplot(df['Size_in_SqFt'], kde=True, color='green')
    plt.title('Distribution of Property Sizes')
    plt.xlabel('Size in SqFt')
    plt.tight_layout()
    plt.savefig('eda_charts/2_size_distribution.png')
    plt.close()

    # 3. Price per sq ft by property type (boxplot)
    plt.figure(figsize=(12,6))
    sns.boxplot(x='Property_Type', y='Price_per_SqFt', data=df)
    plt.title('Price per SqFt by Property Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('eda_charts/3_price_per_sqft_by_property_type.png')
    plt.close()

    # 4. Scatter: Size vs Price
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='Size_in_SqFt', y='Price_in_Lakhs', data=df, alpha=0.5)
    plt.title('Property Size vs Price')
    plt.tight_layout()
    plt.savefig('eda_charts/4_size_vs_price.png')
    plt.close()

    # 5. Outlier detection: boxplots for Price_per_SqFt and Size_in_SqFt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.boxplot(y=df['Price_per_SqFt'], ax=axes[0], color='lightcoral')
    axes[0].set_title('Outliers in Price per SqFt')
    sns.boxplot(y=df['Size_in_SqFt'], ax=axes[1], color='lightblue')
    axes[1].set_title('Outliers in Size in SqFt')
    plt.tight_layout()
    plt.savefig('eda_charts/5_outlier_detection.png')
    plt.close()

    # 6. Average price per sq ft by state (bar chart)
    plt.figure(figsize=(12,6))
    state_avg = df.groupby('State')['Price_per_SqFt'].mean().sort_values(ascending=False).reset_index()
    sns.barplot(x='State', y='Price_per_SqFt', data=state_avg, palette='viridis')
    plt.title('Average Price per Sq Ft by State')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('eda_charts/6_avg_price_per_sqft_state.png')
    plt.close()

    # 7. Average property price by city (top 15)
    plt.figure(figsize=(12,6))
    city_avg = df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(15).reset_index()
    sns.barplot(x='City', y='Price_in_Lakhs', data=city_avg, palette='magma')
    plt.title('Average Property Price by City (Top 15)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('eda_charts/7_avg_price_city.png')
    plt.close()

    # 8. Median age of properties by locality (top 15)
    plt.figure(figsize=(12,6))
    locality_age = df.groupby('Locality')['Age_of_Property'].median().sort_values(ascending=False).head(15).reset_index()
    sns.barplot(x='Locality', y='Age_of_Property', data=locality_age, palette='coolwarm')
    plt.title('Median Age of Properties by Locality (Top 15)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('eda_charts/8_median_age_locality.png')
    plt.close()

    # 9. BHK distribution across cities (heatmap or grouped bar)
    plt.figure(figsize=(14,8))
    bhk_city = pd.crosstab(df['City'], df['BHK'])
    sns.heatmap(bhk_city, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('BHK Distribution Across Cities')
    plt.tight_layout()
    plt.savefig('eda_charts/9_bhk_distribution_cities.png')
    plt.close()

    # 10. Price trends for top 5 most expensive localities
    plt.figure(figsize=(12,6))
    top_localities = df.groupby('Locality')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(5).index
    df_top_loc = df[df['Locality'].isin(top_localities)]
    sns.boxplot(x='Locality', y='Price_in_Lakhs', data=df_top_loc)
    plt.title('Price Distribution for Top 5 Most Expensive Localities')
    plt.tight_layout()
    plt.savefig('eda_charts/10_price_trends_top_localities.png')
    plt.close()

    # 11. Correlation heatmap of all numeric features
    plt.figure(figsize=(16,12))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Numeric Features')
    plt.tight_layout()
    plt.savefig('eda_charts/11_correlation_heatmap.png')
    plt.close()

    # 12. Nearby Schools vs Price_per_SqFt (scatter)
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='Nearby_Schools', y='Price_per_SqFt', data=df, alpha=0.6)
    plt.title('Nearby Schools vs Price per SqFt')
    plt.tight_layout()
    plt.savefig('eda_charts/12_schools_vs_price.png')
    plt.close()

    # 13. Nearby Hospitals vs Price_per_SqFt (scatter)
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='Nearby_Hospitals', y='Price_per_SqFt', data=df, alpha=0.6)
    plt.title('Nearby Hospitals vs Price per SqFt')
    plt.tight_layout()
    plt.savefig('eda_charts/13_hospitals_vs_price.png')
    plt.close()

    # 14. Price by Furnished Status (boxplot)
    plt.figure(figsize=(10,6))
    sns.boxplot(x='Furnished_Status', y='Price_in_Lakhs', data=df)
    plt.title('Price by Furnished Status')
    plt.tight_layout()
    plt.savefig('eda_charts/14_price_by_furnished.png')
    plt.close()

    # 15. Price per sq ft by Facing direction (bar)
    plt.figure(figsize=(10,6))
    facing_avg = df.groupby('Facing')['Price_per_SqFt'].mean().sort_values(ascending=False).reset_index()
    sns.barplot(x='Facing', y='Price_per_SqFt', data=facing_avg, palette='crest')
    plt.title('Price per SqFt by Facing Direction')
    plt.tight_layout()
    plt.savefig('eda_charts/15_price_by_facing.png')
    plt.close()

    # 16. Owner Type distribution (pie/bar)
    plt.figure(figsize=(8,8))
    owner_counts = df['Owner_Type'].value_counts()
    plt.pie(owner_counts, labels=owner_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title('Owner Type Distribution')
    plt.tight_layout()
    plt.savefig('eda_charts/16_owner_type_dist.png')
    plt.close()

    # 17. Availability Status distribution (pie/bar)
    plt.figure(figsize=(8,8))
    avail_counts = df['Availability_Status'].value_counts()
    plt.pie(avail_counts, labels=avail_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set2'))
    plt.title('Availability Status Distribution')
    plt.tight_layout()
    plt.savefig('eda_charts/17_availability_Status_dist.png')
    plt.close()

    # 18. Parking Space vs Price (scatter/bar)
    plt.figure(figsize=(10,6))
    sns.barplot(x='Parking_Space', y='Price_in_Lakhs', data=df, errorbar=None)
    plt.title('Average Price by Parking Space')
    plt.tight_layout()
    plt.savefig('eda_charts/18_parking_vs_price.png')
    plt.close()

    # 19. Amenities vs Price_per_SqFt (bar)
    plt.figure(figsize=(10,6))
    amenities_avg = df.groupby('Amenities')['Price_per_SqFt'].mean().sort_values().reset_index()
    sns.barplot(x='Amenities', y='Price_per_SqFt', data=amenities_avg, palette='Set1')
    plt.title('Average Price per SqFt by Amenities')
    plt.tight_layout()
    plt.savefig('eda_charts/19_amenities_vs_price.png')
    plt.close()

    # 20. Public Transport Accessibility vs Price_per_SqFt (bar)
    plt.figure(figsize=(10,6))
    sns.barplot(x='Public_Transport_Accessibility', y='Price_per_SqFt', data=df, errorbar=None, palette='Spectral')
    plt.title('Price per SqFt by Public Transport Accessibility')
    plt.tight_layout()
    plt.savefig('eda_charts/20_transport_vs_price.png')
    plt.close()
    
    print("EDA completed: 20 charts saved in eda_charts/ directory.")

if __name__ == "__main__":
    if not os.path.exists("cleaned_data.csv"):
        print("Error: cleaned_data.csv not found. Run preprocessing.py first.")
    else:
        generate_eda("cleaned_data.csv")
