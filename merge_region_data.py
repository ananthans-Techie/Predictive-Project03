import pandas as pd

# Load your main dataset
main_df = pd.read_csv(r"C:/Users/sanan/OneDrive/Documents/Predictive_analysis/Project 03/water_potability.csv")

# Example: Load a region-specific dataset (replace with your file path)
region_df = pd.read_csv("region_water_quality.csv")

# Add a 'region' column to both if not present
if 'region' not in main_df.columns:
    main_df['region'] = 'DefaultRegion'
if 'region' not in region_df.columns:
    region_df['region'] = 'NewRegion'

# Ensure columns match (except for region)
common_cols = [col for col in main_df.columns if col in region_df.columns]

# Concatenate datasets
combined_df = pd.concat([
    main_df[common_cols],
    region_df[common_cols]
], ignore_index=True)

# Save the improved dataset
combined_df.to_csv("water_potability_with_regions.csv", index=False)

print("Combined dataset shape:", combined_df.shape)
print(combined_df['region'].value_counts())