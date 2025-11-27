import pandas as pd
import numpy as np

# Load all_data.xlsx
all_data = pd.read_excel('/Users/dusanbaek/mal-prediction-model/LMM_model/all_data.xlsx')
print(f"all_data shape: {all_data.shape}")
print(f"all_data columns: {all_data.columns.tolist()}")

# Load augmented_data.csv
augmented_data = pd.read_csv('/Users/dusanbaek/mal-prediction-model/LMM_model/augmented_data.csv')
print(f"\naugmented_data shape: {augmented_data.shape}")
print(f"augmented_data columns: {augmented_data.columns.tolist()[:5]}...")  # Show first 5 columns

# Add query_id to all_data (row index starting from 1)
all_data['query_id'] = range(1, len(all_data) + 1)

# Rename columns for consistency
all_data = all_data.rename(columns={'participant': 'participant_id'})

# Join the two datasets on 'queries'
merged_data = all_data.merge(augmented_data, on='queries', how='left', suffixes=('', '_aug'))

# Select and reorder columns
# participant_id, query_id, MAL, and all features (feat1~featN)
feature_cols = [col for col in augmented_data.columns if col.startswith('QL_') or col.startswith('feat')]
final_columns = ['participant_id', 'query_id', 'MAL'] + feature_cols

# If there are duplicate MAL columns, use the one from all_data
if 'MAL_aug' in merged_data.columns:
    merged_data = merged_data.drop(columns=['MAL_aug'])

# Select final columns
final_data = merged_data[final_columns]

print(f"\nFinal data shape: {final_data.shape}")
print(f"Final data columns: {final_data.columns.tolist()[:10]}...")  # Show first 10 columns
print(f"\nFirst few rows:")
print(final_data.head())

# Check for any missing values
print(f"\nMissing values per column:")
print(final_data.isnull().sum()[final_data.isnull().sum() > 0])

# Save the final dataset
output_path = '/Users/dusanbaek/mal-prediction-model/LMM_model/final_dataset.csv'
final_data.to_csv(output_path, index=False)
print(f"\nDataset saved to: {output_path}")
print(f"Total rows: {len(final_data)}")
print(f"Total columns: {len(final_data.columns)}")
