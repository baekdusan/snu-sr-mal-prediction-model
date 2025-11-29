import pandas as pd
import numpy as np

# Load all_data.xlsx (2560 rows: 256 queries × 10 participants)
all_data = pd.read_excel('/Users/dusanbaek/mal-prediction-model/projects/LMM_model/data/all_data.xlsx')
print(f"all_data shape: {all_data.shape}")
print(f"all_data columns: {all_data.columns.tolist()}")

# Load augmented_data.csv (256 rows: unique queries with features)
augmented_data = pd.read_csv('/Users/dusanbaek/mal-prediction-model/projects/LMM_model/data/augmented_data.csv')
print(f"\naugmented_data shape: {augmented_data.shape}")
print(f"augmented_data columns: {augmented_data.columns.tolist()[:5]}...")  # Show first 5 columns

# Rename columns for consistency
all_data = all_data.rename(columns={'participant': 'participant_id'})

# Add query_id based on unique queries (1-256)
unique_queries = augmented_data['queries'].tolist()
query_to_id = {query: idx + 1 for idx, query in enumerate(unique_queries)}
all_data['query_id'] = all_data['queries'].map(query_to_id)

# Extract feature columns from augmented_data (exclude 'queries' and 'MAL')
feature_cols = [col for col in augmented_data.columns if col not in ['queries', 'MAL']]

# Join: keep all_data's MAL values, add features from augmented_data
merged_data = all_data.merge(
    augmented_data[['queries'] + feature_cols],
    on='queries',
    how='left'
)

# Select and reorder columns: participant_id, query_id, MAL, features
final_columns = ['participant_id', 'query_id', 'MAL'] + feature_cols
final_data = merged_data[final_columns]

print(f"\nFinal data shape: {final_data.shape}")
print(f"Final data columns: {final_data.columns.tolist()[:10]}...")  # Show first 10 columns
print(f"\nFirst few rows:")
print(final_data.head())

# Check for any missing values
print(f"\nMissing values per column:")
missing_summary = final_data.isnull().sum()
if missing_summary.sum() > 0:
    print(missing_summary[missing_summary > 0])
else:
    print("No missing values!")

# Verify data integrity
print(f"\nData integrity check:")
print(f"- Unique participants: {final_data['participant_id'].nunique()}")
print(f"- Unique queries: {final_data['query_id'].nunique()}")
print(f"- Expected rows (256 queries × 10 participants): 2560")
print(f"- Actual rows: {len(final_data)}")

# Save the final dataset
output_path = '/Users/dusanbaek/mal-prediction-model/projects/LMM_model/data/final_dataset.csv'
final_data.to_csv(output_path, index=False)
print(f"\nDataset saved to: {output_path}")
print(f"Total rows: {len(final_data)}")
print(f"Total columns: {len(final_data.columns)}")
