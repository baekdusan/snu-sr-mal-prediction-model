#!/usr/bin/env python3
"""
Clean augmented_data_v2.csv - Fill missing values
"""

import pandas as pd
import numpy as np


def print_section(title):
    """Print section header"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    """Clean and fill missing values"""

    # ========================================
    # 1. Load data
    # ========================================
    print_section("STEP 1: Loading Data")

    INPUT_FILE = 'augmented_data_v2.csv'
    df = pd.read_csv(INPUT_FILE)

    print(f"✅ Loaded {len(df)} rows from {INPUT_FILE}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Shape: {df.shape}")

    # ========================================
    # 2. Analyze missing values
    # ========================================
    print_section("STEP 2: Missing Value Analysis")

    total_missing = df.isnull().sum().sum()
    print(f"   Total missing values: {total_missing}")

    if total_missing > 0:
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]

        print(f"\n   Columns with missing values: {len(missing_cols)}")
        for col, count in missing_cols.items():
            pct = (count / len(df)) * 100
            print(f"   - {col}: {count} ({pct:.1f}%)")

    # ========================================
    # 3. Fill missing values
    # ========================================
    print_section("STEP 3: Filling Missing Values")

    exclude_cols = ['queries', 'MAL']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    filled_count = 0

    for col in feature_cols:
        missing_count = df[col].isnull().sum()
        if missing_count == 0:
            continue

        dtype = df[col].dtype

        if dtype in ['int64', 'float64']:
            # Numeric: fill with 0
            df[col] = df[col].fillna(0)
            print(f"   ✅ {col}: filled {missing_count} with 0")
        else:
            # Categorical: fill with 'none'
            df[col] = df[col].fillna('none')
            print(f"   ✅ {col}: filled {missing_count} with 'none'")

        filled_count += 1

    print(f"\n   Total columns fixed: {filled_count}")

    # ========================================
    # 4. Verify and save
    # ========================================
    print_section("STEP 4: Verification and Save")

    remaining_missing = df.isnull().sum().sum()

    if remaining_missing > 0:
        print(f"❌ Still have {remaining_missing} missing values!")
        print("   Showing remaining missing columns:")
        missing_counts = df.isnull().sum()
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"   - {col}: {count}")
        return
    else:
        print(f"✅ No missing values remaining!")

    # Save cleaned data
    OUTPUT_FILE = 'augmented_data_v2_cleaned.csv'
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

    print(f"\n✅ Saved to: {OUTPUT_FILE}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Features: {len(feature_cols)}")

    # ========================================
    # 5. Feature summary
    # ========================================
    print_section("STEP 5: Feature Summary")

    # Count by prefix
    categories = {}
    for col in feature_cols:
        prefix = col.split('_')[0] if '_' in col else 'OTHER'
        categories[prefix] = categories.get(prefix, 0) + 1

    print(f"\n📊 Feature Distribution by Category:")
    for cat, count in sorted(categories.items()):
        print(f"   {cat:12s}: {count:3d} features")

    print(f"\n   Total features: {len(feature_cols)}")

    # Data type summary
    print(f"\n📊 Feature Data Types:")
    type_counts = {}
    for col in feature_cols:
        dtype = str(df[col].dtype)
        type_counts[dtype] = type_counts.get(dtype, 0) + 1

    for dtype, count in sorted(type_counts.items()):
        print(f"   {dtype:12s}: {count:3d} columns")

    # ========================================
    # Summary
    # ========================================
    print_section("🎉 CLEANING COMPLETE")

    print(f"\n📝 Summary:")
    print(f"   - Input: {INPUT_FILE} ({len(df)} rows)")
    print(f"   - Output: {OUTPUT_FILE}")
    print(f"   - Missing values filled: {filled_count} columns")
    print(f"   - Total features: {len(feature_cols)}")
    print(f"   - Ready for training: ✅ YES")
    print(f"\n✨ Next step: Update train_model_augmented.py to use '{OUTPUT_FILE}'")
    print("=" * 80)


if __name__ == '__main__':
    main()
