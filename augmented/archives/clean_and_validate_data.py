#!/usr/bin/env python3
"""
Data Cleaning and Validation Script
- Remove columns with type inconsistencies
- Remove columns with too many missing values
- Ensure data quality for model training
"""

import pandas as pd
import numpy as np
import os
import json


def print_section(title):
    """Print section header"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def detect_column_type(series):
    """Detect the intended type of a column"""
    # Remove NaN values for type detection
    non_null = series.dropna()

    if len(non_null) == 0:
        return 'empty'

    # Check if numeric
    try:
        pd.to_numeric(non_null)
        return 'numeric'
    except (ValueError, TypeError):
        pass

    # Check if boolean-like
    unique_vals = set(str(v).lower() for v in non_null.unique())
    if unique_vals.issubset({'0', '1', 'true', 'false', '0.0', '1.0'}):
        return 'boolean'

    # Otherwise it's categorical/string
    return 'categorical'


def check_type_consistency(df, col):
    """Check if a column has consistent types"""
    non_null = df[col].dropna()

    if len(non_null) == 0:
        return False, "all_missing"

    # Detect primary type
    primary_type = detect_column_type(df[col])

    if primary_type == 'empty':
        return False, "empty"

    if primary_type == 'numeric':
        # Check if all non-null values can be converted to numeric
        try:
            pd.to_numeric(non_null)
            return True, "numeric"
        except (ValueError, TypeError):
            return False, "mixed_types"

    elif primary_type == 'boolean':
        # Boolean is consistent
        return True, "boolean"

    else:  # categorical
        # Check if there are too many unique values (might indicate inconsistency)
        unique_ratio = len(non_null.unique()) / len(non_null)
        if unique_ratio > 0.9:  # More than 90% unique values
            return False, "too_diverse"
        return True, "categorical"


def analyze_data_quality(df):
    """Analyze data quality and identify problematic columns"""
    print_section("DATA QUALITY ANALYSIS")

    total_rows = len(df)
    print(f"Total rows: {total_rows}")
    print(f"Total columns: {len(df.columns)}")

    results = {
        'good_columns': [],
        'bad_columns': [],
        'column_stats': {}
    }

    print(f"\n{'Column':<40} {'Type':<15} {'Missing %':<12} {'Status':<15}")
    print("─" * 85)

    for col in df.columns:
        # Skip queries and MAL - these are essential
        if col in ['queries', 'MAL']:
            results['good_columns'].append(col)
            print(f"{col:<40} {'essential':<15} {'0.0%':<12} {'✓ KEEP':<15}")
            continue

        # Calculate missing percentage
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / total_rows) * 100

        # Check type consistency
        is_consistent, type_info = check_type_consistency(df, col)

        # Determine if column should be kept
        reasons = []
        should_keep = True

        # Rule 1: Too many missing values (>80%)
        if missing_pct > 80:
            should_keep = False
            reasons.append(f"missing>{missing_pct:.1f}%")

        # Rule 2: Type inconsistency
        if not is_consistent:
            should_keep = False
            reasons.append(f"inconsistent:{type_info}")

        # Rule 3: All same value (no variance)
        if should_keep and type_info in ['numeric', 'boolean', 'categorical']:
            non_null = df[col].dropna()
            if len(non_null) > 0 and len(non_null.unique()) == 1:
                should_keep = False
                reasons.append("no_variance")

        # Store results
        status = "✓ KEEP" if should_keep else "✗ REMOVE"
        reason_str = ", ".join(reasons) if reasons else type_info

        results['column_stats'][col] = {
            'missing_pct': missing_pct,
            'type': type_info,
            'consistent': is_consistent,
            'keep': should_keep,
            'reasons': reasons
        }

        if should_keep:
            results['good_columns'].append(col)
        else:
            results['bad_columns'].append(col)

        print(f"{col:<40} {type_info:<15} {missing_pct:>5.1f}%{'':<6} {status:<15} {reason_str if not should_keep else ''}")

    print("─" * 85)
    print(f"\n✓ Columns to keep: {len(results['good_columns'])}")
    print(f"✗ Columns to remove: {len(results['bad_columns'])}")

    return results


def clean_dataframe(df, quality_results):
    """Clean dataframe by removing bad columns and fixing data types"""
    print_section("CLEANING DATA")

    # Keep only good columns
    good_cols = quality_results['good_columns']
    cleaned_df = df[good_cols].copy()

    print(f"Removed {len(quality_results['bad_columns'])} problematic columns")
    print(f"Retained {len(good_cols)} good columns")

    # Convert data types appropriately
    print(f"\n📊 Converting data types...")
    for col in cleaned_df.columns:
        if col in ['queries', 'MAL']:
            continue

        col_info = quality_results['column_stats'].get(col, {})
        col_type = col_info.get('type', 'unknown')

        try:
            if col_type == 'numeric':
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            elif col_type == 'boolean':
                # Convert to 0/1
                cleaned_df[col] = cleaned_df[col].apply(
                    lambda x: 1 if str(x).lower() in ['1', 'true', '1.0'] else (0 if str(x).lower() in ['0', 'false', '0.0'] else np.nan)
                )
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        except Exception as e:
            print(f"   ⚠️  Warning: Could not convert {col}: {e}")

    # Fill remaining missing values
    print(f"\n🔧 Handling missing values...")
    for col in cleaned_df.columns:
        if col in ['queries', 'MAL']:
            continue

        col_type = quality_results['column_stats'].get(col, {}).get('type', 'unknown')

        if col_type in ['numeric', 'boolean']:
            # Fill numeric with median
            median_val = cleaned_df[col].median()
            filled_count = cleaned_df[col].isnull().sum()
            if filled_count > 0:
                cleaned_df[col].fillna(median_val, inplace=True)
                print(f"   {col}: filled {filled_count} values with median ({median_val})")
        elif col_type == 'categorical':
            # Fill categorical with mode
            if cleaned_df[col].isnull().sum() > 0:
                mode_val = cleaned_df[col].mode()[0] if len(cleaned_df[col].mode()) > 0 else 'unknown'
                filled_count = cleaned_df[col].isnull().sum()
                cleaned_df[col].fillna(mode_val, inplace=True)
                print(f"   {col}: filled {filled_count} values with mode ({mode_val})")

    return cleaned_df


def generate_report(original_df, cleaned_df, quality_results, output_file):
    """Generate cleaning report"""
    print_section("CLEANING REPORT")

    report = {
        'original_shape': original_df.shape,
        'cleaned_shape': cleaned_df.shape,
        'removed_columns': quality_results['bad_columns'],
        'retained_columns': quality_results['good_columns'],
        'removed_column_details': {}
    }

    # Details of removed columns
    for col in quality_results['bad_columns']:
        report['removed_column_details'][col] = quality_results['column_stats'][col]

    # Data quality summary
    original_missing = original_df.isnull().sum().sum()
    original_cells = original_df.shape[0] * original_df.shape[1]
    original_missing_pct = (original_missing / original_cells * 100) if original_cells > 0 else 0

    cleaned_missing = cleaned_df.isnull().sum().sum()
    cleaned_cells = cleaned_df.shape[0] * cleaned_df.shape[1]
    cleaned_missing_pct = (cleaned_missing / cleaned_cells * 100) if cleaned_cells > 0 else 0

    print(f"\n📊 Summary:")
    print(f"   Original: {original_df.shape[0]} rows × {original_df.shape[1]} columns")
    print(f"   Cleaned:  {cleaned_df.shape[0]} rows × {cleaned_df.shape[1]} columns")
    print(f"   Removed:  {len(quality_results['bad_columns'])} columns")
    print(f"\n   Original missing: {original_missing_pct:.2f}%")
    print(f"   Cleaned missing:  {cleaned_missing_pct:.2f}%")

    report['quality'] = {
        'original_missing_pct': original_missing_pct,
        'cleaned_missing_pct': cleaned_missing_pct
    }

    # Save report
    report_file = output_file.replace('.csv', '_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Report saved to: {report_file}")

    # Print removed columns summary
    if quality_results['bad_columns']:
        print(f"\n🗑️  Removed columns ({len(quality_results['bad_columns'])}):")
        for col in quality_results['bad_columns'][:20]:  # Show first 20
            reasons = quality_results['column_stats'][col]['reasons']
            print(f"   - {col}: {', '.join(reasons)}")
        if len(quality_results['bad_columns']) > 20:
            print(f"   ... and {len(quality_results['bad_columns']) - 20} more")

    return report


def main():
    """Main cleaning pipeline"""

    # ========================================
    # Configuration
    # ========================================
    INPUT_FILE = 'augmented_data_v3.csv'
    OUTPUT_FILE = 'augmented_data_v3_cleaned.csv'

    print_section("DATA CLEANING PIPELINE")
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    # ========================================
    # Load Data
    # ========================================
    print_section("LOADING DATA")

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found!")
        print(f"   Available files:")
        for f in os.listdir('.'):
            if f.endswith('.csv'):
                print(f"   - {f}")
        return

    df = pd.read_csv(INPUT_FILE, index_col=False)
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

    # ========================================
    # Analyze Quality
    # ========================================
    quality_results = analyze_data_quality(df)

    # ========================================
    # Clean Data
    # ========================================
    cleaned_df = clean_dataframe(df, quality_results)

    # ========================================
    # Save Results
    # ========================================
    print_section("SAVING RESULTS")

    cleaned_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"✅ Cleaned data saved to: {OUTPUT_FILE}")

    # ========================================
    # Generate Report
    # ========================================
    report = generate_report(df, cleaned_df, quality_results, OUTPUT_FILE)

    # ========================================
    # Final Summary
    # ========================================
    print_section("✅ CLEANING COMPLETE")

    print(f"\n📁 Output files:")
    print(f"   - Cleaned data: {OUTPUT_FILE}")
    print(f"   - Report: {OUTPUT_FILE.replace('.csv', '_report.json')}")
    print(f"\n✨ Data is ready for model training!")
    print("=" * 80)


if __name__ == '__main__':
    main()
