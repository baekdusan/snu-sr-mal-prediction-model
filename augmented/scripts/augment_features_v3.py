#!/usr/bin/env python3
"""
Feature Augmentation Script v3 - Two-Stage Processing
Stage 1: Analyze entire dataset and define feature schema
Stage 2: Apply schema to batches for consistent features
"""

import pandas as pd
import numpy as np
import os
import re
from openai import OpenAI
import time
import json


def print_section(title):
    """Print section header"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_prompt_template():
    """Load the augment_prompt_v2.md file"""
    # Get script directory and construct path to prompts
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file = os.path.join(script_dir, '..', 'prompts', 'augment_prompt_v2.md')

    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"❌ Error: {prompt_file} not found!")

    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    return prompt_template


def stage1_define_features(client, prompt_template, sample_data):
    """Stage 1: Define feature schema by analyzing sample data"""
    print_section("STAGE 1: Defining Feature Schema")

    # Take a diverse sample (first 10, middle 10, last 10)
    total = len(sample_data)
    sample_indices = list(range(10)) + list(range(total//2-5, total//2+5)) + list(range(total-10, total))
    sample = sample_data.iloc[sample_indices].copy()

    sample_csv = sample[['queries', 'MAL']].to_csv(index=False)

    print(f"📊 Analyzing {len(sample)} sample queries to define features...")

    schema_prompt = f"""{prompt_template}

**TASK: FEATURE SCHEMA DEFINITION**

Analyze this sample dataset and define a COMPLETE, CONSISTENT feature schema following the guidelines above.

Sample Dataset (CSV format):
```csv
{sample_csv}
```

INSTRUCTIONS:
1. Follow the feature extraction guidelines in the system prompt above
2. Extract a RICH set of features (aim for 30+ features as described in the prompt)
3. Cover ALL required feature categories (A-H)
4. Use clear, systematic naming with category prefixes

OUTPUT REQUIREMENTS:
- Output ONLY a CSV with header + 3 example rows
- Format: queries,MAL,feature1,feature2,...
- NO explanations, NO Feature_Specification table, NO additional text
- Every row MUST have a value for EVERY feature (use 0, "none", or appropriate default if not applicable)
- BE CONSISTENT: This exact schema will be applied to ALL batches!

Output the feature schema as CSV (header + 3 example rows):"""

    try:
        response = client.responses.create(
            model="gpt-5.1",  # Use full model for schema definition (accuracy matters)
            input=schema_prompt,
            temperature=0.1,
            max_output_tokens=4000
        )

        response_text = response.output_text
        print(f"✅ Feature schema generated")
        print(f"   Tokens used: {response.usage.total_tokens}")

        # Extract CSV - try multiple methods
        csv_content = None

        # Method 1: Try code block format
        pattern = r'```(?:csv)?\n(.*?)\n```'
        matches = re.findall(pattern, response_text, re.DOTALL)
        if matches:
            csv_content = matches[0].strip()

        # Method 2: If no code block, check if response starts with CSV directly
        if not csv_content and response_text.strip().startswith('queries,'):
            csv_content = response_text.strip()

        # Method 3: Find the first line starting with "queries,"
        if not csv_content:
            lines = response_text.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('queries,'):
                    # Take from this line onwards
                    csv_content = '\n'.join(lines[i:]).strip()
                    break

        if csv_content:
            header_line = csv_content.split('\n')[0]
            features = header_line.split(',')

            print(f"\n✅ Feature schema defined:")
            print(f"   Total features: {len(features)}")

            # Group by category
            categories = {}
            for feat in features:
                if feat in ['queries', 'MAL']:
                    continue
                prefix = feat.split('_')[0] if '_' in feat else 'OTHER'
                categories[prefix] = categories.get(prefix, 0) + 1

            print(f"\n   Feature distribution:")
            for cat, count in sorted(categories.items()):
                print(f"     {cat:10s}: {count:3d} features")

            return features, response_text
        else:
            raise ValueError("Could not extract feature schema from response")

    except Exception as e:
        print(f"❌ Stage 1 failed: {e}")
        raise


def stage2_apply_schema(client, prompt_template, dataset_csv, feature_schema, batch_num):
    """Stage 2: Apply predefined feature schema to batch data"""

    print(f"\n🔄 Processing batch {batch_num} with predefined schema...")
    print(f"   Dataset size: {len(dataset_csv.split(chr(10)))-1} rows")

    # Create schema description
    schema_header = ','.join(feature_schema)

    # Count input rows for validation
    input_row_count = len(dataset_csv.split('\n')) - 1  # -1 for header

    batch_prompt = f"""You MUST use this EXACT feature schema (same features, same order):

{schema_header}

Now extract features for this dataset:

```csv
{dataset_csv}
```

CRITICAL REQUIREMENTS:
1. Output ONLY CSV data - NO explanations, NO analysis, NO markdown
2. First line MUST be the header (exactly as shown above)
3. Include ALL {input_row_count} input rows (NO EXCEPTIONS!)
4. Calculate values for ALL features in the schema
5. Use proper CSV quoting for text fields containing commas
6. IMPORTANT: NEVER leave a cell empty. If a feature doesn't apply:
   - For numeric features: use 0
   - For categorical/text features: use "none" or "neutral"
   - EVERY cell MUST have a value

OUTPUT FORMAT:
Start your response with the CSV header line, then provide all {input_row_count} data rows.
Do NOT include explanations before or after the CSV.
Do NOT use markdown code blocks.
Output ONLY the raw CSV data.

EXAMPLE OF CORRECT ROW (all cells filled):
"What movies did I watch last week?",5.2,28,24,6,4.0,1,1,0,0.86,1,interrogative,past,0,0,none,1,1,5,0.2,media,entertainment,0,1,0,1,5,8,2,3,search,1,0,0,text,0,1,relative,3,1,0,0,2,1,neutral,6,2,7,8,medium,2,0,1"""

    try:
        response = client.responses.create(
            model="gpt-5-mini",  # Use mini for batch processing (cost-effective)
            input=batch_prompt,
            max_output_tokens=8000
        )

        response_text = response.output_text

        print(f"   ✅ Batch processed")
        print(f"   Tokens used: {response.usage.total_tokens}")

        return response_text

    except Exception as e:
        print(f"   ❌ Batch {batch_num} failed: {e}")
        raise


def extract_csv_from_response(response_text):
    """Extract CSV data from LLM response with robust parsing"""
    # First try: code blocks with ```csv or ```
    pattern = r'```(?:csv)?\n(.*?)\n```'
    matches = re.findall(pattern, response_text, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Second try: Find lines starting with "queries,"
    lines = response_text.split('\n')
    csv_lines = []
    in_csv = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Start of CSV (header line)
        if stripped.startswith('queries,'):
            in_csv = True
            csv_lines.append(line)
            continue

        # If we're in CSV section
        if in_csv:
            # Stop conditions
            if (stripped == '' or
                stripped.startswith('#') or
                stripped.startswith('**') or
                stripped.startswith('---') or
                stripped.startswith('Note:') or
                (i > 0 and not ',' in stripped and len(stripped) > 20)):  # Likely prose
                break
            csv_lines.append(line)

    if csv_lines and len(csv_lines) > 1:  # At least header + 1 data row
        return '\n'.join(csv_lines)

    raise ValueError("❌ Could not find CSV data in LLM response")


def validate_batch_alignment(augmented_df, batch_data, feature_schema):
    """
    Validate if batch output aligns with feature schema.
    Returns (is_valid, validation_errors)
    """
    validation_errors = []

    # Check 1: Row count must match exactly
    expected_rows = len(batch_data)
    actual_rows = len(augmented_df)

    if actual_rows != expected_rows:
        validation_errors.append(f"Row count mismatch: expected {expected_rows}, got {actual_rows}")
        return False, validation_errors

    # Check 2: All feature columns must exist
    missing_cols = [col for col in feature_schema if col not in augmented_df.columns]
    if missing_cols:
        validation_errors.append(f"Missing {len(missing_cols)} columns: {missing_cols[:5]}...")
        return False, validation_errors

    # Check 3: No extra columns
    extra_cols = [col for col in augmented_df.columns if col not in feature_schema]
    if extra_cols:
        validation_errors.append(f"Extra {len(extra_cols)} columns: {extra_cols[:5]}...")
        return False, validation_errors

    # Check 4: Column order must match schema
    if list(augmented_df.columns) != feature_schema:
        validation_errors.append("Column order does not match schema")
        return False, validation_errors

    # Check 5: No excessive missing values (>50% per column)
    # Note: Some features like COMP_* may legitimately be 0/"none" for simple queries
    # We only check for truly missing (NaN) values, not 0/"none" values
    missing_per_col = augmented_df.isnull().sum()
    high_missing_cols = missing_per_col[missing_per_col > expected_rows * 0.5]  # >50% missing

    if len(high_missing_cols) > 0:
        validation_errors.append(f"Data quality issue: {len(high_missing_cols)} columns have >50% missing values")
        for col_name, missing_count in high_missing_cols.head(5).items():
            validation_errors.append(f"   - {col_name}: {missing_count}/{expected_rows} missing ({missing_count/expected_rows*100:.1f}%)")
        return False, validation_errors

    # Check 6: queries column must match original
    if 'queries' in augmented_df.columns and 'queries' in batch_data.columns:
        original_queries = batch_data['queries'].values
        augmented_queries = augmented_df['queries'].values

        mismatches = sum(
            1 for i in range(len(original_queries))
            if pd.isna(augmented_queries[i]) or
            str(original_queries[i]).strip() != str(augmented_queries[i]).strip()
        )

        if mismatches > expected_rows * 0.1:  # >10% mismatch is a problem
            validation_errors.append(f"Query mismatch: {mismatches}/{expected_rows} queries don't match ({mismatches/expected_rows*100:.1f}%)")
            return False, validation_errors

    return True, []


def validate_and_fix_batch(augmented_df, batch_data, feature_schema):
    """Validate batch output and fix common issues"""
    issues = []
    is_corrupted = False

    # Issue 1: Row count mismatch
    expected_rows = len(batch_data)
    actual_rows = len(augmented_df)

    if actual_rows != expected_rows:
        issues.append(f"Row mismatch: expected {expected_rows}, got {actual_rows}")

        # If fewer rows, pad with NaN
        if actual_rows < expected_rows:
            missing_rows = expected_rows - actual_rows
            # Create empty rows with same columns
            empty_df = pd.DataFrame(
                [[np.nan] * len(augmented_df.columns)] * missing_rows,
                columns=augmented_df.columns
            )
            augmented_df = pd.concat([augmented_df, empty_df], ignore_index=True)
            issues.append(f"⚠️  Padded {missing_rows} missing rows with NaN")
            is_corrupted = True

        # If too many rows, truncate
        elif actual_rows > expected_rows:
            augmented_df = augmented_df.iloc[:expected_rows]
            issues.append(f"⚠️  Truncated {actual_rows - expected_rows} extra rows")
            is_corrupted = True

    # Issue 2: Column count mismatch
    expected_cols = len(feature_schema)
    actual_cols = len(augmented_df.columns)

    if actual_cols != expected_cols:
        issues.append(f"Column mismatch: expected {expected_cols}, got {actual_cols}")

        # If missing columns, add them with NaN
        if actual_cols < expected_cols:
            for col in feature_schema:
                if col not in augmented_df.columns:
                    augmented_df[col] = np.nan
                    issues.append(f"⚠️  Added missing column: {col}")

            # Reorder columns to match schema
            augmented_df = augmented_df[feature_schema]
            is_corrupted = True

        # If too many columns, remove extras
        elif actual_cols > expected_cols:
            augmented_df = augmented_df[feature_schema]
            issues.append(f"⚠️  Removed {actual_cols - expected_cols} extra columns")
            is_corrupted = True

    # Issue 3: Check for excessive missing values (data quality)
    missing_per_col = augmented_df.isnull().sum()
    high_missing_cols = missing_per_col[missing_per_col > expected_rows * 0.5]  # >50% missing

    if len(high_missing_cols) > 0:
        issues.append(f"❌ Data quality issue: {len(high_missing_cols)} columns have >50% missing values")
        for col_name, missing_count in high_missing_cols.items():
            issues.append(f"   - {col_name}: {missing_count}/{expected_rows} missing ({missing_count/expected_rows*100:.1f}%)")
        is_corrupted = True

    # Issue 4: Ensure queries column matches original
    if 'queries' in augmented_df.columns and 'queries' in batch_data.columns:
        original_queries = batch_data['queries'].values
        augmented_queries = augmented_df['queries'].values

        # Check if queries match (allowing for some LLM variation)
        mismatches = 0
        for i in range(min(len(original_queries), len(augmented_queries))):
            if pd.isna(augmented_queries[i]) or str(original_queries[i]).strip() != str(augmented_queries[i]).strip():
                # Replace with original query
                augmented_df.loc[i, 'queries'] = original_queries[i]
                mismatches += 1

        if mismatches > 0:
            issues.append(f"⚠️  Fixed {mismatches} mismatched queries")

    # Issue 5: Replace original MAL values to ensure consistency
    if 'MAL' in augmented_df.columns and 'MAL' in batch_data.columns:
        augmented_df['MAL'] = batch_data['MAL'].values[:len(augmented_df)]

    return augmented_df, issues, is_corrupted


def main():
    """Main two-stage augmentation pipeline"""

    # ========================================
    # Setup
    # ========================================
    print_section("SETUP: Two-Stage Feature Augmentation")

    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    results_dir = os.path.join(script_dir, '..', 'results')

    # Load data
    rawdata_path = os.path.join(data_dir, 'rawdata.csv')
    data = pd.read_csv(rawdata_path)
    print(f"✅ Loaded {len(data)} queries from {rawdata_path}")

    # Setup OpenAI client
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        return

    client = OpenAI(api_key=api_key)
    print("✅ OpenAI client initialized")

    # Load prompt template
    try:
        prompt_template = load_prompt_template()
        print("✅ Loaded prompt template from augment_prompt_v2.md")
    except Exception as e:
        print(f"❌ Error loading prompt: {e}")
        return

    # ========================================
    # STAGE 1: Define Feature Schema
    # ========================================
    try:
        feature_schema, schema_response = stage1_define_features(
            client, prompt_template, data
        )

        # Save schema
        schema_output_path = os.path.join(results_dir, 'feature_schema_v3.json')
        with open(schema_output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'features': feature_schema,
                'total_features': len(feature_schema),
                'response': schema_response
            }, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Feature schema saved to: {schema_output_path}")

    except Exception as e:
        print(f"❌ Stage 1 failed, cannot continue: {e}")
        return

    # ========================================
    # STAGE 2: Apply Schema to Batches
    # ========================================
    print_section("STAGE 2: Applying Schema to Batches")

    BATCH_SIZE = 50  # Process 50 queries at a time
    MAX_RETRIES = 5  # Maximum number of retries per batch
    total_batches = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"   Total queries: {len(data)}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Total batches: {total_batches}")
    print(f"   Feature schema: {len(feature_schema)} features")
    print(f"   Model: gpt-5-mini (cost-effective)")
    print(f"   Max retries per batch: {MAX_RETRIES}")

    all_augmented_dfs = []
    batch_responses = []

    for batch_idx in range(total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, len(data))
        batch_data = data.iloc[start_idx:end_idx]

        print(f"\n{'─' * 80}")
        print(f"📦 Batch {batch_idx + 1}/{total_batches}: Rows {start_idx}-{end_idx-1} ({len(batch_data)} queries)")
        print(f"{'─' * 80}")

        # Prepare batch CSV once (used for all retries)
        dataset_csv = batch_data[['queries', 'MAL']].to_csv(index=False)

        # Retry loop
        retry_count = 0
        success = False

        while retry_count < MAX_RETRIES and not success:
            if retry_count > 0:
                print(f"\n   🔄 Retry attempt {retry_count}/{MAX_RETRIES-1} (using gpt-5.1-mini)...")

            try:
                # Apply schema to batch
                response_text = stage2_apply_schema(
                    client, prompt_template, dataset_csv, feature_schema, batch_idx + 1
                )

                # Extract CSV
                augmented_csv = extract_csv_from_response(response_text)

                # Parse to DataFrame
                from io import StringIO
                try:
                    # Try parsing with proper CSV handling (quoted fields)
                    augmented_df = pd.read_csv(
                        StringIO(augmented_csv),
                        index_col=False,
                        quoting=1,  # QUOTE_ALL
                        skipinitialspace=True,
                        on_bad_lines='warn'
                    )
                except Exception as parse_error:
                    print(f"   ⚠️  CSV parsing error: {parse_error}")
                    # Fallback to basic parsing
                    augmented_df = pd.read_csv(StringIO(augmented_csv), index_col=False)

                print(f"   📊 Raw output: {len(augmented_df)} rows, {len(augmented_df.columns)} columns")

                # Validate alignment with feature schema
                is_valid, validation_errors = validate_batch_alignment(
                    augmented_df, batch_data, feature_schema
                )

                if is_valid:
                    print(f"   ✅ Validation passed: Output aligns with feature schema")

                    # Apply minor fixes (queries/MAL restoration)
                    augmented_df, issues, _ = validate_and_fix_batch(
                        augmented_df, batch_data, feature_schema
                    )

                    if issues:
                        print(f"   🔧 Minor fixes applied:")
                        for issue in issues:
                            print(f"      {issue}")

                    print(f"   ✅ Batch {batch_idx + 1} completed: {len(augmented_df)} rows, {len(augmented_df.columns)} columns")

                    # Store results
                    all_augmented_dfs.append(augmented_df)
                    batch_responses.append({
                        'batch': batch_idx + 1,
                        'rows': f"{start_idx}-{end_idx-1}",
                        'response': response_text,
                        'retries': retry_count
                    })

                    success = True

                else:
                    # Validation failed
                    print(f"   ❌ Validation failed:")
                    for error in validation_errors:
                        print(f"      {error}")

                    retry_count += 1

                    if retry_count >= MAX_RETRIES:
                        print(f"   ❌ Max retries reached. Skipping batch {batch_idx + 1}.")
                        print(f"   💡 This batch will need manual processing")
                    else:
                        print(f"   🔄 Retrying batch {batch_idx + 1}...")
                        time.sleep(1)  # Brief delay before retry

            except Exception as e:
                print(f"   ❌ Error processing batch {batch_idx + 1}: {e}")
                retry_count += 1

                if retry_count >= MAX_RETRIES:
                    print(f"   ❌ Max retries reached. Skipping batch {batch_idx + 1}.")
                    print(f"   Continuing with next batch...")
                else:
                    print(f"   🔄 Retrying batch {batch_idx + 1}...")
                    time.sleep(1)  # Brief delay before retry

        # Small delay between batches to avoid rate limiting
        if batch_idx < total_batches - 1 and success:
            time.sleep(0.5)

    # ========================================
    # Combine and Save
    # ========================================
    print_section("FINALIZING: Combining Results")

    if not all_augmented_dfs:
        print("❌ No batches were successfully processed!")
        return

    # Concatenate all dataframes
    combined_df = pd.concat(all_augmented_dfs, ignore_index=True)

    print(f"✅ Combined all batches")
    print(f"   Total rows: {len(combined_df)}")
    print(f"   Total columns: {len(combined_df.columns)}")

    # Check for missing data
    if len(combined_df) != len(data):
        print(f"   ⚠️  Warning: Expected {len(data)} rows, got {len(combined_df)}")

        # Pad with NaN if needed
        if len(combined_df) < len(data):
            missing = len(data) - len(combined_df)
            print(f"   🔧 Padding {missing} missing rows with NaN")
            empty_df = pd.DataFrame(
                [[np.nan] * len(combined_df.columns)] * missing,
                columns=combined_df.columns
            )
            combined_df = pd.concat([combined_df, empty_df], ignore_index=True)

        # Truncate if too many
        elif len(combined_df) > len(data):
            extra = len(combined_df) - len(data)
            print(f"   🔧 Truncating {extra} extra rows")
            combined_df = combined_df.iloc[:len(data)]

    # Ensure original queries and MAL values are preserved
    print(f"\n🔧 Restoring original queries and MAL values...")
    combined_df['queries'] = data['queries'].values[:len(combined_df)]
    combined_df['MAL'] = data['MAL'].values[:len(combined_df)]

    # Missing value analysis
    missing_count = combined_df.isnull().sum().sum()
    total_cells = combined_df.shape[0] * combined_df.shape[1]
    missing_pct = (missing_count / total_cells * 100) if total_cells > 0 else 0

    print(f"\n📊 Data Quality:")
    print(f"   Missing values: {missing_count} / {total_cells} ({missing_pct:.2f}%)")

    # Save results
    output_csv_path = os.path.join(data_dir, 'augmented_data_v3.csv')
    combined_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"\n✅ Saved to: {output_csv_path}")

    # Save batch responses
    response_json_path = os.path.join(results_dir, 'batch_responses_v3.json')
    with open(response_json_path, 'w', encoding='utf-8') as f:
        json.dump(batch_responses, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved batch responses to: {response_json_path}")

    # ========================================
    # Summary
    # ========================================
    print_section("🎉 TWO-STAGE PROCESSING COMPLETE")

    print(f"\n📝 Summary:")
    print(f"   - Feature schema: {len(feature_schema)} features")
    print(f"   - Batches processed: {len(all_augmented_dfs)}/{total_batches}")
    print(f"   - Total rows: {len(combined_df)}/{len(data)}")
    print(f"   - Missing data: {missing_pct:.2f}%")
    print(f"   - Output: {output_csv_path}")
    print("\n✨ Two-stage processing ensures consistent features across all batches!")
    print("=" * 80)


if __name__ == '__main__':
    main()
