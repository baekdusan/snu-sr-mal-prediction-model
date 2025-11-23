#!/usr/bin/env python3
"""
Feature Augmentation Script v2 - Batch Processing
Generates RICH augmented features (25-35+) for MAL prediction model
Processes in batches to avoid timeout/token limits
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
    prompt_file = 'augment_prompt_v2.md'

    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"❌ Error: {prompt_file} not found!")

    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    return prompt_template


def prepare_dataset_for_llm(data):
    """Convert DataFrame to CSV string for LLM input"""
    # Only include queries and MAL columns
    subset = data[['queries', 'MAL']].copy()
    csv_string = subset.to_csv(index=False)
    return csv_string


def extract_csv_from_response(response_text):
    """Extract CSV data from LLM response"""
    # Look for CSV code block
    pattern = r'```(?:csv)?\n(.*?)\n```'
    matches = re.findall(pattern, response_text, re.DOTALL)

    if not matches:
        # Try to find CSV without code block markers
        lines = response_text.split('\n')
        csv_lines = []
        in_csv = False

        for line in lines:
            if line.strip().startswith('queries,'):
                in_csv = True
            if in_csv:
                csv_lines.append(line)
                if line.strip() == '' or line.startswith('#') or line.startswith('**'):
                    break

        if csv_lines:
            return '\n'.join(csv_lines)

        raise ValueError("❌ Could not find CSV data in LLM response")

    return matches[0].strip()


def call_openai_api_batch(client, prompt_template, dataset_csv, batch_num):
    """Call OpenAI API to generate RICH augmented features for a batch"""
    print(f"\n🔄 Calling OpenAI API for batch {batch_num} (gpt-5.1)...")
    print(f"   Dataset size: {len(dataset_csv.split(chr(10)))-1} rows")

    user_message = f"""{prompt_template}

Please analyze this dataset and generate RICH augmented features (minimum 25-35 features).

Dataset (CSV format):
```csv
{dataset_csv}
```

IMPORTANT:
- Extract AT LEAST 25-35 features covering all categories (A-H)
- Use systematic naming with category prefixes (TXT_, LING_, SEM_, TASK_, TEMP_, INTENT_, INFO_, COMP_)
- Be thorough and creative - extract features at multiple levels
- Output the COMPLETE expanded dataset with ALL new features in CSV format
- Include ALL rows from input dataset
- Ensure consistent feature names across all batches"""

    try:
        response = client.responses.create(
            model="gpt-5.1",
            input=user_message,
            temperature=0.3,
            max_output_tokens=16000
        )

        response_text = response.output_text

        print(f"✅ API call successful")
        print(f"   Tokens used: {response.usage.total_tokens}")

        return response_text

    except Exception as e:
        print(f"❌ API call failed: {e}")
        raise


def main():
    """Main augmentation pipeline with batch processing"""

    # ========================================
    # 1. Load raw data
    # ========================================
    print_section("STEP 1: Loading Raw Data")

    data = pd.read_csv('rawdata.csv')
    print(f"✅ Loaded {len(data)} queries from rawdata.csv")

    # ========================================
    # 2. Setup OpenAI client
    # ========================================
    print_section("STEP 2: OpenAI Client Setup")

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        return

    client = OpenAI(api_key=api_key)
    print("✅ OpenAI client initialized")

    # ========================================
    # 3. Load enhanced prompt template
    # ========================================
    print_section("STEP 3: Loading Enhanced Prompt Template (v2)")

    try:
        prompt_template = load_prompt_template()
        print("✅ Loaded enhanced prompt template from augment_prompt_v2.md")
    except Exception as e:
        print(f"❌ Error loading prompt: {e}")
        return

    # ========================================
    # 4. Process in batches
    # ========================================
    print_section("STEP 4: Batch Processing")

    BATCH_SIZE = 50  # Process 50 queries at a time
    total_batches = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"   Total queries: {len(data)}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Total batches: {total_batches}")

    all_augmented_dfs = []
    batch_responses = []

    for batch_idx in range(total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, len(data))
        batch_data = data.iloc[start_idx:end_idx]

        print(f"\n{'─' * 80}")
        print(f"📦 Batch {batch_idx + 1}/{total_batches}: Rows {start_idx}-{end_idx-1} ({len(batch_data)} queries)")
        print(f"{'─' * 80}")

        try:
            # Prepare batch CSV
            dataset_csv = prepare_dataset_for_llm(batch_data)

            # Call API
            response_text = call_openai_api_batch(
                client, prompt_template, dataset_csv, batch_idx + 1
            )

            # Extract CSV
            augmented_csv = extract_csv_from_response(response_text)

            # Parse to DataFrame
            from io import StringIO
            augmented_df = pd.read_csv(StringIO(augmented_csv), index_col=False)

            print(f"   ✅ Batch processed: {len(augmented_df)} rows, {len(augmented_df.columns)} columns")

            # Store results
            all_augmented_dfs.append(augmented_df)
            batch_responses.append({
                'batch': batch_idx + 1,
                'rows': f"{start_idx}-{end_idx-1}",
                'response': response_text
            })

            # Small delay to avoid rate limiting
            if batch_idx < total_batches - 1:
                time.sleep(2)

        except Exception as e:
            print(f"   ❌ Error processing batch {batch_idx + 1}: {e}")
            print(f"   Continuing with next batch...")
            continue

    # ========================================
    # 5. Combine all batches
    # ========================================
    print_section("STEP 5: Combining Batches")

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

    # Feature summary
    feature_cols = [col for col in combined_df.columns if col not in ['queries', 'MAL', 'query', 'mal']]
    categories = {}
    for col in feature_cols:
        prefix = col.split('_')[0] if '_' in col else 'OTHER'
        categories[prefix] = categories.get(prefix, 0) + 1

    print(f"\n📊 Feature Distribution:")
    for cat, count in sorted(categories.items()):
        print(f"   {cat:12s}: {count:3d} features")

    # ========================================
    # 6. Save results
    # ========================================
    print_section("STEP 6: Saving Results")

    OUTPUT_FILE = 'augmented_data_v2.csv'
    combined_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"✅ Saved to: {OUTPUT_FILE}")

    # Save batch responses
    RESPONSE_FILE = 'batch_responses_v2.json'
    with open(RESPONSE_FILE, 'w', encoding='utf-8') as f:
        json.dump(batch_responses, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved batch responses to: {RESPONSE_FILE}")

    # ========================================
    # Summary
    # ========================================
    print_section("🎉 BATCH PROCESSING COMPLETE")

    print(f"\n📝 Summary:")
    print(f"   - Total batches processed: {len(all_augmented_dfs)}/{total_batches}")
    print(f"   - Total rows: {len(combined_df)}/{len(data)}")
    print(f"   - Total features: {len(feature_cols)}")
    print(f"   - Output: {OUTPUT_FILE}")
    print(f"\n✨ Next step: Run clean script to handle missing values!")
    print("=" * 80)


if __name__ == '__main__':
    main()
