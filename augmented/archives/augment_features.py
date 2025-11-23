#!/usr/bin/env python3
"""
Feature Augmentation Script using OpenAI API
Generates augmented features for MAL prediction model
"""

import pandas as pd
import numpy as np
import os
import re
from openai import OpenAI


def print_section(title):
    """Print section header"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_prompt_template():
    """Load the augment_prompt.md file"""
    prompt_file = 'augment_prompt.md'

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
        # Look for header line starting with "queries"
        lines = response_text.split('\n')
        csv_lines = []
        in_csv = False

        for line in lines:
            if line.strip().startswith('queries,'):
                in_csv = True
            if in_csv:
                csv_lines.append(line)
                # Stop if we hit an empty line or markdown section
                if line.strip() == '' or line.startswith('#') or line.startswith('**'):
                    break

        if csv_lines:
            return '\n'.join(csv_lines)

        raise ValueError("❌ Could not find CSV data in LLM response")

    # Return the first (and hopefully only) CSV block
    return matches[0].strip()


def call_openai_api(client, prompt_template, dataset_csv):
    """Call OpenAI API to generate augmented features"""
    print("\n🔄 Calling OpenAI API (gpt-4o-mini)...")
    print(f"   Dataset size: {len(dataset_csv.split(chr(10)))-1} rows")

    # Construct user message
    user_message = f"""Please analyze this dataset and generate augmented features.

Dataset (CSV format):
```csv
{dataset_csv}
```

Please output the complete expanded dataset with all new features in CSV format."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt_template},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,  # Lower temperature for more consistent feature extraction
            max_tokens=16000  # Enough for 256 rows + features
        )

        response_text = response.choices[0].message.content

        print(f"✅ API call successful")
        print(f"   Tokens used: {response.usage.total_tokens}")
        print(f"   Model: {response.model}")

        return response_text

    except Exception as e:
        print(f"❌ API call failed: {e}")
        raise


def main():
    """Main augmentation pipeline"""

    # ========================================
    # 1. Load raw data
    # ========================================
    print_section("STEP 1: Loading Raw Data")

    data = pd.read_csv('rawdata.csv')
    print(f"✅ Loaded {len(data)} queries from rawdata.csv")
    print(f"   Columns: {list(data.columns)}")


    # ========================================
    # 2. Setup OpenAI client
    # ========================================
    print_section("STEP 2: OpenAI Client Setup")

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("   Please set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-key-here'")
        return

    client = OpenAI(api_key=api_key)
    print("✅ OpenAI client initialized")


    # ========================================
    # 3. Load prompt template
    # ========================================
    print_section("STEP 3: Loading Prompt Template")

    try:
        prompt_template = load_prompt_template()
        print("✅ Loaded prompt template from augment_prompt.md")
        print(f"   Template length: {len(prompt_template)} characters")
    except Exception as e:
        print(f"❌ Error loading prompt: {e}")
        return


    # ========================================
    # 4. Prepare dataset for LLM
    # ========================================
    print_section("STEP 4: Preparing Dataset")

    dataset_csv = prepare_dataset_for_llm(data)
    print("✅ Dataset prepared for LLM")
    print(f"   CSV size: {len(dataset_csv)} characters")
    print("\nFirst 3 rows preview:")
    print('\n'.join(dataset_csv.split('\n')[:4]))


    # ========================================
    # 5. Call OpenAI API
    # ========================================
    print_section("STEP 5: Calling OpenAI API")

    try:
        response_text = call_openai_api(client, prompt_template, dataset_csv)
    except Exception as e:
        print(f"❌ Failed to get response from OpenAI: {e}")
        return


    # ========================================
    # 6. Extract and validate CSV
    # ========================================
    print_section("STEP 6: Extracting Augmented Dataset")

    try:
        augmented_csv = extract_csv_from_response(response_text)
        print("✅ Successfully extracted CSV from response")

        # Parse CSV to validate
        from io import StringIO
        augmented_df = pd.read_csv(StringIO(augmented_csv))

        print(f"\n📊 Augmented Dataset Info:")
        print(f"   Total rows: {len(augmented_df)}")
        print(f"   Total columns: {len(augmented_df.columns)}")
        print(f"   Columns: {list(augmented_df.columns)}")

        # Validate row count
        if len(augmented_df) != len(data):
            print(f"\n⚠️  Warning: Row count mismatch!")
            print(f"   Original: {len(data)} rows")
            print(f"   Augmented: {len(augmented_df)} rows")

        print("\nFirst 3 rows:")
        print(augmented_df.head(3).to_string())

    except Exception as e:
        print(f"❌ Error extracting CSV: {e}")
        print("\n--- Full Response ---")
        print(response_text)
        return


    # ========================================
    # 7. Check for data leakage and clean
    # ========================================
    print_section("STEP 7: Data Leakage Detection")

    # Find MAL column
    mal_column = 'MAL' if 'MAL' in augmented_df.columns else 'mal'
    mal_values = augmented_df[mal_column].values

    # Check all feature columns for leakage
    exclude_cols = ['queries', 'MAL', 'query', 'mal']
    feature_cols = [col for col in augmented_df.columns if col not in exclude_cols]

    leaked_features = []
    for col in feature_cols:
        # Only check numeric columns
        if augmented_df[col].dtype in ['int64', 'float64']:
            # Check if values are identical to MAL
            if np.allclose(augmented_df[col].values, mal_values, rtol=1e-9, atol=1e-9):
                leaked_features.append(col)
                print(f"   ⚠️  LEAKAGE DETECTED: '{col}' is identical to MAL")

    if leaked_features:
        print(f"\n🗑️  Removing {len(leaked_features)} leaked feature(s): {leaked_features}")
        augmented_df = augmented_df.drop(columns=leaked_features)
        print(f"   ✅ Cleaned dataset: {len(augmented_df.columns)} columns remaining")
    else:
        print(f"   ✅ No data leakage detected - all features are valid")


    # ========================================
    # 8. Save augmented dataset
    # ========================================
    print_section("STEP 8: Saving Augmented Dataset")

    OUTPUT_FILE = 'augmented_data.csv'

    try:
        augmented_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        print(f"✅ Augmented dataset saved successfully!")
        print(f"   File: {OUTPUT_FILE}")
        print(f"   Rows: {len(augmented_df)}")
        print(f"   Columns: {len(augmented_df.columns)}")

        # Also save the full response for reference
        RESPONSE_FILE = 'llm_response.txt'
        with open(RESPONSE_FILE, 'w', encoding='utf-8') as f:
            f.write(response_text)
        print(f"\n💾 Full LLM response saved to: {RESPONSE_FILE}")
        print("   (Contains feature explanations and modeling notes)")

    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return


    # ========================================
    # Summary
    # ========================================
    print_section("🎉 FEATURE AUGMENTATION COMPLETE")

    print(f"\n📝 Summary:")
    print(f"   - Input: {len(data)} queries from rawdata.csv")
    print(f"   - Output: {len(augmented_df)} rows with {len(augmented_df.columns)} columns")
    print(f"   - New features added: {len(augmented_df.columns) - len(data.columns)}")
    if leaked_features:
        print(f"   - Leaked features removed: {len(leaked_features)} ({leaked_features})")
    print(f"   - Saved to: {OUTPUT_FILE}")
    print(f"\n✨ Next step: Run 'train_model_augmented.py' to train models!")
    print("=" * 80)


if __name__ == '__main__':
    main()
