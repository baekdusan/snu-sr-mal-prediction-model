"""
MAL Data Augmentation Pipeline (JSON-based with Structure Validation)

Input: rawdata.csv (256 rows)
Output: augmented_data.csv (256 rows + new features)

Pipeline:
1. Feature Design (GPT-5.1) - Analyze all 256 rows → Feature Specification
2. Data Gen Batch 1 (GPT-5.1) - Process rows 1-32 → JSON + Feature_Value_Reference
3. Data Gen Batch 2-8 (GPT-5-mini) - Process rows 33-256 → JSON only
4. Validate all batches (field consistency check)
5. Merge all batches → augmented_data.csv

Key improvement: JSON output + structure validation prevents field count errors
No hardcoded schema - validates based on batch 1's actual structure
"""

import os
import csv
import json
import re
from pathlib import Path
from openai import OpenAI
import pandas as pd

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

client = OpenAI(api_key=OPENAI_API_KEY)

# Model configuration
MODEL_HIGH = "gpt-5.1"  # For feature design and batch 1
MODEL_LOW = "gpt-5-mini"  # For batch 2-8

# File paths
RAWDATA_PATH = "rawdata.csv"
FEATURE_DESIGN_PROMPT_PATH = "feature_design_prompt.md"
DATA_GEN_PROMPT_PATH = "data_generation_prompt.md"
OUTPUT_PATH = "augmented_data.csv"

# Batch configuration
BATCH_SIZE = 32
TOTAL_ROWS = 256

# Checkpoint configuration
CHECKPOINT_DIR = "checkpoints"

# Batch responses directory
BATCH_RESPONSES_DIR = "batch_responses"


def load_prompt(prompt_path: str) -> str:
    """Load prompt from markdown file."""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_csv(file_path: str) -> list[dict]:
    """Load CSV file and return list of rows as dictionaries."""
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return list(reader)


def format_csv_for_prompt(rows: list[dict]) -> str:
    """Format CSV rows as a string for the prompt."""
    if not rows:
        return ""

    fieldnames = rows[0].keys()
    lines = [",".join(fieldnames)]

    for row in rows:
        lines.append(",".join(f'"{row[field]}"' for field in fieldnames))

    return "\n".join(lines)


def call_openai(system_prompt: str, user_message: str, model: str) -> str:
    """Call OpenAI API and return response."""
    print(f"Calling OpenAI API with model: {model}")

    # gpt-5-mini doesn't support temperature parameter
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    }

    # Only add temperature for models that support it
    if model != MODEL_LOW:
        kwargs["temperature"] = 0.7

    response = client.chat.completions.create(**kwargs)

    return response.choices[0].message.content


def step1_feature_design(all_rows: list[dict]) -> str:
    """Step 1: Feature Design using GPT-5.1"""
    print("\n" + "="*60)
    print("STEP 1: Feature Design (GPT-5.1)")
    print("="*60)

    prompt = load_prompt(FEATURE_DESIGN_PROMPT_PATH)
    dataset_text = format_csv_for_prompt(all_rows)

    user_message = f"""I have a dataset with {len(all_rows)} queries and their Maximum Acceptable Latency (MAL) values.

Please analyze this dataset and design meaningful features for MAL prediction.

Dataset (CSV format):
```csv
{dataset_text}
```

Please provide the Feature Specification following the output format specified in the prompt."""

    response = call_openai(prompt, user_message, MODEL_HIGH)

    # Save feature specification
    with open("feature_specification.md", 'w', encoding='utf-8') as f:
        f.write(response)

    print("✓ Feature Specification saved to: feature_specification.md")
    return response


def step2_batch1_generation(rows: list[dict], feature_spec: str) -> tuple[str, str]:
    """Step 2: Generate Batch 1 with GPT-5.1 to create Feature_Value_Reference"""
    print("\n" + "="*60)
    print("STEP 2: Data Generation Batch 1/8 (GPT-5.1)")
    print("="*60)

    prompt = load_prompt(DATA_GEN_PROMPT_PATH)
    dataset_text = format_csv_for_prompt(rows)

    user_message = f"""This is batch 1/8.

I have the Feature Specification and the first batch of queries (rows 1-32).

Feature Specification:
{feature_spec}

Dataset (CSV format):
```csv
{dataset_text}
```

Please generate the expanded dataset following the output format. Remember to include the Feature_Value_Reference section since this is batch 1/8."""

    response = call_openai(prompt, user_message, MODEL_HIGH)

    # Extract CSV and Feature_Value_Reference from response
    # Save full response
    os.makedirs(BATCH_RESPONSES_DIR, exist_ok=True)
    with open(os.path.join(BATCH_RESPONSES_DIR, "batch_1_response.md"), 'w', encoding='utf-8') as f:
        f.write(response)

    print(f"✓ Batch 1 response saved to: {BATCH_RESPONSES_DIR}/batch_1_response.md")

    # Extract Feature_Value_Reference (text between the section header and next section)
    # This is a simple extraction - you may need to adjust based on actual response format
    return response, response


def step3_subsequent_batches(batch_num: int, rows: list[dict], feature_spec: str, feature_ref: str) -> str:
    """Step 3: Generate subsequent batches with GPT-5-mini"""
    print("\n" + "="*60)
    print(f"STEP 3: Data Generation Batch {batch_num}/8 (GPT-5-mini)")
    print("="*60)

    prompt = load_prompt(DATA_GEN_PROMPT_PATH)
    dataset_text = format_csv_for_prompt(rows)

    start_row = (batch_num - 1) * BATCH_SIZE + 1
    end_row = batch_num * BATCH_SIZE

    user_message = f"""This is batch {batch_num}/8.

I have the Feature Specification, Feature_Value_Reference from batch 1, and the current batch of queries (rows {start_row}-{end_row}).

Feature Specification:
{feature_spec}

Feature_Value_Reference (from batch 1):
{feature_ref}

Dataset (CSV format):
```csv
{dataset_text}
```

Please generate the expanded dataset following the EXACT same rules as batch 1. Use the Feature_Value_Reference to ensure consistency."""

    response = call_openai(prompt, user_message, MODEL_LOW)

    # Save response
    os.makedirs(BATCH_RESPONSES_DIR, exist_ok=True)
    with open(os.path.join(BATCH_RESPONSES_DIR, f"batch_{batch_num}_response.md"), 'w', encoding='utf-8') as f:
        f.write(response)

    print(f"✓ Batch {batch_num} response saved to: {BATCH_RESPONSES_DIR}/batch_{batch_num}_response.md")
    return response


def extract_json_from_response(response: str) -> list[dict]:
    """Extract JSON content from markdown code block in response and validate it."""
    # Look for ```json code blocks
    pattern = r'```json\s*\n(.*?)\n```'
    match = re.search(pattern, response, re.DOTALL)

    if not match:
        # Try without language tag
        pattern = r'```\s*\n(\[.*?\])\s*\n```'
        match = re.search(pattern, response, re.DOTALL)

    if not match:
        raise ValueError("No JSON code block found in response")

    json_text = match.group(1)

    # Parse JSON
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in response: {e}")

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data)}")

    return data


def validate_batch_structure(batch_num: int, data: list[dict], expected_fields: set = None) -> tuple[list[dict], set]:
    """
    Validate batch data structure (field consistency).

    Args:
        batch_num: Batch number for logging
        data: List of dictionaries to validate
        expected_fields: Expected field names (from batch 1). If None, this IS batch 1.

    Returns:
        Tuple of (validated data, field set)

    Raises:
        ValueError: If structure validation fails
    """
    if not data:
        raise ValueError(f"Batch {batch_num}: No data provided")

    print(f"\n  Validating batch {batch_num} structure...")
    print(f"    - {len(data)} rows to validate")

    # Get field set from first row
    first_row_fields = set(data[0].keys())

    if expected_fields is None:
        # This is batch 1 - establish the schema
        print(f"    - Batch 1: Establishing schema with {len(first_row_fields)} fields")
        expected_fields = first_row_fields
    else:
        # Subsequent batches - check consistency with batch 1
        if first_row_fields != expected_fields:
            missing = expected_fields - first_row_fields
            extra = first_row_fields - expected_fields
            error_msg = f"Batch {batch_num} field mismatch with batch 1.\n"
            if missing:
                error_msg += f"      Missing fields: {missing}\n"
            if extra:
                error_msg += f"      Extra fields: {extra}"
            raise ValueError(error_msg)

    # Validate all rows have same structure
    for row_idx, row in enumerate(data):
        row_fields = set(row.keys())
        if row_fields != expected_fields:
            missing = expected_fields - row_fields
            extra = row_fields - expected_fields
            error_msg = f"Batch {batch_num}, Row {row_idx}: Field mismatch.\n"
            if missing:
                error_msg += f"      Missing: {missing}\n"
            if extra:
                error_msg += f"      Extra: {extra}"
            raise ValueError(error_msg)

    print(f"    ✓ All rows have consistent {len(expected_fields)} fields")
    return data, expected_fields


def merge_and_validate_batches(batch_responses: list[str], output_path: str):
    """
    Extract JSON from all batches, validate structure consistency, and merge into CSV.

    This function ensures all batches have the same field structure (based on batch 1).
    """
    print("\n" + "="*60)
    print("STEP 4: Extracting, Validating, and Merging all batches")
    print("="*60)

    all_data = []
    expected_fields = None  # Will be set by batch 1

    for i, response in enumerate(batch_responses, 1):
        print(f"\nProcessing batch {i}...")

        # Extract JSON from response
        try:
            json_data = extract_json_from_response(response)
            print(f"  ✓ Extracted {len(json_data)} rows from JSON")
        except ValueError as e:
            print(f"  ✗ Failed to extract JSON: {e}")
            raise

        # Validate structure
        try:
            validated_data, field_set = validate_batch_structure(i, json_data, expected_fields)

            # Batch 1 establishes the schema
            if i == 1:
                expected_fields = field_set
                print(f"    - Schema established: {sorted(expected_fields)[:5]}... (+{len(expected_fields)-5} more)")

            all_data.extend(validated_data)

        except ValueError as e:
            print(f"  ✗ Batch {i} validation failed:")
            print(f"      {str(e)}")
            raise

    print(f"\n{'='*60}")
    print(f"Validation Summary:")
    print(f"  Total batches: {len(batch_responses)}")
    print(f"  Total validated rows: {len(all_data)}")
    print(f"  Field count: {len(expected_fields)}")
    print(f"{'='*60}")

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Save to CSV
    df.to_csv(output_path, index=False, encoding='utf-8')

    print(f"\n✓ Merged and validated CSV saved to: {output_path}")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns[:5])}... (+{df.shape[1]-5} more)")

    return df


def check_existing_files() -> dict:
    """Check which steps have already been completed."""
    status = {
        'feature_spec': os.path.exists('feature_specification.md'),
        'batch_1': os.path.exists(os.path.join(BATCH_RESPONSES_DIR, 'batch_1_response.md')),
        'batches': {}
    }

    for i in range(2, 9):
        status['batches'][i] = os.path.exists(os.path.join(BATCH_RESPONSES_DIR, f'batch_{i}_response.md'))

    return status


def load_existing_feature_spec() -> str:
    """Load existing feature specification."""
    with open('feature_specification.md', 'r', encoding='utf-8') as f:
        return f.read()


def load_existing_batch(batch_num: int) -> str:
    """Load existing batch response."""
    with open(os.path.join(BATCH_RESPONSES_DIR, f'batch_{batch_num}_response.md'), 'r', encoding='utf-8') as f:
        return f.read()


def main(resume: bool = True):
    """Main pipeline execution.

    Args:
        resume: If True, resume from existing files. If False, start from scratch.
    """
    print("\n" + "="*60)
    print("MAL DATA AUGMENTATION PIPELINE")
    print("="*60)

    # Load raw data
    print(f"\nLoading data from: {RAWDATA_PATH}")
    all_rows = load_csv(RAWDATA_PATH)
    print(f"✓ Loaded {len(all_rows)} rows")

    # Check existing files if resume mode
    status = check_existing_files() if resume else {'feature_spec': False, 'batch_1': False, 'batches': {}}

    if resume:
        print("\nChecking for existing progress...")
        print(f"  - feature_specification.md: {'✓ Found' if status['feature_spec'] else '✗ Missing'}")
        print(f"  - batch_1_response.md: {'✓ Found' if status['batch_1'] else '✗ Missing'}")
        for i in range(2, 9):
            if status['batches'].get(i):
                print(f"  - batch_{i}_response.md: ✓ Found")

    # Step 1: Feature Design
    if status['feature_spec']:
        print("\n" + "="*60)
        print("STEP 1: Feature Design (SKIPPED - using existing)")
        print("="*60)
        feature_spec = load_existing_feature_spec()
        print("✓ Loaded existing feature_specification.md")
    else:
        feature_spec = step1_feature_design(all_rows)

    # Step 2: Batch 1 (with Feature_Value_Reference)
    if status['batch_1']:
        print("\n" + "="*60)
        print("STEP 2: Data Generation Batch 1/8 (SKIPPED - using existing)")
        print("="*60)
        batch_1_response = load_existing_batch(1)
        feature_ref = batch_1_response
        print("✓ Loaded existing batch_1_response.md")
    else:
        batch_1_rows = all_rows[:BATCH_SIZE]
        batch_1_response, feature_ref = step2_batch1_generation(batch_1_rows, feature_spec)

    batch_responses = [batch_1_response]

    # Step 3: Batches 2-8
    num_batches = (len(all_rows) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(2, num_batches + 1):
        if status['batches'].get(batch_num):
            print("\n" + "="*60)
            print(f"STEP 3: Data Generation Batch {batch_num}/8 (SKIPPED - using existing)")
            print("="*60)
            response = load_existing_batch(batch_num)
            print(f"✓ Loaded existing batch_{batch_num}_response.md")
        else:
            start_idx = (batch_num - 1) * BATCH_SIZE
            end_idx = min(batch_num * BATCH_SIZE, len(all_rows))
            batch_rows = all_rows[start_idx:end_idx]

            response = step3_subsequent_batches(batch_num, batch_rows, feature_spec, feature_ref)

        batch_responses.append(response)

    # Step 4: Validate and merge all batches
    merge_and_validate_batches(batch_responses, OUTPUT_PATH)

    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nOutput file: {OUTPUT_PATH}")
    print("\nGenerated files:")
    print("  - feature_specification.md")
    print("  - batch_1_response.md (includes Feature_Value_Reference)")
    print("  - batch_2_response.md ... batch_8_response.md")
    print(f"  - {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
