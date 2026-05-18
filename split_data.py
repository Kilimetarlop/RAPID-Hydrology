import os
import pandas as pd
from pathlib import Path

# ==================== Configuration (update to actual paths) ====================
source_dir = Path("./dataset")  # Source data directory (containing CSV files directly)
target_parent_A = Path("./dataset/train_dataset")  # Directory for first keep_rows rows
target_parent_B = Path("./dataset/test_dataset")  # Directory for remaining rows
keep_rows = 15088  # Number of rows to keep in training set
# ==============================================================================

# Check if source directory exists
if not source_dir.exists():
    print(f"Error: source directory does not exist - {source_dir}")
    exit(1)

# Ensure target directories A and B exist
for target_dir in [target_parent_A, target_parent_B]:
    target_dir.mkdir(exist_ok=True, parents=True)

# Statistics counters
total_csv = 0
success_A = 0  # Files successfully saved to A
success_B = 0  # Files successfully saved to B
failed_files = []

print("Starting data processing...")
print(f"Source directory: {source_dir}")
print(f"First {keep_rows} rows saved to: {target_parent_A}")
print(f"Remaining rows saved to: {target_parent_B}\n")

# Get all CSV files from source directory
csv_files = list(source_dir.glob("*.csv"))
total_csv = len(csv_files)

print(f"=== Processing directory: {source_dir.name} ===")
print(f"CSV files found: {len(csv_files)}")

for csv_file in csv_files:
    try:
        # Read CSV (handle encoding automatically)
        df = pd.read_csv(csv_file, encoding_errors='ignore')
        total_rows = len(df)

        # Split data: first keep_rows rows to A, remainder to B
        df_A = df.head(keep_rows)  # First 15088 rows
        df_B = df.iloc[keep_rows:]  # Remaining rows starting from 15089

        # Save to A
        target_A = target_parent_A / csv_file.name
        df_A.to_csv(target_A, index=False, encoding='utf-8')
        success_A += 1

        # Save to B (if there are remaining rows)
        if not df_B.empty:
            target_B = target_parent_B / csv_file.name
            df_B.to_csv(target_B, index=False, encoding='utf-8')
            success_B += 1
        else:
            print(f"  {csv_file.name} has no remaining rows (total <= {keep_rows}), skipped B")

        # Progress indicator
        current_total = success_A + success_B
        if current_total % 10 == 0:
            print(f"  Processed {current_total} files (A: {success_A}, B: {success_B})")

    except Exception as e:
        failed_files.append({
            "file": str(csv_file),
            "error": str(e)
        })
        print(f"  Processing failed: {csv_file.name} - {str(e)}")

# Output summary
print("\n" + "=" * 60)
print("Data splitting complete!")
print("=" * 60)
print(f"Source directory: {total_csv} CSV files total")
print(f"Successfully saved to folder A (first {keep_rows} rows): {success_A} files")
print(f"Successfully saved to folder B (remaining rows): {success_B} files")
print(f"Failed files: {len(failed_files)}")

if failed_files:
    print("\nFailed file list:")
    for i, fail in enumerate(failed_files, 1):
        print(f"  {i}. File: {fail['file'].split('/')[-1]}")
        print(f"     Error: {fail['error']}")

print(f"\nFolder A path: {target_parent_A}")
print(f"Folder B path: {target_parent_B}")
print("Note: All CSV files are saved directly in the target directories, without subdirectory structure.")
