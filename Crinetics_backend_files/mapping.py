import pandas as pd
import uuid
import os


# =========================================================
# FUNCTION 1: Create Mapping + Anonymize Selected Columns
# =========================================================

def create_mapping_and_anonymize(df, columns_to_mask, mapping_file="id_mapping.csv"):
    """
    Reversible anonymization for selected VARCHAR columns only.
    
    Parameters:
    df (pd.DataFrame): Original dataframe
    columns_to_mask (list): Columns to anonymize
    mapping_file (str): CSV file path to store mapping
    
    Returns:
    df_anonymized (pd.DataFrame)
    mapping_df (pd.DataFrame)
    """

    df_anonymized = df.copy()

    # Load existing mapping (incremental support)
    if os.path.exists(mapping_file):
        mapping_df = pd.read_csv(mapping_file, dtype=str)
    else:
        mapping_df = pd.DataFrame(columns=["column_name", "original_value", "dummy_value"])

    new_records = []

    for col in columns_to_mask:

        # Skip if column not present in dataframe
        if col not in df_anonymized.columns:
            print(f"⚠ Column '{col}' not found in dataframe. Skipping.")
            continue

        # Ensure VARCHAR behavior
        df_anonymized[col] = df_anonymized[col].astype(str)

        # Get existing mappings for this column
        existing_col_mapping = mapping_df[mapping_df["column_name"] == col]
        existing_dict = dict(zip(existing_col_mapping["original_value"],
                                 existing_col_mapping["dummy_value"]))

        unique_values = df_anonymized[col].dropna().unique()
        col_mapping = {}

        for val in unique_values:
            if val in existing_dict:
                col_mapping[val] = existing_dict[val]
            else:
                dummy_id = str(uuid.uuid4())
                col_mapping[val] = dummy_id

                new_records.append({
                    "column_name": col,
                    "original_value": val,
                    "dummy_value": dummy_id
                })

        # Replace ONLY this column
        df_anonymized[col] = df_anonymized[col].map(col_mapping)

    # Append new mappings if needed
    if new_records:
        new_mapping_df = pd.DataFrame(new_records)
        mapping_df = pd.concat([mapping_df, new_mapping_df], ignore_index=True)

    # Always save
    mapping_df.to_csv(mapping_file, index=False)

    return df_anonymized, mapping_df


# =========================================================
# FUNCTION 2: Reverse Mapping
# =========================================================

def reverse_mapping(df_masked, mapping_file="id_mapping.csv"):
    """
    Restores original values using mapping CSV.
    """

    if not os.path.exists(mapping_file):
        raise FileNotFoundError("Mapping file not found.")

    mapping_df = pd.read_csv(mapping_file, dtype=str)
    df_original = df_masked.copy()

    for col in mapping_df["column_name"].unique():

        if col in df_original.columns:

            col_map = mapping_df[mapping_df["column_name"] == col]
            reverse_dict = dict(zip(col_map["dummy_value"],
                                    col_map["original_value"]))

            df_original[col] = df_original[col].map(reverse_dict)

    return df_original



df = pd.read_csv("Geron_DDD_Data_Gen_AI.csv", low_memory=False)

columns_to_mask = ["campus_id", "parent_id"]

df_masked, mapping_df = create_mapping_and_anonymize(df, columns_to_mask)


df_masked.to_csv("rytelo_DDD_masked.csv", index=False)
print(df_masked)

print("\n📁 Mapping Table:")
print(mapping_df)

#     # Reverse
# df_restored = reverse_mapping(df_masked)

# print("\n🔄 Restored Data:")
# print(df_restored)