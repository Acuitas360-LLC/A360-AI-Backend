import pandas as pd
import os


class FileMasker:
    def __init__(self, feature_config, mapping_csv="mask_mapping.csv"):
        self.feature_config = feature_config
        self.mapping_csv = mapping_csv
        self.mapping_df = self._load_mapping()

    def _load_mapping(self):
        if os.path.exists(self.mapping_csv):
            return pd.read_csv(self.mapping_csv)
        return pd.DataFrame(columns=["column_name", "original_value", "masked_value"])

    def _get_next_id(self, col):
        col_data = self.mapping_df[self.mapping_df["column_name"] == col]

        if col_data.empty:
            return 1

        ids = col_data["masked_value"].str.extract(r'(\d+)$')[0].dropna().astype(int)

        if ids.empty:
            return 1

        return ids.max() + 1

    def process_file(self, input_path, output_path=None):
        df = pd.read_csv(input_path)

        if output_path is None:
            base = os.path.splitext(input_path)[0]
            output_path = f"{base}_masked.xlsx"

        df_masked = df.copy()
        new_rows = []

        for col, pattern in self.feature_config.items():

            if col not in df.columns:
                continue

            # Fill missing values
            df[col] = df[col].fillna("UNKNOWN").astype(str)

            # Existing mappings for this column
            existing = self.mapping_df[self.mapping_df["column_name"] == col]
            value_map = dict(zip(existing["original_value"], existing["masked_value"]))

            # 🔥 FIX: Maintain local running ID
            current_max_id = self._get_next_id(col)

            for val in df[col].unique():

                if val not in value_map:
                    masked_val = pattern.replace("{id}", str(current_max_id))

                    value_map[val] = masked_val

                    new_rows.append({
                        "column_name": col,
                        "original_value": val,
                        "masked_value": masked_val
                    })

                    current_max_id += 1  # ✅ critical fix

            # Apply mapping
            df_masked[col] = df[col].map(value_map)

        # Save updated mapping
        if new_rows:
            self.mapping_df = pd.concat(
                [self.mapping_df, pd.DataFrame(new_rows)],
                ignore_index=True
            )
            self.mapping_df.to_csv(self.mapping_csv, index=False)

        # Save masked Excel
        #df_masked.to_excel(output_path, index=False)

        print(f"✅ Masked file saved: {output_path}")
        print(f"✅ Mapping file updated: {self.mapping_csv}")


# =========================
# 🔧 USAGE
# =========================

if __name__ == "__main__":

    # feature_config = {
    #     "campus_region_id": "campus_region_id_{id}",
    #     "campus_region": "campus_region_{id}",
    #     "campus_territory_id":"campus_territory_id_{id}",
    #     "campus_territory":"campus_territory_{id}",
    #     "campus_id":"campus_id_{id}",
    #     "campus_account_name":"campus_account_name_{id}",
    #     "parent_id":"parent_id_{id}",
    #     "parent_account_name":"parent_account_name_{id}"
    # }

    feature_config = {
    
        "parent_name":"parent_name_{id}"
    }

    masker = FileMasker(feature_config)

    masker.process_file("data_867_final.csv")  # 🔁 replace with your file