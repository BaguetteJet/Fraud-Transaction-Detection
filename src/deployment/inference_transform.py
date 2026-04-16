import pandas as pd
import numpy as np
import sqlite3
import joblib

"""
    Rebuilds the features from RollingFeatureExtract.ipynb for one singular input

    We make a assumption in the dataset that the latest index in the data is the latest input
    even if time(step) is the same
"""
class InferencePreprocessor:
    def __init__(self, db_file, table_name, preprocessor_path):
        self.db_file = db_file
        self.table_name = table_name
        self.preprocessor = joblib.load(preprocessor_path)

        self.input_row = None
        self.cust_row = None
        self.merch_row = None
        self.cust_merch_row = None
        self.cust_cat_row = None
        self.last_row = None

    def get_last_entry(self, filters=None):
        filters = filters or {}
        conn = sqlite3.connect(self.db_file)

        # Joined clause to filter rows together
        where_clause = " AND ".join([f"{col} = ?" for col in filters.keys()]) if filters else "1=1"
        values = tuple(filters.values())

        query = f"""
            SELECT *
            FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY rowid DESC
            LIMIT 1
        """

        df = pd.read_sql_query(query, conn, params=values)
        conn.close()
        return df

    def load_reference_rows(self):
        # Used to build all customer features
        self.cust_row = self.get_last_entry({"Customer": self.input_row.loc[0, "customer"]})
        # Used to build all merchant features
        self.merch_row = self.get_last_entry({"Merchant": self.input_row.loc[0, "merchant"]})
        # Used to build linked customer-merchant features (eg. last time a customer shopped at this merchant)
        self.cust_merch_row = self.get_last_entry({
            "Customer": self.input_row.loc[0, "customer"],
            "Merchant": self.input_row.loc[0, "merchant"]
        })
        # Used to build linked customer-category features (eg. last time a customer shopped this category)
        self.cust_cat_row = self.get_last_entry({
            "Customer": self.input_row.loc[0, "customer"],
            f"category_{self.input_row.loc[0, 'category']}": 1
        })
        # Get very last entry to build global features
        self.last_row = self.get_last_entry({})

    def hot_encode(self):
        # Rebuilds the one hot encoded columns within the database
        for col in self.cust_row.columns:
            if col.startswith("category_") or col.startswith("age_") or col.startswith("gender_"):
                self.input_row[col] = 0

        cat_col = f"category_{self.input_row.loc[0, 'category']}"
        age_col = f"age_{self.input_row.loc[0, 'age']}"
        gender_col = f"gender_{self.input_row.loc[0, 'gender']}"

        if cat_col in self.input_row.columns:
            self.input_row.loc[0, cat_col] = 1
        if age_col in self.input_row.columns:
            self.input_row.loc[0, age_col] = 1
        if gender_col in self.input_row.columns:
            self.input_row.loc[0, gender_col] = 1

        self.input_row = self.input_row.reindex(columns=self.cust_row.columns, fill_value=0)

    def customer_features(self):
        """
        Rebuilds the customer features for the incoming row
        """
        self.input_row["customer_prev_step"] = self.cust_row["step"]
        self.input_row["customer_time_since_last_transaction"] = self.input_row["step"] - self.input_row["customer_prev_step"]

        self.input_row["customer_transaction_count"] = self.cust_row["customer_transaction_count"] + 1

        self.input_row["customer_time_since_last_transaction"] = self.input_row["customer_time_since_last_transaction"].fillna(-1)
        self.input_row["customer_prev_step"] = self.input_row["customer_prev_step"].fillna(-1)

        self.input_row["customer_amount_sum"] = self.cust_row["customer_amount_sum"] + self.input_row["customer_amount"]
        self.input_row["customer_amount_sum"] = self.input_row["customer_amount_sum"].fillna(0)

        self.input_row["customer_avg_amount"] = self.input_row["customer_amount_sum"] / self.input_row["customer_transaction_count"]
        self.input_row["customer_avg_amount"] = self.input_row["customer_avg_amount"].fillna(self.input_row["customer_amount"])

        self.input_row["customer_amount_sq_sum"] = self.cust_row["customer_amount_sq_sum"] + (self.input_row["customer_amount"] ** 2)
        self.input_row["customer_amount_sq_sum"] = self.input_row["customer_amount_sq_sum"].fillna(0)

        self.input_row["customer_M2_amount"] = (
            self.input_row["customer_amount_sq_sum"]
            - (self.input_row["customer_amount_sum"] ** 2) / self.input_row["customer_transaction_count"].replace(0, np.nan)
        )
        self.input_row["customer_M2_amount"] = self.input_row["customer_M2_amount"].fillna(0).clip(lower=0)

        self.input_row["customer_std_amount"] = np.sqrt(
            self.input_row["customer_M2_amount"] / self.cust_row["customer_transaction_count"].replace(0, np.nan)
        ).fillna(0)

        self.input_row["customer_avg_amount_ratio"] = self.input_row["customer_amount"] / self.input_row["customer_avg_amount"]
        self.input_row["customer_log_amount_ratio"] = np.log1p(self.input_row["customer_avg_amount_ratio"])

        self.input_row["customer_zscore"] = (
            self.input_row["customer_amount"] - self.input_row["customer_avg_amount"]
        ) / self.input_row["customer_std_amount"]
        self.input_row["customer_zscore"] = self.input_row["customer_zscore"].replace([np.inf, -np.inf], 0).fillna(0)

        self.input_row["customer_merchant_count"] = self.cust_merch_row["customer_merchant_count"] + 1
        self.input_row["customer_category_count"] = self.cust_cat_row["customer_category_count"] + 1

        self.input_row["customer_prev_merchant_step"] = self.cust_merch_row["step"]
        self.input_row["customer_time_since_last_merchant_transaction"] = (
            self.input_row["step"] - self.input_row["customer_prev_merchant_step"]
        )
        self.input_row["customer_time_since_last_merchant_transaction"] = (
            self.input_row["customer_time_since_last_merchant_transaction"].fillna(-1)
        )
        self.input_row["customer_prev_merchant_step"] = self.input_row["customer_prev_merchant_step"].fillna(-1)

    def merchant_features(self):
        """
        Rebuilds the merchant features for the incoming row
        """
        self.input_row["merchant_transaction_count"] = self.merch_row["merchant_transaction_count"] + 1
        self.input_row["merchant_amount_sum"] = self.merch_row["merchant_amount_sum"] + self.input_row["customer_amount"]
        self.input_row["merchant_avg_amount"] = self.input_row["merchant_amount_sum"] / self.input_row["merchant_transaction_count"]

        self.input_row["merchant_amount_sq_sum"] = self.merch_row["merchant_amount_sq_sum"] + (self.input_row["customer_amount"] ** 2)

        self.input_row["merchant_M2_amount"] = (
            self.input_row["merchant_amount_sq_sum"]
            - (self.input_row["merchant_amount_sum"] ** 2) / self.input_row["merchant_transaction_count"].replace(0, np.nan)
        )
        self.input_row["merchant_M2_amount"] = self.input_row["merchant_M2_amount"].fillna(0).clip(lower=0)

        self.input_row["merchant_std_amount"] = np.sqrt(
            self.input_row["merchant_M2_amount"] / self.merch_row["merchant_transaction_count"].replace(0, np.nan)
        ).fillna(0)

        self.input_row["merchant_avg_amount_ratio"] = self.input_row["customer_amount"] / self.input_row["merchant_avg_amount"]
        self.input_row["merchant_log_amount_ratio"] = np.log1p(self.input_row["merchant_avg_amount_ratio"])

        self.input_row["merchant_amount_zscore"] = (
            self.input_row["customer_amount"] - self.input_row["merchant_avg_amount"]
        ) / self.input_row["merchant_std_amount"]
        self.input_row["merchant_amount_zscore"] = self.input_row["merchant_amount_zscore"].replace([np.inf, -np.inf], 0).fillna(0)

        self.input_row["merchant_fraud_count"] = self.merch_row["merchant_fraud_count"] + self.input_row["fraud"]
        self.input_row["merchant_fraud_rate"] = (
            self.input_row["merchant_fraud_count"] / self.input_row["merchant_transaction_count"]
        ).fillna(0)

        self.input_row["merchant_prev_step"] = self.merch_row["step"]
        self.input_row["merchant_time_since_last_transaction"] = self.input_row["step"] - self.input_row["merchant_prev_step"]
        self.input_row["merchant_time_since_last_transaction"] = self.input_row["merchant_time_since_last_transaction"].fillna(-1)
        self.input_row["merchant_prev_step"] = self.input_row["merchant_prev_step"].fillna(-1)

    def global_features(self):
        """
        Rebuilds the global features for the incoming row
        """
        self.input_row["global_transaction_count"] = self.last_row["global_transaction_count"] + 1

        self.input_row["global_amount_sum"] = self.last_row["global_amount_sum"] + self.input_row["customer_amount"]
        self.input_row["global_amount_sum"] = self.input_row["global_amount_sum"].fillna(0)

        self.input_row["global_avg_amount"] = self.input_row["global_amount_sum"] / self.input_row["global_transaction_count"]
        self.input_row["global_avg_amount"] = self.input_row["global_avg_amount"].fillna(self.input_row["customer_amount"])

        self.input_row["global_amount_ratio"] = self.input_row["customer_amount"] / self.input_row["global_avg_amount"]
        self.input_row["global_log_amount_ratio"] = np.log1p(self.input_row["global_amount_ratio"])

        self.input_row["global_amount_sq_sum"] = self.last_row["global_amount_sq_sum"] + (self.input_row["customer_amount"] ** 2)
        self.input_row["global_amount_sq_sum"] = self.input_row["global_amount_sq_sum"].fillna(0)

        self.input_row["global_M2_amount"] = (
            self.input_row["global_amount_sq_sum"]
            - (self.input_row["global_amount_sum"] ** 2) / self.input_row["global_transaction_count"].replace(0, np.nan)
        )
        self.input_row["global_M2_amount"] = self.input_row["global_M2_amount"].fillna(0).clip(lower=0)

        self.input_row["global_std_amount"] = np.sqrt(
            self.input_row["global_M2_amount"] / self.last_row["global_transaction_count"].replace(0, np.nan)
        ).fillna(0)

        self.input_row["global_z_score"] = (
            self.input_row["customer_amount"] - self.input_row["global_avg_amount"]
        ) / self.input_row["global_std_amount"]
        self.input_row["global_z_score"] = self.input_row["global_z_score"].replace([np.inf, -np.inf], 0).fillna(0)

        self.input_row["global_median_amount"] = self.last_row["global_median_amount"]
        self.input_row["global_median_amount"] = self.input_row["global_median_amount"].fillna(self.input_row["customer_amount"])

        self.input_row["global_median_amount_ratio"] = self.input_row["customer_amount"] / self.input_row["global_median_amount"]
        self.input_row["global_log_median_amount_ratio"] = np.log1p(self.input_row["global_median_amount_ratio"])

    def drop_columns(self):
        # Time based columns as they are arbitrary as time grows and time_since_last_transactions captures better meaning
        self.input_row.drop(columns=["customer_prev_step", "customer_prev_merchant_step", "merchant_prev_step"], inplace=True)

        # Sums and averages are captured in the ratios and z-scores, so can drop them to reduce dimensionality
        self.input_row.drop(columns=["customer_amount_sum", "customer_avg_amount", "merchant_amount_sum", "merchant_avg_amount", "global_amount_sum"], inplace=True)

        # Was only used to dervive fraud rate, can drop it now
        self.input_row.drop(columns=["merchant_fraud_count"], inplace=True)

        # Dropping from data analyisis, median captures more of the more common transaction amounts, compared to the mean
        self.input_row.drop(columns=["global_avg_amount", "global_amount_ratio", "global_log_amount_ratio", "global_median_amount"], inplace=True)

        # From Permutation graphs log values tend to do better
        self.input_row.drop(columns=["customer_avg_amount_ratio", "merchant_avg_amount_ratio", "global_median_amount_ratio"], inplace=True)

        # These columns are only used to build std during inference
        self.input_row.drop(columns=["customer_amount_sq_sum", "customer_M2_amount", "merchant_amount_sq_sum","merchant_M2_amount", "global_amount_sq_sum", "global_M2_amount"], inplace=True)

        # Index isnt perserved in the DB
        self.input_row.drop(columns=["global_transaction_count"], inplace=True)

    def standardize(self):
        scaled = self.preprocessor.transform(self.input_row)
        return pd.DataFrame(scaled, columns=self.preprocessor.get_feature_names_out())

    # def transform(self, sample, columns):
    #     self.input_row = pd.DataFrame([sample], columns=columns)

    #     self.load_reference_rows()
    #     self.hot_encode()
    #     self.customer_features()
    #     self.merchant_features()
    #     self.global_features()
    #     self.drop_columns()

    #     return self.standardize()
    
    def transform(self, df):
        self.input_row = df.copy()

        self.load_reference_rows()
        self.hot_encode()
        self.customer_features()
        self.merchant_features()
        self.global_features()
        self.drop_columns()

        self.input_row = self.input_row.apply(pd.to_numeric, errors="coerce")

        return self.standardize()