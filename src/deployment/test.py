from inference_transform import InferencePreprocessor
import pandas as pd

DB_FILE = "data/database.db"
TABLE_NAME = "records"
PREPROCESSOR_FILE = "data/processed/preprocessor.pkl"
COLUMNS = ["step","customer","age","gender","zipcodeori","merchant","zipmerchant","category","customer_amount"]

def supervised(transformer, df):
    frame = df

    cols_to_drop = ["customer", "merchant"] + [col for col in frame.columns if col.startswith("category")]
    frame.drop(columns=cols_to_drop,inplace = True)
    return frame

def unsupervised(transformer, df):
    frame = df

    frame.drop(columns=["fraud"], inplace=True)
    return frame

if __name__ == "__main__":
    sample = [0,'C352968107',"'2'","'M'",'28007','M348934600','28007','es_transportation',39.68]
    transformer = InferencePreprocessor(DB_FILE, TABLE_NAME, PREPROCESSOR_FILE)

    print(supervised(transformer, pd.DataFrame([sample], columns=COLUMNS)))
    print(unsupervised(transformer, pd.DataFrame([sample], columns=COLUMNS)))
