from inference_transform import InferencePreprocessor

DB_FILE = "../../data/database.db"
TABLE_NAME = "records"
PREPROCESSOR_FILE = "data\processed\preprocessor.pkl"
COLUMNS = ["step","customer","age","gender","zipcodeori","merchant","zipmerchant","category","customer_amount","fraud"]

def supervised(transformer, sample, columns):
    frame = transformer.transform(sample, columns)

    cols_to_drop = ["step", "customer", "merchant"] + [col for col in frame.columns if col.startswith("category")]
    frame.drop(columns=[cols_to_drop],inplace = True)
    return frame

def unsupervised(transformer, sample, columns):
    frame = transformer.transform(sample, columns)

    frame.drop(columns=["step","fraud"], inplace=True)
    return frame

if __name__ == "__main__":
    sample = [0,'C352968107',"'2'","'M'",'28007','M348934600','28007','es_transportation',39.68,0]
    transformer = InferencePreprocessor(DB_FILE, TABLE_NAME, PREPROCESSOR_FILE)

    print(supervised(transformer, sample, COLUMNS))
