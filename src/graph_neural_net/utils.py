# Weights for training from importance and premutation of other models
def get_weight(col):
    if col in [
        "customer_log_amount_ratio",
        "merchant_transaction_count",
        "merchant_fraud_rate",
        "customer_amount"
    ]:
        return 2.5
    
    elif col in [
        "customer_transaction_count",
        "merchant_std_amount",
        "global_z_score",
        "merchant_amount_zscore"
    ]:
        return 1.5
    
    else:
        return 1.0
    
