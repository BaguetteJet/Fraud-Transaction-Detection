import torch
import pandas as pd
from torch_geometric.data import HeteroData

def build_mappings(df: pd.DataFrame):
    customer_ids = df['customer'].unique().tolist()
    merchant_ids = df['merchant'].unique().tolist()
    
    # Graph doesnt support strings for connecting nodes to map to indices
    customer_map = {cid: idx for idx, cid in enumerate(customer_ids)}
    merchant_map = {mid: idx for idx, mid in enumerate(merchant_ids)}

    return customer_map, merchant_map

def customer_features(df: pd.DataFrame, customer_map):
    # Creates customer nodes (index, age, gender)
    ages = [col for col in df if col.startswith('age_')]
    genders = [col for col in df if col.startswith('gender_')]
    
    customer_df = df[["customer"] + ages + genders].drop_duplicates(subset="customer").copy()
    
    customer_df["node_idx"] = customer_df["customer"].map(customer_map)
    customer_df = customer_df.sort_values("node_idx")

    x = customer_df[ages + genders].to_numpy(dtype="float32")
    return torch.tensor(x, dtype=torch.float32)

def merchant_features(df: pd.DataFrame, merchant_map):
    # Creates merchant nodes of (index, category)
    categories = [col for col in df if col.startswith('category_')]

    merchant_df = df[["merchant"] + categories].drop_duplicates(subset="merchant").copy()
    
    merchant_df["node_idx"] = merchant_df["merchant"].map(merchant_map)
    merchant_df = merchant_df.sort_values("node_idx")

    x = merchant_df[categories].to_numpy(dtype="float32")
    return torch.tensor(x, dtype=torch.float32)

def transaction_features(df: pd.DataFrame, feature_cols):
    # Creates transaction nodes (index, features)
    txn_df = df[feature_cols].copy()
    txn_df["node_idx"] = df.index
    txn_df = txn_df.sort_values("node_idx")

    x = txn_df[feature_cols].to_numpy(dtype="float32")
    return torch.tensor(x, dtype=torch.float32)

def build_graph(df: pd.DataFrame, feature_cols):
    customer_map, merchant_map = build_mappings(df)

    data = HeteroData()

    # Creates the nodes
    data["customer"].x = customer_features(df, customer_map)
    data["merchant"].x = merchant_features(df, merchant_map)
    data["transaction"].x = transaction_features(df, feature_cols)

    # Gets indexes to match each txn to corresponding nodes
    customer_idx = [customer_map[c] for c in df["customer"]]
    merchant_idx = [merchant_map[m] for m in df["merchant"]]
    txn_idx = df.index.tolist()

    # Each node is connected double edged
    data["customer", "makes", "transaction"].edge_index = torch.tensor(
        [customer_idx, txn_idx], dtype=torch.long
    )

    data["transaction", "made_by", "customer"].edge_index = torch.tensor(
        [txn_idx, customer_idx], dtype=torch.long
    )

    data["transaction", "at", "merchant"].edge_index = torch.tensor(
        [txn_idx, merchant_idx], dtype=torch.long
    )

    data["merchant", "receives", "transaction"].edge_index = torch.tensor(
        [merchant_idx, txn_idx], dtype=torch.long
    )

    # Creates lables for nodes to verify if fraud/non-fraud
    if "fraud" in df.columns:
        data["transaction"].y = torch.tensor(df["fraud"].to_numpy(), dtype=torch.float32)

    return data
