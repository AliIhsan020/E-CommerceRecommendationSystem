import numpy as np
import pandas as pd

def load_articles_data(path):
    df = pd.read_csv(path)
    return df

def clean_articles_data_silver(df):
    df_silver = df.copy()

    # Drop rows with -1 values (encoded missing values, _name equivalents are also Unknown)
    cols_with_minus1 = [
        'product_type_no', 'graphical_appearance_no',
        'colour_group_code', 'perceived_colour_value_id',
        'perceived_colour_master_id'
    ]
    df_silver = df_silver[~(df_silver[cols_with_minus1] == -1).any(axis=1)]

    return df_silver

def clean_articles_data_gold(df_silver):
    df_gold = df_silver.copy()

    # Drop redundant columns
    cols_to_drop = [
        'product_type_name', 'graphical_appearance_name',
        'colour_group_name', 'perceived_colour_value_name',
        'perceived_colour_master_name', 'department_name',
        'index_name', 'index_group_name', 'section_name',
        'garment_group_name', 'prod_name', 'detail_desc'
    ]
    df_gold = df_gold.drop(columns=cols_to_drop)

    # Convert categorical columns to category type
    cat_cols = [
        'product_type_no', 'graphical_appearance_no',
        'colour_group_code', 'perceived_colour_value_id',
        'perceived_colour_master_id', 'department_no',
        'index_group_no', 'section_no', 'garment_group_no',
        'index_code', 'product_group_name'
    ]
    df_gold[cat_cols] = df_gold[cat_cols].astype('category')

    return df_gold

def run_articles_pipeline(path, silver_path, gold_path):
    df = load_articles_data(path)

    df_silver = clean_articles_data_silver(df)
    df_silver.to_parquet(silver_path, index=False)
    print("Silver saved:", df_silver.shape)

    df_gold = clean_articles_data_gold(df_silver)
    df_gold.to_parquet(gold_path, index=False)
    print("Gold saved:", df_gold.shape)

if __name__ == "__main__":
    run_articles_pipeline(
        path="data/bronze/articles.csv",
        silver_path="data/silver/articles.parquet",
        gold_path="data/gold/articles.parquet"
    )