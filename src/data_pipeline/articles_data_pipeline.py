import numpy as np
import pandas as pd

def load_articles_data(path):
    df = pd.read_csv(path)
    return df

def clean_articles_data_silver(df):
    df_silver = df.copy()

    # Drop rows with -1 values 
    cols_with_minus1 = [
        'product_type_no', 'graphical_appearance_no',
        'colour_group_code', 'perceived_colour_value_id',
        'perceived_colour_master_id'
    ]
    df_silver = df_silver[~(df_silver[cols_with_minus1] == -1).any(axis=1)]

    return df_silver

def run_articles_pipeline(path):
    df = load_articles_data(path)
    df = clean_articles_data_silver(df)
    return df

if __name__ == "__main__":
    path = "../../data/articles.csv"
    articles_df = run_articles_pipeline(path)