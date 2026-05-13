import pandas as pd


def load_articles_data(path):
    df = pd.read_csv(path)
    return df


def clean_articles_data_silver(df):
    df_silver = df.copy()

    # Text columns: missing values should not break feature engineering
    text_cols = ['prod_name', 'detail_desc']
    df_silver[text_cols] = df_silver[text_cols].fillna('').astype(str)

    return df_silver


def clean_articles_data_gold(df_silver):
    df_gold = df_silver.copy()

    # Keep only columns that will be useful and interpretable
    cols_to_keep = [
        'article_id',
        'product_code',
        'prod_name',
        'detail_desc',
        'product_type_name',
        'product_group_name',
        'graphical_appearance_name',
        'colour_group_name',
        'perceived_colour_value_name',
        'perceived_colour_master_name',
        'department_name',
        'index_name',
        'index_group_name',
        'section_name',
        'garment_group_name'
    ]

    df_gold = df_gold[cols_to_keep].copy()

    # Fill missing categorical values
    categorical_cols = [
        'product_type_name',
        'product_group_name',
        'graphical_appearance_name',
        'colour_group_name',
        'perceived_colour_value_name',
        'perceived_colour_master_name',
        'department_name',
        'index_name',
        'index_group_name',
        'section_name',
        'garment_group_name'
    ]

    df_gold[categorical_cols] = (
        df_gold[categorical_cols]
        .fillna('Unknown')
        .astype('category')
    )

    # Ensure text columns are clean
    text_cols = ['prod_name', 'detail_desc']
    df_gold[text_cols] = df_gold[text_cols].fillna('').astype(str)

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