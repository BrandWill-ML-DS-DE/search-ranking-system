def group_data(df):
    # Required for LightGBM ranking
    return df.groupby("query_id").size().to_list()
