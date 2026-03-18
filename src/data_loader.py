import pandas as pd

def load_sample_data():
    # Simulated ranking dataset (replace with MS MARCO later)
    data = {
        "query_id": [1, 1, 1, 2, 2],
        "doc_id": [101, 102, 103, 201, 202],
        "feature1": [0.2, 0.1, 0.4, 0.3, 0.8],
        "feature2": [0.5, 0.3, 0.2, 0.7, 0.6],
        "label": [1, 0, 2, 0, 1],
    }
    return pd.DataFrame(data)
