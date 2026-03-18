import lightgbm as lgb
from src.data_loader import load_sample_data
from src.preprocess import group_data
import os

def train():
    df = load_sample_data()

    X = df[["feature1", "feature2"]]
    y = df["label"]
    group = group_data(df)

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=100
    )

    model.fit(X, y, group=group)

    os.makedirs("models", exist_ok=True)
    model.booster_.save_model("models/lgbm_ranker.txt")

    print("Model trained and saved")

if __name__ == "__main__":
    train()
