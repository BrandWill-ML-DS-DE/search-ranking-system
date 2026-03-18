import lightgbm as lgb

class Ranker:
    def __init__(self, path):
        self.model = lgb.Booster(model_file=path)

    def predict(self, X):
        return self.model.predict(X)
