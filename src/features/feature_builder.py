def build_features(query, docs):
    feats = []
    for doc, score in docs:
        feats.append([
            len(query),
            len(doc),
            score
        ])
    return feats
