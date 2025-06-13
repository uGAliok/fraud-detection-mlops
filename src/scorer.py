import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import logging
from lightgbm import LGBMClassifier
import joblib

logger = logging.getLogger(__name__)
logger.info('Importing pretrained model...')

model = LGBMClassifier()
model = joblib.load('./models/mylightgbm_model.pkl')

model_th = 0.686
logger.info('Pretrained model imported successfully...')

def make_pred(dt, path_to_file):
    scores = model.predict_proba(dt)[:, 1]
    ids = pd.read_csv(path_to_file).index
    assert len(scores) == len(ids), (
        f"Preds len={len(scores)}   IDs len={len(ids)} – "
        "индексы разошлись, проверьте preprocessing."
    )

    submission = pd.DataFrame({
        'index':  pd.read_csv(path_to_file).index,
        'prediction': (scores > model_th) * 1
    })
    logger.info('Prediction complete for file: %s', path_to_file)

    return submission, scores, model

def save_feature_importance(model, output_path, timestamp, top_k=5):
    names = model.booster_.feature_name()
    imps = model.booster_.feature_importance(importance_type="gain")
    top_indices = np.argsort(imps)[::-1][:top_k]
    top_features = {names[i]: float(imps[i]) for i in top_indices}
    filename = f"feature_importances_{timestamp}.json"
    with open(os.path.join(output_path, filename), "w") as f:
        json.dump(top_features, f, indent=2)

def save_score_distribution(scores, output_path, timestamp):
    plt.figure()
    plt.hist(scores, bins=50, density=True)
    plt.title("Distribution of predicted probabilities")
    plt.xlabel("Probability")
    plt.ylabel("Density")
    filename = f"scores_density_{timestamp}.png"
    plt.savefig(os.path.join(output_path, filename))
    plt.close()
