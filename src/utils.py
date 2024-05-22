import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm

def evaluate_Challenge_B(df_true, df_score, method='pearson'):
    score_list = []
    df_true = df_true.values
    df_score = df_score.values
    for i in range(df_score.shape[1]):
        y_pred = df_score[:, i]
        y_true = df_true[:, i]
        score_list.append(np.corrcoef(y_true, y_pred)[0, 1])
    return score_list


def analyze_predictions(pred, true, plot=False, transpose=False, bins=100):
    if transpose:
        true = true.T
        pred = pred.T
    score_list = evaluate_Challenge_B(true, pred)
    score_list = [score if not np.isnan(score) else 0 for score in score_list]
    mean = np.mean(score_list)
    print("Mean corr:", mean)
    mse = mean_squared_error(true, pred)
    print("Mean mse:", mse)
    print("Prop > 0.1:", np.mean([1 if score > 0.1 else 0 for score in score_list]))
    print("Prop > 0.2:", np.mean([1 if score > 0.2 else 0 for score in score_list]))
    print("Prop > 0.3:", np.mean([1 if score > 0.3 else 0 for score in score_list]))
    if plot:
        plt.figure(figsize=(10, 5), dpi=200)
        plt.hist(score_list, bins=bins)
        plt.axvline(x=mean, color='red')
        plt.xlim(-1, 1)
        plt.show()

def zscore(df):
    return pd.DataFrame(
        StandardScaler().fit_transform(df),
        index=df.index,
        columns=df.columns,
    )