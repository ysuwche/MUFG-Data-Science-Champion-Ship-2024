# 加重平均を計算する関数
def weighted_average(pred_proba):
    class_labels = np.arange(pred_proba.shape[1])  # クラスラベル（0, 1, 2, 3, 4）
    weighted_avg = np.sum(pred_proba * class_labels, axis=1)  # 加重平均を計算
    return weighted_avg
