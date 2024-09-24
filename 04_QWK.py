from sklearn.metrics import cohen_kappa_score

def qwk(y_true, y_pred):
    """
    Quadratic Weighted Kappa (QWK) を計算する関数。
    """
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')
