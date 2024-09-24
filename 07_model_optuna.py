from sklearn.model_selection import KFold
import lightgbm as lgb
import optuna
import numpy as np
from sklearn.metrics import mean_squared_error

kf = KFold(n_splits=4, shuffle=True, random_state=72)

def objective(trial):
        params = {
            'boosting_type': 'gbdt',
            'objective':'multiclass',
            'num_class': 5,
            'metric': 'multi_logloss',
            'learning_rate': 0.01,
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 5,100),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
            'verbose': -1,
            'n_estimators':10000
        }

        qwk_scores = []
        estimator_list = []

        for train_idx, valid_idx in kf.split(X):
          X_train_fold, X_valid_fold = X[train_idx], X[valid_idx]
          y_train_fold, y_valid_fold = y[train_idx], y[valid_idx]

          model = lgb.LGBMClassifier(**params)
          model.fit(X_train_fold, y_train_fold, eval_set=[(X_valid_fold, y_valid_fold)], callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

          estimator_list.append(model.best_iteration_)

          X_weighted = weighted_average(model.predict_proba(X_train_fold))

          y_pred_valid = model.predict_proba(X_valid_fold)
          y_pred_valid_weight = weighted_average(y_pred_valid)

          # 最適な閾値を求める
          optR = OptimizedRounder()
          optR.fit(X_weighted, y_train_fold)
          optimized_thresholds = optR.coefficients()
          y_val_pred = optR.predict(y_pred_valid_weight, optimized_thresholds)

          qwk_score = qwk(y_valid_fold, y_val_pred)
          qwk_scores.append(qwk_score)

          # 平均を取って最適なn_estimatorsを決定
          optimal_n_estimators = int(np.mean(estimator_list))
          print(f'Optimal n_estimators: {optimal_n_estimators}')

        return np.mean(qwk_scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
