best_params = study.best_params

#ここの490は先ほどのモデルを動かして、ベストな結果を出したハイパーパラメータにおいてealrystoppingされたn_estimatorの出力を見て決定
best_params['n_estimator'] = 490

best_params['verbose'] = -1
best_params['metric'] = 'multi_logloss'
best_params['objective'] = 'multiclass'
best_params['boosting_type'] = 'gbdt'
best_params['num_class'] = 5
model = lgb.LGBMClassifier(**best_params)
model.fit(X, y)
X_weighted = weighted_average(model.predict_proba(X))
X_test_weighted = weighted_average(model.predict_proba(X_test))

# 最適な閾値を求める
optR = OptimizedRounder()
optR.fit(X_weighted, y)
optimized_thresholds = optR.coefficients()
y_pred_adjusted = optR.predict(X_test_weighted, optimized_thresholds)
y_pred_adjusted = y_pred_adjusted.astype(int)
