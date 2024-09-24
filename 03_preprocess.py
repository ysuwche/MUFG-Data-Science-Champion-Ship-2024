import re
import numpy as np

###TimeToReplyの前処理###

train['timeToReply'] = train['timeToReply'].astype(str)
test['timeToReply'] = test['timeToReply'].astype(str)

# 正規表現パターン

# 変換関数
def convert_to_minutes(time_str):
    # 正規表現で日、時間、分、秒を抽出
    match = re.match(r'(\d+)\s+days\s+(\d{2}):(\d{2}):(\d{2})', time_str)
    if match:
        days, hours, minutes, seconds = map(int, match.groups())
        # 分に変換
        total_minutes = days * 24 * 60 + hours * 60 + minutes + seconds / 60
        return total_minutes
    else:
        return float('nan')

train['timeToReply'] = train['timeToReply'].apply(convert_to_minutes)
test['timeToReply'] = test['timeToReply'].apply(convert_to_minutes)

#trainとテストを結合し、timeToReplyの中央値で欠損値を補完し、再び分割
df2 = pd.concat([train, test])
df2['timeToReply'] = df2['timeToReply'].fillna(df2['timeToReply'].median())
train['timeToReply'] = df2['timeToReply'][:len(train)]
test['timeToReply'] = df2['timeToReply'][len(train):]


###reviewCreatedVersionの前処理###
from sklearn.preprocessing import LabelEncoder

train['reviewCreatedVersion'] = train['reviewCreatedVersion'].astype(str)
test['reviewCreatedVersion'] = test['reviewCreatedVersion'].astype(str)

label_encoder = LabelEncoder()


####前処理した数値データとテキストデータから抽出した特徴量を結合して最終的な特徴量を定義####
tr_numrical = train[['thumbsUpCount', 'reviewCreatedVersion', 'timeToReply']].values
te_numrical = test[['thumbsUpCount', 'reviewCreatedVersion', 'timeToReply']].values

X = np.hstack((train_features, tr_numrical))
X_test = np.hstack((test_features, te_numrical))
y = train['score'].values
