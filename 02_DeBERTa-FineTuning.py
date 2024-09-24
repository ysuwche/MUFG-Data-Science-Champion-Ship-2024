!pip install transformers
!pip install datasets

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from datasets import Dataset

# レビューとリプライを明確に分けて結合
train['text'] = '[REVIEW] ' + train['review'] + ' [REPLY] ' + train['replyContent']
test['text'] = '[REVIEW] ' + test['review'] + ' [REPLY] ' + test['replyContent']

# 2. DeBERTaのトークナイザーを用意
model_name = 'microsoft/deberta-v3-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# トークン化関数
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

# 3. 4-foldクロスバリデーションの設定
skf = StratifiedKFold(n_splits=4)

# 4. データセットをHugging Faceのデータセット形式に変換
train_dataset = Dataset.from_pandas(train)
test_dataset = Dataset.from_pandas(test)

# トークン化を適用
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# ここでlabelsとしてscore（0～4の値）を渡す
train_dataset = train_dataset.rename_column("score", "labels")

# 必要なカラムのみ残す
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# 5. モデルとトレーニング設定
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)  # 5クラス分類

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy='epoch'
)

# 6. 4-foldクロスバリデーションのループ
train_features = np.zeros((len(train), 5))  # trainデータの特徴量を保存する配列
for fold, (train_index, val_index) in enumerate(skf.split(train, train["score"])):
    print(f"Fold {fold + 1}")
    train_fold = train_dataset.select(train_index)
    val_fold = train_dataset.select(val_index)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_fold,
        eval_dataset=val_fold,
        tokenizer=tokenizer
    )

    trainer.train()

    # 各分割ごとにtrainデータから特徴量を抽出
    val_dataloader = torch.utils.data.DataLoader(val_fold, batch_size=16)
    model.eval()
    fold_features = []
    with torch.no_grad():
        for batch in val_dataloader:
            inputs = {k: v.to('cuda') for k, v in batch.items() if k != 'labels'}
            outputs = model(**inputs)
            fold_features.append(outputs.logits.cpu().numpy())

    fold_features = np.concatenate(fold_features, axis=0)
    # 各 fold の特徴量を train_features に格納
    train_features[val_index] = fold_features

# 7. テストデータの特徴量抽出
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

model.eval()
test_features = []
with torch.no_grad():
    for batch in test_dataloader:
        inputs = {k: v.to('cuda') for k, v in batch.items()}
        outputs = model(**inputs)
        test_features.append(outputs.logits.cpu().numpy())
