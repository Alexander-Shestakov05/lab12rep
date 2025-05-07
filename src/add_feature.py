import pandas as pd

# Загрузка данных
X_train = pd.read_csv('data/prepare/train_normalized.csv')
X_val = pd.read_csv('data/prepare/val_normalized.csv')

# Находим числовые столбцы
numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns

# Добавление нового признака для каждого числового столбца (возведение в квадрат)
for col in numeric_cols:
    X_train[f'{col}_squared'] = X_train[col] ** 2
    X_val[f'{col}_squared'] = X_val[col] ** 2

# Сохранение обновленных данных
X_train.to_csv('data/prepare/train_final.csv', index=False)
X_val.to_csv('data/prepare/val_final.csv', index=False)

print("Новые признаки добавлены и сохранены в data/prepare/")
