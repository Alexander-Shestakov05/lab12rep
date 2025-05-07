import pandas as pd
from sklearn.preprocessing import StandardScaler

# Загрузка данных
X_train = pd.read_csv('data/prepare/train.csv')
X_val = pd.read_csv('data/prepare/val.csv')

# Находим все числовые столбцы
numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns

# Нормализация всех числовых столбцов
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])

# Сохранение нормализованных данных
X_train.to_csv('data/prepare/train_normalized.csv', index=False)
X_val.to_csv('data/prepare/val_normalized.csv', index=False)

print("Числовые столбцы нормализованы и сохранены в data/prepare/")
