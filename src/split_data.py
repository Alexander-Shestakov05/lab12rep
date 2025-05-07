import pandas as pd
from sklearn.model_selection import train_test_split

# Загрузка очищенных данных
data = pd.read_csv('data/prepare/cleaned_23.csv')

# Поиск колонки с типом 'category' — считаем её целевой переменной
categorical_cols = data.select_dtypes(['category', 'object']).columns

if len(categorical_cols) == 0:
    raise ValueError("Не найдена колонка с типом 'category' или 'object' для использования как целевая переменная")

target_col = categorical_cols[0]  # Берём первую подходящую колонку
print(f"Целевая переменная: {target_col}")

# Разделение на признаки и целевую переменную
X = data.drop(target_col, axis=1)
y = data[target_col]

# Разделение на train+val и test (80% / 20%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Разделение train+val на train и val (80% / 20%)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Сохранение всех выборок
X_train.to_csv('data/prepare/train.csv', index=False)
X_val.to_csv('data/prepare/val.csv', index=False)
X_test.to_csv('data/prepare/test.csv', index=False)
y_train.to_csv('data/prepare/y_train.csv', index=False)
y_val.to_csv('data/prepare/y_val.csv', index=False)
y_test.to_csv('data/prepare/y_test.csv', index=False)

print("✅ Данные разделены и сохранены в папке data/prepare/")
