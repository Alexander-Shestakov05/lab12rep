import pandas as pd

# Загрузка CSV без заголовков
data = pd.read_csv('data/raw/23.csv', header=None)

# Автоматическая генерация имён колонок: col_0, col_1, ...
data.columns = [f'col_{i}' for i in range(data.shape[1])]

# Очистка: удаление строк с пропусками
data = data.dropna()

# Предположим, колонка col_1 — это 'diagnosis'
data['col_1'] = data['col_1'].astype('category')

# Сохранение
data.to_csv('data/prepare/cleaned_23.csv', index=False)
print("Данные очищены и сохранены в data/prepare/cleaned_23.csv")
