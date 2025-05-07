import pandas as pd

# Загрузка данных
X_train = pd.read_csv('data/prepare/train_normalized.csv')
X_val = pd.read_csv('data/prepare/val_normalized.csv')
y_train = pd.read_csv('data/prepare/y_train.csv')
y_val = pd.read_csv('data/prepare/y_val.csv')

# Сохранение в HDF5
with pd.HDFStore('data/prepare/data.h5') as store:
    store['X_train'] = X_train
    store['X_val'] = X_val
    store['y_train'] = y_train
    store['y_val'] = y_val

print("Данные сохранены в HDF5 формате в data/prepare/data.h5")
