import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

def load_data(fname):
    
    """
    Ini adalah fungsi untuk membaca data

    Parameter
    -fname: str
    Path yang menunjukkan posisi file

    Return
    data  : DataFrame 
    berisi kolom dan baris data
    
    """
    
    data = pd.read_csv(fname)
    print(f"Data Shape: {data.shape}")

    return data


def split_input_output(data, target_col):
    
    """
    Memisahkan data input menjadi fitur (X) dan target (y).

    Parameter
    - data       : DataFrame
    Data input yang berisi fitur dan target.
    
    - target_col : string
    Nama kolom yang digunakan sebagai variabel target.

    Return
    - X : DataFrame
    Data fitur
    
    - y : Series
    Data target
    
    """
        
    X = data.drop(target_col, axis = 1)
    y = data[target_col]
    
    print(f'Original data shape: {data.shape}')
    print(f'X data shape       : {X.shape}')
    print(f'y data shape       : {y.shape}')
    
    return X, y


def split_train_test(X, y, test_size, random_state = None):
    
    """
    Memisahkan data X dan y menjadi data train dan data test.

    Parameter
    - X           : DataFrame
    Data input yang berisi fitur.
    
    - y           : Series
    Data target yang berisi target.
    
    - test_size   : float
    Besaran proporsi untuk pembagian train dan test
    
    - random_state: int
    Angka acak agar pembagian data konsisten setiap kali fungsi dijalankan.

    Return
    - X_train, X_test, y_train, y_test : DataFrame
    Data train dan test untuk fitur dan target
    
    """
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state, stratify = y)
    print('X train shape:', X_train.shape)
    print('X test shape :', X_test.shape)
    print('y train shape:', y_train.shape)
    print('y test shape :', y_test.shape)

    return X_train, X_test, y_train, y_test


def serialize_data(data, path): 
    
    """
    Buat serial data ke penyimpanan yang ditentukan menggunakan joblib.
    
    Parameter
    - data   : DataFrame
    Data yang akan diserialkan.
    
    - path   : str
    lokasi tempat data serial akan disimpan.
    
    Return
    - no name: str
    lokasi file tempat data diserialkan.
    """
    return joblib.dump(data, path)


def deserialize_data(path):
    """
    Deserialisasi data dari lokasi yang ditentukan dengan menggunakan joblib.
    
    Parameter
    path : str
    Lokasi data yang akan dideserialisasikan.
    
    Return
    data : DataFrame 
    Data yang sudah diserialisasikan.
    """
    data = joblib.load(path)
    return data