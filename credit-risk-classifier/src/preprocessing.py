import pandas as pd

from sklearn.preprocessing import OneHotEncoder

def ohe_transform(dataset, subset, prefix, ohe):
   
    """
    Melakukan One-Hot Encoding pada kolom kategorik dalam dataset.

    Parameter:
    -----------
    dataset: pd.DataFrame
             DataFrame yang berisi data yang ingin dilakukan pengkodean.
    subset : str
             Nama kolom dalam dataset yang berisi data kategorik yang akan di-encode.
    prefix : str
             Nama awalan yang akan disematkan pada kolom hasil pengkodean.
    ohe    : OneHotEncoder
             Encoder yang telah dilatih sebelumnya dengan kategori khusus.

    Returns:
    --------
    pd.DataFrame
        DataFrame yang telah dilakukan pengkodean pada kolom kategorik yang ditentukan.
    
    Raises:
    -------
    RuntimeError
        Jika parameter `dataset` bukan bertipe DataFrame.
        Jika parameter `subset` bukan bertipe str.
        Jika parameter `prefix` bukan bertipe str.
        Jika parameter `ohe` bukan bertipe OneHotEncoder.
        Jika kolom yang ditentukan oleh parameter `subset` tidak ditemukan dalam dataset.
    """
    
    # Validasi parameter dataset
    if not isinstance(dataset, pd.DataFrame):
        raise RuntimeError("Fungsi ohe_transform: parameter dataset harus bertipe DataFrame!")

    # Validasi parameter ohe
    if not isinstance(ohe, OneHotEncoder):
        raise RuntimeError("Fungsi ohe_transform: parameter ohe harus bertipe OneHotEncoder!")

    # Validasi parameter prefix
    if not isinstance(prefix, str):
        raise RuntimeError("Fungsi ohe_transform: parameter prefix harus bertipe str!")

    # Validasi parameter subset
    if not isinstance(subset, str):
        raise RuntimeError("Fungsi ohe_transform: parameter subset harus bertipe str!")

    # Validasi keberadaan subset dalam dataset
    try:
        dataset.columns.tolist().index(subset)
    except ValueError:
        raise RuntimeError("Fungsi ohe_transform: parameter subset string tidak ditemukan dalam daftar kolom yang terdapat pada parameter dataset.")

    print("Fungsi ohe_transform: parameter telah divalidasi.")
    
    # Membuat duplikat dataset
    dataset = dataset.copy()

    # Menampilkan daftar nama kolom sebelum pengkodean
    print(f"Fungsi ohe_transform: daftar nama kolom sebelum dilakukan pengkodean adalah {list(dataset.columns)}.")
    
    # Membuat nama kolom untuk hasil pengkodean
    col_names = [f"{prefix}_{col_name}" for col_name in ohe.categories_[0].tolist()]
    
    # Melakukan pengkodean
    encoded = pd.DataFrame(
        ohe.transform(dataset[[subset]]).toarray(),
        columns=col_names,
        index=dataset.index
    )

    # Menggabungkan hasil pengkodean dengan dataset
    dataset = pd.concat([dataset, encoded], axis=1)
    
    # Menghapus kolom asli yang sudah dikodekan
    dataset.drop(columns=[subset], inplace=True)
    
    # Menampilkan daftar nama kolom setelah pengkodean
    print(f"Fungsi ohe_transform: daftar nama kolom setelah dilakukan pengkodean adalah {list(dataset.columns)}.")
    print()
    
    return dataset
