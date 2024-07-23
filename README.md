# streamlit_fastapi_credit_risk
 Advancement of credit_risk_classification_model with fastapi and streamlit

Ini adalah pengembangan lanjutan dari model klasifikasi credit_risk yang sudah dikembangkan dalam repository lain. Silakan cek di credit_risk_classification_model. Pengembangan terbesarnya ada pada pemenggalan proses persiapan, pembersihan, penyimpanan, pemodelan, dan evaluasi yang disimpan dalam file dan folder yang berbeda. Dengan file dan folder yang berbeda, proses pengubahan code dapat berlangsung dalam fase yang lebih ringkas. Berikut adalah struktur foldernya:

<img width="300" alt="image" src="https://github.com/user-attachments/assets/be1604ee-7034-4c43-862f-a2767b434832">



Pengembangan lainnya adalah pembuatan backend dengan fastapi dan frontend dengan streamlit dengan menggunakan port 8000 dan 8501 (localhost). Apabila code-nya dijalankan, teman-teman dapat melakukan input ke masing-masing kolom dan melihat status prediksi yang dikeluarkan model. Andaikata teman-teman adalah seorang peminjam, model akan mendefinisikan, apakah teman-teman adalah calon peminjam yang berpotensi gagal bayar atau tidak. Berikut tampilan sederhananya.

<img width="300" alt="image" src="https://github.com/user-attachments/assets/8f4cc5dd-3c36-4d22-9a1b-2e6496fb04b6">


Silakan dinikmati! 

Menariknya model terbaik versi lama dan yang sekarang tetap pada RandomForestClassifier. Perbedaannya yang lama menghasilkan skor 0.82, yang baru meningkatkan performanya dan mencatat skor 0.85. Ini berarti performa model dalam melakukan klasifikasi kelas semakin baik. Rupanya proses feature engineering pada model baru, yaitu dengan melakukan over sampling pada kelas yang memiliki data yang lebih sedikit memainkan peranan yang signifikan. 

