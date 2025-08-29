# Turkish E-commerce Sentiment Analysis / Türkçe E-ticaret Duygu Analizi

## 📖 Project Description / Proje Açıklaması
🇹🇷 Bu proje, Türkçe e-ticaret ürün yorumlarını kullanarak duygu analizi yapmayı amaçlamaktadır. Kullanıcı yorumları 1 ile 5 arasında puanlanmıştır ve bu puanlara göre yorumlar Negatif, Nötr ve Pozitif olarak sınıflandırılmıştır.

🇬🇧 This project aims to perform sentiment analysis on Turkish e-commerce product reviews. User reviews are rated on a scale of 1 to 5, and based on these ratings, reviews are classified into Negative, Neutral, and Positive categories.

## 📊 Dataset / Veri Seti

🇹🇷 Veri seti, Türkçe e-ticaret ürün yorumlarını içermektedir ve projeyi geliştiren kişi tarafından hazırlanmıştır. Her satır bir kullanıcı yorumunu ve ilgili puanını içerir. Tekrarlayan yorumlar çıkarılmış ve satırlar rastgele karıştırılmıştır.

🇬🇧 The dataset contains Turkish e-commerce product reviews and was prepared by the project developer. Each row consists of a user review and its corresponding rating. Duplicate reviews were removed, and the rows were randomly shuffled.

📂 Download / İndir: [Kaggle - Turkish E-commerce Reviews](https://www.kaggle.com/datasets/alpersah11/turkish-ecommerce-reviews-csv/data)


## ⚙️ Technologies Used / Kullanılan Teknolojiler

- Python
- TensorFlow / Keras
- scikit-learn
- pandas
- NumPy

## 🧪 Model / Model

🇹🇷 Model, LSTM tabanlı sinir ağı mimarisi kullanılarak geliştirilmiştir:

-Bidirectional LSTM katmanları
-AdamW optimizasyonu
-Erken durdurma (early stopping)
-Öğrenme oranı azaltma mekanizmaları

🇬🇧 The model was built using an LSTM-based neural network architecture:

-Bidirectional LSTM layers
-AdamW optimizer
-Early stopping
-Learning rate reduction mechanisms


## 🚀 Usage / Kullanım

🇹🇷 Gerekli Python paketlerini yüklemek için:
🇬🇧 Install the required Python packages:
```bash
pip install -r requirements.txt
```

🇹🇷 Eğitim ve tahmin işlemleri için turkish_reviews_nlp.py dosyasını kullanabilirsiniz.
🇬🇧 Use the turkish_reviews_nlp.py file for both training and prediction.


