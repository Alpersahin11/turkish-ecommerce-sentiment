# Turkish E-commerce Sentiment Analysis

Bu proje, Türkçe e-ticaret ürün yorumlarını kullanarak duygu analizi yapmayı amaçlamaktadır. Kullanıcı yorumları, 1 ile 5 arasında puanlanmış ve bu puanlara göre yorumlar "Negatif", "Nötr" ve "Pozitif" olarak sınıflandırılmıştır.

## 📊 Veri Seti

Bu veri seti, Türkçe e-ticaret ürün yorumlarını içermektedir ve projeyi geliştiren kişi tarafından hazırlanmıştır.  
Her satır bir kullanıcı yorumunu ve yorum puanını içerir. Tekrarlayan yorumlar çıkarılmış ve satırlar rastgele karıştırılmıştır.  

Veri setini buradan indirebilirsiniz: [Kaggle - Turkish E-commerce Reviews](https://www.kaggle.com/datasets/alpersah11/turkish-ecommerce-reviews-csv/data)

## ⚙️ Kullanılan Teknolojiler

- Python
- TensorFlow / Keras
- scikit-learn
- pandas
- NumPy

## 🧪 Model

- LSTM tabanlı sinir ağı modeli
- Bidirectional LSTM katmanları
- AdamW optimizasyonu
- Erken durdurma ve öğrenme oranı azaltma mekanizmaları


## Kullanım

Gerekli Python paketlerini yüklemek için:
```bash
pip install -r requirements.txt
```

- Eğitim ve tahmin işlemleri için turkish_reviews_nlp.py dosyasını kullanabilirsiniz.


