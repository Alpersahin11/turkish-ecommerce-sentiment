import os
import warnings
import pandas as pd
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# ⚡ Uyarıları gizle
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU kullanma
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow uyarıları gizle


# -------------------- Fonksiyonlar --------------------

def temizle(text):
    """Metni küçük harfe çevir, Türkçe karakterleri koru, özel karakterleri kaldır."""
    text = str(text).lower()
    text = re.sub(r'[^a-zçğıöşü0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def puan_to_sentiment(rating):
    """Rating'i duygu etiketine çevir: 0=Negatif, 1=Nötr, 2=Pozitif"""
    if rating >= 4:
        return 2
    elif rating == 3:
        return 1
    else:
        return 0


def veri_yukle_ve_hazirla(csv_path):
    """CSV'yi yükle, temizle ve duygu etiketlerini oluştur."""
    df = pd.read_csv(csv_path)
    df = df[["comment", "rating"]].dropna()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].astype("int16")
    df["clean_comment"] = df["comment"].apply(temizle)
    df["sentiment"] = df["rating"].apply(puan_to_sentiment)
    return df


def veri_tokenize_ve_bol(df, max_words=5000, max_len=70, test_size=0.2):
    """Tokenizer uygula, padding yap, one-hot encode ve veri böl."""
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(df["clean_comment"].values)
    sequences = tokenizer.texts_to_sequences(df["clean_comment"].values)
    X = pad_sequences(sequences, maxlen=max_len, padding="post")
    y = pd.get_dummies(df["sentiment"]).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    class_weights_values = compute_class_weight('balanced', classes=np.unique(df['sentiment']), y=df['sentiment'])
    class_weights = dict(zip(np.unique(df['sentiment']), class_weights_values))

    return X_train, X_test, y_train, y_test, tokenizer, class_weights


def model_olustur(max_words=5000, max_len=70):
    """Bidirectional LSTM tabanlı model oluştur."""
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=128))
    model.add(Bidirectional(LSTM(64, dropout=0.3, return_sequences=True)))
    model.add(Bidirectional(LSTM(32, dropout=0.3)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(3, activation='softmax'))

    adamw = AdamW(learning_rate=1e-3, weight_decay=1e-5)
    model.compile(loss='categorical_crossentropy', optimizer=adamw, metrics=['accuracy'])
    return model


def model_egit(model, X_train, y_train, X_val, y_val, class_weights=None, epochs=50, batch_size=64):
    """Modeli eğit."""
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr]
    )
    return history


def tahmin_et(model, tokenizer, yorum, max_len=70):
    """Verilen yorumu tahmin et ve olasılıkları göster."""
    classes = {0: "Negatif", 1: "Nötr", 2: "Pozitif"}
    seq = tokenizer.texts_to_sequences([yorum])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded, verbose=0)
    print("-" * 10)
    print(yorum)
    print(f"Olasılıklar: {pred}")
    print(f"Tahmin: {classes[pred.argmax(axis=1)[0]]}")
    print("-" * 10)


def rapor_yazdir(model, X_test, y_test):
    """Classification report yazdır."""
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    print(classification_report(y_true, y_pred_classes, target_names=["Negatif", "Nötr", "Pozitif"]))


def model_kaydet(model, path_h5="sentiment_model.h5", path_keras="sentiment_model.keras"):
    model.save(path_h5)
    model.save(path_keras)


# -------------------- Kullanım --------------------
csv_path = "turkish_ecommerce_reviews.csv"
df = veri_yukle_ve_hazirla(csv_path)
X_train, X_test, y_train, y_test, tokenizer, class_weights = veri_tokenize_ve_bol(df)
model = model_olustur()
history = model_egit(model, X_train, y_train, X_test, y_test, class_weights=class_weights)

# Örnek tahminler
tahmin_et(model, tokenizer, "iade ettim beğenmedim")
tahmin_et(model, tokenizer, "Fena değil eksik parçaları var iade etmedim kullanırım")
tahmin_et(model, tokenizer, "Ürün harika bayıldım")

# Rapor ve kaydetme
rapor_yazdir(model, X_test, y_test)
model_kaydet(model)
