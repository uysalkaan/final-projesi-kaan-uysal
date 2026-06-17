# final-projesi-kaan-uysal
Delivery delay risk prediction system using machine learning
# Smart Delivery Risk Predictor

## Problem Tanımı

Online siparişlerde teslimat gecikmeleri müşteri memnuniyetini azaltmaktadır. Bu proje, sipariş bilgilerini kullanarak teslimatın gecikme riski taşıyıp taşımadığını önceden tahmin etmektedir.

## Hedef Kullanıcı

* Operasyon yöneticileri
* Lojistik ekipleri
* E-ticaret şirketleri
* Market teslimat operasyonları

## Çözümün Kısa Açıklaması

Sistem, siparişe ait mesafe, trafik durumu, hava durumu, kurye deneyimi ve hazırlık süresi gibi bilgileri kullanarak teslimat riskini tahmin eder.

## Kullanılan Teknolojiler

* Python
* Pandas
* Scikit-Learn
* Logistic Regression
* Random Forest
* Streamlit
* Joblib

## Sistem Mimarisi

1. Veri seti okunur
2. Eksik veriler temizlenir
3. Risk değişkeni oluşturulur
4. Model eğitilir
5. Kullanıcı Streamlit arayüzünden veri girer
6. Risk tahmini üretilir

## Kurulum

```bash
pip install pandas scikit-learn streamlit joblib
```

## Çalıştırma

```bash
py train_model.py
py -m streamlit run app.py
```

## Test Sonuçları

Logistic Regression Accuracy: 0.91

Random Forest Accuracy: 0.865

En başarılı model Logistic Regression olarak belirlenmiştir.

## Bilinen Sınırlılıklar

* Veri seti yalnızca 1000 kayıt içermektedir.
* Gerçek zamanlı trafik verisi kullanılmamaktadır.

## Gelecek Çalışmalar

* SHAP açıklamaları eklenebilir.
* Gerçek zamanlı trafik API entegrasyonu yapılabilir.
* Daha büyük veri setleriyle model geliştirilebilir.

## Yapay Zeka Araçları

Kod geliştirme ve dokümantasyon süreçlerinde yapay zeka araçlarından destek alınmıştır.
