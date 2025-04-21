# 🛍️ Market Analiz API

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68.1-green)
![License](https://img.shields.io/badge/license-MIT-orange)

Market verilerini DBSCAN algoritması kullanarak analiz eden ve görselleştiren modern bir web API.

## 🌟 Özellikler

- **Ürün Segmentasyonu**: Ürünleri satış performanslarına göre gruplandırma
- **Tedarikçi Analizi**: Tedarikçileri performans metriklerine göre sınıflandırma
- **Ülke Bazlı Analiz**: Ülkeleri satış desenlerine göre segmentlere ayırma
- **İnteraktif Görselleştirmeler**: Plotly ile etkileşimli grafikler
- **Dinamik Parametre Ayarları**: Kaydırma çubuğu ile kolay parametre optimizasyonu
- **Detaylı İstatistikler**: Küme bazlı detaylı metrikler ve analizler

## 🚀 Hızlı Başlangıç

### Gereksinimler

- Python 3.8+
- pip (Python paket yöneticisi)

### Kurulum

1. Projeyi klonlayın:
```bash
git clone https://github.com/kullaniciadi/market-analysis-api.git
cd market-analysis-api
```

2. Sanal ortam oluşturun ve aktifleştirin:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

4. Uygulamayı başlatın:
```bash
uvicorn market_analysis_api:app --reload
```

5. Tarayıcınızda açın:
```
http://localhost:8000
```

## 📊 Kullanım

### Web Arayüzü

1. Ana sayfada üç farklı analiz seçeneği bulunur:
   - Ürün Analizi
   - Tedarikçi Analizi
   - Ülke Analizi
     ![2025-04-21_21-52-53](https://github.com/user-attachments/assets/6ce5291e-62ab-42ed-8ec9-20c11d1f33d4)

2. Her analiz için:
   - Kaydırma çubuğu ile minimum örnek sayısını ayarlayın
   - "Analiz Et" butonuna tıklayın
   - Sonuçları görselleştirmeler ve tablolar halinde inceleyin
![2025-04-21_21-53-23](https://github.com/user-attachments/assets/dce3ecf8-45ae-4aff-a330-2ce2aa151270)

![2025-04-21_21-53-54](https://github.com/user-attachments/assets/cfbd5691-ba4c-4a1c-877a-f847fcd56019)


### API Endpoints

- `GET /analyze/products?min_samples=3`: Ürün analizi
- `GET /analyze/suppliers?min_samples=3`: Tedarikçi analizi
- `GET /analyze/countries?min_samples=3`: Ülke analizi

API dokümantasyonu için: `http://localhost:8000/docs`

![2025-04-21_21-54-18](https://github.com/user-attachments/assets/a99e0935-40da-4f1c-9541-4ba855184718)


## 🛠️ Teknik Detaylar

### Kullanılan Teknolojiler

- **Backend**: FastAPI, Python
- **Veri Analizi**: Pandas, NumPy, Scikit-learn
- **Görselleştirme**: Plotly
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Veritabanı**: SQL (Northwind örnek veritabanı)

### Mimari

- **Modüler Yapı**: Her analiz türü için ayrı fonksiyonlar
- **DBSCAN Algoritması**: Gürültüye dayanıklı kümeleme
- **Dinamik Parametreler**: Minimum örnek sayısı optimizasyonu
- **İnteraktif Arayüz**: Kullanıcı dostu web arayüzü

## 📈 Analiz Sonuçları

Her analiz sonucunda şu bilgiler sunulur:

- Küme dağılımları
- Aykırı değerler
- İstatistiksel metrikler
- İnteraktif görselleştirmeler:
  - Scatter plot
  - Box plot
  - 3D görselleştirme

## 🤝 Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🙏 Teşekkürler

- [FastAPI](https://fastapi.tiangolo.com/) ekibine
- [Plotly](https://plotly.com/) ekibine
- [Bootstrap](https://getbootstrap.com/) ekibine

---

<div align="center">
  <sub>Built with ❤️ by <a href="https://github.com/kullaniciadi">Your Name</a></sub>
</div> "# DBSCAN_Voyager" 
