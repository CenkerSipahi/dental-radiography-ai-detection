# 🦷 Diş Röntgeni Anomali Tespiti — YOLOv8

Panoramik diş röntgenleri üzerinde nesne tespiti yapan YOLOv8 tabanlı yapay zeka projesi. Model, röntgen görüntülerinde dört farklı dental bulguyu tespit etmek üzere eğitilmiştir:

- **Implant** — İmplant
- **Fillings** — Dolgu
- **Impacted Tooth** — Gömülü diş
- **Cavity** — Çürük

Bu repo, GitHub ve LinkedIn portföy sunumu için düzenlenmiş profesyonel bir açık kaynak proje yapısı içerir. Hazır eğitilmiş model `models/best.pt` içinde yer alır. Eğitim süreci ayrı bir notebook ile belgelenmiştir; demo notebook ise `test_pics/` klasöründen rastgele bir röntgen görseli seçip model çıktısını doğrudan notebook içinde gösterir.

> ⚠️ **Klinik Uyarı:** Bu proje araştırma ve portföy prototipidir. Model çıktıları klinik tanı, tedavi planı veya hekim değerlendirmesinin yerine geçmez.

---

## Proje Tanıtımı

Bu proje, panoramik diş röntgenlerinde belirli dental bulguları otomatik olarak işaretlemek için geliştirilmiş bir derin öğrenme uygulamasıdır. Model, röntgen üzerindeki olası bulguları bounding box ile gösterir, tahmin edilen sınıfı ve güven skorunu üretir.

---

## Projenin Amacı

Bu projenin amacı:

1. Panoramik diş röntgenlerinde nesne tespiti yapmak,
2. Eğitilmiş YOLOv8 modelini sade bir notebook akışıyla test edilebilir hale getirmek,
3. `test_pics/` klasöründeki örnek görseller üzerinden terminal kullanmadan tahmin almak,
4. Model çıktısını sınıf, güven skoru ve işaretlenmiş görüntü ile birlikte göstermek,
5. GitHub ve LinkedIn için profesyonel bir yapay zeka portföy projesi oluşturmaktır.

---

## Özellikler

- YOLOv8 tabanlı nesne tespiti modeli
- Hazır eğitilmiş model ağırlığı: `models/best.pt`
- Model eğitim sürecini belgeleyen ayrı eğitim notebook'u
- Basit ve sorunsuz demo notebook akışı
- `test_pics/` klasöründen rastgele test görseli seçme
- Kullanıcının kod içinden belirli test görselini seçebilmesi
- Tahmin edilen sınıf, güven skoru ve bounding box bilgilerini tablo olarak gösterme
- Orijinal görüntü ve model çıktısını yan yana görselleştirme
- Komut satırı ile tek görsel üzerinde çıkarım yapabilme

---

## Kullanılan Teknolojiler

| Teknoloji | Kullanım Amacı |
|---|---|
| Python | Ana programlama dili |
| Ultralytics YOLOv8 | Nesne tespiti modeli |
| PyTorch | YOLOv8 arka plan derin öğrenme altyapısı |
| OpenCV | Görsel okuma, işleme ve çıktı üretimi |
| Pandas | Tahmin sonuçlarını tablo olarak gösterme |
| Matplotlib | Notebook içinde görselleştirme |
| JupyterLab | Notebook çalıştırma ortamı |
| PyYAML | YOLO veri seti yapılandırması |

---

## Klasör Yapısı

```text
.
├── data/
│   └── dental_data.yaml              # YOLO veri seti yapılandırması
├── docs/
│   └── results/                      # Eğitim grafikleri, confusion matrix ve örnek tahmin çıktısı
├── models/
│   └── best.pt                       # Eğitilmiş model ağırlığı
├── notebooks/
│   ├── README.md                     # Notebook kullanım rehberi
│   ├── 01_model_training.ipynb       # Model eğitim ve veri dönüşüm notebook'u
│   └── 02_model_demo.ipynb           # Basit model test/demo notebook'u
├── src/
│   └── dental_ai/
│       ├── __init__.py
│       └── inference.py              # Ortak model yükleme, tahmin ve görselleştirme fonksiyonları
├── test_pics/                        # Notebook ve CLI için örnek test görselleri
├── predict.py                        # Komut satırı tahmin betiği
├── requirements.txt                  # Python bağımlılıkları
├── .gitignore                        # GitHub'a eklenmemesi gereken dosyalar
├── LICENSE                           # MIT lisansı
└── README.md                         # Proje dokümantasyonu
```

---

## Kurulum

Aşağıdaki adımlar sıfırdan kurulum yapan bir kullanıcı için hazırlanmıştır.

### 1. Repoyu indirin

```bash
git clone <repo-url>
cd dental_dl_project_model
```

> `<repo-url>` kısmını GitHub'a yükledikten sonra kendi repo adresinizle değiştirin.

### 2. Sanal ortam oluşturun

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. pip'i güncelleyin

```bash
python -m pip install --upgrade pip
```

### 4. Bağımlılıkları kurun

```bash
python -m pip install -r requirements.txt
```

### 5. Notebook'u başlatın

```bash
jupyter lab
```

Ardından hızlı demo için `notebooks/02_model_demo.ipynb` dosyasını açın. Model eğitim sürecini incelemek veya ham veri setiyle yeniden eğitim yapmak için `notebooks/01_model_training.ipynb` dosyasını kullanın.

---

## Gereksinimler

Önerilen çalışma ortamı:

- Python **3.10 - 3.13**
- pip
- JupyterLab veya Jupyter Notebook
- En az 4 GB RAM
- CPU ile çıkarım yapılabilir; yeniden eğitim için GPU önerilir

Temel Python paketleri `requirements.txt` içinde tanımlanmıştır:

```text
ultralytics
opencv-python-headless
numpy
pandas
pyyaml
matplotlib
jupyterlab
Pillow
```

---

## Notebook Kullanımı

Projede iki ayrı notebook vardır:

```text
notebooks/01_model_training.ipynb
notebooks/02_model_demo.ipynb
```

| Notebook | Açıklama |
|---|---|
| `01_model_training.ipynb` | Modelin eğitim sürecini, CSV etiketlerinin YOLO formatına dönüşümünü, eğitim komutlarını ve performans görselleştirmesini içerir. |
| `02_model_demo.ipynb` | Eğitilmiş `models/best.pt` modelini terminal kullanmadan test eden sade demo notebook'udur. |

### Demo notebook kullanımı

`02_model_demo.ipynb` şu şekilde çalışır:

1. Proje yollarını ve yardımcı fonksiyonları hazırlar.
2. `models/best.pt` modelini yükler.
3. `test_pics/` klasöründeki desteklenen görselleri listeler.
4. Bu klasörden rastgele bir görsel seçer.
5. Seçilen görseli modele gönderir.
6. Tahmin edilen sınıfı, güven skorunu, kutu koordinatlarını ve öneri notunu tablo olarak gösterir.
7. Orijinal röntgen ve işaretlenmiş model çıktısını yan yana görüntüler.

Belirli bir görseli test etmek için notebook içindeki **Ayarlar** hücresinde şu satırı değiştirin:

```python
SELECTED_IMAGE_NAME = "0001.jpg"
```

Tekrar rastgele seçim yapmak için:

```python
SELECTED_IMAGE_NAME = None
```

Güven eşiğini değiştirmek için:

```python
CONFIDENCE_THRESHOLD = 0.25
```

Notebook'u kullanan kişi terminal komutu yazmak zorunda değildir. Hücreleri sırayla çalıştırması yeterlidir.

---

## Model Nasıl Çalıştırılır

Projede iki farklı kullanım yolu vardır.

### 1. Notebook ile çalıştırma

En pratik yol `notebooks/02_model_demo.ipynb` dosyasını açmaktır.

Notebook içinde:

1. Hücreleri yukarıdan aşağıya çalıştırın.
2. Notebook otomatik olarak `test_pics/` klasöründen bir görsel seçsin.
3. Seçilen görsel modele gönderilsin.
4. Tahmin tablosunu ve işaretlenmiş çıktıyı notebook içinde görüntüleyin.

Belirli bir görsel kullanmak isterseniz **Ayarlar** hücresindeki `SELECTED_IMAGE_NAME` değerini değiştirmeniz yeterlidir.

### 2. Komut satırı ile çalıştırma

Hazır test görsellerinden biriyle örnek kullanım:

```bash
python predict.py --image test_pics/0001.jpg
```

Güven eşiğini değiştirmek için:

```bash
python predict.py --image test_pics/0001.jpg --conf 0.25
```

Sonucu belirli bir dosyaya kaydetmek için:

```bash
python predict.py --image test_pics/0001.jpg --output docs/results/prediction_0001.jpg --no-show
```

---

## Test Görselleri

Demo ve komut satırı tahminleri için örnek röntgen görselleri `test_pics/` klasöründe yer alır.

Desteklenen formatlar:

- `.jpg`
- `.jpeg`
- `.png`
- `.webp`

Yeni test görselleri eklemek için dosyaları `test_pics/` klasörüne koyabilirsiniz. Demo notebook, desteklenen uzantıya sahip dosyaları otomatik olarak listeler.

---

## Örnek Çıktılar

Model çıktıları şu bilgileri içerir:

- Tahmin edilen sınıf
- Türkçe sınıf adı
- Güven skoru
- Güven yüzdesi
- Bounding box koordinatları
- Kısa klinik değerlendirme notu
- Orijinal görsel ve işaretlenmiş model çıktısı

Örnek çıktı dosyası:

```text
docs/results/example_prediction_0001.jpg
```

Eğitim sürecine ait grafikler de `docs/results/` klasöründe tutulur.

---

## Kullanılan Veri Seti

Model, panoramik diş röntgenleri üzerinde dental bulguları tespit edecek şekilde YOLO formatına dönüştürülmüş etiketlerle eğitilmiştir.

Bu repoda örnek test görselleri ve eğitilmiş model ağırlığı bulunur. Ham eğitim veri seti boyut, lisans veya paylaşım kısıtları nedeniyle repoya dahil edilmemiş olabilir. Eğitim notebook'u, veri seti uygun klasör yapısıyla eklendiğinde CSV etiketlerini YOLO formatına dönüştürme ve modeli yeniden eğitme akışını göstermektedir.

Beklenen eğitim veri yapısı genel olarak şu şekildedir:

```text
data/
└── dataset/
    ├── images/
    ├── labels/
    └── annotations.csv
```

Veri setini yeniden kullanmadan önce ilgili veri kaynağının lisans koşulları kontrol edilmelidir.

---

## Model Bilgileri

- Model ailesi: **YOLOv8**
- Görev tipi: **Object Detection / Nesne Tespiti**
- Model dosyası: `models/best.pt`
- Girdi: Panoramik diş röntgeni görseli
- Çıktı: Sınıf etiketi, güven skoru ve bounding box koordinatları
- Hedef sınıflar:
  - `Implant`
  - `Fillings`
  - `Impacted Tooth`
  - `Cavity`

Model çıktısı, `src/dental_ai/inference.py` içindeki yardımcı fonksiyonlarla özetlenir ve notebook/CLI üzerinde okunabilir hale getirilir.

---
