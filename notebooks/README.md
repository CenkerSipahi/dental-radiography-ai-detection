# Notebook Rehberi

Bu klasörde proje için iki ayrı notebook bulunur:

```text
01_model_training.ipynb
02_model_demo.ipynb
```

## 01_model_training.ipynb

Modelin eğitim sürecini belgeleyen notebook'tur. CSV etiketlerinin YOLO formatına dönüştürülmesi, veri yapılandırması, YOLOv8 eğitimi ve eğitim sonuçlarının incelenmesi bu notebook içinde yer alır.

Ham eğitim veri seti repoya dahil edilmemişse eğitim hücreleri veri bulunmadığını belirten uyarılar verebilir. Bu normaldir. Veri seti uygun klasör yapısıyla eklendiğinde notebook yeniden eğitim için kullanılabilir.

## 02_model_demo.ipynb

GitHub ve LinkedIn portföy sunumu için önerilen demo notebook'tur.

Bu notebook:

1. `test_pics/` klasöründeki görselleri bulur,
2. klasörden rastgele bir görsel seçer,
3. seçilen görseli `models/best.pt` modeline gönderir,
4. tahmin tablosunu oluşturur,
5. orijinal görsel ve model çıktısını yan yana gösterir.

Belirli bir görsel test edilmek istenirse `02_model_demo.ipynb` içindeki **Ayarlar** hücresinde şu satır değiştirilebilir:

```python
SELECTED_IMAGE_NAME = "0001.jpg"
```

Tekrar rastgele seçim için:

```python
SELECTED_IMAGE_NAME = None
```

Güven eşiği de aynı hücreden değiştirilebilir:

```python
CONFIDENCE_THRESHOLD = 0.25
```

Demo notebook herhangi bir widget, buton veya dosya yükleme paneli kullanmaz. Bu nedenle Jupyter arayüzünde ekstra eklenti gerektirmeden çalışır.
