Cross-Sell Model Özeti
Bu doküman, baskets.parquet verisi kullanılarak hazırlanan cross-sell önerisi çalışmasında yapılan adımları, kullanılan model tipini, train/test ayrımını, üretilen dosyaları ve elde edilen metrikleri özetler.

Kullandığımız Model
Problem tipi klasik sınıflandırma ya da regresyon değil, önerilecek ürünleri bulma problemidir.

Kullanılan model:

Plaintext
Cross-Sell Association Rules Model
Teknik yaklaşım:

Plaintext
Association Rule Mining
Market Basket Analysis
Item-to-Item Co-occurrence Scoring
Modelin temel mantığı şudur:

A ürünü ile B ürünü aynı sepetlerde sık birlikte görülüyorsa, A ürünü sepetteyken B ürünü cross-sell önerisi olarak verilir.

Örnek:
Sepet: [A, B, C]
Bu sepetten model şu ilişkileri öğrenir: A -> B, A -> C, B -> A, B -> C, C -> A, C -> B

Kullanılan Veri
Ana veri dosyası: data/gold/baskets.parquet

Veri Formatı:

t_dat

customer_id

articles (Ürün listesi)

basket_size

Cross-sell modeli için articles kolonu kullanıldı. Bu kolon bir sepet içindeki ürün listesini temsil eder.

Train/Test Ayrımı
Train/test ayrımı rastgele yapılmadı. Öneri sistemlerinde daha doğru bir test kurgusu için zamana göre ayrım yapıldı.

Mantık:

Geçmiş dönem verisi -> Train

En son dönem verisi -> Test

Bu projede son 7 gün test seti olarak ayrıldı.

Veri İstatistikleri:

Train Aralığı: 2018-09-20 - 2020-09-15

Test Aralığı: 2020-09-16 - 2020-09-22

Toplam İşlenen Sepet: 6,134,581

Unique Müşteri Sayısı: 1,174,992

Ortalama Sepet Boyutu: 4.21

Feature ve Model Akışı
Hazırlanan akış üç ana notebooktan oluşur:

1. Feature ve Split Notebooku
Dosya: notebooks/02_features/cross_sell_basket_features.ipynb

baskets.parquet dosyasını okur.

articles kolonunu normalize eder.

Zamana göre train/test ayrımı yapar.

Ürün bazlı temel feature dosyası oluşturur.

2. Baseline Model Notebooku
Dosya: notebooks/03_modeling/cross_sell/cross_sell_association_rules_baseline.ipynb

Train sepetlerinden ürün çiftlerini çıkarır.

Ürünlerin birlikte görülme sayılarını hesaplar.

Association rule metriklerini (support, confidence, lift, jaccard) hesaplar.

Test setinde modeli değerlendirir.

Baseline Sonuçları:

precision@12 = 0.0161

recall@12 = 0.1133

hit_rate@12 = 0.1724

3. Tuning Notebooku
Dosya: notebooks/03_modeling/cross_sell/cross_sell_association_rules_tuning.ipynb

Farklı parametre kombinasyonlarını (min_item_support, min_pair_support, score_formula) dener.

En iyi sonucu veren modeli kaydeder.

En İyi Parametreler:

min_item_support = 10

min_pair_support = 3

score_formula = confidence_lift

En İyi Tuning Sonucu:

precision@12 = 0.0171

recall@12 = 0.1199

hit_rate@12 = 0.1820

Metriklerin Yorumu
precision@12: Modelin verdiği 12 önerinin ne kadarının test sepetindeki gerçek ürünlerle eşleştiğini ölçer.

recall@12: Test sepetinde alınmış ürünlerin ne kadarının 12 önerilik liste içinde yakalandığını ölçer.

hit_rate@12: Modelin verdiği 12 öneride en az bir doğru ürün yakalama oranını ölçer.

En iyi model için hit_rate@12 = 0.1820 sonucu şu anlama gelir:
Model 12 ürün önerdiğinde, test sepetlerinin yaklaşık %18.2'sinde en az bir doğru cross-sell ürününü yakalamıştır.

Genel Sonuç
Bu çalışmada cross-sell için association rules tabanlı bir önerici hazırlandı. Model, sepetlerde birlikte görülen ürünleri öğrenerek ürün bazlı cross-sell önerileri üretiyor. Çalışma kapsamında train/test ayrımı yapıldı, baseline model eğitildi, tuning işlemleri tamamlandı ve en iyi model ile metrikler kayıt altına alındı.