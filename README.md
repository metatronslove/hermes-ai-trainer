# 🔮 HERMES ULTIMATE V15.0 - AI Eğitim Sistemi

**70B→7B Akıllı Veri Distilasyonu + QLoRA + DeepSpeed + Okült Rehberlik**  
*2× NVIDIA P104-100 8GB Optimizasyonu + Zengin Terminal Arayüzü*

---

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Sistem Gereksinimleri](#-sistem-gereksinimleri)
- [Kurulum](#-kurulum)
- [Hızlı Başlangıç](#-hızlı-başlangıç)
- [Kullanım Kılavuzu](#-kullanım-kılavuzu)
- [Yapılandırma Profilleri](#-yapılandırma-profilleri)
- [Model Eğitimi](#-model-eğitimi)
- [Sorun Giderme](#-sorun-giderme)
- [Katkıda Bulunma](#-katkıda-bulunma)

---

## 🎯 Özellikler

### 🤖 Gelişmiş AI Eğitim Sistemi
- **70B→7B Akıllı Distilasyon**: Büyük veri setlerini optimize ederek küçük modellerde kullanma
- **QLoRA + 4-bit Quantization**: Düşük bellek tüketimi ile verimli fine-tuning
- **DeepSpeed Zero3**: Çoklu GPU desteği ve bellek optimizasyonu
- **Flash Attention**: Hızlandırılmış dikkat mekanizması

### 🔮 Okült Rehberlik Entegrasyonu
- **Hermetik Bilgelik**: Simya, numeroloji, gematria entegrasyonu
- **Sembolik Analiz**: Metaforik ve arketipik yorumlama
- **Çoklu Hesaplama Yöntemleri**: 9 farklı analiz tekniği
- **Bağlamsal Rehberlik**: 8 farklı konuşma stili ipucu

### 💻 Sistem Optimizasyonu
- **2× NVIDIA P104-100 Optimizasyonu**: 8GB VRAM için özel ayarlar
- **Zengin Terminal Arayüzü**: Renkli, interaktif kontrol paneli
- **Akıllı Kaynak Yönetimi**: Otomatik bellek optimizasyonu
- **Gerçek Zamanlı İzleme**: GPU, CPU, bellek takibi

### 📊 Veri İşleme
- **Akıllı Distilasyon**: Kalite tabanlı otomatik filtreleme
- **Çoklu Format Desteği**: JSONL, TXT, CSV, HuggingFace Datasets
- **Gelişmiş Augmentasyon**: Bağlamsal veri zenginleştirme
- **Otomatik Ön İşleme**: Tokenizasyon ve format dönüşümü

---

## 🛠 Sistem Gereksinimleri

### Minimum Gereksinimler
- **İşletim Sistemi**: Windows 10/11, Linux Ubuntu 18.04+, macOS 12+
- **Python**: 3.9, 3.10 veya 3.11
- **RAM**: 16 GB
- **Depolama**: 50 GB boş alan
- **GPU**: NVIDIA GPU (8GB+ VRAM önerilir)

### Önerilen Sistem
- **İşletim Sistemi**: Windows 11 / Linux Ubuntu 20.04+
- **Python**: 3.10
- **RAM**: 32 GB
- **Depolama**: 100 GB SSD
- **GPU**: 2× NVIDIA P104-100 (8GB) veya eşdeğeri

---

## 📥 Kurulum

### 1. Miniconda Kurulumu
```bash
# Windows: https://docs.conda.io/en/latest/miniconda.html
# Linux/macOS: 
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh
```

### 2. Otomatik Kurulum (Windows)
```bash
# install_hermes.bat dosyasını çalıştırın
install_hermes.bat
```

### 3. Manuel Kurulum
```bash
# Conda ortamı oluştur
conda create -n ai-env python=3.10 -y
conda activate ai-env

# PyTorch ve temel paketler
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8 cudatoolkit=11.8

# Diğer paketler
pip install transformers datasets accelerate peft bitsandbytes deepspeed rich zstandard pandas numpy scikit-learn

# Dizin yapısı oluştur
mkdir -p data/{raw,distilled,augmented,cache} models/hermes_final profiles logs configs checkpoints
```

---

## 🚀 Hızlı Başlangıç

### 1. Ortamı Aktif Et
```bash
conda activate ai-env
```

### 2. Programı Başlat
```bash
python ai_trainer_15.py
```

### 3. İlk Yapılandırma
1. Ana menüden `1` seçerek Konfigürasyon Editörü'nü açın
2. Model ayarlarını yapılandırın
3. Eğitim parametrelerini ayarlayın
4. `S` tuşu ile profili kaydedin

### 4. Örnek Veri ile Test
```bash
python ai_trainer_15.py --quick --data "https://huggingface.co/datasets/trivia_qa"
```

---

## 📖 Kullanım Kılavuzu

### Ana Menü Özellikleri

#### 🎛️ Konfigürasyon Editörü (Seçenek 1)
- **Model Ayarları**: Temel model, mimari, sequence length
- **Eğitim Parametreleri**: Learning rate, batch size, epoch
- **QLoRA/LoRA**: Rank, alpha, dropout ayarları
- **Okült Özellikler**: Rehberlik ağırlıkları, prompt şablonları
- **Sistem Optimizasyon**: GPU, DeepSpeed, mixed precision

#### 💾 Profil Yönetimi (Seçenek 2)
```bash
# Profil kategorileri:
- occult: Okült Rehberlik
- scientific: Bilimsel Analiz  
- creative: Yaratıcı Üretim
- technical: Teknik Danışman
- balanced: Dengeli Mod
```

#### 📥 Veri İndirme (Seçenek 3)
Desteklenen veri kaynakları:
- **URL'ler**: HTTP/HTTPS üzerinden raw text
- **Yerel Dosyalar**: JSONL, TXT, CSV formatları
- **HuggingFace**: Dataset isimleri (örn: `trivia_qa`)

#### ⚡ Hızlı Başlat (Seçenek 4)
Mevcut konfigürasyonla direkt eğitim başlatma

#### 🔮 Tam Pipeline (Seçenek 5)
70B→7B distilasyon + Okült augmentasyon + Model eğitimi

### Komut Satırı Argümanları

```bash
# Temel kullanım
python ai_trainer_15.py

# Profil ile başlatma
python ai_trainer_15.py --profile "okult-rehber"

# Hızlı başlatma
python ai_trainer_15.py --quick --data "path/to/data" "https://dataset.org"

# Özel konfigürasyon
python ai_trainer_15.py --config "custom_config.json"
```

---

## ⚙️ Yapılandırma Profilleri

### 1. Okült Rehberlik Profili
```json
{
  "profile_name": "okult-rehber",
  "model_architecture": "llama-7b",
  "base_model": "meta-llama/Llama-2-7b-hf",
  "training_strategy": "qlora-4bit",
  "integrate_occult": true,
  "lora_rank": 32,
  "occult_coherence_weight": 0.7,
  "conversation_tips": {
    "okült": "Sembolik yorumlama ve kadim bilgelik entegrasyonu",
    "spiritüel": "Sezgisel rehberlik ile pratik temellendirme"
  }
}
```

### 2. Bilimsel Danışman Profili
```json
{
  "profile_name": "bilimsel-danisman", 
  "base_model": "codellama/CodeLlama-7b-hf",
  "training_strategy": "lora-8bit",
  "integrate_occult": false,
  "scientific_reasoning": {
    "methodology": "empirical_analysis",
    "principles": ["falsifiability", "reproducibility", "peer_review"]
  }
}
```

---

## 🧠 Model Eğitimi

### Eğitim Aşamaları

#### 1. Veri Hazırlama
```python
# Otomatik veri indirme ve işleme
processor = AdvancedDataProcessor(config)
raw_texts = processor.download_and_prepare_data(data_sources)
```

#### 2. Akıllı Distilasyon
- Kalite skorlama (0.0-1.0)
- Önceliklendirme
- Hedef boyut optimizasyonu

#### 3. Okült Augmentasyon
```python
# Otomatik prompt şablonu uygulama
augmented_texts = processor.apply_occult_augmentation(distilled_texts)
```

#### 4. Model Eğitimi
```python
# QLoRA + DeepSpeed optimizasyonu
trainer = OccultEnhancedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)
trainer.train()
```

### Optimizasyon İpuçları

#### P104-100 için Özel Ayarlar
```python
batch_size = 1  # 8GB VRAM için
gradient_accumulation_steps = 16  # Effective batch size = 16
max_sequence_length = 2048  # Bellek optimizasyonu
use_4bit_quantization = True  # Zorunlu
```

#### Performans Optimizasyonu
```python
gradient_checkpointing = True  # %25 bellek tasarrufu
use_flash_attention = True  # %30 hız artışı
mixed_precision = True  # FP16 eğitim
```

---

## 🐛 Sorun Giderme

### Sık Karşılaşılan Sorunlar

#### 1. GPU Bellek Hatası
```python
# Çözümler:
batch_size = 1  # Azalt
gradient_accumulation_steps = 32  # Artır
max_sequence_length = 1024  # Azalt
use_gradient_checkpointing = True  # Aktif et
```

#### 2. Model Yüklenemiyor
```bash
# Çözüm:
python ai_trainer_15.py --profile "balanced" --local_files_only false
```

#### 3. CUDA Out of Memory
```python
# Acil çözüm:
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
```

#### 4. Veri İndirme Hatası
```python
# Alternatif:
config.local_files_only = True
# Manuel olarak veriyi data/raw/ dizinine koyun
```

### Performans İpuçları

#### VRAM Optimizasyonu
```python
# P104-100 (8GB) için ideal ayarlar:
config.batch_size = 1
config.gradient_accumulation_steps = 16
config.max_sequence_length = 2048
config.use_4bit_quantization = True
config.gradient_checkpointing = True
```

#### Eğitim Hızlandırma
```python
config.use_flash_attention = True
config.mixed_precision = True
config.use_deepspeed = True
config.num_gpus = 2  # 2× P104-100
```

---

## 🤝 Katkıda Bulunma

### Geliştirme Kurulumu
```bash
# Geliştirme bağımlılıkları
pip install -U black flake8 mypy pytest pre-commit

# Kod formatı
black ai_trainer_15.py

# Testler
pytest tests/
```

### Katkı Yönergeleri
1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Branch'i push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

---

## 📞 Destek

### Yararlı Bağlantılar
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [DeepSpeed Documentation](https://deepspeed.ai/)

### Soru & Cevap
**S: P104-100 dışında GPU kullanabilir miyim?**  
C: Evet, batch size ve sequence length'i GPU belleğinize göre ayarlayın.

**S: Kendi veri setimi nasıl kullanırım?**  
C: Verilerinizi `data/raw/` dizinine koyun ve `--data "data/raw"` argümanını kullanın.

**S: Model eğitimi ne kadar sürer?**  
C: Veri boyutuna ve GPU sayısına bağlı. 2B token için 2× P104-100'de ~24-48 saat.

---

## 📜 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

## 🙏 Teşekkür

- **HuggingFace** 🤗 - Transformers ve datasets kütüphaneleri
- **Microsoft** - DeepSpeed optimizasyonu
- **Meta AI** - LLaMA model mimarisi
- **Hermetik Gelenek** - Okült bilgelik entegrasyonu

---

**🔮 HERMES ULTIMATE V15.0 - Akıl ve Bilgelik Sentezi**  
*"Kadim bilgelik ile modern teknolojiyi birleştiren AI rehberi"*

## 3. Ek Dosyalar

Ayrıca şu dosyaları da oluşturmanızı öneririm:

### `.gitignore`
```
# Veri dosyaları
data/raw/*
data/distilled/*
data/augmented/*
data/cache/*

# Model dosyaları
models/*
!models/.gitkeep

# Loglar
logs/*

# Checkpoints
checkpoints/*

# Profiller (opsiyonel - backup için saklanabilir)
# profiles/*

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
```

### `requirements.txt` (Opsiyonel - pip için)
```
transformers>=4.36.2
datasets>=2.15.0
accelerate>=0.25.0
peft>=0.7.1
bitsandbytes>=0.41.3
deepspeed>=0.12.6
rich>=13.7.0
zstandard>=0.22.0
torch>=2.1.2
pandas>=2.0.3
numpy>=1.24.3
scikit-learn>=1.3.2
tqdm>=4.66.1
wandb>=0.16.1
tensorboard>=2.15.1
psutil>=5.9.6
requests>=2.31.0
beautifulsoup4>=4.12.2
html2text>=2020.1.16
```

## 🚀 Kurulum ve Kullanım Özeti

1. **Miniconda kurulu olduğundan emin olun**
2. **`install_hermes.bat` dosyasını çalıştırın**
3. **Ortamı aktif edin:** `conda activate ai-env`
4. **Programı başlatın:** `python ai_trainer_15.py`
5. **Konfigürasyon editöründen ayarlarınızı yapın**
6. **Veri kaynaklarınızı belirleyin**
7. **Eğitimi başlatın!**


Bu kurulum, HERMES Ultimate V15.0'ı tam fonksiyonlu olarak çalıştıracak ve tüm gelişmiş özellikleri kullanmanıza olanak sağlayacaktır! 🎉

## ☕ Destek Olun / Support

Projemi beğendiyseniz, bana bir kahve ısmarlayarak destek olabilirsiniz!

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://buymeacoffee.com/metatronslove)

Teşekkürler! 🙏
