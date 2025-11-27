@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo.
echo ==================================================
echo    HERMES ULTIMATE V15.0 - KURULUM SİSTEMİ
echo ==================================================
echo.

echo [1/6] Miniconda ortamı kontrol ediliyor...
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Miniconda bulunamadı!
    echo 📥 Lütfen Miniconda yükleyin: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo ✅ Miniconda bulundu

echo [2/6] ai-env ortamı kontrol ediliyor...
conda info --envs | findstr "ai-env" >nul
if %errorlevel% neq 0 (
    echo 🔄 Yeni conda ortamı oluşturuluyor: ai-env
    conda create -n ai-env python=3.10 -y
) else (
    echo ✅ ai-env ortamı zaten mevcut
)

echo [3/6] Conda paketleri yükleniyor...
call conda activate ai-env
if %errorlevel% neq 0 (
    echo ❌ ai-env ortamı aktifleştirilemedi
    pause
    exit /b 1
)

conda install -y -c conda-forge -c pytorch -c nvidia ^
    python=3.10 ^
    pytorch=2.1.2 ^
    torchvision=0.16.2 ^
    torchaudio=2.1.2 ^
    pytorch-cuda=11.8 ^
    cudatoolkit=11.8 ^
    numpy=1.24.3 ^
    pandas=2.0.3 ^
    matplotlib=3.7.2 ^
    jupyter=1.0.0 ^
    pip=23.3.1

echo ✅ Conda paketleri yüklendi

echo [4/6] Pip paketleri yükleniyor...
pip install --upgrade pip

pip install -U ^
    transformers==4.36.2 ^
    datasets==2.15.0 ^
    accelerate==0.25.0 ^
    peft==0.7.1 ^
    bitsandbytes==0.41.3 ^
    deepspeed==0.12.6 ^
    rich==13.7.0 ^
    zstandard==0.22.0 ^
    scikit-learn==1.3.2 ^
    tqdm==4.66.1 ^
    wandb==0.16.1 ^
    tensorboard==2.15.1 ^
    psutil==5.9.6 ^
    requests==2.31.0 ^
    beautifulsoup4==4.12.2 ^
    html2text==2020.1.16

echo ✅ Pip paketleri yüklendi

echo [5/6] Dizin yapısı oluşturuluyor...
if not exist "data\raw" mkdir data\raw
if not exist "data\distilled" mkdir data\distilled
if not exist "data\augmented" mkdir data\augmented
if not exist "data\cache" mkdir data\cache
if not exist "models\hermes_final" mkdir models\hermes_final
if not exist "profiles" mkdir profiles
if not exist "logs" mkdir logs
if not exist "configs" mkdir configs
if not exist "checkpoints" mkdir checkpoints

echo ✅ Dizin yapısı oluşturuldu

echo [6/6] Sistem test ediliyor...
python -c "
import sys
try:
    import torch, transformers, datasets, peft, bitsandbytes, rich, deepspeed
    print('✅ Tüm kütüphaneler başarıyla yüklendi!')
    print(f'   PyTorch: {torch.__version__}')
    print(f'   CUDA Kullanılabilir: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   GPU Sayısı: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
    else:
        print('   ⚠️ CUDA kullanılamıyor - CPU modunda çalışılacak')
except ImportError as e:
    print(f'❌ Hata: {e}')
    sys.exit(1)
"

echo.
echo ==================================================
echo    KURULUM BAŞARIYLA TAMAMLANDI! 🎉
echo ==================================================
echo.
echo 🚀 HERMES'i başlatmak için:
echo    conda activate ai-env
echo    python ai_trainer_15.py
echo.
echo 📚 Detaylı kullanım için README.md dosyasını okuyun
echo.

timeout /t 10