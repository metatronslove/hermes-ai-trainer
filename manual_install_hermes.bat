@echo off
chcp 65001 >nul

echo ==================================================
echo    HERMES MANUEL KURULUM
echo ==================================================

echo 1. Conda paketleri yükleniyor...
conda install -y -c conda-forge -c pytorch -c nvidia python=3.10 pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 pytorch-cuda=11.8 cudatoolkit=11.8 numpy=1.24.3 pandas=2.0.3 matplotlib=3.7.2 jupyter=1.0.0 pip=23.3.1

echo 2. Pip paketleri yükleniyor...
pip install --upgrade pip
pip install transformers==4.36.2 datasets==2.15.0 accelerate==0.25.0 peft==0.7.1 bitsandbytes==0.41.3 deepspeed==0.12.6 rich==13.7.0 zstandard==0.22.0 scikit-learn==1.3.2

echo 3. Ek pip paketleri yükleniyor...
pip install tqdm==4.66.1 wandb==0.16.1 tensorboard==2.15.1 psutil==5.9.6 requests==2.31.0 beautifulsoup4==4.12.2 html2text==2020.1.16

echo 4. Dizin yapısı oluşturuluyor...
mkdir data\raw 2>nul
mkdir data\distilled 2>nul
mkdir data\augmented 2>nul
mkdir data\cache 2>nul
mkdir models\hermes_final 2>nul
mkdir profiles 2>nul
mkdir logs 2>nul
mkdir configs 2>nul
mkdir checkpoints 2>nul

echo 5. Test ediliyor...
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo.
echo ✅ Manuel kurulum tamamlandı!
pause