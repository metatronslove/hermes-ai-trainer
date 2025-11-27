import subprocess
import sys
import os

def run_command(command, description):
    print(f"🔧 {description}...")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ {description} tamamlandı")
        return True
    else:
        print(f"❌ {description} başarısız: {result.stderr}")
        return False

def main():
    print("🔮 HERMES ULTIMATE - MANUEL KURULUM")
    print("=" * 50)
    
    # Conda paketleri
    conda_packages = [
        "python=3.10",
        "pytorch=2.1.2",
        "torchvision=0.16.2", 
        "torchaudio=2.1.2",
        "pytorch-cuda=11.8",
        "cudatoolkit=11.8",
        "numpy=1.24.3",
        "pandas=2.0.3",
        "matplotlib=3.7.2",
        "jupyter=1.0.0",
        "pip=23.3.1"
    ]
    
    # Pip paketleri
    pip_packages = [
        "transformers==4.36.2",
        "datasets==2.15.0",
        "accelerate==0.25.0",
        "peft==0.7.1", 
        "bitsandbytes==0.41.3",
        "deepspeed==0.12.6",
        "rich==13.7.0",
        "zstandard==0.22.0",
        "scikit-learn==1.3.2",
        "tqdm==4.66.1",
        "wandb==0.16.1",
        "tensorboard==2.15.1",
        "psutil==5.9.6",
        "requests==2.31.0",
        "beautifulsoup4==4.12.2",
        "html2text==2020.1.16"
    ]
    
    # Conda paketlerini yükle
    for package in conda_packages:
        cmd = f'conda install -y -c conda-forge -c pytorch -c nvidia {package}'
        if not run_command(cmd, f"{package} yükleniyor"):
            print("Kurulum başarısız!")
            return
    
    # Pip paketlerini yükle
    run_command("pip install --upgrade pip", "Pip güncelleniyor")
    for package in pip_packages:
        cmd = f'pip install -U {package}'
        if not run_command(cmd, f"{package} yükleniyor"):
            print("Pip kurulumunda hata!")
    
    # Dizinleri oluştur
    directories = [
        "data/raw", "data/distilled", "data/augmented", "data/cache",
        "models/hermes_final", "profiles", "logs", "configs", "checkpoints"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ {directory} dizini oluşturuldu")
    
    # Test
    print("🧪 Sistem test ediliyor...")
    try:
        import torch, transformers, datasets, peft, bitsandbytes, rich, deepspeed
        print("✅ Tüm kütüphaneler başarıyla yüklendi!")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.device_count()} adet")
    except ImportError as e:
        print(f"❌ İçe aktarma hatası: {e}")
    
    print("\n🎉 KURULUM TAMAMLANDI!")

if __name__ == "__main__":
    main()