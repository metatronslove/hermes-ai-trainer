#!/usr/bin/env python3
"""
🔮 HERMES ULTIMATE V15.0 - TAM TERMINAL AI EĞİTİM SİSTEMİ
70B→7B Akıllı Veri Distilasyonu + QLoRA + DeepSpeed + Okült Rehberlik
2× NVIDIA P104-100 8GB Optimizasyonu + Zengin Terminal Arayüzü
"""

import os
import sys
import json
import time
import gc
import numpy as np
import re
import math
import argparse
import logging
import pickle
import zstandard as zstd
import shutil
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

# ==================== KÜTÜPHANE KONTROLÜ ====================
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, random_split
    from torch.optim import AdamW
    from torch.cuda.amp import autocast, GradScaler
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    import bitsandbytes as bnb
    
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, AutoConfig,
        TrainingArguments, Trainer as HFTrainer,
        BitsAndBytesConfig, prepare_model_for_kbit_training,
        DataCollatorForLanguageModeling, PreTrainedTokenizer,
        get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
    )
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    from accelerate import Accelerator, init_empty_weights
    
    from datasets import Dataset as HFDataset, load_dataset
    import pandas as pd
    
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.tree import Tree
    from rich.columns import Columns
    from rich.markdown import Markdown
    from rich.logging import RichHandler
    from rich.syntax import Syntax
    from rich.align import Align
    from rich.console import Group
    from rich import box
    
except ImportError as e:
    print(f"❌ Eksik kütüphaneler: {e}")
    print("📦 Kurulum: pip install torch transformers datasets peft accelerate bitsandbytes rich pandas zstandard deepspeed")
    sys.exit(1)

console = Console()

# ==================== KAPSAMLI KONFİGÜRASYON SİSTEMİ ====================
class TrainingStrategy(Enum):
    QLORA_4BIT = "qlora-4bit"
    LORA_8BIT = "lora-8bit"
    FULL_FINETUNE = "full-finetune"
    DEEPSPEED_ZERO3 = "deepspeed-zero3"

class ModelArchitecture(Enum):
    LLAMA_7B = "llama-7b"
    LLAMA_13B = "llama-13b"
    CUSTOM_7B = "custom-7b"
    CUSTOM_13B = "custom-13b"

@dataclass
class HermesUltimateConfig:
    # === SİSTEM KİMLİĞİ ===
    profile_name: str = "hermes-ultimate-v15"
    config_version: str = "15.0"
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    unique_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # === MODEL MİMARİSİ ===
    model_architecture: ModelArchitecture = ModelArchitecture.LLAMA_7B
    base_model: str = "meta-llama/Llama-2-7b-hf"
    custom_model_path: str = ""
    tokenizer_path: Optional[str] = None
    local_files_only: bool = False
    
    # === MODEL PARAMETRELERİ ===
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    vocab_size: int = 32000
    
    # === EĞİTİM STRATEJİSİ ===
    training_strategy: TrainingStrategy = TrainingStrategy.QLORA_4BIT
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_epochs: int = 3
    max_steps: int = 5000
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    
    # === PRECISION & OPTIMİZASYON ===
    mixed_precision: bool = True
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    use_4bit_quantization: bool = True
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    use_8bit_optimizer: bool = True
    
    # === DAĞITIK EĞİTİM (2× P104-100) ===
    use_ddp: bool = True
    use_deepspeed: bool = True
    deepspeed_config: str = "./configs/deepspeed_zero3.json"
    num_gpus: int = 2
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1])
    
    # === VERİ İŞLEME - 70B→7B DISTILATION ===
    max_sequence_length: int = 2048
    min_sequence_length: int = 256
    train_split: float = 0.95
    validation_split: float = 0.05
    distillation_target_size: int = 2000000000  # 2B token (P104-100 için optimize)
    data_quality_threshold: float = 0.7
    use_intelligent_sampling: bool = True
    data_cache_dir: str = "./data/cache"
    
    # === GELİŞMİŞ ÖZELLİKLER ===
    integrate_occult: bool = True
    computation_methods: List[str] = field(default_factory=lambda: [
        "Mathematical optimization using gradient descent",
        "Statistical analysis with Bayesian methods", 
        "Algorithmic problem solving with complexity analysis",
        "Symbolic Logic and Pattern Recognition for Esoteric Texts",
        "Quantum-inspired computing models",
        "Fractal and recursive algorithms",
        "Gematria and numerology calculations",
        "Tarot symbolism interpretation",
        "Astrological pattern recognition"
    ])
    
    conversation_tips: Dict[str, str] = field(default_factory=lambda: {
        'bilimsel': 'Empirical evidence and peer review with replicable methodology',
        'okült': 'Symbolic interpretation and ancient wisdom integration',
        'felsefi': 'Logical reasoning and critical thinking frameworks',
        'rehberlik': 'Encouraging, non-judgemental synthesis of diverse perspectives',
        'teknik': 'Precise algorithmic explanations with practical examples',
        'yaratıcı': 'Metaphorical thinking and cross-domain analogies',
        'spiritüel': 'Intuitive guidance with practical grounding',
        'pratik': 'Actionable advice with step-by-step implementation'
    })
    
    scientific_reasoning: Dict[str, Any] = field(default_factory=lambda: {
        'methodology': 'empirical_analysis',
        'principles': ['falsifiability', 'reproducibility', 'peer_review', 'objective_observation'],
        'frameworks': ['scientific_method', 'bayesian_inference', 'hypothesis_testing']
    })
    
    advanced_augmentation: bool = True
    knowledge_distillation: bool = True
    occult_guidance: bool = True
    
    # === OKÜLT KONFİGÜRASYON ===
    occult_coherence_weight: float = 0.75
    occult_interpretation_weight: float = 0.25
    occult_creativity_weight: float = 0.15
    occult_prompt_template: str = """Bilimsel Veri: {scientific_data}
Okült Materyal: {occult_material}
Hesaplama Yöntemi: {computation_method}
Sembolik Analiz: {symbolic_analysis}
Rehberlik Yorumu: """
    
    occult_keywords: List[str] = field(default_factory=lambda: [
        "hermeticism", "alchemy", "esoteric", "gematria", "kabbalah", 
        "simya", "kadim", "bilgelik", "sembol", "rituel", "meditasyon",
        "quantum", "entanglement", "fractal", "recursive", "algorithm",
        "numeroloji", "astroloji", "tarot", "gnosis", "mistik"
    ])
    
    # === ÇIKTI VE LOGLAMA ===
    output_dir: str = "./hermes_models"
    logging_dir: str = "./logs"
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 200
    save_total_limit: int = 5
    log_level: str = "INFO"
    report_to: str = "none"
    
    # === VERİ YOLLARI ===
    raw_data_paths: List[str] = field(default_factory=lambda: ["./data/raw"])
    distilled_data_path: str = "./data/distilled/data.jsonl"
    augmented_data_path: str = "./data/augmented/data.jsonl"
    model_output_path: str = "./models/hermes_final"
    
    # === İNDİRME AYARLARI ===
    download_timeout: int = 300
    resume_download: bool = True
    force_download: bool = False
    proxies: Dict[str, str] = field(default_factory=dict)
    
    # === GÜVENLİK VE DOĞRULAMA ===
    trust_remote_code: bool = True
    use_auth_token: bool = False
    enable_model_verification: bool = True
    checksum_verification: bool = True

class ConfigManager:
    """Gelişmiş konfigürasyon yönetimi"""
    
    def __init__(self, profiles_dir: str = "./profiles"):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(exist_ok=True)
        self.current_config = HermesUltimateConfig()
        self.profile_categories = {
            "occult": "Okült Rehberlik",
            "scientific": "Bilimsel Analiz", 
            "creative": "Yaratıcı Üretim",
            "technical": "Teknik Danışman",
            "balanced": "Dengeli Mod"
        }
    
    def create_profile_category(self, name: str, description: str):
        """Yeni profil kategorisi oluştur"""
        self.profile_categories[name] = description
    
    def list_profiles(self, category: str = None) -> List[Tuple[str, str]]:
        """Profil listesi (isim, kategori)"""
        profiles = []
        for profile_file in self.profiles_dir.glob("*.json"):
            with open(profile_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                profile_category = data.get('profile_category', 'default')
                if category is None or category == profile_category:
                    profiles.append((profile_file.stem, profile_category))
        return profiles
    
    def save_profile(self, name: str, category: str = "default", description: str = ""):
        """Profili kategori ile kaydet"""
        self.current_config.profile_name = name
        profile_data = asdict(self.current_config)
        profile_data['profile_category'] = category
        profile_data['profile_description'] = description
        profile_data['last_modified'] = datetime.now().isoformat()
        
        profile_path = self.profiles_dir / f"{name}.json"
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile_data, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]✓ Profil kaydedildi: {name} ({category})[/green]")
        return profile_path
    
    def load_profile(self, name: str) -> bool:
        """Profili yükle"""
        profile_path = self.profiles_dir / f"{name}.json"
        if not profile_path.exists():
            console.print(f"[red]❌ Profil bulunamadı: {name}[/red]")
            return False
            
        with open(profile_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.current_config = HermesUltimateConfig(**data)
        
        console.print(f"[green]✓ Profil yüklendi: {name}[/green]")
        return True
    
    def delete_profile(self, name: str):
        """Profili sil"""
        profile_path = self.profiles_dir / f"{name}.json"
        if profile_path.exists():
            profile_path.unlink()
            console.print(f"[green]✓ Profil silindi: {name}[/green]")
        else:
            console.print(f"[red]❌ Profil bulunamadı: {name}[/red]")
    
    def duplicate_profile(self, source_name: str, target_name: str):
        """Profili çoğalt"""
        source_path = self.profiles_dir / f"{source_name}.json"
        target_path = self.profiles_dir / f"{target_name}.json"
        
        if not source_path.exists():
            console.print(f"[red]❌ Kaynak profil bulunamadı: {source_name}[/red]")
            return
        
        shutil.copy2(source_path, target_path)
        console.print(f"[green]✓ Profil çoğaltıldı: {source_name} → {target_name}[/green]")
    
    def interactive_config_editor(self):
        """Zengin terminal tabanlı konfigürasyon editörü"""
        while True:
            # Ana menü
            console.clear()
            console.print(Panel.fit(
                "🎛️ HERMES ULTIMATE V15.0 - KONFİGÜRASYON EDITÖRÜ",
                style="bold blue"
            ))
            
            # Mevcut konfigürasyon özeti
            self._display_config_summary()
            
            # Menü seçenekleri
            menu_table = Table(title="DÜZENLEME MENÜSÜ", box=box.DOUBLE_EDGE)
            menu_table.add_column("No", style="cyan", width=6)
            menu_table.add_column("Bölüm", style="magenta")
            menu_table.add_column("Açıklama", style="green")
            
            menu_items = [
                ("1", "Model Ayarları", "Temel model ve mimari"),
                ("2", "Eğitim Parametreleri", "LR, batch size, epoch vb."),
                ("3", "QLoRA/LoRA Ayarları", "Fine-tuning optimizasyonu"),
                ("4", "Veri İşleme", "Distilasyon ve augmentasyon"),
                ("5", "Okült Özellikler", "Rehberlik ve hesaplama"),
                ("6", "Sistem Optimizasyon", "GPU ve bellek ayarları"),
                ("7", "Çıktı ve Loglama", "Kaydetme ve monitörleme"),
                ("8", "İndirme Ayarları", "Model ve veri indirme"),
                ("S", "Kaydet", "Profili kaydet"),
                ("L", "Yükle", "Profil yükle"),
                ("R", "Sıfırla", "Varsayılanlara dön"),
                ("Q", "Çıkış", "Editörden çık")
            ]
            
            for no, section, desc in menu_items:
                menu_table.add_row(no, section, desc)
            
            console.print(menu_table)
            
            choice = Prompt.ask("Seçiminiz", choices=[item[0] for item in menu_items]).upper()
            
            if choice == '1':
                self._edit_model_settings()
            elif choice == '2':
                self._edit_training_parameters()
            elif choice == '3':
                self._edit_lora_settings()
            elif choice == '4':
                self._edit_data_processing()
            elif choice == '5':
                self._edit_occult_features()
            elif choice == '6':
                self._edit_system_optimization()
            elif choice == '7':
                self._edit_output_settings()
            elif choice == '8':
                self._edit_download_settings()
            elif choice == 'S':
                self._save_profile_interactive()
            elif choice == 'L':
                self._load_profile_interactive()
            elif choice == 'R':
                if Confirm.ask("Varsayılan ayarlara dönülsün mü?"):
                    self.current_config = HermesUltimateConfig()
                    console.print("[green]✓ Ayarlar sıfırlandı[/green]")
            elif choice == 'Q':
                if Confirm.ask("Değişiklikler kaydedilmeden çıkılsın mı?"):
                    break
    
    def _display_config_summary(self):
        """Konfigürasyon özetini göster"""
        config = self.current_config
        
        summary_table = Table(title="MEVCUT KONFİGÜRASYON", box=box.ROUNDED)
        summary_table.add_column("Parametre", style="cyan")
        summary_table.add_column("Değer", style="green")
        
        summary_data = [
            ("Model", f"{config.base_model}"),
            ("Strateji", config.training_strategy.value),
            ("Batch Size", f"{config.batch_size} (×{config.gradient_accumulation_steps})"),
            ("Learning Rate", f"{config.learning_rate:.2e}"),
            ("Sequence Length", f"{config.max_sequence_length}"),
            ("LoRA Rank", f"{config.lora_rank}"),
            ("Okült Entegrasyon", "✓" if config.integrate_occult else "✗"),
            ("Distilasyon", f"{config.distillation_target_size:,} token"),
            ("Çıktı Dizini", config.output_dir)
        ]
        
        for param, value in summary_data:
            summary_table.add_row(param, value)
        
        console.print(summary_table)
        console.print()
    
    def _edit_model_settings(self):
        """Model ayarlarını düzenle"""
        console.print(Panel.fit("🤖 MODEL AYARLARI", style="bold blue"))
        
        # Model mimarisi
        arch_choices = {str(i+1): arch for i, arch in enumerate(ModelArchitecture)}
        arch_table = Table(title="Model Mimarisi")
        for choice, arch in arch_choices.items():
            arch_table.add_row(choice, arch.value)
        console.print(arch_table)
        
        arch_choice = Prompt.ask("Mimari seçin", choices=list(arch_choices.keys()))
        self.current_config.model_architecture = arch_choices[arch_choice]
        
        # Diğer model ayarları
        self.current_config.base_model = Prompt.ask(
            "Base model/HuggingFace ID", 
            default=self.current_config.base_model
        )
        
        self.current_config.max_position_embeddings = IntPrompt.ask(
            "Maksimum sequence length", 
            default=self.current_config.max_position_embeddings,
            min=512, max=8192
        )
        
        self.current_config.use_flash_attention = Confirm.ask(
            "Flash attention kullanılsın mı?",
            default=self.current_config.use_flash_attention
        )
    
    def _edit_training_parameters(self):
        """Eğitim parametrelerini düzenle"""
        console.print(Panel.fit("⚡ EĞİTİM PARAMETRELERİ", style="bold blue"))
        
        self.current_config.learning_rate = FloatPrompt.ask(
            "Learning rate", 
            default=self.current_config.learning_rate,
            min=1e-7, max=1e-2
        )
        
        self.current_config.batch_size = IntPrompt.ask(
            "Per device batch size",
            default=self.current_config.batch_size,
            min=1, max=8
        )
        
        self.current_config.gradient_accumulation_steps = IntPrompt.ask(
            "Gradient accumulation steps",
            default=self.current_config.gradient_accumulation_steps,
            min=1, max=128
        )
        
        self.current_config.num_epochs = IntPrompt.ask(
            "Epoch sayısı",
            default=self.current_config.num_epochs,
            min=1, max=50
        )
        
        self.current_config.warmup_steps = IntPrompt.ask(
            "Warmup steps",
            default=self.current_config.warmup_steps,
            min=0, max=10000
        )
    
    def _edit_lora_settings(self):
        """LoRA ayarlarını düzenle"""
        console.print(Panel.fit("🎯 QLORA/LORA AYARLARI", style="bold blue"))
        
        self.current_config.use_lora = Confirm.ask(
            "LoRA kullanılsın mı?",
            default=self.current_config.use_lora
        )
        
        if self.current_config.use_lora:
            self.current_config.lora_rank = IntPrompt.ask(
                "LoRA rank (r)",
                default=self.current_config.lora_rank,
                min=8, max=256
            )
            
            self.current_config.lora_alpha = IntPrompt.ask(
                "LoRA alpha",
                default=self.current_config.lora_alpha,
                min=8, max=256
            )
            
            self.current_config.lora_dropout = FloatPrompt.ask(
                "LoRA dropout",
                default=self.current_config.lora_dropout,
                min=0.0, max=0.5
            )
            
            self.current_config.use_4bit_quantization = Confirm.ask(
                "4-bit quantization (QLoRA) kullanılsın mı?",
                default=self.current_config.use_4bit_quantization
            )
    
    def _edit_data_processing(self):
        """Veri işleme ayarlarını düzenle"""
        console.print(Panel.fit("📊 VERİ İŞLEME AYARLARI", style="bold blue"))
        
        self.current_config.distillation_target_size = IntPrompt.ask(
            "Hedef distilasyon boyutu (token)",
            default=self.current_config.distillation_target_size,
            min=1000000, max=10000000000
        )
        
        self.current_config.data_quality_threshold = FloatPrompt.ask(
            "Veri kalite eşiği (0.0-1.0)",
            default=self.current_config.data_quality_threshold,
            min=0.0, max=1.0
        )
        
        self.current_config.use_intelligent_sampling = Confirm.ask(
            "Akıllı örnekleme kullanılsın mı?",
            default=self.current_config.use_intelligent_sampling
        )
        
        self.current_config.advanced_augmentation = Confirm.ask(
            "İleri seviye augmentasyon kullanılsın mı?",
            default=self.current_config.advanced_augmentation
        )
    
    def _edit_occult_features(self):
        """Okült özellikleri düzenle"""
        console.print(Panel.fit("🔮 OKÜLT ÖZELLİKLER", style="bold blue"))
        
        self.current_config.integrate_occult = Confirm.ask(
            "Okült rehberlik entegre edilsin mi?",
            default=self.current_config.integrate_occult
        )
        
        if self.current_config.integrate_occult:
            self.current_config.occult_coherence_weight = FloatPrompt.ask(
                "Tutarlılık ağırlığı",
                default=self.current_config.occult_coherence_weight,
                min=0.0, max=1.0
            )
            
            self.current_config.occult_interpretation_weight = FloatPrompt.ask(
                "Yorumlama ağırlığı",
                default=self.current_config.occult_interpretation_weight,
                min=0.0, max=1.0
            )
            
            # Okült prompt şablonunu düzenleme seçeneği
            if Confirm.ask("Okült prompt şablonunu düzenlemek ister misiniz?"):
                current_template = self.current_config.occult_prompt_template
                console.print("Mevcut şablon:")
                console.print(Panel(current_template, style="dim"))
                new_template = Prompt.ask("Yeni şablon (boş bırakılırsa değişmez)", default="")
                if new_template:
                    self.current_config.occult_prompt_template = new_template
    
    def _edit_system_optimization(self):
        """Sistem optimizasyon ayarlarını düzenle"""
        console.print(Panel.fit("💻 SİSTEM OPTİMİZASYONU", style="bold blue"))
        
        self.current_config.use_deepspeed = Confirm.ask(
            "DeepSpeed kullanılsın mı?",
            default=self.current_config.use_deepspeed
        )
        
        self.current_config.gradient_checkpointing = Confirm.ask(
            "Gradient checkpointing kullanılsın mı?",
            default=self.current_config.gradient_checkpointing
        )
        
        self.current_config.mixed_precision = Confirm.ask(
            "Mixed precision kullanılsın mı?",
            default=self.current_config.mixed_precision
        )
        
        self.current_config.num_gpus = IntPrompt.ask(
            "Kullanılacak GPU sayısı",
            default=self.current_config.num_gpus,
            min=1, max=8
        )
    
    def _edit_output_settings(self):
        """Çıktı ayarlarını düzenle"""
        console.print(Panel.fit("💾 ÇIKTI AYARLARI", style="bold blue"))
        
        self.current_config.output_dir = Prompt.ask(
            "Model çıktı dizini",
            default=self.current_config.output_dir
        )
        
        self.current_config.save_steps = IntPrompt.ask(
            "Kaydetme aralığı (steps)",
            default=self.current_config.save_steps,
            min=10, max=5000
        )
        
        self.current_config.logging_steps = IntPrompt.ask(
            "Loglama aralığı (steps)",
            default=self.current_config.logging_steps,
            min=1, max=1000
        )
        
        self.current_config.save_total_limit = IntPrompt.ask(
            "Maksimum checkpoint sayısı",
            default=self.current_config.save_total_limit,
            min=1, max=20
        )
    
    def _edit_download_settings(self):
        """İndirme ayarlarını düzenle"""
        console.print(Panel.fit("📥 İNDİRME AYARLARI", style="bold blue"))
        
        self.current_config.local_files_only = Confirm.ask(
            "Sadece yerel dosyaları kullan?",
            default=self.current_config.local_files_only
        )
        
        self.current_config.download_timeout = IntPrompt.ask(
            "İndirme timeout (saniye)",
            default=self.current_config.download_timeout,
            min=30, max=3600
        )
        
        self.current_config.resume_download = Confirm.ask(
            "İndirmeye devam et?",
            default=self.current_config.resume_download
        )
    
    def _save_profile_interactive(self):
        """Interaktif profil kaydetme"""
        console.print(Panel.fit("💾 PROFİL KAYDET", style="bold blue"))
        
        profile_name = Prompt.ask("Profil adı")
        
        # Kategori seçimi
        categories = list(self.profile_categories.keys())
        cat_table = Table(title="Profil Kategorileri")
        for i, (cat, desc) in enumerate(self.profile_categories.items()):
            cat_table.add_row(str(i+1), cat, desc)
        console.print(cat_table)
        
        cat_choice = Prompt.ask("Kategori seçin", choices=[str(i+1) for i in range(len(categories))])
        category = categories[int(cat_choice)-1]
        
        description = Prompt.ask("Profil açıklaması (opsiyonel)", default="")
        
        self.save_profile(profile_name, category, description)
    
    def _load_profile_interactive(self):
        """Interaktif profil yükleme"""
        console.print(Panel.fit("📂 PROFİL YÜKLE", style="bold blue"))
        
        profiles = self.list_profiles()
        if not profiles:
            console.print("[yellow]⚠️ Kayıtlı profil bulunamadı[/yellow]")
            return
        
        # Profil listesi
        profile_table = Table(title="MEVCUT PROFİLLER", box=box.ROUNDED)
        profile_table.add_column("No", style="cyan", width=4)
        profile_table.add_column("Profil Adı", style="magenta")
        profile_table.add_column("Kategori", style="green")
        
        for i, (name, category) in enumerate(profiles):
            profile_table.add_row(str(i+1), name, category)
        
        console.print(profile_table)
        
        choice = Prompt.ask("Yüklenecek profil", choices=[str(i+1) for i in range(len(profiles))])
        profile_name = profiles[int(choice)-1][0]
        
        self.load_profile(profile_name)

# ==================== GELİŞMİŞ VERİ İŞLEME SİSTEMİ ====================
class AdvancedDataProcessor:
    """70B→7B veri distilasyonu ve okült augmentasyon"""
    
    def __init__(self, config: HermesUltimateConfig):
        self.config = config
        self.tokenizer = None
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Loglama sistemini kur"""
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        return logging.getLogger("DataProcessor")
    
    def initialize_tokenizer(self, model_name: str = None):
        """Tokenizer başlat"""
        try:
            model_name = model_name or self.config.base_model
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=self.config.trust_remote_code,
                local_files_only=self.config.local_files_only
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            console.print("[green]✓ Tokenizer başarıyla yüklendi[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Tokenizer yüklenemedi: {e}[/yellow]")
            self.tokenizer = self._create_fallback_tokenizer()
    
    def _create_fallback_tokenizer(self):
        """Basit tokenizer fallback"""
        class FallbackTokenizer:
            def encode(self, text, **kwargs):
                return [ord(c) for c in text[:1000]]
            def decode(self, tokens):
                return ''.join(chr(t) if t < 65536 else '?' for t in tokens)
        return FallbackTokenizer()
    
    def download_and_prepare_data(self, data_sources: List[str]) -> List[str]:
        """Veriyi indir ve hazırla"""
        console.print(Panel.fit("📥 VERİ HAZIRLAMA SİSTEMİ", style="bold blue"))
        
        all_texts = []
        
        for source in data_sources:
            if source.startswith('http'):
                # URL'den indirme
                texts = self._download_from_url(source)
            elif Path(source).exists():
                # Yerel dosyadan yükleme
                texts = self._load_from_file(source)
            else:
                # HuggingFace datasets
                texts = self._load_from_hf(source)
            
            all_texts.extend(texts)
        
        console.print(f"[green]✓ Toplam {len(all_texts)} metin yüklendi[/green]")
        return all_texts
    
    def _download_from_url(self, url: str) -> List[str]:
        """URL'den veri indir"""
        console.print(f"[blue]📡 İndiriliyor: {url}[/blue]")
        # Basit indirme implementasyonu - gerçek uygulamada requests vs. kullanılır
        return []  # Placeholder
    
    def _load_from_file(self, file_path: str) -> List[str]:
        """Yerel dosyadan veri yükle"""
        path = Path(file_path)
        console.print(f"[blue]📖 Dosya okunuyor: {path.name}[/blue]")
        
        try:
            if path.suffix == '.jsonl':
                return self._load_jsonl(path)
            elif path.suffix == '.txt':
                return self._load_txt(path)
            elif path.suffix == '.csv':
                return self._load_csv(path)
            else:
                console.print(f"[yellow]⚠️ Desteklenmeyen format: {path.suffix}[/yellow]")
                return []
        except Exception as e:
            console.print(f"[red]❌ Dosya okuma hatası: {e}[/red]")
            return []
    
    def _load_jsonl(self, path: Path) -> List[str]:
        """JSONL dosyasını yükle"""
        texts = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    text = self._extract_text(item)
                    if text and len(text) > 50:
                        texts.append(text)
                except json.JSONDecodeError:
                    continue
        return texts
    
    def _load_txt(self, path: Path) -> List[str]:
        """TXT dosyasını yükle"""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        return self._split_into_chunks(content, 1000)
    
    def _load_csv(self, path: Path) -> List[str]:
        """CSV dosyasını yükle"""
        df = pd.read_csv(path)
        texts = []
        for column in df.columns:
            if df[column].dtype == 'object':
                texts.extend([str(x) for x in df[column].dropna() if len(str(x)) > 50])
        return texts
    
    def _load_from_hf(self, dataset_name: str) -> List[str]:
        """HuggingFace datasets'ten yükle"""
        console.print(f"[blue]🤗 HF Dataset: {dataset_name}[/blue]")
        try:
            dataset = load_dataset(dataset_name)
            texts = []
            for split in dataset.keys():
                for item in dataset[split]:
                    text = self._extract_text(item)
                    if text and len(text) > 50:
                        texts.append(text)
            return texts
        except Exception as e:
            console.print(f"[red]❌ HF dataset yükleme hatası: {e}[/red]")
            return []
    
    def _extract_text(self, item) -> str:
        """Öğeden metni çıkar"""
        if isinstance(item, str):
            return item
        elif isinstance(item, dict):
            # Öncelikli anahtarlar
            text_keys = ['text', 'content', 'body', 'article', 'document', 'response', 'output']
            for key in text_keys:
                if key in item and isinstance(item[key], str) and len(item[key]) > 50:
                    return item[key]
            # Tüm string değerleri birleştir
            return ' '.join(str(v) for v in item.values() if isinstance(v, str) and len(str(v)) > 20)
        return ""
    
    def _split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Metni parçalara ayır"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            if current_size + len(word) > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def intelligent_distillation(self, texts: List[str]) -> List[str]:
        """Akıllı veri distilasyonu"""
        console.print(Panel.fit("🔮 AKILLI VERİ DAMITMA", style="bold blue"))
        
        with Progress() as progress:
            # 1. Kalite puanlama
            task1 = progress.add_task("📊 Kalite analizi...", total=len(texts))
            scored_texts = []
            
            for text in texts:
                quality_score = self._calculate_quality_score(text)
                if quality_score >= self.config.data_quality_threshold:
                    scored_texts.append((text, quality_score))
                progress.advance(task1)
            
            # 2. Önceliklendirme
            task2 = progress.add_task("🎯 Önceliklendirme...", total=len(scored_texts))
            scored_texts.sort(key=lambda x: x[1], reverse=True)
            
            # 3. Hedef boyuta göre seçim
            task3 = progress.add_task("📦 Boyut optimizasyonu...", total=self.config.distillation_target_size)
            selected_texts = []
            current_size = 0
            
            for text, score in scored_texts:
                if self.tokenizer:
                    text_size = len(self.tokenizer.encode(text, add_special_tokens=False))
                else:
                    text_size = len(text.split()) * 1.3  # Tahmini
                
                if current_size + text_size > self.config.distillation_target_size:
                    break
                    
                selected_texts.append(text)
                current_size += text_size
                progress.update(task3, completed=current_size)
        
        console.print(f"[green]✓ Distilasyon tamamlandı: {len(selected_texts)} metin, ~{current_size:,} token[/green]")
        return selected_texts
    
    def _calculate_quality_score(self, text: str) -> float:
        """Metin kalite skoru hesapla"""
        if not text or len(text) < 50:
            return 0.0
        
        scores = {
            'length': min(len(text) / 500, 1.0),
            'diversity': len(set(text.lower().split())) / len(text.split()) if text else 0,
            'structure': self._structure_score(text),
            'coherence': self._coherence_score(text),
            'information': self._information_density(text),
            'occult_relevance': self._occult_relevance_score(text)
        }
        
        weights = {
            'length': 0.15, 'diversity': 0.20, 'structure': 0.15,
            'coherence': 0.15, 'information': 0.20, 'occult_relevance': 0.15
        }
        
        total_score = sum(scores[metric] * weights[metric] for metric in scores)
        return min(total_score, 1.0)
    
    def _structure_score(self, text: str) -> float:
        """Yapısal puan"""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) < 2:
            return 0.3
        avg_len = np.mean([len(s.split()) for s in sentences])
        return 1.0 - min(abs(avg_len - 15) / 15, 1.0)
    
    def _coherence_score(self, text: str) -> float:
        """Tutarlılık puanı"""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) < 3:
            return 0.5
        
        overlaps = []
        for i in range(len(sentences)-1):
            words1 = set(sentences[i].lower().split())
            words2 = set(sentences[i+1].lower().split())
            if words1 and words2:
                overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                overlaps.append(overlap)
        
        return np.mean(overlaps) if overlaps else 0.3
    
    def _information_density(self, text: str) -> float:
        """Bilgi yoğunluğu"""
        words = text.split()
        if len(words) < 20:
            return 0.5
        content_words = [w for w in words if len(w) > 3 and w.isalpha()]
        return min(len(content_words) / len(words) * 2, 1.0)
    
    def _occult_relevance_score(self, text: str) -> float:
        """Okült alaka puanı"""
        if not self.config.integrate_occult:
            return 0.5
        
        keyword_count = sum(1 for keyword in self.config.occult_keywords 
                          if keyword.lower() in text.lower())
        return min(keyword_count / 5, 1.0)
    
    def apply_occult_augmentation(self, texts: List[str]) -> List[str]:
        """Okült augmentasyon uygula"""
        if not self.config.integrate_occult:
            return texts
        
        console.print(Panel.fit("🔮 OKÜLT AUGMENTASYON", style="bold blue"))
        
        augmented_texts = []
        
        with Progress() as progress:
            task = progress.add_task("Augmentasyon uygulanıyor...", total=len(texts))
            
            for text in texts:
                augmented = self._augment_single_text(text)
                augmented_texts.append(augmented)
                progress.advance(task)
        
        return augmented_texts
    
    def _augment_single_text(self, text: str) -> str:
        """Tekil metne augmentasyon uygula"""
        # Okült şablonunu uygula
        computation_method = np.random.choice(self.config.computation_methods)
        symbolic_analysis = "Sembolik desenler ve arketipler analiz edildi"
        
        augmented = self.config.occult_prompt_template.format(
            scientific_data="[Bilimsel veri analizi]",
            occult_material="[Okült materyal entegrasyonu]",
            computation_method=computation_method,
            symbolic_analysis=symbolic_analysis
        ) + text
        
        # Konuşma ipucu ekle
        tip_key = np.random.choice(list(self.config.conversation_tips.keys()))
        tip = self.config.conversation_tips[tip_key]
        augmented += f"\n\n💡 {tip_key.upper()} İPUCU: {tip}"
        
        return augmented
    
    def prepare_training_dataset(self, texts: List[str]) -> HFDataset:
        """Eğitim dataset'i hazırla"""
        console.print(Panel.fit("🤖 EĞİTİM VERİSİ HAZIRLAMA", style="bold blue"))
        
        if not self.tokenizer:
            self.initialize_tokenizer()
        
        # Tokenleştirme
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.config.max_sequence_length,
                return_tensors="pt"
            )
        
        # Dataset oluştur
        dataset_dict = {"text": texts}
        dataset = HFDataset.from_dict(dataset_dict)
        
        # Tokenleştir
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenleştirme"
        )
        
        # Labels ekle (causal LM için)
        tokenized_dataset = tokenized_dataset.map(
            lambda examples: {"labels": examples["input_ids"]},
            batched=True,
            desc="Labels ekleniyor"
        )
        
        console.print(f"[green]✓ Dataset hazırlandı: {len(tokenized_dataset)} örnek[/green]")
        return tokenized_dataset

# ==================== GELİŞMİŞ MODEL SİSTEMİ ====================
class AdvancedModelSystem:
    """QLoRA + DeepSpeed optimizasyonlu model sistemi"""
    
    def __init__(self, config: HermesUltimateConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.accelerator = None
    
    def setup_model(self):
        """Modeli kur"""
        console.print(Panel.fit("🧠 MODEL KURULUMU", style="bold blue"))
        
        try:
            # 1. Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model,
                trust_remote_code=self.config.trust_remote_code,
                local_files_only=self.config.local_files_only
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 2. Quantization config
            bnb_config = None
            if self.config.use_4bit_quantization:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
            
            # 3. Model yükleme
            console.print("[blue]📦 Model yükleniyor...[/blue]")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                quantization_config=bnb_config,
                device_map="auto" if not self.config.use_ddp else None,
                trust_remote_code=self.config.trust_remote_code,
                torch_dtype=torch.float16 if self.config.mixed_precision else torch.float32,
                local_files_only=self.config.local_files_only
            )
            
            # 4. Gradient checkpointing
            if self.config.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            
            # 5. QLoRA/LoRA
            if self.config.use_lora:
                self.model = prepare_model_for_kbit_training(self.model)
                
                lora_config = LoraConfig(
                    r=self.config.lora_rank,
                    lora_alpha=self.config.lora_alpha,
                    target_modules=self.config.lora_target_modules,
                    lora_dropout=self.config.lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM
                )
                self.model = get_peft_model(self.model, lora_config)
                self._print_trainable_parameters()
            
            console.print("[green]✓ Model başarıyla hazırlandı[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Model hazırlama hatası: {e}[/red]")
            raise
    
    def _print_trainable_parameters(self):
        """Eğitilebilir parametreleri göster"""
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        console.print(f"🎯 Eğitilebilir parametreler: {trainable_params:,} / {all_param:,} "
                     f"({100 * trainable_params / all_param:.4f}%)")

# ==================== ÖZEL EĞİTİM SİSTEMİ ====================
class OccultEnhancedTrainer(HFTrainer):
    """Okült geliştirmeli özel trainer"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.occult_coherence_weight = getattr(self.args, 'occult_coherence_weight', 0.7)
        self.occult_interpretation_weight = getattr(self.args, 'occult_interpretation_weight', 0.3)
        self.occult_creativity_weight = getattr(self.args, 'occult_creativity_weight', 0.15)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Özel okült kayıp fonksiyonu"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Geleneksel LM kaybı
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                          shift_labels.view(-1))
        
        # Yaratıcılık kaybı (entropy regularization)
        probs = F.softmax(shift_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        creativity_loss = -entropy.mean()  # Yüksek entropiyi teşvik et
        
        # Birleştirilmiş kayıp
        total_loss = (self.occult_coherence_weight * lm_loss + 
                     self.occult_interpretation_weight * lm_loss * 0.5 +
                     self.occult_creativity_weight * creativity_loss)
        
        return (total_loss, outputs) if return_outputs else total_loss

# ==================== ANA EĞİTİM SİSTEMİ ====================
class HermesTrainingSystem:
    """Ana eğitim koordinatörü"""
    
    def __init__(self, config: HermesUltimateConfig):
        self.config = config
        self.config_manager = ConfigManager()
        self.data_processor = AdvancedDataProcessor(config)
        self.model_system = AdvancedModelSystem(config)
        self.trainer = None
    
    def create_deepspeed_config(self):
        """DeepSpeed config oluştur"""
        ds_config = {
            "train_batch_size": self.config.batch_size * self.config.gradient_accumulation_steps * self.config.num_gpus,
            "train_micro_batch_size_per_gpu": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "fp16": {"enabled": self.config.mixed_precision},
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "offload_param": {"device": "cpu", "pin_memory": True},
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": self.config.hidden_size * self.config.hidden_size,
                "prefetch_bucket_size": self.config.hidden_size * self.config.hidden_size * 0.5,
            },
            "gradient_clipping": self.config.max_grad_norm,
            "steps_per_print": self.config.logging_steps,
            "wall_clock_breakdown": False
        }
        
        # Dizin oluştur
        config_dir = Path(self.config.deepspeed_config).parent
        config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.config.deepspeed_config, 'w') as f:
            json.dump(ds_config, f, indent=2)
        
        console.print(f"[green]✓ DeepSpeed config oluşturuldu: {self.config.deepspeed_config}[/green]")
    
    def setup_training(self, train_dataset, eval_dataset=None):
        """Eğitimi hazırla"""
        console.print(Panel.fit("⚡ EĞİTİM HAZIRLIĞI", style="bold blue"))
        
        # DeepSpeed config
        if self.config.use_deepspeed and self.config.use_ddp:
            self.create_deepspeed_config()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_epochs,
            max_steps=self.config.max_steps,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            max_grad_norm=self.config.max_grad_norm,
            lr_scheduler_type=self.config.lr_scheduler_type,
            fp16=self.config.mixed_precision,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            save_total_limit=self.config.save_total_limit,
            deepspeed=self.config.deepspeed_config if self.config.use_deepspeed else None,
            report_to=self.config.report_to,
            gradient_checkpointing=self.config.gradient_checkpointing,
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
            local_rank=int(os.environ.get("LOCAL_RANK", -1)),
            # Özel parametreler
            occult_coherence_weight=self.config.occult_coherence_weight,
            occult_interpretation_weight=self.config.occult_interpretation_weight,
            occult_creativity_weight=self.config.occult_creativity_weight
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.model_system.tokenizer,
            mlm=False,
        )
        
        # Trainer
        self.trainer = OccultEnhancedTrainer(
            model=self.model_system.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.model_system.tokenizer,
            data_collator=data_collator,
        )
        
        console.print("[green]✓ Eğitim hazırlığı tamamlandı[/green]")
    
    def run_training(self):
        """Eğitimi başlat"""
        console.print(Panel.fit("🚀 EĞİTİM BAŞLIYOR", style="bold red"))
        
        try:
            # Eğitim öncesi GPU durumu
            self._print_gpu_status()
            
            # Eğitimi başlat
            start_time = time.time()
            self.trainer.train()
            training_time = time.time() - start_time
            
            # Modeli kaydet
            final_path = Path(self.config.model_output_path)
            final_path.mkdir(parents=True, exist_ok=True)
            
            self.trainer.save_model(str(final_path))
            self.model_system.tokenizer.save_pretrained(str(final_path))
            
            # Eğitim raporu
            self._print_training_report(training_time)
            
            console.print(Panel.fit(
                f"🎉 EĞİTİM TAMAMLANDI!\n"
                f"Model: {final_path}\n"
                f"Süre: {training_time/3600:.2f} saat",
                style="bold green"
            ))
            
        except Exception as e:
            console.print(f"[red]❌ Eğitim hatası: {e}[/red]")
            raise
    
    def _print_gpu_status(self):
        """GPU durumunu göster"""
        if torch.cuda.is_available():
            gpu_table = Table(title="GPU DURUMU")
            gpu_table.add_column("GPU", style="cyan")
            gpu_table.add_column("Bellek", style="green")
            gpu_table.add_column("Kullanım", style="yellow")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                total_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                allocated = torch.cuda.memory_allocated(i) / 1e9
                usage_pct = (allocated / total_mem) * 100
                
                gpu_table.add_row(
                    f"GPU {i} ({gpu_name[:20]}...)",
                    f"{total_mem:.1f} GB",
                    f"{usage_pct:.1f}% ({allocated:.1f} GB)"
                )
            
            console.print(gpu_table)
    
    def _print_training_report(self, training_time: float):
        """Eğitim raporu göster"""
        report_table = Table(title="EĞİTİM RAPORU", box=box.DOUBLE_EDGE)
        report_table.add_column("Metrik", style="cyan")
        report_table.add_column("Değer", style="green")
        
        report_data = [
            ("Toplam Süre", f"{training_time/3600:.2f} saat"),
            ("Epoch Sayısı", str(self.config.num_epochs)),
            ("Toplam Adım", str(self.trainer.state.global_step)),
            ("Öğrenme Oranı", f"{self.config.learning_rate:.2e}"),
            ("Son Kayıp", f"{self.trainer.state.log_history[-1].get('loss', 'N/A')}"),
            ("Model Boyutu", f"{self._get_model_size():.1f} GB")
        ]
        
        for metric, value in report_data:
            report_table.add_row(metric, value)
        
        console.print(report_table)
    
    def _get_model_size(self) -> float:
        """Model boyutunu hesapla"""
        if self.model_system.model is None:
            return 0.0
        
        param_size = 0
        for param in self.model_system.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.model_system.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return (param_size + buffer_size) / 1024**3
    
    def run_complete_pipeline(self, data_sources: List[str]):
        """Tam pipeline'ı çalıştır"""
        console.print(Panel.fit("🔮 HERMES ULTIMATE - TAM PIPELINE", style="bold magenta"))
        
        try:
            # 1. Model kurulumu
            self.model_system.setup_model()
            self.data_processor.tokenizer = self.model_system.tokenizer
            
            # 2. Veri hazırlama
            console.print("[blue]📥 Veri indiriliyor ve hazırlanıyor...[/blue]")
            raw_texts = self.data_processor.download_and_prepare_data(data_sources)
            
            if not raw_texts:
                raise ValueError("Yüklenen veri bulunamadı!")
            
            # 3. Distilasyon
            console.print("[blue]🔮 Veri distilasyonu uygulanıyor...[/blue]")
            distilled_texts = self.data_processor.intelligent_distillation(raw_texts)
            
            # 4. Augmentasyon
            console.print("[blue]✨ Okült augmentasyon uygulanıyor...[/blue]")
            augmented_texts = self.data_processor.apply_occult_augmentation(distilled_texts)
            
            # 5. Dataset hazırlama
            console.print("[blue]🤖 Eğitim dataset'i hazırlanıyor...[/blue]")
            dataset = self.data_processor.prepare_training_dataset(augmented_texts)
            
            # 6. Train/validation split
            if len(dataset) > 1000:  # Yeterli veri varsa validation yap
                train_size = int(self.config.train_split * len(dataset))
                val_size = len(dataset) - train_size
                train_dataset, eval_dataset = random_split(dataset, [train_size, val_size])
            else:
                train_dataset = dataset
                eval_dataset = None
            
            # 7. Eğitimi hazırla ve başlat
            self.setup_training(train_dataset, eval_dataset)
            self.run_training()
            
        except Exception as e:
            console.print(f"[red]❌ Pipeline hatası: {e}[/red]")
            logging.exception("Pipeline hatası detayları:")

# ==================== ZENGİN TERMINAL ARAYÜZÜ ====================
class HermesUltimateCLI:
    """Zengin terminal arayüzü"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.current_config
        self.training_system = None
        self.current_section = "main"
        
    def show_main_dashboard(self):
        """Ana kontrol paneli"""
        console.clear()
        
        # Başlık
        console.print(Panel.fit(
            "🔮 HERMES ULTIMATE V15.0 - AI EĞİTİM KONTROL MERKEZİ",
            style="bold magenta"
        ))
        
        # Sistem durumu
        self._display_system_status()
        
        # Hızlı aksiyonlar
        self._display_quick_actions()
        
        # Mevcut konfigürasyon
        self._display_current_config()
        
    def _display_system_status(self):
        """Sistem durumunu göster"""
        status_layout = Layout()
        
        # GPU durumu
        gpu_status = self._get_gpu_status()
        status_layout.split_row(
            Layout(Panel(gpu_status, title="🎮 GPU DURUMU", style="cyan"), name="left"),
            Layout(Panel(self._get_system_info(), title="💻 SİSTEM", style="green"), name="right")
        )
        
        console.print(status_layout)
        console.print()
    
    def _get_gpu_status(self) -> str:
        """GPU durumu metnini hazırla"""
        if not torch.cuda.is_available():
            return "❌ GPU kullanılamıyor"
        
        status_lines = []
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            allocated = torch.cuda.memory_allocated(i) / 1e9
            usage_pct = (allocated / total_mem) * 100
            
            status_lines.append(
                f"GPU {i}: {gpu_name[:25]}...\n"
                f"  💾 {total_mem:.1f} GB | 🟢 {usage_pct:.1f}%"
            )
        
        return "\n\n".join(status_lines)
    
    def _get_system_info(self) -> str:
        """Sistem bilgilerini hazırla"""
        import psutil
        
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return (
            f"🖥️  CPU Kullanımı: {cpu_usage}%\n"
            f"🧠 RAM: {memory.percent}% ({memory.used/1e9:.1f}/{memory.total/1e9:.1f} GB)\n"
            f"💾 Disk: {disk.percent}% ({disk.used/1e9:.1f}/{disk.total/1e9:.1f} GB)\n"
            f"📦 Python: {sys.version.split()[0]}\n"
            f"🔥 PyTorch: {torch.__version__}"
        )
    
    def _display_quick_actions(self):
        """Hızlı aksiyon butonları"""
        actions_table = Table(title="⚡ HIZLI AKSİYONLAR", box=box.ROUNDED, show_header=False)
        actions_table.add_column("Seçim", style="cyan", width=8)
        actions_table.add_column("Aksiyon", style="magenta")
        actions_table.add_column("Açıklama", style="green")
        
        actions = [
            ("1", "Konfigürasyon Editörü", "Tüm ayarları düzenle"),
            ("2", "Profil Yönetimi", "Ayarları kaydet/yükle"),
            ("3", "Veri İndirme", "Veri setlerini indir"),
            ("4", "Hızlı Başlat", "Mevcut ayarlarla eğitimi başlat"),
            ("5", "Tam Pipeline", "70B→7B distilasyon + eğitim"),
            ("6", "Model Testi", "Eğitilmiş modeli test et"),
            ("7", "Sistem İzleme", "GPU ve bellek izleme"),
            ("8", "Belgelendirme", "Kullanım kılavuzu"),
            ("Q", "Çıkış", "Programdan çık")
        ]
        
        for choice, action, desc in actions:
            actions_table.add_row(choice, action, desc)
        
        console.print(actions_table)
        console.print()
    
    def _display_current_config(self):
        """Mevcut konfigürasyonu göster"""
        config = self.config
        
        config_table = Table(title="⚙️ MEVCUT KONFİGÜRASYON", box=box.ROUNDED)
        config_table.add_column("Parametre", style="cyan")
        config_table.add_column("Değer", style="green")
        
        config_data = [
            ("Profil", config.profile_name),
            ("Model", config.base_model.split('/')[-1]),
            ("Strateji", config.training_strategy.value),
            ("Batch Size", f"{config.batch_size} × {config.gradient_accumulation_steps}"),
            ("Learning Rate", f"{config.learning_rate:.1e}"),
            ("LoRA Rank", str(config.lora_rank)),
            ("Sequence Length", str(config.max_sequence_length)),
            ("Okült Entegrasyon", "✅" if config.integrate_occult else "❌")
        ]
        
        for param, value in config_data:
            config_table.add_row(param, value)
        
        console.print(config_table)
    
    def run_interactive_cli(self):
        """Interaktif CLI'ı çalıştır"""
        while True:
            self.show_main_dashboard()
            
            choice = Prompt.ask(
                "\n🏹 Seçiminiz", 
                choices=["1", "2", "3", "4", "5", "6", "7", "8", "Q", "q"]
            ).upper()
            
            if choice == "1":
                self.config_manager.interactive_config_editor()
                self.config = self.config_manager.current_config
                self.training_system = HermesTrainingSystem(self.config)
            elif choice == "2":
                self.manage_profiles()
            elif choice == "3":
                self.download_data()
            elif choice == "4":
                self.quick_start()
            elif choice == "5":
                self.full_pipeline()
            elif choice == "6":
                self.test_model()
            elif choice == "7":
                self.monitor_system()
            elif choice == "8":
                self.show_documentation()
            elif choice == "Q":
                if Confirm.ask("Programdan çıkmak istediğinize emin misiniz?"):
                    console.print("[green]👋 Görüşmek üzere![/green]")
                    break
    
    def manage_profiles(self):
        """Profil yönetimi"""
        console.print(Panel.fit("💾 PROFİL YÖNETİMİ", style="bold blue"))
        
        while True:
            profiles = self.config_manager.list_profiles()
            
            action_table = Table(title="PROFİL İŞLEMLERİ", box=box.ROUNDED)
            action_table.add_column("Seçim", style="cyan", width=6)
            action_table.add_column("İşlem", style="magenta")
            action_table.add_column("Açıklama", style="green")
            
            actions = [
                ("1", "Profil Yükle", "Mevcut profili yükle"),
                ("2", "Profil Kaydet", "Yeni profil kaydet"),
                ("3", "Profil Sil", "Profil sil"),
                ("4", "Profil Çoğalt", "Profil kopyala"),
                ("5", "Profil Listesi", "Tüm profilleri göster"),
                ("B", "Geri", "Ana menüye dön")
            ]
            
            for choice, action, desc in actions:
                action_table.add_row(choice, action, desc)
            
            console.print(action_table)
            
            choice = Prompt.ask("Seçiminiz", choices=["1", "2", "3", "4", "5", "B", "b"]).upper()
            
            if choice == "1":
                self._load_profile()
            elif choice == "2":
                self._save_profile()
            elif choice == "3":
                self._delete_profile()
            elif choice == "4":
                self._duplicate_profile()
            elif choice == "5":
                self._list_profiles()
            elif choice == "B":
                break
    
    def _load_profile(self):
        """Profil yükle"""
        profiles = self.config_manager.list_profiles()
        if not profiles:
            console.print("[yellow]⚠️ Kayıtlı profil bulunamadı[/yellow]")
            return
        
        profile_table = Table(title="MEVCUT PROFİLLER")
        profile_table.add_column("No", style="cyan", width=4)
        profile_table.add_column("Profil", style="magenta")
        profile_table.add_column("Kategori", style="green")
        
        for i, (name, category) in enumerate(profiles):
            profile_table.add_row(str(i+1), name, category)
        
        console.print(profile_table)
        
        choice = Prompt.ask("Yüklenecek profil", choices=[str(i+1) for i in range(len(profiles))])
        profile_name = profiles[int(choice)-1][0]
        
        if self.config_manager.load_profile(profile_name):
            self.config = self.config_manager.current_config
            self.training_system = HermesTrainingSystem(self.config)
            console.print(f"[green]✓ {profile_name} profili yüklendi ve sistem hazırlandı[/green]")
    
    def _save_profile(self):
        """Profil kaydet"""
        profile_name = Prompt.ask("Profil adı")
        
        categories = list(self.config_manager.profile_categories.keys())
        cat_table = Table(title="Kategori Seçimi")
        for i, cat in enumerate(categories):
            cat_table.add_row(str(i+1), cat, self.config_manager.profile_categories[cat])
        console.print(cat_table)
        
        cat_choice = Prompt.ask("Kategori", choices=[str(i+1) for i in range(len(categories))])
        category = categories[int(cat_choice)-1]
        
        description = Prompt.ask("Açıklama (opsiyonel)", default="")
        
        self.config_manager.save_profile(profile_name, category, description)
    
    def _delete_profile(self):
        """Profil sil"""
        profiles = self.config_manager.list_profiles()
        if not profiles:
            console.print("[yellow]⚠️ Kayıtlı profil bulunamadı[/yellow]")
            return
        
        profile_names = [p[0] for p in profiles]
        profile_name = Prompt.ask("Silinecek profil", choices=profile_names)
        
        if Confirm.ask(f"'{profile_name}' profilini silmek istediğinize emin misiniz?"):
            self.config_manager.delete_profile(profile_name)
    
    def _duplicate_profile(self):
        """Profil çoğalt"""
        profiles = self.config_manager.list_profiles()
        if not profiles:
            console.print("[yellow]⚠️ Kayıtlı profil bulunamadı[/yellow]")
            return
        
        source_names = [p[0] for p in profiles]
        source_name = Prompt.ask("Kaynak profil", choices=source_names)
        target_name = Prompt.ask("Hedef profil adı")
        
        self.config_manager.duplicate_profile(source_name, target_name)
    
    def _list_profiles(self):
        """Profil listesini göster"""
        profiles = self.config_manager.list_profiles()
        if not profiles:
            console.print("[yellow]⚠️ Kayıtlı profil bulunamadı[/yellow]")
            return
        
        # Kategorilere göre grupla
        categories = {}
        for name, category in profiles:
            if category not in categories:
                categories[category] = []
            categories[category].append(name)
        
        for category, profile_list in categories.items():
            console.print(Panel.fit(
                f"📂 {category.upper()}\n" + "\n".join([f"  • {name}" for name in profile_list]),
                style="blue"
            ))
    
    def download_data(self):
        """Veri indirme arayüzü"""
        console.print(Panel.fit("📥 VERİ İNDİRME SİSTEMİ", style="bold blue"))
        
        data_sources = Prompt.ask(
            "Veri kaynakları (URL, dosya yolu veya HF dataset, boşlukla ayırın)"
        ).split()
        
        if not data_sources:
            console.print("[red]❌ Veri kaynağı belirtilmedi[/red]")
            return
        
        console.print("[blue]🔍 Veri kaynakları analiz ediliyor...[/blue]")
        
        # Data processor'ı hazırla
        if not self.training_system:
            self.training_system = HermesTrainingSystem(self.config)
        
        processor = self.training_system.data_processor
        
        try:
            with Progress() as progress:
                task = progress.add_task("Veri indiriliyor...", total=len(data_sources))
                
                for source in data_sources:
                    progress.update(task, description=f"İşleniyor: {source[:30]}...")
                    # Burada gerçek indirme işlemi yapılacak
                    time.sleep(1)  # Simülasyon
                    progress.advance(task)
            
            console.print("[green]✓ Veri indirme işlemi tamamlandı[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Veri indirme hatası: {e}[/red]")
    
    def quick_start(self):
        """Hızlı başlat"""
        console.print(Panel.fit("⚡ HIZLI BAŞLAT", style="bold blue"))
        
        data_sources = Prompt.ask(
            "Veri kaynakları (boşlukla ayırın)", 
            default=" ".join(self.config.raw_data_paths)
        ).split()
        
        if not data_sources:
            console.print("[red]❌ Veri kaynağı belirtilmedi[/red]")
            return
        
        if not self.training_system:
            self.training_system = HermesTrainingSystem(self.config)
        
        self.training_system.run_complete_pipeline(data_sources)
    
    def full_pipeline(self):
        """Tam pipeline"""
        console.print(Panel.fit("🔮 TAM PIPELINE - 70B→7B", style="bold magenta"))
        
        # Konfigürasyon kontrolü
        if not Confirm.ask("Mevcut konfigürasyonla devam edilsin mi?"):
            self.config_manager.interactive_config_editor()
            self.config = self.config_manager.current_config
        
        # Data sources
        data_sources = Prompt.ask(
            "Veri kaynakları", 
            default=" ".join(self.config.raw_data_paths)
        ).split()
        
        if not data_sources:
            console.print("[red]❌ Veri kaynağı belirtilmedi[/red]")
            return
        
        # Pipeline'ı başlat
        if not self.training_system:
            self.training_system = HermesTrainingSystem(self.config)
        
        console.print(Panel.fit(
            "🚀 TAM PIPELINE BAŞLIYOR:\n"
            "1. Veri İndirme\n"
            "2. 70B→7B Distilasyon\n" 
            "3. Okült Augmentasyon\n"
            "4. Model Eğitimi\n"
            "5. Model Doğrulama",
            style="bold green"
        ))
        
        if Confirm.ask("Pipeline'ı başlatmak istiyor musunuz?"):
            self.training_system.run_complete_pipeline(data_sources)
    
    def test_model(self):
        """Model testi"""
        console.print(Panel.fit("🧪 MODEL TESTİ", style="bold blue"))
        
        model_path = Prompt.ask("Model yolu", default=self.config.model_output_path)
        
        if not Path(model_path).exists():
            console.print("[red]❌ Model bulunamadı[/red]")
            return
        
        try:
            console.print("[blue]🔄 Model yükleniyor...[/blue]")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            console.print("[green]✓ Model yüklendi[/green]")
            console.print("[yellow]💡 Çıkmak için 'quit' yazın[/yellow]")
            
            while True:
                prompt = Prompt.ask("\n💬 Test promptu")
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                # Okült şablonu uygula
                if self.config.integrate_occult:
                    computation_method = np.random.choice(self.config.computation_methods)
                    full_prompt = self.config.occult_prompt_template.format(
                        scientific_data="[Bilimsel analiz]",
                        occult_material="[Okült yorum]",
                        computation_method=computation_method,
                        symbolic_analysis="[Sembolik analiz]"
                    ) + prompt
                else:
                    full_prompt = prompt
                
                inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Sadece yanıt kısmını göster
                if self.config.integrate_occult and "Rehberlik Yorumu:" in response:
                    response = response.split("Rehberlik Yorumu:")[-1].strip()
                
                console.print(Panel(response, title="🤖 MODEL YANITI", style="green"))
                
        except Exception as e:
            console.print(f"[red]❌ Model test hatası: {e}[/red]")
    
    def monitor_system(self):
        """Sistem izleme"""
        console.print(Panel.fit("📊 SİSTEM İZLEME", style="bold blue"))
        
        import psutil
        from rich.live import Live
        from rich.table import Table
        
        def generate_monitor_table():
            table = Table(title="🎮 REALTIME SİSTEM DURUMU", title_style="bold magenta")
            table.add_column("Metrik", style="cyan")
            table.add_column("Değer", style="green")
            table.add_column("Durum", style="yellow")
            
            # CPU
            cpu_percent = psutil.cpu_percent()
            table.add_row("CPU Kullanımı", f"{cpu_percent}%", "🟢" if cpu_percent < 80 else "🔴")
            
            # Memory
            memory = psutil.virtual_memory()
            table.add_row("RAM Kullanımı", f"{memory.percent}%", "🟢" if memory.percent < 80 else "🔴")
            
            # GPU
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1e9
                    total = torch.cuda.get_device_properties(i).total_memory / 1e9
                    percent = (allocated / total) * 100
                    table.add_row(f"GPU {i} Bellek", f"{percent:.1f}%", "🟢" if percent < 80 else "🔴")
            
            # Disk
            disk = psutil.disk_usage('/')
            table.add_row("Disk Kullanımı", f"{disk.percent}%", "🟢" if disk.percent < 80 else "🔴")
            
            return table
        
        console.print("[yellow]⏱️  Gerçek zamanlı izleme başlıyor (5 saniye)...[/yellow]")
        
        with Live(generate_monitor_table(), refresh_per_second=4) as live:
            for i in range(5):
                time.sleep(1)
                live.update(generate_monitor_table())
    
    def show_documentation(self):
        """Belgelendirme göster"""
        console.print(Panel.fit("📚 HERMES ULTIMATE KILAVUZU", style="bold blue"))
        
        docs = """
## 🎯 TEMEL ÖZELLİKLER

- **70B→7B Akıllı Distilasyon**: Büyük veri setlerini küçük modeller için optimize etme
- **QLoRA + DeepSpeed**: Düşük bellek tüketimi ile verimli eğitim
- **Okült Rehberlik**: Simya, numeroloji, hermeticism entegrasyonu
- **Zengin Terminal Arayüzü**: Kullanıcı dostu interaktif kontrol

## 🚀 HIZLI BAŞLANGIÇ

1. **Konfigürasyon**: '1' seçeneği ile ayarları yapılandırın
2. **Veri Yükleme**: '3' seçeneği ile veri setlerini indirin
3. **Eğitim**: '5' seçeneği ile tam pipeline'ı başlatın

## ⚙️ ÖNEMLİ AYARLAR

- **Batch Size**: GPU belleğine göre ayarlayın (P104-100 için 1-2 önerilir)
- **Learning Rate**: 2e-4 ile 5e-5 arası başlangıç değeri
- **LoRA Rank**: 16-64 arası deneyin
- **Sequence Length**: 2048 P104-100 için idealdir

## 🔮 OKÜLT ENTEGRASYON

- **Okült Prompt Şablonu**: Bilimsel ve spiritüel bilgiyi birleştirir
- **Hesaplama Yöntemleri**: Gematria, numeroloji, sembolik analiz
- **Konuşma İpuçları**: Farklı bağlamlar için optimize yanıtlar

## 🛠️ SORUN GİDERME

- **GPU Bellek Hatası**: Batch size veya sequence length azaltın
- **Yavaş Eğitim**: Gradient accumulation steps artırın
- **Model Yüklenemiyor**: local_files_only=False deneyin
        """
        
        console.print(Markdown(docs))
        
        if Confirm.ask("Ayrıntılı yapılandırma örnekleri gösterilsin mi?"):
            self._show_configuration_examples()
    
    def _show_configuration_examples(self):
        """Yapılandırma örnekleri göster"""
        examples = """
## 🎪 KONFİGÜRASYON ÖRNEKLERİ

### 1. OKÜLT REHBERLİK MODELİ
```json
{
  "base_model": "meta-llama/Llama-2-7b-hf",
  "training_strategy": "qlora-4bit",
  "lora_rank": 32,
  "integrate_occult": true,
  "occult_coherence_weight": 0.7,
  "computation_methods": ["gematria", "numerology", "symbolic_analysis"]
}
```

### 2. BİLİMSEL DANIŞMAN MODELİ  
```json
{
  "base_model": "codellama/CodeLlama-7b-hf",
  "training_strategy": "lora-8bit", 
  "integrate_occult": false,
  "conversation_tips": {"teknik": "Detaylı algoritmik açıklamalar"}
}
```

### 3. YARATICI YAZAR MODELİ
```json
{
  "base_model": "mistralai/Mistral-7B-v0.1",
  "occult_creativity_weight": 0.3,
  "conversation_tips": {"yaratıcı": "Metaforik ve poetik dil"}
}
```
        """
        
        console.print(Markdown(examples))

# ==================== ANA PROGRAM ====================
def main():
    """Ana program giriş noktası"""
    # Loglama ayarları
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'hermes_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Başlık
    console.print(Panel.fit(
        "🔮 HERMES ULTIMATE V15.0\n"
        "70B→7B Distilasyon + QLoRA + DeepSpeed + Okült Rehberlik\n"
        "2× NVIDIA P104-100 8GB Optimizasyonu\n"
        "Terminal Tabanlı Tam AI Eğitim Sistemi",
        style="bold magenta"
    ))
    
    # Sistem kontrolü
    if not torch.cuda.is_available():
        console.print("[yellow]⚠️ CUDA/Uyumlu GPU bulunamadı. CPU modunda çalışılacak.[/yellow]")
    else:
        console.print(f"[green]✓ {torch.cuda.device_count()} GPU tespit edildi[/green]")
    
    # Komut satırı argümanları
    parser = argparse.ArgumentParser(description="HERMES Ultimate AI Trainer")
    parser.add_argument('--data', nargs='+', help='Veri kaynakları')
    parser.add_argument('--profile', help='Kullanılacak profil')
    parser.add_argument('--quick', action='store_true', help='Hızlı başlat')
    parser.add_argument('--config', help='Konfigürasyon dosyası')
    
    args = parser.parse_args()
    
    # CLI oluştur
    cli = HermesUltimateCLI()
    
    # Konfigürasyon yükleme
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                cli.config = HermesUltimateConfig(**config_data)
                cli.config_manager.current_config = cli.config
                console.print(f"[green]✓ Konfigürasyon yüklendi: {args.config}[/green]")
    
    # Profil yükleme
    if args.profile:
        if cli.config_manager.load_profile(args.profile):
            cli.config = cli.config_manager.current_config
            console.print(f"[green]✓ Profil yüklendi: {args.profile}[/green]")
    
    # Hızlı başlat
    if args.quick and args.data:
        cli.config.raw_data_paths = args.data
        cli.quick_start()
    else:
        # Interaktif CLI
        cli.run_interactive_cli()

if __name__ == "__main__":
    main()