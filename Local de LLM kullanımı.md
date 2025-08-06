## Yerel (Local) LLM Çalıştırma Rehberi  
*Türkçe konuşanlar için hazırlanmıştır.  Her adımda kısa açıklamalar ve örnek komutlar bulacaksınız.*

---

## 1. Neye İhtiyacınız Var?

| Bileşen | Minimum Gereksinim | Önerilen/İdeal |
|---------|-------------------|----------------|
| **CPU** | 8‑core (Intel i7 / AMD Ryzen 7) | 12‑core+ |
| **GPU** | 8 GB VRAM (NVIDIA GTX 1660‑Ti vb.) | 12‑24 GB VRAM (RTX 3080/3090, RTX A6000, RTX 4060‑Ti 16 GB vb.) |
| **RAM** | 16 GB | 32 GB+ |
| **Depolama** | 100 GB SSD (model dosyaları sıkıştırılmış) | 500 GB+ NVMe (özellikle 70 GB‑100 GB modeller için) |
| **OS** | Windows 10/11, Ubuntu 20.04+, macOS 12+ (Apple Silicon için farklı çözümler) |
| **Software** | • Python 3.9‑3.11  <br>• CUDA 12.x (GPU’nuz NVIDIA ise) <br>• PyTorch (CUDA destekli) <br>• Git  <br>• (İsteğe bağlı) Docker, Conda, VS Code |

> **Not:** Model lisansları ve kullanım koşulları dikkatle incelenmelidir (Meta LLaMA, Mistral‑7B, Falcon‑180B gibi). Açık‑kaynak ve ticari olmayan modellerle çalışmak genellikle sorunsuzdur.

---

## 2. Çalışma Ortamını Hazırlama

### 2.1 Python ve Sanal Ortam

```bash
# Ubuntu / macOS
sudo apt-get update && sudo apt-get install -y python3-venv git

# Sanal ortam oluştur
python3 -m venv llm-env
source llm-env/bin/activate   # Windows: .\llm-env\Scripts\activate

# pip'i güncelle
pip install --upgrade pip setuptools wheel
```

### 2.2 CUDA ve PyTorch (GPU Kullanımı)

```bash
# CUDA sürümünüzü öğrenin (nvidia-smi)
nvidia-smi
# Çıktı örneği: CUDA Version: 12.2

# PyTorch’u uygun CUDA versiyonu ile kurun
pip install torch==2.3.0+cu122 torchvision==0.18.0+cu122 torchaudio==2.3.0+cu122 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

> **Alternatif:** Eğer GPU yoksa **CPU‑only** PyTorch kurabilirsiniz:
> `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`

### 2.3 Diğer Gereksinimler

```bash
pip install transformers==4.41.0 \
    accelerate==0.30.1 \
    bitsandbytes==0.43.1 \
    einops sentencepiece tqdm
```

- **`bitsandbytes`**: 4‑bit / 8‑bit quantization (GPU‑RAM tasarrufu).  
- **`accelerate`**: Çoklu GPU / CPU dağıtımı için yardımcı araç.

---

## 3. Hangi Modeli Kullanacaksınız?

| Model | Parametre | Önerilen VRAM | Lisans | Kullanım Örneği |
|-------|-----------|---------------|-------|-----------------|
| **LLaMA‑2 7B** | 7 B | 8 GB (4‑bit) / 12 GB (FP16) | Meta Research (non‑commercial) | `meta-llama/Llama-2-7b-hf` (HuggingFace) |
| **Mistral‑7B‑Instruct** | 7 B | 8 GB (4‑bit) | Apache‑2.0 | `mistralai/Mistral-7B-Instruct-v0.2` |
| **Falcon‑7B‑Instruct** | 7 B | 8 GB (4‑bit) | Apache‑2.0 | `tiiuae/falcon-7b-instruct` |
| **Phi‑3‑mini‑4k‑instruct** | 3.8 B | 4 GB (FP16) | Apache‑2.0 | `microsoft/phi-3-mini-4k-instruct` |
| **Gemma‑2‑9B** | 9 B | 12 GB (FP16) | Apache‑2.0 | `google/gemma-2-9b` |

> **Kısa Not:** 4‑bit quantization (`bitsandbytes`) ile 7 B modelini 8 GB VRAM’da rahatça çalıştırabilirsiniz. 12 GB+ VRAM varsa FP16 (half‑precision) tercih edin, daha stabil ve hızlı olur.

---

## 4. Modeli İndirme & Quantize Etme

### 4.1 HuggingFace Hub’dan İndirme

```bash
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
cd Mistral-7B-Instruct-v0.2
```

> **Alternatif (direct `transformers` download):**
> ```python
> from transformers import AutoModelForCausalLM, AutoTokenizer
> model_name = "mistralai/Mistral-7B-Instruct-v0.2"
> tokenizer = AutoTokenizer.from_pretrained(model_name)
> model = AutoModelForCausalLM.from_pretrained(model_name)
> ```

### 4.2 4‑bit Quantization (bitsandbytes)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,          # 4‑bit quantization
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # n‑f4 (en iyi kalite‑performans dengesi)
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",          # GPU/CPU otomatik dağıtım
    quantization_config=quant_config,
    trust_remote_code=True,
)
```

**`device_map="auto"`** → Model parçaları otomatik olarak GPU‑VRAM’a sığacak şekilde yerleştirilir. Tek GPU’da çalışıyorsanız `"balanced"` veya `"sequential"` da kullanılabilir.

### 4.3 8‑bit Quantization (daha az kalite, daha hızlı)

```python
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype="float16"
)
```

---

## 5. Basit Inference (Tahmin) Örneği

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# 4‑bit quantization yapılandırması
quant_cfg = BitsAndBytesConfig(load_in_4bit=True,
                               bnb_4bit_compute_dtype="float16",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_quant_type="nf4")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quant_cfg,
    trust_remote_code=True,
)

def ask(prompt: str, max_new_tokens: int = 256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

# Örnek kullanım
print(ask("Türkiye'nin başkenti neresidir?"))
```

**Çıktı (örnek):**
```
Türkiye'nin başkenti Ankara'dır. ...
```

---

## 6. Performans & Bellek Yönetimi İpuçları

| Teknik | Açıklama | Ne Zaman Kullanmalı? |
|--------|----------|----------------------|
| **`torch.cuda.empty_cache()`** | Boş GPU bellek bloklarını temizler. | Tekrarlayan inference sırasında hafıza biriktiğinde |
| **`torch.compile()` (PyTorch 2.0+)** | JIT‑açılım, hız artışı (≈%20‑30). | Model büyükse ve CPU/GPU büyük derecede meşgulse |
| **`accelerate launch`** | Çoklu GPU dağıtımı ve mixed‑precision otomatik. | 2‑4+ GPU’ya sahipseniz |
| **`llama.cpp` / `ggml`** | C‑tabanlı, yalnızca CPU veya düşük‑VRAM GPU’da çalışır, 4‑bit/8‑bit quant. | GPU yoksa veya çok düşük VRAM (≤4 GB) varsa |
| **`Ollama` veya `LM Studio`** | GUI + API, modelleri tek komutla kurar. | Hızlı deneme ve UI istiyorsanız |

---

## 7. GUI / API Çözümleri (Opsiyonel)

### 7.1 LM Studio (Ücretsiz, Çapraz‑platform)

1. <https://lmstudio.ai> sitesinden DMG (macOS), EXE (Windows) ya da AppImage (Linux) indirin.  
2. Kurulum sonrası “Model Gallery”den **Mistral‑7B‑Instruct**, **LLaMA‑2‑7B** vb. indirect olarak seçip “Download & Load”e tıklayın.  
3. **Local Server** (REST/HTTPS) özelliğini açın → `http://127.0.0.1:1234/v1/chat/completions` gibi bir endpoint alırsınız.  
4. Python’da `openai` paketini kullanarak çağrı yapabilirsiniz:

```python
import openai

client = openai.OpenAI(base_url="http://127.0.0.1:1234/v1")
resp = client.chat.completions.create(
    model="mistral-7b-instruct",
    messages=[{"role":"user","content":"Türk tarihindeki en önemli 3 olay nedir?"}]
)
print(resp.choices[0].message.content)
```

### 7.2 Ollama (Docker + CLI)

```bash
# Linux/macOS (Docker kurulu)
curl -fsSL https://ollama.com/install.sh | sh

# Windows (PowerShell)
iwr -useb https://ollama.com/install.ps1 | iex

# Modeli indir (örn. llama2)
ollama pull llama2

# Basit CLI kullanım
ollama run llama2
```

Ollama, `ollama serve` komutuyla bir OpenAI‑compatible API sunar → aynı `openai` kodu çalışır.

---

## 8. Model Lisansları ve Etik Kurallar

1. **Meta LLaMA‑2** – yalnızca “non‑commercial” (ticari olmayan) kullanım izni.  
2. **Mistral‑7B, Falcon‑7B, Phi‑3** – Apache‑2.0 (özgür; ticari de dahil).  
3. **Gemma‑2** – Google’ın Apache‑2.0 lisansı (ticari).  
4. **Veri gizliliği** – Kullanıcı verilerini kaydettiğiniz bir servis (ör. bir web UI) yapacaksanız GDPR/KVK gibi yasaları kontrol edin.  
5. **Telif hakkı / İçerik** – Model çıktıları telifli materyaller içeriyorsa sorumlu kullanım ilkelerini uygulayın.

---

## 9. Sorun Giderme (Common Issues)

| Hata | Muhtemel Neden | Çözüm |
|------|----------------|-------|
| `RuntimeError: CUDA out of memory` | Model VRAM limitini aşıyor. | - 4‑bit quantize edin (`load_in_4bit=True`). <br> - `device_map="auto"` yerine `"cpu"`/`"balanced"` deneyin. <br> - `max_new_tokens` değerini düşürün. |
| `bitsandbytes not compiled with CUDA` | `bitsandbytes` GPU sürümü yüklenmemiş. | `pip uninstall bitsandbytes && pip install bitsandbytes-cu122` (CUDA 12.2 örneği). |
| `OSError: [Errno 2] No such file or directory: 'git'` | Git yüklü değil. | `sudo apt install git` (Linux) / Git for Windows kurun. |
| `ImportError: cannot import name 'AutoModelForCausalLM' from 'transformers'` | Transformers çok eski. | `pip install -U transformers`. |
| `UserWarning: The tokenizers library etc.` | Tokenizer modeli eksik. | `pip install tokenizers` ya da `pip install sentencepiece` (modelin ihtiyacına göre). |

---

## 10. Hızlı Başlangıç – Tek Komutlu Script Örneği (Linux/macOS)

```bash
#!/usr/bin/env bash
set -e

# 1️⃣ Ortam oluştur
python3 -m venv llm && source llm/bin/activate

# 2️⃣ Gereksinimler
pip install --upgrade pip
pip install torch==2.3.0+cu122 torchvision==0.18.0+cu122 torchaudio==2.3.0+cu122 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers accelerate bitsandbytes einops sentencepiece tqdm

# 3️⃣ Modeli indir ve quantize et (Mistral‑7B‑Instruct)
cat > infer.py <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
quant_cfg = BitsAndBytesConfig(load_in_4bit=True,
                               bnb_4bit_compute_dtype="float16",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_quant_type="nf4")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quant_cfg,
    trust_remote_code=True,
)

def chat(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, temperature=0.7, top_p=0.9, do_sample=True)
    return tokenizer.decode(out[0], skip_special_tokens=True)

print(chat("Türk mutfağının en meşhur 5 yemeği hangileridir?"))
PY

# 4️⃣ Çalıştır
python infer.py
```

Bu script’i bir dosyaya (`install_and_run.sh`) yapıştırıp `bash install_and_run.sh` komutuyla çalıştırdığınızda:

- Sanal ortam oluşturulur,
- PyTorch + CUDA‑destekli paketler kurulur,
- `bitsandbytes` ile 4‑bit quantize edilmiş Mistral‑7B‑Instruct modeli indirilir,
- Tek bir soruya yanıt alınır.

---

## 11. Daha İleri Konular (İsteğe Bağlı)

| Konu | Açıklama | Kaynak |
|------|----------|--------|
| **LoRA / PEFT ile İnce Ayar (Fine‑tune)** | Küçük veri setiyle özel görevler için 0.5‑2 GB parametre ekleyin. | `peft` kütüphanesi, <https://github.com/huggingface/peft> |
| **GPU‑offload / CPU‑offload** | Büyük modelleri (13‑30 B) **CPU‑GPU hibrid** yürütme ile çalıştırın. | `accelerate config` → `offload` seçeneği |
| **Distributed Inference** | Çoklu sunucu/ GPU üzerinden artan throughput. | `vLLM` (<https://github.com/vllm-project/vllm>) |
| **OpenAI‑compatible API** | Yerel sunucu ile `openai` paketini tamamen aynı şekilde kullanın. | `text-generation-webui`, `LM Studio`, `Ollama` |
| **RAG (Retrieval‑Augmented Generation)** | Dış veri kaynaklarından bilgi çekip modele “prompt” ekleyin. | Haystack, LangChain, Llama‑Index |

---

## 12. Son Söz

- **Deneme yanılma** en hızlı öğrenme yoludur. Küçük bir 3‑B model (Phi‑3‑mini) ile başlayın; CUDA‑memoria ve quantization ayarlarını kavrayın.  
- **Model seçimi** amacınıza göre değişir: sohbet, kod üretimi, bilimsel metin, vs.  
- **GPU sürücüsü ve CUDA** sürümünün uyumlu olduğundan emin olun; çoğu zaman `nvidia-smi` ve `nvcc --version` çıktılarıyla kontrol edebilirsiniz.

# İkinci Bölüm

## 🎯 “LLM’i yerel (local) makinede çalıştırmak” için 0’dan 1’e adım‑adım rehber  
*(Türkçe açıklamalar, komut satırı ve Python kodları dahil)*  

> **Not:** Aşağıdaki adımlar **Linux/macOS** ve **Windows** (PowerShell/WSL) ortamları için geçerlidir.  
> Çoğu modern LLM’i **GPU** (CUDA) ile çok daha hızlı çalışır. Eğer GPU’nuz yoksa “CPU‑only” versiyonlarını da kullanabilirsiniz – bu durumda biraz daha uzun yanıt süresi olur.

---

## 1️⃣ Sistem Gereksinimleri

| Bileşen | Minimum | Önerilen |
|--------|----------|----------|
| **İşletim Sistemi** | 64‑bit Linux/macOS/Windows | Linux (Ubuntu 22.04) en stabil |
| **CPU** | 4‑çekirdek | 8‑çekirdek+ |
| **RAM** | 8 GB | 16‑32 GB |
| **GPU** (isteğe bağlı) | Yok (CPU‑only) | NVIDIA RTX 3060 + (12 GB VRAM) veya daha yüksek |
| **Disk** | 10 GB boş | 30‑50 GB (model ağırlıkları için) |
| **Python** | 3.9‑3.11 | 3.10 önerilir |

> **GPU‑destekli kurulumda**: **CUDA Toolkit** (≥ 11.7) ve **cuDNN** yüklü olmalı. `nvidia-smi` komutu GPU’nun düzgün kurulduğunu gösterir.

---

## 2️⃣ Ortam (Environment) Oluşturma

### Conda (önerilen)

```bash
# 1. Conda (Miniconda/Anaconda) kurulu olduğunu varsayıyoruz
conda create -n llm_local python=3.10 -y
conda activate llm_local
```

### Venv (alternatif)

```bash
python -m venv llm_local
source llm_local/bin/activate   # Linux/macOS
.\llm_local\Scripts\activate    # Windows PowerShell
```

---

## 3️⃣ Gerekli Kütüphanelerin Kurulması

#### 3.1. PyTorch (GPU varsa CUDA destekli)

```bash
# Linux/macOS (CUDA 12.1 örnek)
pip install torch==2.3.0+cu121 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Windows (CUDA 12.1)
pip install torch==2.3.0+cu121 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
```

> **CPU‑only** için: `pip install torch==2.3.0+cpu torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu`

#### 3.2. Transformers, Accelerate, HuggingFace Hub

```bash
pip install transformersaccelerate sentencepiece tqdm
```

#### 3.3. (Opsiyonel) **bitsandbytes** – 4‑bit quantization, düşük VRAM tüketimi

```bash
pip install bitsandbytes
```

> **Windows**’ta `bitsandbytes` kurulumu bazen hata verir. Bunun yerine `bitsandbytes-windows` (ön‑derlenmiş) paketini deneyebilirsiniz:
> ```bash
> pip install bitsandbytes-windows
> ```

#### 3.4. (Opsiyonel) **PEFT** – LoRA‑tabanlı hafif adaptasyonlar

```bash
pip install peft
```

---

## 4️⃣ Hangi Modeli Kullanacağız?  

Başlangıç için **Mistral‑7B‑Instruct** (7 B parametre, iyi performans).  
Quantized (GGUF) versiyonunu **bitsandbytes**/`transformers` ile doğrudan CPU‑only çalıştırabiliriz.

| Model | HuggingFace Hub adlandırması | Boyut (Quantized) |
|------|-----------------------------|-------------------|
| Mistral‑7B‑Instruct (GGUF, 4‑bit) | `TheBloke/Mistral-7B-Instruct-v0.2-GGUF` | ~5 GB |
| Llama‑2‑7B‑Chat (Q4_K_M) | `TheBloke/Llama-2-7B-Chat-GGUF` | ~5 GB |
| OpenChat‑3.5‑7B (Q4_K_M) | `TheBloke/OpenChat_3.5-1210-GGUF` | ~5 GB |

> **GGUF** = “GPT‑Quantized Unified Format”. `transformers` ≥ 4.41 bu formatı destekler.

### 4.1. Modeli İlk Defa İndirmek

```bash
# HuggingFace Hub'dan model dosyasını (GGUF) indirelim
# –only the .gguf file (örnek: mistral-7b-instruct-v0.2.Q4_K_M.gguf)

python - <<'PY'
from huggingface_hub import snapshot_download

repo_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
# sadece .gguf uzantılı dosyayı indir (en küçük Q4_K_M sürümü)
local_dir = snapshot_download(repo_id,
                              allow_patterns="*.gguf",
                              resume_download=True)
print(f"Model dosyaları indirildi: {local_dir}")
PY
```

> İndirme tamamlandığında bir klasör içinde `*.gguf` dosyası göreceksiniz (örnek: `mistral-7b-instruct-v0.2.Q4_K_M.gguf`).

---

## 5️⃣ Basit Python Kodu ile **Tek‑Satır** Çıktı Alma

Aşağıdaki kod, **transformers** pipelines API’siyle *inference* yapar.

```python
# ------------------------------------------------------------------
# 01_llm_inference.py
# ------------------------------------------------------------------
import os
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ---------- 1️⃣ Model yolu ----------
# snapshot_download çıktısındaki klasör yolu
model_dir = Path("~/.cache/huggingface/hub/TheBloke__Mistral-7B-Instruct-v0.2-GGUF").expanduser()
gguf_file = next(model_dir.glob("*.gguf"))   # en birinci .gguf dosyasını al
print(f"Kullanılan GGUF dosyası: {gguf_file}")

# ---------- 2️⃣ Tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",   # aynı modelin tokenizer (HF’da mevcut)
    trust_remote_code=True                # bazı modeller özel kod gerektirebilir
)

# ---------- 3️⃣ Model (bitsandbytes‑4bit) ----------
# GPU varsa torch_dtype= torch.float16  + device_map="auto"
# CPU (bitsandbytes 4‑bit) için `load_in_4bit=True`

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",                       # GPU varsa otomatik dağıt
    trust_remote_code=True,
    # 4‑bit quantization (CPU‑only) -----------------
    load_in_4bit=True if not torch.cuda.is_available() else False,
    # ------------------------------------------------
)

# ---------- 4️⃣ Pipeline (text generation) ----------
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    device=0 if torch.cuda.is_available() else -1   # -1 → CPU
)

# ---------- 5️⃣ Çalıştırma ----------
prompt = """Soru: Türkiye'nin en uzun nehri hangisidir? Cevap ver ve gerekçeni açıkla."""
outputs = generator(prompt, num_return_sequences=1)
print("\n=== Model Cevabı ===")
print(outputs[0]["generated_text"])
```

### Çalıştırma

```bash
python 01_llm_inference.py
```

#### Beklenen Çıktı (örnek)

```
=== Model Cevabı ===
Soru: Türkiye'nin en uzun nehri hangisidir? Cevap ver ve gerekçeni açıkla.
Cevap: Türkiye'nin en uzun nehri Kızılırmak'tır. Kızılırmak, 1.355 kilometre uzunluğuyla Türkiye'nin en uzun iç

...
```

> **GPU varsa** `device=0` sayesinde model GPU’da (CUDA) çalışır ve yanıt süresi 1‑2 saniye civarındadır.  
> **CPU‑only** kullanımda ise 4‑bit quantizasyon sayesinde (≈ 5 GB VRAM yerine sadece RAM) 5‑15 saniye sürebilir – hâlâ kullanışlıdır.

---

## 6️⃣ Daha Gelişmiş Kullanım Senaryoları  

### 6.1. İnteraktif Konsol (Chat‑like) Döngüsü

```python
# ------------------------------------------------------------------
# 02_llm_chat.py
# ------------------------------------------------------------------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "YOUR_LOCAL_MODEL_PATH"          # 4‑bit GGUF klasör yolu
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
    load_in_4bit= not torch.cuda.is_available()
)

def chat_loop():
    print("🚀 Yerel LLM sohbetine hoş geldiniz! (çıkmak için 'quit' yazın)")
    history = []
    while True:
        user = input("\n🗣️ Sen: ")
        if user.strip().lower() in ("quit", "exit"):
            break
        # Sistem/Prompt formatı (Mistral‑Instruct tarzı)
        prompt = f"""[INST] {user} [/INST]"""
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generation
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Model çıktısında prompt da tekrar yer alabilir → sadece cevabı al
        response = response.split("[/INST]")[-1].strip()
        print(f"🤖 Bot: {response}")

if __name__ == "__main__":
    chat_loop()
```

### 6.2. **LoRA** ile Modelle **İnce‑Ayarlama (Fine‑tuning)**  

> Eğer kendi veri setinizle (örnek: 1‑2 GB metin) modeli hafifçe “öğretmek” isterseniz **PEFT** + **bitsandbytes** ile **LoRA** yaklaşımını deneyebilirsiniz. Aşağıda minimum bir şablon verilmiştir.

```bash
pip install peft datasets tqdm
```

```python
# 03_finetune_lora.py
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    load_in_4bit=True
)

# LoRA ayarları
lora_config = LoraConfig(
    r=16,                # düşük rank
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Mistral’da kullanılanlar
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Küçük örnek veri (JSONL, CSV vs.)
dataset = load_dataset("json", data_files={"train": "my_data/train.jsonl",
                                           "validation": "my_data/val.jsonl"})

def tokenize_fn(example):
    # Örnek: {"prompt":"Soru?", "response":"Cevap"}
    text = f"[INST] {example['prompt']} [/INST] {example['response']}"
    tokenized = tokenizer(text, truncation=True, max_length=1024)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_ds = dataset.map(tokenize_fn, remove_columns=dataset["train"].column_names)

training_args = TrainingArguments(
    output_dir="./lora_mistral",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=200,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()
# LoRA ağırlıklarını kaydedelim
model.save_pretrained("./lora_mistral")
tokenizer.save_pretrained("./lora_mistral")
```

> Fine‑tuning sonrası `./lora_mistral` klasöründeki **adapter** ağırlıklarını, aynı model ile `PeftModel.from_pretrained` kullanarak tekrar yükleyebilir ve hemen sohbet edebilirsiniz.

---

## 7️⃣ Diğer Kullanışlı Araçlar / UI'ler  

| Araç | Açıklama | Kurulum (kısaca) |
|------|----------|-------------------|
| **Oobabooga Text Generation Web UI** | Web‑tabanlı, tek komutla çok sayıda model (GGUF, GPTQ, LoRA) çalıştırır. | `git clone https://github.com/oobabooga/text-generation-webui && cd text-generation-webui && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && python server.py` |
| **LM Studio** (GUI, Windows/macOS/Linux) | Sürükle‑bırak ile model ekleyebilir, sohbet penceresi sunar. | <https://lmstudio.ai> sitesinden .dmg/.exe indir ve kur. |
| **vLLM** | Çok‑iş parçacıklı yüksek hızlı inference (GPU ≥ 12 GB). | `pip install vllm && python -m vllm.entrypoints.openai.api_server --model TheBloke/Mistral-7B-Instruct-v0.2-GGUF` |
| **FastChat** | LLM API‑leri (OpenAI‑style) sunar, birden çok model aynı anda yönetilebilir. | `git clone https://github.com/lm-sys/FastChat && cd FastChat && pip install -e . && python -m fastchat.serve.controller & python -m fastchat.serve.model_worker --model-dir ~/.cache/huggingface/hub/...` |

---

## 8️⃣ Yaygın Hata ve Çözüm Özetleri  

| Hata | Neden | Çözüm |
|------|-------|-------|
| `RuntimeError: CUDA out of memory` | Model VRAM’i aşıyor. | 1) Daha küçük model (4‑bit/8‑bit). 2) `device_map="auto"` yerine `torch_dtype=torch.float16` + `max_memory` ayarları. 3) `mpirun`/`accelerate launch` ile model paralel dağıtımı. |
| `ImportError: bitsandbytes` | `bitsandbytes` uygun CUDA sürümüyle uyumsuz. | PyTorch ve CUDA sürümünü eşleştirin; `pip uninstall bitsandbytes && pip install bitsandbytes==0.44.0` gibi sürüm belirleyin. |
| `OSError: No such file or directory: 'tokenizer.json'` | Tokenizer bulunamadı. | Modelin *config* gibi bir `tokenizer` klasörü yoksa, aynı modelin **HF tokenizer** (`mistralai/Mistral-7B-Instruct-v0.2`) ile `from_pretrained` edin. |
| `torch.cuda.is_available() == False




# BÖLÜM3

Elbette! Sana temel bir **Django projesi** oluşturmak için gerekli olan başlangıç adımlarını, dizin yapısını, temel **Model**, **View** ve **URL yapılandırması** ile birlikte ayrıntılı olarak açıklıyorum.

---

## 🎯 Django Projesi Oluşturmak için Başlangıç

### 1. 🛠️ Gerekli Ortam Kurulumu

İlk olarak Python ve `pip` yüklü olmalı.

```bash
# Sanal ortam oluştur
python -m venv myenv

# Sanal ortamı aktif et (Linux/Mac)
source myenv/bin/activate

# Sanal ortamı aktif et (Windows)
myenv\Scripts\activate
```

### 2. 📦 Django Kurulumu

```bash
pip install django
```

---

## 📁 1. Django Projesi ve Uygulaması Oluşturma

```bash
# Yeni bir Django projesi oluştur
django-admin startproject myproject

# Projeye gir
cd myproject

# Bir uygulama oluştur (örn. blog)
python manage.py startapp blog
```

---

## 📁 2. Proje Dizin Yapısı

Aşağıdaki gibi bir dizin yapısı oluşur:

```
myproject/
│
├── manage.py
├── myproject/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
│
└── blog/
    ├── __init__.py
    ├── admin.py
    ├── apps.py
    ├── models.py
    ├── views.py
    ├── urls.py     # (Bu dosyayı biz oluşturacağız)
    ├── migrations/
    └── tests.py
```

---

## 📦 3. Uygulamayı Tanıt (settings.py)

`myproject/settings.py` dosyasına uygulamayı ekle:

```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'blog',  # 👈 Uygulama burada eklenir
]
```

---

## 🧱 4. Model Oluşturma (models.py)

`blog/models.py` içeriği:

```python
from django.db import models

class Post(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
```

### Modeli veritabanına ekle:

```bash
python manage.py makemigrations
python manage.py migrate
```

---

## 👁️ 5. View Oluşturma (views.py)

`blog/views.py` içeriği:

```python
from django.shortcuts import render
from django.http import HttpResponse
from .models import Post

def post_list(request):
    posts = Post.objects.all()
    return render(request, 'blog/post_list.html', {'posts': posts})

def home(request):
    return HttpResponse("Hoş geldiniz! Bu Django anasayfasıdır.")
```

---

## 🔗 6. URL Yapılandırması

### a) `blog/urls.py` dosyasını oluştur:

```python
# blog/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('posts/', views.post_list, name='post_list'),
]
```

### b) Ana URL yapılandırması (`myproject/urls.py`):

```python
# myproject/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('blog.urls')),
]
```

---

## 🖼️ 7. Template Oluşturma (HTML Sayfaları)

### a) `blog/templates/blog/` klasörünü oluştur:

```
blog/
└── templates/
    └── blog/
        └── post_list.html
```

### b) `blog/templates/blog/post_list.html` içeriği:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Blog Yazıları</title>
</head>
<body>
    <h1>Blog Yazıları</h1>
    <ul>
        {% for post in posts %}
            <li>{{ post.title }} - {{ post.created_at }}</li>
        {% endfor %}
    </ul>
</body>
</html>
```

### c) `settings.py` içine template ayarlarını ekle (gerekirse):

```python
TEMPLATES = [
    {
        ...
        'DIRS': [BASE_DIR / 'templates'],  # Eğer genel template klasörü varsa
        ...
    },
]
```

---

## 🧪 8. Admin Paneline Model Ekleme

`blog/admin.py` dosyasını düzenle:

```python
from django.contrib import admin
from .models import Post

admin.site.register(Post)
```

### Admin kullanıcısı oluştur:

```bash
python manage.py createsuperuser
```

---

## ▶️ 9. Sunucuyu Çalıştır

```bash
python manage.py runserver
```

Tarayıcında `http://127.0.0.1:8000/` adresine git.

- Admin paneli için: `http://127.0.0.1:8000/admin/`

---

## ✅ Özet

| Bileşen      | Açıklama |
|--------------|----------|
| `models.py`  | Veritabanı tablolarını tanımlar |
| `views.py`   | Kullanıcı isteklerini işler |
| `urls.py`    | URL'leri view fonksiyonlarına bağlar |
| Templates    | HTML sayfalarını oluşturur |
| Admin Panel  | Modelleri yönetim arayüzünden yönetir |

---

Eğer özel bir proje (örn. blog, e-ticaret, sosyal medya) üzerine kurmak istersen, ona özel model ve view yapılarını da hazırlayabilirim.

İstersen bir sonraki adım olarak veri ekleme, form işlemleri veya kullanıcı yetkilendirme gibi konulara da geçebiliriz. Yardımcı olmamı ister misin? 😊

Click to add a cell.
