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

Click to add a cell.
