# LangChain ile RAG Pipeline — Sıfırdan Uca
> Mesaj yapısından vektör aramasına kadar her adımı sıfırdan öğreten, Türkçe kaynak

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yasir-alrawi-12814521a/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/yasir237)

---

## Bu Seri Ne Anlatıyor?

RAG (Retrieval-Augmented Generation), bir LLM'e sormadan önce ilgili bilgiyi bulup ona göre cevap ürettirme yöntemidir. Bu seri, RAG'ı oluşturan her parçayı ayrı ayrı öğretir — teoride değil, çalışan kodla.

Her adım bir sonrakine köprü kurar. Atlama.

---

## Seri Haritası

```
[1] Mesaj yapısı ve LLM bağlantısı
        │
        ▼
[2] PromptTemplate ile şablonlu prompt
        │
        ▼
[3] Çoklu zincir — zincirleri birbirine bağlama
        │
        ▼
[4] Metin bölme (chunking) + Embedding
        │
        ▼
[5] ChromaDB vektör store + benzerlik araması
        │
        ▼
[6] ← şu an buradasın — serinin özeti ve öğrendiklerin
```

| Adım | Repo | Ne Öğrenirsin |
|---|---|---|
| 1 | [rag-langchain-1](https://github.com/yasir237/rag-langchain-1) | SystemMessage, HumanMessage, AIMessage yapısı |
| 2 | [rag-langchain-2](https://github.com/yasir237/rag-langchain-2) | PromptTemplate, LCEL, chain.invoke() |
| 3 | [rag-langchain-3](https://github.com/yasir237/rag-langchain-3) | RunnableLambda, RunnablePassthrough, çoklu zincir |
| 4 | [rag-langchain-4](https://github.com/yasir237/rag-langchain-4) | RecursiveCharacterTextSplitter, Gemini Embedding |
| 5 | [rag-langchain-5](https://github.com/yasir237/rag-langchain-5) | ChromaDB, similarity_search(), k parametresi |

---

## Uçtan Uca Pipeline

```
Kullanıcı sorusu
      │
      ▼
┌─────────────────────────────────────────────────────┐
│                   RAG Pipeline                      │
│                                                     │
│  1. Ham metin → RecursiveCharacterTextSplitter      │
│                        │                            │
│                        ▼                            │
│             Temiz chunk'lar (300 karakter)          │
│                        │                            │
│                        ▼                            │
│  2. Her chunk → Gemini Embedding → 3072 vektör      │
│                        │                            │
│                        ▼                            │
│  3. Vektörler → ChromaDB'ye kaydedilir              │
│                        │                            │
│  4. Soru → Embed et → Chroma'da ara → k chunk bul  │
│                        │                            │
│                        ▼                            │
│  5. Bulunan chunk'lar + soru → PromptTemplate       │
│                        │                            │
│                        ▼                            │
│  6. Groq (Llama 3.1) → Cevap üretir                │
└─────────────────────────────────────────────────────┘
      │
      ▼
Kaynağa dayalı, güvenilir cevap
```

---

## Embedding Modeli Nasıl Seçilir?

| Model | Şirket | Vektör Boyutu | Türkçe | Ücret | Ne Zaman Kullan |
|---|---|---|---|---|---|
| `paraphrase-multilingual-MiniLM` | HuggingFace | 384 | ✅ Çok iyi | Ücretsiz, local | İnternet olmadan, hızlı prototip |
| `text-embedding-ada-002` | OpenAI | 1536 | ⚠️ Orta | Ücretli | Büyük İngilizce veri setleri |
| `text-embedding-3-small` | OpenAI | 1536 | ✅ İyi | Ucuz | Production, İngilizce ağırlıklı |
| `embedding-001` | Google/Gemini | 768 | ✅ İyi | Ücretsiz limit | Hızlı başlangıç |
| `gemini-embedding-001` | Google/Gemini | 3072 | ✅ İyi | Ücretsiz limit | Öğrenme + kalite bir arada |

**Kural:** Türkçe metin kullanıyorsan OpenAI'ın avantajı kaybolur. Gemini veya HuggingFace multilingual modeller daha uygun seçimdir.

---

## Vektör Store Nasıl Seçilir?

| | ChromaDB | FAISS | Pinecone |
|---|---|---|---|
| Kurulum | pip, sıfır config | pip | Bulut, kayıt gerekir |
| Kalıcı depolama | Opsiyonel | ❌ | ✅ |
| Ücretsiz | ✅ Tamamen | ✅ Tamamen | ⚠️ Limit var |
| LangChain entegrasyonu | ✅ Tek satır | ✅ Tek satır | ✅ Tek satır |
| Öğrenme için | ✅ İdeal | ⚠️ Orta | ❌ Erken |

**Kural:** Öğrenme aşamasında ChromaDB. Production'a geçince ihtiyaca göre Pinecone veya FAISS.

---

## Chunk Stratejisi — En Önemli Karar

RAG'da en çok yapılan hata: embedding modeline odaklanıp chunk stratejisini geçiştirmek. Oysa **en iyi embedding, kötü chunk'ı kurtaramaz.**

### Chunk Size Nasıl Seçilir?

Elle denemek yerine veriyi kendine sordur:

```python
def optimal_chunk_size_bul(metin):
    cumleler = [c.strip() for c in metin.split(".") if len(c.strip()) > 10]
    ortalama = sum(len(c) for c in cumleler) / len(cumleler)

    print(f"Ortalama cümle uzunluğu : {ortalama:.0f} karakter")
    print(f"Önerilen chunk_size     : {ortalama * 2:.0f}  (2 cümle)")
    print(f"Önerilen chunk_overlap  : {ortalama * 0.2:.0f} (chunk'ın %20'si)")

optimal_chunk_size_bul(metin)
```

### Akıllı Hile — Chunk Kalitesini Ölç

En iyi chunk_size'ı bulmak için komşu chunk'ların birbirine ne kadar benzediğine bak. **Benzerlik en düşük olan bölme en iyisidir** — chunk'lar birbirinden anlamca ayrılmış demektir.

```python
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def chunk_kalitesi_test_et(metin, chunk_sizes=[50, 100, 200, 300]):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    for size in chunk_sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=int(size * 0.2),
        )
        chunks = splitter.create_documents([metin])
        vektorler = np.array(embeddings.embed_documents(
            [c.page_content for c in chunks]
        ))

        benzerlikler = []
        for i in range(len(vektorler) - 1):
            v1, v2 = vektorler[i], vektorler[i + 1]
            b = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            benzerlikler.append(b)

        print(f"chunk_size={size:3d} | "
              f"chunk sayısı={len(chunks):2d} | "
              f"ort. benzerlik={np.mean(benzerlikler):.3f}")
```

Örnek çıktı:
```
chunk_size= 50 | chunk sayısı=18 | ort. benzerlik=0.923  ← fazla bölmüş
chunk_size=100 | chunk sayısı=10 | ort. benzerlik=0.847  ← iyi
chunk_size=200 | chunk sayısı= 5 | ort. benzerlik=0.761  ← iyi
chunk_size=300 | chunk sayısı= 3 | ort. benzerlik=0.912  ← az bölmüş
```

### Overlap Altın Kuralı

```
chunk_overlap = chunk_size * 0.20
```

- **%0 overlap** → cümleler kopuk, anlam kaybı
- **%20 overlap** → bağlam korunur, tekrar minimum ✅
- **%50 overlap** → çok fazla tekrar, vektör store şişer

### Separators Sırası

```python
separators=["\n\n", "\n", ". ", " "]
```

LangChain bu listeyi soldan sağa dener. Önce paragraftan kesmek ister, olmuyorsa satırdan, olmuyorsa cümleden. Böylece cümle asla ortadan kesilmez.

---

## Chunk Temizliği — Kaynakta Çöz

Chunk'ları embed ettikten sonra temizlemeye çalışmak yerine LLM'e baştan temiz metin ürettir. Prompt'una tek satır ekle:

```
- Başlık, bölüm adı veya markdown kullanma, düz metin yaz
```

Bu sayede `**Anlatım**`, `**Özet**` gibi başlıklar chunk'lara karışmaz. Sonradan regex ile temizlemeye gerek kalmaz.

---

## Kurulum

```bash
pip install langchain langchain-core langchain-groq
pip install langchain-text-splitters langchain-google-genai
pip install langchain-chroma chromadb
```

### API Anahtarları

Google Colab **Secrets** sekmesine ekle:
- `GROQ_API_KEY` → [console.groq.com](https://console.groq.com)
- `GOOGLE_API_KEY` → [aistudio.google.com](https://aistudio.google.com)

---

## Bağlantı

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yasir-alrawi-12814521a/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yasir237)

---

## Lisans

MIT
