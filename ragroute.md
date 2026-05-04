# RAGRoute — 구현 레퍼런스

> 논문: "Efficient Federated Search for Retrieval-Augmented Generation" (EuroMLSys '25, EPFL)
> 원본 레포: https://github.com/sacs-epfl/ragroute
> 목표: 논문 재현 (MIRAGE + MMLU 벤치마크), 단일 프로세스 구조로 재구현
> `claude.md`와 함께 읽는다. 충돌 시 `claude.md` 우선.

---

## 0. 원본 레포 vs 이 구현의 차이

원본 레포는 EPFL NFS 클러스터 전용 코드다. 그대로 실행 불가.

| 항목 | 원본 레포 | 이 구현 |
|------|---------|--------|
| 아키텍처 | ZMQ 기반 멀티프로세스 분산 서버 | 단일 프로세스 클래스 구조 |
| 경로 | `/mnt/nfs/home/dpetresc` 하드코딩 | 로컬 상대경로 |
| 사전 계산 파일 | 저자 생성 `.pkl`, `.pth`, `emb_queries/*.npy` 의존 | 파이프라인으로 직접 생성 |
| 실행 방식 | `main.py` 서버 + `run_benchmark.py` 분리 | 스크립트 순차 실행 |
| 임베딩 클래스 | `CustomizeSentenceTransformer` (CLS pooling 커스텀) | HuggingFace 직접 호출 + CLS pooling |

**레포에서 그대로 가져오는 것**: `CorpusRoutingNN` 아키텍처, feature 구성, 학습 하이퍼파라미터, data split 방식, config 값들.
**레포에 이미 있어서 그대로 사용하는 것**: `data/benchmark/MIRAGE.json`, `data/question_order_*.json`

---

## 1. 프로젝트 구조

```
ragroute/
├── ragroute.md
├── claude.md
├── data/
│   ├── raw/
│   │   └── mirage/                    # MedRAG corpora
│   ├── embeddings/
│   │   ├── mirage/
│   │   │   ├── {corpus}_article_embeddings.npy   # (N, 768) float32
│   │   │   ├── {corpus}_chunks.json              # len==N, index 1:1 대응
│   │   │   └── {benchmark}_query_embeddings.npy  # (Q, 768) float32
│   │   └── mmlu/
│   │       ├── cluster_{0-9}_embeddings.npy      # (N_c, 768) float32
│   │       └── cluster_{0-9}_chunks.json
│   ├── processed/
│   │   ├── mirage/
│   │   │   ├── train_test_split.json   # {q_id: "train"/"val"/"test"} (재현성)
│   │   │   ├── X_train.npy, y_train.npy
│   │   │   ├── X_val.npy,   y_val.npy
│   │   │   └── X_test.npy,  y_test.npy
│   │   └── mmlu/
│   │       └── (동일 구조)
│   ├── stats/
│   │   ├── pubmed_stats.json           # {"centroid": [...], "num_documents": N}
│   │   ├── statpearls_stats.json
│   │   ├── textbooks_stats.json
│   │   ├── wikipedia_stats.json        # (medrag corpus)
│   │   └── cluster_stats.json          # wikipedia(mmlu): [{"centroid": [...]}, ...]
│   └── benchmark/                      # 원본 레포에서 그대로 복사
│       ├── MIRAGE.json
│       │   # keys: medqa(1273), medmcqa(4183), pubmedqa(500), bioasq(618), mmlu(1089)
│       │   # 총 7663 questions
│       └── question_order_*.json       # 평가 순서 고정 (재현성 필수)
├── src/
│   ├── config.py
│   ├── data_source.py
│   ├── embedding_model.py
│   ├── feature_extractor.py
│   ├── router_model.py
│   ├── router_trainer.py
│   ├── rag_router.py
│   ├── federated_retriever.py
│   └── utils.py
├── scripts/
│   ├── 01_download_data.sh
│   ├── 02_build_embeddings.py       # article + query 분리, mmlu k-means 포함
│   ├── 03_build_index.py            # FAISS index + stats.json
│   ├── 04_generate_train_data.py    # label 생성 + question-level split
│   ├── 05_train_router.py
│   └── 06_evaluate.py
├── experiments/
│   ├── mirage_top32.yaml
│   └── mmlu_top10.yaml
├── checkpoints/
├── results/
├── tests/
└── requirements.txt
```

---

## 2. 시스템 아키텍처

```
[User Query]
    │
    ▼
[EmbeddingModel.encode_query()]     # MedCPT-Query-Encoder (CLS) / DPR (pooler_output)
    │
    ├──────────────────────────────────────────────────┐
    ▼                                                  ▼
[RAGRouter.route()]                        (Training Phase)
    │  CorpusRoutingNN                     [RouterTrainer]
    │  batch all N sources → 1 forward         │
    │  sigmoid(logit) > 0.5 → select           │  generate_labels()
    ▼                                           │  train()
[Selected Sources m ≤ n]                       ▼
    │  (빈 리스트 가능, fallback 없음)  [checkpoints/*.pth + *.pkl]
    ▼
[FederatedRetriever.retrieve()]
    │  per-source top-K_RETRIEVE(50) → merge → global top-k_eval(10 or 32)
    ▼
[LLM]                               # Ollama + LLaMA 3.1 (disable-rerank 모드)
```

**K 값 구분** (혼동 주의):

| 상수 | 값 | 용도 |
|------|----|------|
| `LABEL_K` | 15 | Router 학습용 label 생성 global top-k |
| `K_RETRIEVE` | 50 | Inference 시 per-source retrieval top-k |
| `k_eval` | 10 or 32 | 논문 실험의 최종 global top-k |

---

## 3. 핵심 설계 결정 (재현 실패 시 가장 먼저 점검)

### 3.1 Feature 불일치 (논문 vs 코드)

**논문 기술 feature** (Section 3.2.1):
- query embedding, centroid, dist(query, centroid), source size, source density

**실제 코드 feature**:
- query embedding (768) + centroid (768) + one-hot(source_id) → 1540/1546 dim

이건 단순 차이가 아니라 **모델 입력 의미 자체가 다르다**:
- 논문: explicit geometric features (dist, size, density)
- 코드: implicit — query/centroid embedding에서 distance를 스스로 학습

이 구현은 **실제 저자 코드 기준**이므로 코드 방식을 따른다.
재현 목표도 논문 수치가 아니라 **저자 코드가 실제로 달성한 수치**다.

### 3.2 Routing threshold

현재 구현: `sigmoid > 0.5`

원본 코드에 tuning 흔적 있음 (주석 처리된 0.4924, 0.4872).
최종 사용값은 0.5로 확정.

**threshold는 recall vs efficiency trade-off에 직접 영향**:
- 낮추면 → recall 올라가고 query reduction 감소
- 올리면 → recall 내려가고 query reduction 증가

재현 시 기본값 0.5 사용. recall이 크게 낮으면 0.4~0.5 범위에서 val 기반 조정 가능.

### 3.3 FAISS metric별 정렬 방향 (치명적 버그 포인트)

FAISS IndexFlatL2와 IndexFlatIP는 반환 방향이 다르다:

| 인덱스 타입 | FAISS search 반환 | 좋은 방향 | merge 정렬 |
|------------|-----------------|---------|-----------|
| IndexFlatL2 (medrag) | L2 거리 (작을수록 좋음) | 오름차순 | `sort(key=dist, reverse=False)` |
| IndexFlatIP (wikipedia) | inner product (클수록 좋음) | 내림차순 | `sort(key=score, reverse=True)` |

**label 생성과 retrieval merge 모두 데이터셋에 따라 정렬 방향이 달라야 한다.**

```python
# medrag label 생성: dist 오름차순
all_results.sort(key=lambda x: x[0], reverse=False)

# wikipedia label 생성: score 내림차순
all_results.sort(key=lambda x: x[0], reverse=True)
```

이걸 통일하면 label 전체가 오염된다.

### 3.4 Query ordering alignment (전 단계 일관성 필수)

다음 4개가 동일한 question 순서를 공유해야 한다:

```
MIRAGE.json의 question 순서
    ↓ (동일 순서 유지)
{benchmark}_query_embeddings.npy  ← row i = question i
    ↓ (동일 q_id 기준)
train_test_split.json             ← {q_id: "train"/"val"/"test"}
    ↓ (동일 순서)
question_order_*.json             ← 06_evaluate.py의 평가 순서
```

**어느 하나라도 순서가 다르면 label, train, eval 전체가 오염된다.**

구현 요구사항:
- `{benchmark}_query_embeddings.npy`를 생성할 때 MIRAGE.json의 question 순서 그대로 저장
- `train_test_split.json`은 `{q_id: split}` dict 형태로 저장 (index 기반 아님)
- `04_generate_train_data.py`는 q_id를 키로 데이터를 조립한 후 split 기준으로 분리
- `06_evaluate.py`는 `question_order_*.json` 파일의 순서로 실행

### 3.5 Empty routing case

router가 아무 source도 선택하지 않을 수 있다 (sigmoid < 0.5 전부).

이 경우:
- retriever는 빈 청크 반환
- LLM은 context 없이 query만으로 답변 (No RAG와 동일)
- **fallback으로 all_sources 선택 금지** (원본 코드 정책)

empty rate가 높으면 threshold 조정 또는 학습 문제를 의심한다.

### 3.6 MMLU 비교 주의

논문: 10 subjects, 2313 questions
원본 코드: 8 subjects (TARGET_SUBJECTS)

**이 구현은 코드 기준(8개)을 따르므로 논문 Table 1의 MMLU 수치와 직접 비교 불가.**
상대적 성능 경향(RAGRoute ≈ all corpora) 확인에 집중한다.

---

## 4. 컴포넌트 스펙

### 4.1 config.py

```python
DATA_SOURCES = {
    "medrag":    ["pubmed", "statpearls", "textbooks", "wikipedia"],
    "wikipedia": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
}
# 주의: 논문 "five corpora" = 4 corpus + MedCorp(합집합). 실제 사용은 4개.

MEDRAG_SOURCE_TO_ID = {"pubmed": 0, "statpearls": 1, "textbooks": 2, "wikipedia": 3}

INPUT_DIM = {"medrag": 1540, "wikipedia": 1546}

LABEL_K    = 15
K_RETRIEVE = 50

TRAIN_CONFIG = {
    "medrag": {
        "seed": 12, "batch_size": 128, "epochs": 150,
        "lr_base": 1e-3, "lr_max": 5e-3, "weight_decay": 3e-5,
        "grad_clip": 1.0, "cyclic_step_up": 10, "cyclic_mode": "triangular2",
        "cyclic_cutoff": 115, "step_lr_step": 50, "step_lr_gamma": 0.05,
        "best_metric": "val_auc", "use_pos_weight": False,
    },
    "wikipedia": {
        "seed": 42, "batch_size": 128, "epochs": 150,
        "lr_base": 1e-3, "lr_max": 5e-3, "weight_decay": 1e-5,
        "grad_clip": 1.0, "cyclic_step_up": 10, "cyclic_mode": "triangular2",
        "cyclic_cutoff": 115, "step_lr_step": 50, "step_lr_gamma": 0.05,
        "best_metric": "val_f1", "use_pos_weight": True, "pos_weight_scale": 5.0,
    },
}

MMLU_TARGET_SUBJECTS = {
    "high_school_microeconomics", "international_law", "college_biology",
    "miscellaneous", "prehistory", "philosophy",
    "professional_psychology", "high_school_mathematics",
}

LLM_CONFIG = {
    "ollama_model":        "llama3.1_extended",
    "hf_name":             "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "docs_context_length": 128000,
    "base_url":            "http://localhost:11434",
    "disable_rerank":      True,
}

DATA_DIR = "data"; CHECKPOINT_DIR = "checkpoints"; RESULTS_DIR = "results"
```

### 4.2 DataSource (`src/data_source.py`)

```python
@dataclass
class DataSource:
    source_id: str
    dataset: str            # "medrag" or "wikipedia"
    chunks: List[Any]       # medrag: dict / wikipedia: (title, text) tuple
    index: faiss.Index      # medrag=IndexFlatL2 / wikipedia=normalized IndexFlatIP
    centroid: np.ndarray    # float32, (768,)
    size: int

    def search(self, query_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        # medrag:    index.search(query_vec.reshape(1,-1), k)
        # wikipedia: query_copy = query_vec.reshape(1,-1).copy()
        #            faiss.normalize_L2(query_copy); index.search(query_copy, k)
        # returns (scores[0], indices[0]) — shape (k,)
        # medrag scores = L2 distances (작을수록 좋음)
        # wikipedia scores = inner products (클수록 좋음)

    @classmethod
    def from_files(cls, source_id, dataset, index_path, chunks, stats_path):
        # medrag:    stats = json.load(f); centroid = np.array(stats["centroid"])
        # wikipedia: stats_list = json.load(f); centroid = np.array(stats_list[int(source_id)]["centroid"])
```

### 4.3 EmbeddingModel (`src/embedding_model.py`)

```python
class EmbeddingModel:
    # medrag query:   "ncbi/MedCPT-Query-Encoder", CLS pooling
    # medrag article: "ncbi/MedCPT-Article-Encoder", CLS pooling (corpus 인코딩용)
    # wikipedia:      DPRQuestionEncoder("facebook/dpr-question_encoder-single-nq-base")
    #                 → pooler_output
    #
    # MMLU query 포맷: question + "\n" + " | ".join(choices)
    # MIRAGE query 포맷: question text 그대로

    def encode_query(self, text: str) -> np.ndarray:     # (768,) float32
    def encode_batch(self, texts: List[str], batch_size=128) -> np.ndarray:  # (N, 768) float32
```

### 4.4 RouterFeatureExtractor (`src/feature_extractor.py`)

```python
class RouterFeatureExtractor:
    def extract(self, query_vec, centroid, source_id, dataset, all_source_ids) -> np.ndarray:
        # [0:768]    query_vec
        # [768:1536] centroid
        # [1536:]    np.eye(n_sources)[source_to_id[source_id]]
        # returns (1540,) medrag / (1546,) wikipedia, float32
```

### 4.5 CorpusRoutingNN (`src/router_model.py`)

```python
class CorpusRoutingNN(nn.Module):
    def __init__(self, input_dim: int):
        # fc1:    Linear(input_dim, 128) → LayerNorm(128) → ReLU → Dropout(0.4)
        # fc2:    Linear(128, 64)        → LayerNorm(64)  → ReLU → Dropout(0.4)
        # fc3:    Linear(64, 32)         → LayerNorm(32)  → ReLU → Dropout(0.4)
        # fc_out: Linear(32, 1)          ← raw logit

    def forward(self, x):  # (batch, input_dim) → (batch, 1)
```

> **주의**: router.py의 `CorpusRoutingNN`은 `dropout=0.5`를 파라미터로 받지만, 실제 train scripts에서는 파라미터 없이 내부 0.4 고정으로 초기화한다. inference 시 `model.eval()`이면 dropout은 비활성화되므로 load 시 동일 클래스 정의를 사용하면 된다.

### 4.6 RouterTrainer (`src/router_trainer.py`)

```python
class RouterTrainer:
    def generate_labels(self, query_vecs, sources, dataset, k=LABEL_K):
        # ⚠️ 정렬 방향이 데이터셋마다 다름 (Section 3.3 필수 참고)
        # medrag:    all_results.sort(key=lambda x: x[0], reverse=False)  # L2 오름차순
        # wikipedia: all_results.sort(key=lambda x: x[0], reverse=True)   # IP 내림차순
        #
        # per query:
        #   1. 모든 source search → (score, source_id, local_idx) 수집
        #   2. dataset별 방향으로 정렬 → global top-k source_id set
        #   3. label[i] = 1 if source_i in top_k else 0
        # returns X (Q*N, input_dim) float32, y (Q*N,) float32

    def split_questions(self, all_question_ids, dataset, save_path):
        # question-level split (row-level 절대 금지)
        # train_qs, test_qs = train_test_split(all_qs, test_size=0.6, random_state=SEED)
        # train_qs, val_qs  = train_test_split(train_qs, test_size=0.1, random_state=SEED)
        # → train ~36% / val ~4% / test 60% (전체 기준)
        # 주의: 논문 기술 30/10/60과 다름 (val이 train의 10%, 즉 전체 ~4%)
        # JSON 저장 형식: {q_id: "train"/"val"/"test"} (index 기반 금지)

    def train(self, X_train, y_train, X_val, y_val, dataset):
        # cfg = TRAIN_CONFIG[dataset]
        # 1. scaler.fit_transform(X_train); scaler.transform(X_val)
        # 2. medrag: BCEWithLogitsLoss()
        #    wikipedia: BCEWithLogitsLoss(pos_weight=5 * neg/pos)
        # 3. Adam(lr=1e-3, weight_decay=cfg["weight_decay"])
        # 4. epoch<115:  CyclicLR(base=1e-3, max=5e-3, step_up=10, mode="triangular2", cycle_momentum=False)
        #    epoch>=115: StepLR(step_size=50, gamma=0.05)
        # 5. clip_grad_norm_(max_norm=1.0)
        # 6. medrag: best=val_auc, wikipedia: best=val_f1
        # returns (best_model, fitted_scaler)
```

**feature 행렬 question-level 마스킹**:
```python
n_sources = len(sources)
# train_test_split.json에서 q_id → split 매핑 로드
train_mask = np.repeat([split[qid] == "train" for qid in q_ids], n_sources)
```

### 4.7 RAGRouter (`src/rag_router.py`)

```python
class RAGRouter:
    def route(self, query_vec: np.ndarray) -> List[DataSource]:
        # batch all sources → single forward → sigmoid > 0.5
        # 빈 리스트 반환 가능, fallback 없음
        # "sub-millisecond" = 이 forward만 해당 (encode_query 제외)

    @classmethod
    def load(cls, checkpoint_path, scaler_path, sources, dataset, threshold=0.5):
        ...
```

### 4.8 FederatedRetriever (`src/federated_retriever.py`)

```python
class FederatedRetriever:
    # __init__(self, k_retrieve=50, k_global=32)

    def retrieve(self, query_vec, selected_sources) -> List[Any]:
        # ⚠️ merge 정렬도 dataset에 따라 다름 (Section 3.3)
        # medrag:    (score, src, idx) sort ascending
        # wikipedia: (score, src, idx) sort descending
        # global top k_global 반환

    def retrieve_with_stats(self, query_vec, selected_sources, all_sources) -> dict:
        # {"chunks", "n_queries", "n_queries_naive", "query_reduction_pct"}
        # comm_volume 비교: ZMQ 환경과 직접 비교 불가 → query_reduction_pct에 집중
```

### 4.9 LLM 및 평가 (`utils.py` 또는 `rag_pipeline.py`)

**Prompt 형식** (원본 `config.py` 그대로):

medrag system prompt:
```
You are a helpful medical expert, and your task is to answer a multi-choice medical question
using the relevant documents. Please first think step-by-step and then choose the answer
from the provided options.
Output JSON: {"step_by_step_thinking": "...", "answer_choice": "A/B/C/D"}
```

medrag user prompt:
```
Here are the relevant documents:
{context}
Here is the question: {question}
Here are the potential choices: {options}
```

**Answer extraction**:
```python
# llm_output에서 '"answer_choice": "' 뒤 파싱
ans = llm_output.split('"answer_choice": "')[-1].strip()
# 이후 regex로 A/B/C/D 추출
```

**Context truncation** (llm_message.py 방식):
```python
tokenizer = AutoTokenizer.from_pretrained(hf_name)  # HF_TOKEN 필요
encoded = tokenizer.encode("\n".join(contexts))[:docs_context_length]
context = tokenizer.decode(encoded)
```

---

## 5. 데이터 파이프라인

### Stage 2: Embedding

**medrag - 인코더 분리 필수, 포맷 다름**:
```python
# article (corpus): MedCPT-Article-Encoder, CLS, max_length=512
# query: MedCPT-Query-Encoder, CLS, 질문 text 그대로

# 저장: {benchmark}_query_embeddings.npy 생성 시 MIRAGE.json 순서 그대로 유지
# → alignment 보장
```

**mmlu - query 포맷 주의**:
```python
# DPR query encoder
# 포맷: question + "\n" + " | ".join(choices)
formatted_q = item["question"] + "\n" + " | ".join(item["choices"])
q_emb = encode_query_dpr(formatted_q)

# question index: enumerate(dataset)의 i (q_id = f"question_{i}")
```

**mmlu k-means**:
```python
kmeans = MiniBatchKMeans(n_clusters=10, random_state=42)
cluster_labels = kmeans.fit_predict(embs)  # 클러스터 크기: 49k~148k 검증
```

### Stage 3: FAISS + Stats

```python
# medrag: IndexFlatL2
index = faiss.IndexFlatL2(768)
index.add(embeddings.astype(np.float32))
# stats: {"centroid": centroid.tolist(), "num_documents": N} (corpus별 개별 파일)

# wikipedia: normalize 후 IndexFlatIP
embs_copy = embeddings.copy().astype(np.float32)
faiss.normalize_L2(embs_copy)
index = faiss.IndexFlatIP(768)
index.add(embs_copy)
# stats: cluster_stats.json = [{"centroid": [...]}, ...] (list, index=cluster_id)
```

### Stage 4: Label Generation

```python
# ⚠️ 데이터셋별 정렬 방향 다름 (Section 3.3)

# medrag
for q_id, q_vec in zip(q_ids, q_vecs):
    all_results = [(d, src.source_id, i) for src in sources for d,i in zip(*src.search(q_vec, LABEL_K))]
    all_results.sort(key=lambda x: x[0], reverse=False)  # L2 오름차순
    top_k_sources = {sid for _, sid, _ in all_results[:LABEL_K]}
    for src in sources:
        feat = extractor.extract(q_vec, src.centroid, src.source_id, "medrag", source_ids)
        label = 1.0 if src.source_id in top_k_sources else 0.0

# wikipedia (mmlu)
# MMLU_TARGET_SUBJECTS 8개만, q_id = f"question_{enumerate_index}"
for i, item in enumerate(dataset):
    if item["subject"] not in MMLU_TARGET_SUBJECTS: continue
    q_vec = encode_query_dpr(item["question"] + "\n" + " | ".join(item["choices"]))
    all_results = [(s, cid, li) for cid, src in enumerate(sources) for s,li in zip(*src.search(q_vec, LABEL_K))]
    all_results.sort(key=lambda x: x[0], reverse=True)  # IP 내림차순
    top_k_clusters = {cid for _, cid, _ in all_results[:LABEL_K]}
    for cid, src in enumerate(sources):
        feat = extractor.extract(q_vec, src.centroid, str(cid), "wikipedia", cluster_ids)
        label = 1.0 if cid in top_k_clusters else 0.0
```

### Stage 6: Evaluation

```python
# question_order_*.json 순서 사용 필수 (재현성)
with open(f"data/question_order_MIRAGE_{benchmark}.json") as f:
    ordered_q_ids = json.load(f)

# accuracy 계산: check_mirage_answer() / check_mmlu_answer()
# LLM output에서 '"answer_choice": "' 이후 파싱
# No RAG baseline도 함께 측정 (context=[] 전달)
```

---

## 6. 환경 설정

```
torch>=2.1.0        numpy>=1.24.0       scikit-learn>=1.3.0
faiss-cpu>=1.7.4    transformers>=4.36.0  sentence-transformers>=4.0.0
accelerate>=0.25.0  datasets>=2.16.0    tqdm>=4.66.0
pyyaml>=6.0         scipy>=1.11.0       ollama>=0.4.6
python-liquid       pytest>=7.4.0
```

```bash
conda create -n ragroute python=3.9 && conda activate ragroute
pip install -r requirements.txt
```

**LLM**:
```bash
ollama serve
ollama pull llama3.1:8b
# 128k context 버전 (논문 원본)
cat > Modelfile << 'EOF'
FROM llama3.1:8b
PARAMETER num_ctx 131072
EOF
ollama create llama3.1_extended -f Modelfile
export HF_TOKEN=hf_...   # tokenizer 로드용
```

**데이터**:
```bash
cp -r /path/to/ragroute/data/benchmark data/
cp /path/to/ragroute/data/question_order_*.json data/
git clone https://github.com/Teddy-XiongGZ/MedRAG.git
```

**실행 순서**:
```bash
python scripts/02_build_embeddings.py --dataset medrag
python scripts/02_build_embeddings.py --dataset mmlu
python scripts/03_build_index.py --dataset medrag
python scripts/03_build_index.py --dataset mmlu
python scripts/04_generate_train_data.py --config experiments/mirage_top32.yaml
python scripts/05_train_router.py --config experiments/mirage_top32.yaml
python scripts/06_evaluate.py --config experiments/mirage_top32.yaml
```

---

## 7. 논문 재현 목표 수치

### Classification (Table 1)

| 실험 | Accuracy | Recall | F1 | AUC |
|------|---------|--------|----|-----|
| MIRAGE Top-32 | 85.63 ± 3.92 | 85.47 ± 3.61 | 85.79 ± 2.45 | 92.60 ± 2.33 |
| MIRAGE Top-10 | 87.30 ± 6.10 | 88.32 ± 3.96 | 85.43 ± 4.18 | 93.67 ± 3.33 |
| MMLU Top-10 ★ | 90.06 ± 5.04 | 76.23 ± 6.64 | 78.29 ± 7.59 | 92.88 ± 3.29 |

★ 이 구현은 8개 subjects → 논문 MMLU 수치와 직접 비교 불가. 경향만 참고.

### Retrieval Recall (Figure 4)

| 실험 | RAGRoute recall 범위 |
|------|---------------------|
| MIRAGE top-10 | 95.3% ~ 99.0% |
| MIRAGE top-32 | 96.7% ~ 98.5% |
| MMLU top-10   | ~90% |

### Efficiency (Section 4.4)

| 지표 | 목표 | 비고 |
|------|------|------|
| Query reduction (MMLU) | 77.5% | 직접 재현 가능 |
| Query reduction (MIRAGE PubMedQA) | 71.3% (top-10) | 직접 재현 가능 |
| Comm. reduction (MMLU) | 76.2% | ZMQ 환경 측정, 이 구현에서 직접 재현 불가 |
| Routing selection_time (CPU) | < 0.8ms | encode_query 시간 제외 |

### End-to-end (Table 2)

| 설정 | Accuracy |
|------|---------|
| No RAG | 67.04 ± 7.66 |
| RAG all | 72.22 ± 9.86 |
| RAGRoute | 72.24 ± 9.36 |

---

## 8. 논문 기술 vs 실제 코드 차이 요약

| 항목 | 논문 기술 | 실제 코드 | 채택 |
|------|---------|---------|-----|
| RouterModel 레이어 | 256→128→1 | **128→64→32→1** | 실제 코드 |
| Dropout | 0.1 | **0.4** | 실제 코드 |
| Features | dist/size/density 포함 | **query+centroid+one-hot** | 실제 코드 |
| pos_weight | "with positional weight" | **medrag=미전달, wiki=5×neg/pos** | 실제 코드 |
| Best model (medrag) | val accuracy | **val AUC** | 실제 코드 |
| Best model (wiki) | val accuracy | **val F1** | 실제 코드 |
| Epochs | 50 | **150** | 실제 코드 |
| LR schedule | CyclicLR 단일 | **CyclicLR(0~114) + StepLR(115~)** | 실제 코드 |
| LR mode | triangular | **triangular2** | 실제 코드 |
| weight_decay | 미명시 | **medrag=3e-5, wiki=1e-5** | 실제 코드 |
| Random seed | 미명시 | **medrag=12, wiki=42** | 실제 코드 |
| Embedding pooling | 미명시 | **CLS** | 실제 코드 |
| wikipedia index | IndexFlatL2 | **normalized + IndexFlatIP** | 실제 코드 |
| MMLU subjects | 10개 | **8개** | 실제 코드 |
| data split | 30/10/60 | **~36/~4/60** | 실제 코드 |
| corpora 수 | "five corpora" | **4개** (MedCorp는 baseline용) | 실제 코드 |
| label_k vs K_RETRIEVE | 미구분 | **label_k=15, K_RETRIEVE=50** | 실제 코드 |
