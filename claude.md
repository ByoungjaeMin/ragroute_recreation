# claude.md — RAGRoute 프로젝트 Claude Code 행동 규칙

이 파일은 이 프로젝트에서 Claude Code의 모든 행동을 통제한다.
`ragroute.md`의 구현 스펙과 함께 읽는다. 충돌 시 이 파일이 우선한다.

---

## 0. 최우선 원칙

**이 프로젝트의 목적은 논문의 완전한 재현이다.**
구현 세부사항의 기준은 원본 레포 코드(https://github.com/sacs-epfl/ragroute)다.
논문 텍스트와 실제 코드가 다를 경우 항상 실제 코드를 따른다.
빠른 완성보다 정확한 구현이 항상 우선한다.

---

## 1. 절대 금지 행위

아래 행위는 어떤 이유로도 허용되지 않는다. 위반 시 구현을 중단하고 즉시 알린다.

### 1-1. Fallback / 우회 구현 금지

겉으로 돌아가는 것처럼 보이지만 실제로 다른 동작을 하는 코드는 전부 금지다.

```python
# 금지: 에러를 조용히 삼키고 다른 값 반환
try:
    result = real_implementation()
except Exception:
    result = []                       # ← 금지. raise로 올려야 함

# 금지: 빈 routing 결과를 전체 소스로 대체
selected = router.route(q_vec)
if not selected:
    selected = all_sources            # ← 금지. 빈 리스트 그대로 반환

# 금지: 실제 계산 없이 상수 반환
def extract(self, ...):
    return np.zeros(1540)             # ← 금지

# 금지: FAISS 없이 임의 인덱스 반환
def search(self, query_vec, k):
    return np.zeros(k), np.arange(k)  # ← 금지

# 금지: 학습 없이 미초기화 모델 반환
def train(self, ...):
    return CorpusRoutingNN(input_dim), StandardScaler()  # ← 금지
```

**원칙**: 무언가 작동하지 않으면 조용히 넘어가지 말고 `raise`로 명시적 에러를 낸다.

### 1-2. 원본 코드 스펙 임의 변경 금지

| 항목 | 원본 코드 값 | 변경 금지 이유 |
|------|------------|--------------|
| RouterModel 레이어 | 128→64→32→1 | 원본 `CorpusRoutingNN` 그대로 |
| Dropout | 0.4 전 레이어 고정 | 파라미터로 받지 않음 |
| 정규화 방식 | LayerNorm | BatchNorm 대체 금지 |
| Loss | BCEWithLogitsLoss | BCE 직접 대체 금지 |
| Feature 구성 | query + centroid + one-hot | dist/size/density 추가 금지 |
| Input dim medrag | 1540 | 변경 시 checkpoint 호환 깨짐 |
| Input dim wikipedia | 1546 | 동일 |
| Embedding pooling | CLS | MEAN pooling 절대 금지 |
| LR schedule | CyclicLR(0~114) + StepLR(115~) | 단일 scheduler로 단순화 금지 |
| LR mode | triangular2 | triangular로 변경 금지 |
| Epochs | 150 | 임의 단축 금지 |
| Best model (medrag) | val AUC | val accuracy/F1로 변경 금지 |
| Best model (wikipedia) | val F1 | val accuracy/AUC로 변경 금지 |
| Seed (medrag) | 12 | 임의 변경 금지 |
| Seed (wikipedia) | 42 | 임의 변경 금지 |
| weight_decay (medrag) | 3e-5 | wiki 값으로 통일 금지 |
| weight_decay (wikipedia) | 1e-5 | medrag 값으로 통일 금지 |
| pos_weight (medrag) | BCEWithLogitsLoss에 미전달 | 전달로 변경 금지 |
| pos_weight (wikipedia) | 5 × neg/pos | 계수 변경 금지 |
| Data split | question-level, test 60%, val=train의 10%(전체 ~4%) | row-level 금지 |
| LABEL_K | 15 | 변경 금지 |
| K_RETRIEVE | 50 | LABEL_K와 혼동 금지 |
| Routing threshold | 0.5 | 임의 변경 금지 |
| medrag FAISS | IndexFlatL2 | IVF/HNSW 대체 금지 |
| wikipedia FAISS | normalize_L2 후 IndexFlatIP | IndexFlatL2 대체 금지 |
| medrag merge 정렬 | L2 오름차순 (reverse=False) | 내림차순으로 변경 금지 |
| wikipedia merge 정렬 | IP 내림차순 (reverse=True) | 오름차순으로 변경 금지 |
| MedCPT 인코더 | query/article 분리 | 단일 인코더 통합 금지 |
| MMLU query 포맷 | question + "\n" + " \| ".join(choices) | 변경 금지 |
| MMLU 학습 subjects | TARGET_SUBJECTS 8개 | 임의 추가/제거 금지 |
| medrag stats.json | corpus별 개별 파일 | wikipedia list 방식으로 통일 금지 |
| wikipedia cluster_stats.json | list, index=cluster_id | medrag 방식으로 변경 금지 |
| 평가 순서 | question_order_*.json 기준 | 임의 순서 사용 금지 |
| LLM output format | JSON {"answer_choice": "A/B/C/D"} | 변경 금지 |

### 1-3. 지정되지 않은 파일 수정 금지

- 요청된 파일과 함수만 수정한다.
- "연관이 있어서" 다른 파일을 함께 고치지 않는다.
- 수정 전 항상 현재 파일 내용을 먼저 확인한다.

### 1-4. 테스트를 통과시키기 위한 하드코딩 금지

```python
# 금지
def evaluate():
    return {"accuracy": 0.856, "auc": 0.926}

# 금지: 기댓값에 맞춰 결과를 조작
```

### 1-5. 원본 레포의 EPFL 경로 하드코딩 금지

```python
# 금지
MODELS_USR_DIR = "/mnt/nfs/home/dpetresc"
BASE_DIR       = "/mnt/nfs/home/dpetresc/MedRAG"
ROUTING_DIR    = "/mnt/nfs/home/dpetresc/MedRAG/routing/"
WIKI_DIR       = "/mnt/nfs/home/dpetresc/wiki_dataset/dpr_wiki_index"
```

모든 경로는 `src/config.py`의 로컬 상수를 사용한다.

---

## 2. 구현 품질 요구사항

### 2-1. 에러 처리 원칙

```python
# 올바른 방식: 명시적 에러
def search(self, query_vec: np.ndarray, k: int):
    if self.index is None:
        raise RuntimeError(f"DataSource '{self.source_id}': index not loaded.")
    if query_vec.dtype != np.float32:
        raise TypeError(f"query_vec must be float32, got {query_vec.dtype}")
    ...

# 올바른 방식: 빈 결과는 그대로
def route(self, query_vec):
    selected = [src for src, score in zip(self.sources, scores) if score > self.threshold]
    return selected   # 빈 리스트여도 그대로
```

### 2-2. dtype과 shape 일관성

- 모든 임베딩: `float32`
- FAISS 입력 전 `.astype(np.float32)` 확인
- `search()` 반환: `(scores[0], indices[0])`, 각 shape `(k,)`
- `extract()` 반환: medrag `(1540,)`, wikipedia `(1546,)`, `float32`
- wikipedia search: `query_copy = query_vec.reshape(1,-1).copy()` → `faiss.normalize_L2(query_copy)` → search
- StandardScaler: `fit_transform`은 X_train에만, val/test는 `transform`만

### 2-3. 원본과 다르게 구현하는 항목 (승인된 deviation)

이 범위만 허용. 벗어나면 즉시 알린다.

| 원본 | 이 구현 | 이유 |
|------|--------|------|
| ZMQ 멀티프로세스 서버 | 단일 프로세스 클래스 | 로컬 환경 |
| `/mnt/nfs/...` 경로 | `data/`, `checkpoints/` 상대경로 | 로컬 환경 |
| 사전 계산 파일 의존 | 파이프라인으로 직접 생성 | 저자 파일 없음 |
| `CustomizeSentenceTransformer` | AutoModel + CLS pooling 직접 구현 | 의존성 단순화 |
| `emb_queries/{q_id}.npy` 개별 파일 | 배치 일괄 저장 | 효율성 |
| `relevant_top_15.json` / `question_{i}_cluster_ids.txt` | FAISS search로 직접 계산 | 저자 파일 없음 |
| communication volume: ZMQ raw bytes | query_reduction_pct에 집중 | ZMQ 없음 |

코드 주석 형식:
```python
# DEVIATION FROM ORIGINAL: [이유]
# Original: [원본 방식]
# This impl: [이 구현 방식]
```

---

## 3. 작업 프로토콜

### 3-1. 파일 수정 전

1. 해당 파일 현재 내용 확인
2. 수정할 함수/클래스 범위 명시
3. 변경 이유 한 줄 설명

### 3-2. 작업 완료 보고

```
변경된 함수: [함수명 목록]
변경되지 않은 파일: [명시]
원본 코드 일치 여부: [일치 / 불일치 시 DEVIATION 주석 위치]
```

### 3-3. 의존성 임포트 방향

```
config.py               → 외부 의존성 없음
data_source.py          → numpy, faiss
router_model.py         → torch.nn
feature_extractor.py    → numpy, config
federated_retriever.py  → numpy, faiss, data_source (torch 없음)
embedding_model.py      → torch, transformers
router_trainer.py       → torch, sklearn, router_model, feature_extractor, data_source, config
rag_router.py           → torch, sklearn, router_model, feature_extractor, data_source, config
rag_pipeline.py         → rag_router, federated_retriever, embedding_model, config
```

역방향 import 금지.

---

## 4. 테스트 요구사항

### 4-1. 단위 테스트 원칙

- 실제 데이터/모델 다운로드 없이 실행 가능
- random seed 고정 (numpy 42, torch 42)
- 가짜 DataSource: `np.random.randn(100, 768).astype(np.float32)`

### 4-2. 통합 테스트 최소 조건

`test_integration.py`의 `test_ragroute_end_to_end()`:
- feature shape: medrag `(1540,)`, wikipedia `(1546,)`
- `router.route()`: `List[DataSource]`, len ≥ 0 (빈 리스트 허용)
- `retriever.retrieve()`: `List`, len == k_global
- label에 0과 1 모두 포함 (all-zero → label 생성 버그)
- model `fc1.in_features` == `INPUT_DIM[dataset]`
- medrag/wikipedia merge 정렬 방향 각각 테스트

### 4-3. 재현성 검증 기준

Table 1 수치와 비교:
- Accuracy ±3% 이내 → 재현 성공
- 벗어날 경우 점검 순서:
  1. merge 정렬 방향 (CLS pooling 다음으로 가장 치명적)
  2. CLS pooling 여부
  3. question-level split 여부
  4. feature one-hot 구성
  5. pos_weight 설정
  6. seed

---

## 5. 자주 발생하는 실수

| 실수 | 결과 | 올바른 방법 |
|------|------|-----------|
| **wikipedia merge를 오름차순으로 정렬** | **label 전체 오염 (최치명)** | **IP → 내림차순 (reverse=True)** |
| query/article 인코더 혼용 | recall 급락 | MedCPT Query / Article 분리 |
| MEAN pooling 사용 | 임베딩 불일치 | CLS pooling 필수 |
| question ordering mismatch | label/train/eval 전체 오염 | q_id 기반 dict 관리, question_order.json 준수 |
| row-level data split | data leakage | question-level split 필수 |
| RouterModel에 sigmoid 추가 | loss 오계산 | raw logit 반환 |
| StandardScaler fit을 val/test에도 적용 | test leakage | X_train에만 fit |
| FAISS에 float64 입력 | 런타임 에러 | `.astype(np.float32)` 후 입력 |
| wikipedia search에 normalize 누락 | 검색 결과 오염 | `faiss.normalize_L2` 후 search |
| one-hot 대신 dist/size/density 사용 | input_dim 불일치 | 원본 코드대로 one-hot |
| medrag에 pos_weight 전달 | 원본과 다른 학습 | `BCEWithLogitsLoss()` pos_weight 없이 |
| CyclicLR만 150 epoch 사용 | 원본과 다른 학습 곡선 | epoch 115부터 StepLR 전환 |
| weight_decay 동일하게 | medrag/wiki 차이 무시 | medrag=3e-5, wiki=1e-5 각각 |
| LABEL_K와 K_RETRIEVE 혼동 | label 오류 또는 retrieval 저하 | LABEL_K=15, K_RETRIEVE=50 구분 |
| route() 빈 결과를 all_sources로 대체 | fallback 금지 | 빈 리스트 그대로 반환 |
| MMLU query 포맷 누락 | 임베딩 불일치 | question + "\n" + choices 포맷 |
| EPFL NFS 경로 하드코딩 | 로컬 실행 불가 | config.py 로컬 상수 사용 |
| < 0.8ms를 전체 routing 시간으로 오해 | 성능 기준 오류 | selection_time만 (encode_query 제외) |
| wikipedia stats를 corpus별 파일로 생성 | from_files() 로딩 오류 | cluster_stats.json list 형태 |
| question_order 파일 무시 | 평가 재현성 깨짐 | 06_evaluate.py에서 반드시 사용 |
