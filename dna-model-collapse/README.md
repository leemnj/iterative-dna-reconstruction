# DNA Sequence Evolution Pipeline - Modular Version

## 프로젝트 개요

DNA 시퀀스의 반복적 진화를 시뮬레이션하고, 기초 모델(Foundation Models)의 의미적 표현 변화를 분석하는 파이프라인입니다. 원본 Jupyter Notebook을 효율적인 모듈 구조로 리팩토링했습니다.

## 폴더 구조

```
DNA_Model_Collapse/
├── preparation.py              # 환경 설정, 모델 로드
├── sequence_generation.py      # 시퀀스 생성, 임베딩 추출
├── visualization.py            # 결과 시각화, 분석
├── guidebook.ipynb            # 통합 실행 가이드
└── README.md                  # 이 파일
```

## 모듈 설명

### 1. preparation.py
**역할**: 환경 초기화 및 모델 로드

**주요 함수**:
- `get_device()`: CUDA/MPS/CPU 자동 감지
- `force_patch_triton_config()`: DNABERT-2 Triton 패치 (MPS 호환성)
- `SequenceEvolver` 클래스: 시퀀스 진화 엔진
  - `get_embedding()`: 의미적 임베딩 추출
  - `decode()`: 디코딩 전략 (greedy/sampling/top-k)
  - `evolve_step()`: 단계별 시퀀스 진화
  - `run()`: 반복적 진화 실행
- `load_models()`: DNABERT-2, Nucleotide Transformer 로드

**사용 예**:
```python
from preparation import get_device, load_models
device = get_device()
models = load_models(device)
```

### 2. sequence_generation.py
**역할**: 유전자 수집, 시퀀스 생성, 임베딩 추출

**주요 함수**:
- `fetch_gene_sequences()`: NCBI Entrez에서 유전자 수집
- `sort_genes_by_length()`: 시퀀스 길이 기준 정렬
- `generate_and_embed_sequences()`: 메인 생성 파이프라인
- `save_sequences_compressed()`: 압축 저장 (gzip/pickle)
- `load_sequences_compressed()`: 압축 파일 로드
- `load_from_parts()`: 부분 저장 디렉토리 구조에서 로드

**전략 및 유전자 설정**:
```python
DEFAULT_DECODING_STRATEGIES = {
    "greedy": {"type": "greedy", "temperatures": [1.0]},
    "sampling": {"type": "sampling", "temperatures": [0.5, 1.0, 1.5]}
}

DEFAULT_GENE_UIDS = {
    'GAPDH': 'NM_002046.7',
    'STAT3': 'NM_139276.3',
    'GAPDHP1': 'NG_001123.6'
}

DEFAULT_GENES_TO_SEARCH = ['H4C1', 'TP53', 'NORAD']
```

**사용 예**:
```python
from sequence_generation import fetch_gene_sequences, generate_and_embed_sequences

gene_selection = fetch_gene_sequences("your_email@example.com")
all_seqs, all_embs = generate_and_embed_sequences(
    gene_selection, models, DEFAULT_DECODING_STRATEGIES
)
```

### 3. visualization.py
**역할**: 결과 시각화 및 분석

**주요 함수**:
- `cosine_series_from_embeddings()`: 임베딩 유사도 계산
- `calculate_shannon_entropy()`: 시퀀스 엔트로피 계산
- `ResultsLoader` 클래스: 디스크/메모리에서 데이터 로드
- `plot_semantic_similarity_overview()`: 의미적 유사도 개요 (2x2 그리드)
- `plot_gene_pair_comparison()`: 유전자 쌍 비교 (유사도+엔트로피+길이)

**사용 예**:
```python
from visualization import ResultsLoader, plot_semantic_similarity_overview

loader = ResultsLoader(output_dir=Path('output'))
embeddings = loader.load_embeddings()
plot_semantic_similarity_overview(embeddings, model_label="DNABERT-2")
```

## guidebook.ipynb 사용법

노트북은 다음 단계를 포함합니다:

### Step 1: 환경 설정 및 의존성 설치
```bash
!pip install transformers einops huggingface_hub matplotlib scikit-learn biopython torch
```

### Step 2: 모듈 임포트
```python
from preparation import get_device, load_models
from sequence_generation import fetch_gene_sequences, generate_and_embed_sequences
from visualization import ResultsLoader, plot_semantic_similarity_overview
```

### Step 3: 디바이스 및 모델 초기화
```python
device = get_device()
models = load_models(device)
```

### Step 4: 유전자 시퀀스 수집
```python
gene_selection = fetch_gene_sequences("your_email@example.com")
```

### Step 5: 시퀀스 생성 및 임베딩
```python
all_sequences, all_embeddings = generate_and_embed_sequences(
    gene_selection, models, DEFAULT_DECODING_STRATEGIES,
    iterations=50, mask_ratio=0.15
)
```

### Step 6: 결과 시각화
```python
plot_semantic_similarity_overview(all_embeddings, model_label="DNABERT-2")
plot_gene_pair_comparison(all_embeddings, all_sequences, 'STAT3', 'NORAD', ...)
```

## 데이터 저장 및 로드

### 저장 형식

결과는 다음과 같이 저장됩니다:

```
output/
├── generated_sequences.json.gz    # 생성된 시퀀스 (gzip 압축)
├── gene_embeddings.pkl.gz         # 임베딩 (pickle 압축)
└── parts/                         # 전략별 개별 저장
    ├── sequences/
    │   ├── GAPDH/
    │   │   ├── DNABERT-2/
    │   │   │   ├── greedy.json.gz
    │   │   │   ├── sampling_t0.5.json.gz
    │   │   │   └── ...
    │   │   └── NT-v2-500m/
    │   │       └── ...
    │   └── ...
    └── embeddings/
        ├── GAPDH/
        │   ├── DNABERT-2/
        │   │   ├── greedy.pkl.gz
        │   │   ├── sampling_t0.5.pkl.gz
        │   │   └── ...
        │   └── NT-v2-500m/
        │       └── ...
        └── ...
```

### 압축 옵션

```python
from sequence_generation import save_sequences_compressed

# gzip 압축 (최소 용량)
save_sequences_compressed(data, filepath, compression='gzip')

# pickle 압축 (빠른 I/O)
save_sequences_compressed(data, filepath, compression='pickle')

# pickle + gzip (가장 작고 빠름)
save_sequences_compressed(data, filepath, compression='pickle_gzip')
```

## 환경 설정

### 필수 패키지
- torch >= 1.9.0
- transformers >= 4.20.0
- einops
- huggingface-hub
- matplotlib
- scikit-learn
- biopython

### 하드웨어 요구사항
- **메모리**: 최소 8GB RAM (권장: 16GB 이상)
- **GPU**: NVIDIA (CUDA) 또는 Apple Silicon (MPS) 권장 (CPU도 가능하지만 느림)
- **저장소**: 시퀀스 저장용 5-10GB

### 설치 방법

```bash
# 기본 설치
pip install torch transformers einops huggingface_hub matplotlib scikit-learn biopython

# M1/M2 Mac 사용자 (MPS 지원)
pip install --upgrade torch torchvision torchaudio
```

## 주요 파라미터

### 시퀀스 생성 파라미터

```python
generate_and_embed_sequences(
    gene_selection,                  # 유전자 딕셔너리
    models,                         # 모델 딕셔너리
    decoding_strategies,            # 디코딩 전략
    iterations=50,                  # 진화 반복 횟수
    mask_ratio=0.15,               # 각 스텝의 마스킹 비율
    save_all_sequences=True,        # 모든 중간 시퀀스 저장
    save_interval=5,                # save_all=False일 때 저장 간격
    output_dir=Path('output'),      # 출력 디렉토리
    store_in_memory=False,          # 전체 데이터 메모리 보관
    save_each_strategy=True,        # 전략별 개별 파일 저장
    device='cuda',                  # 연산 디바이스
    use_notebook_tqdm=True          # 노트북 진행률 표시
)
```

## 성능 최적화

### 메모리 절약
```python
# 전략별 개별 저장 + 부분 로드
generate_and_embed_sequences(
    ...,
    store_in_memory=False,
    save_each_strategy=True
)

# 필요한 데이터만 로드
loader = ResultsLoader(output_dir)
embeddings = loader.load_embeddings()  # 임베딩만 로드
```

### 속도 향상
```python
# GPU 사용
device = "cuda"

# 더 큰 배치 처리 (메모리 충분할 경우)
mask_ratio=0.2  # 더 빠른 수렴

# 더 적은 반복
iterations=30
```

## 문제 해결

### DNABERT-2 로드 실패
```
❌ DNABERT-2 Load Fail: CUDA out of memory
```
→ CPU 사용 또는 배치 크기 감소

### MPS 호환성 오류
```
RuntimeError: MPS does not support flash attention
```
→ `force_patch_triton_config()` 자동 실행됨

### NCBI 연결 오류
```
❌ Error fetching gene: HTTP Error 400
```
→ 이메일 주소 확인, 네트워크 연결 확인

## 참고 자료

- [DNABERT-2 논문](https://arxiv.org/abs/2306.15022)
- [Nucleotide Transformer](https://github.com/InstaDeep/nucleotide-transformer)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [NCBI Entrez API](https://www.ncbi.nlm.nih.gov/books/NBK25499/)

## 라이센스

이 프로젝트는 연구 목적으로 작성되었습니다.

## 기여 및 피드백

개선사항이나 버그 보고는 이슈로 등록해주세요.
