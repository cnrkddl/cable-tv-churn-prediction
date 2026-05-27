# 🔴 LG HelloVision 고객 이탈 예측 앙상블 모델

> 케이블TV 가입자 **약 200만 명** 대상 해지 위험 고객 사전 탐지 시스템  
> 머신러닝 + 규칙 기반 앙상블로 전체 해지율(1.4%) 대비 최대 **14.2배 Lift** 달성

---

## 📌 프로젝트 배경

LG HelloVision은 국내 주요 케이블TV 사업자로, 가입자 이탈은 직접적인 매출 손실로 이어집니다.
기존에는 이탈 후 사후 대응이 주를 이뤘으나, **해지 예고 기간(30일) 이전에 선제적으로 위험 고객을 식별**하여 리텐션 캠페인을 실행할 필요가 있었습니다.

| 항목 | 내용 |
|---|---|
| **목표** | 고위험 해지 고객 사전 식별 → 선제적 리텐션 대응 |
| **데이터 기간** | 2023년 2월 ~ 11월 (10개월) |
| **검증 대상** | 2023년 11월 가입자 1,998,550명 |
| **실제 해지자** | 27,931명 (해지율 1.40%) |
| **검증 방식** | Walk-Forward Expanding Window + Hold-out OOT (2023년 11월) |

---

## 🏆 최종 성능 요약

### Soft Voting (상위 N% 추출)

| 구간 | 대상 고객 수 | Precision | Recall | Lift |
|---|---|---|---|---|
| Top 1% | 19,066명 | 6.73% | 4.59% | **4.82x** |
| Top 2% | 38,132명 | 6.09% | 8.31% | 4.36x |
| Top 3% | 57,236명 | 5.83% | 11.95% | 4.17x |
| Top 5% | 96,305명 | 5.30% | 18.28% | 3.79x |

### Hard Voting (3모델 교집합, 초고위험군)

| 그룹 | 대상 고객 수 | Precision | Recall | Lift |
|---|---|---|---|---|
| 🔴 초고위험군 | **469명** | **19.83%** | 0.33% | **14.19x** |
| 🟠 고위험군 | 28,483명 | 7.97% | 8.13% | 5.71x |

> 기준: 전체 해지율 1.40% (27,931명 / 1,998,550명)

---

## 🔧 모델 설계

### 앙상블 구성도

```
┌─────────────────────────────────────────────────────────┐
│                     입력 데이터                          │
│   취향/요금제 피처   │   시청 행동 데이터   │  STB·VOC 이력  │
└────────┬────────────┴──────────┬──────────┴───────┬──────┘
         ▼                       ▼                   ▼
┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│    Model 1      │  │    Model 2       │  │   Pain Score     │
│  XGBoost        │  │  XGBoost         │  │  (DuckDB SQL     │
│  + RFECV        │  │  Walk-Forward    │  │   규칙 기반)     │
│  36→17 피처     │  │  6-Fold 검증     │  │  이탈율 1.4%     │
│  + Platt 보정   │  │  AUC 0.722~0.734 │  │  → 25.1%         │
└────────┬────────┘  └──────────┬───────┘  └────────┬─────────┘
         │                      │                    │
         └──────────────────────┴────────────────────┘
                                ▼
                   ┌─────────────────────────┐
                   │  Optuna 200 trials       │
                   │  가중치 최적화           │
                   │  목적함수: Precision@5%  │
                   └───────────┬─────────────┘
                               ▼
         ┌─────────────────────────────────────────┐
         │          최종 앙상블 점수                │
         │  0.256 × M1 + 0.015 × M2 + 0.730 × Pain │
         └────────────┬──────────────┬─────────────┘
                      ▼              ▼
            [Soft Voting]      [Hard Voting]
            상위 N% 추출       3모델 초고위험군
            Lift 3.8~4.8x      교집합 → Lift 14.2x
```

### Optuna 가중치 최적화 결과

| 구성 요소 | 기존 가중치 | Optuna 최적 가중치 | 변화 |
|---|---|---|---|
| Model 1 (취향/요금제) | 0.40 | **0.256** | ↓ |
| Model 2 (시청 데이터) | 0.40 | **0.015** | ↓↓ |
| Pain Score | 0.20 | **0.730** | ↑↑↑ |
| **Precision@Top5%** | 5.32% | **7.02%** | +1.70%p |

> Pain Score가 ML 모델과 **직교하는 독립 신호**를 제공하여 가중치가 대폭 증가

---

## 🔬 주요 기법 상세

### 1. Walk-Forward Expanding Window 검증

```
Fold 1: Train[02~04월] → Test[05월]
Fold 2: Train[02~05월] → Test[06월]
Fold 3: Train[02~06월] → Test[07월]
Fold 4: Train[02~07월] → Test[08월]
Fold 5: Train[02~08월] → Test[09월]
Fold 6: Train[02~09월] → Test[10월]
OOT:    Train[02~10월] → Test[11월]  ← 최종 평가
```

- 시간 순서를 엄격히 지켜 **미래 데이터 누수(Data Leakage) 완전 방지**
- 6개 Fold AUC: 0.722 ~ 0.734 (안정적 수렴 확인)

### 2. RFECV 피처 선택 (Model 1)

- 초기 피처 36개 → RFECV로 **최적 17개 선택**
- Recall 0.85 → **0.91** 향상 (+7.1%p)
- 불필요한 노이즈 피처 제거로 일반화 성능 개선

### 3. Platt Scaling 확률 보정 (Model 1)

- 훈련 시 1:1 다운샘플링으로 예측 확률이 실제보다 과대추정되는 문제 발생
- Logistic Regression 사후 보정 적용
- **ECE 0.3835 → 0.0004 (99.9% 개선)**

### 4. Pain Score (DuckDB SQL 규칙 기반)

STB(셋톱박스) 장애 이력 + VOC 데이터를 SQL로 집계하여 불만 지수 산출

```sql
Pain Score = STB 장애 횟수 × 가중치
           + VOC 전체 접수 건수 × 가중치
           + VOC 해지/불만 관련 건수 × 높은 가중치
```

- Pain Score 0점 구간 해지율: **1.4%** (base rate 수준)
- Pain Score 최고 구간 해지율: **25.1%** (17.9배 차이)

---

## 📁 디렉토리 구조

```
cable-tv-churn-prediction/
│
├── app.py                              # Streamlit 대시보드 메인 앱
├── requirements.txt                    # 의존성 패키지
├── .gitignore
│
├── src/                                # 공통 모듈
│   ├── config.py                       # 경로 및 전역 설정
│   ├── common_utils.py                 # 공통 유틸리티 함수
│   ├── data_processing.py              # TPS 해지 데이터 전처리
│   ├── data_processing_sha.py          # SHA 고객 데이터 전처리
│   └── data_merge.py                   # 다중 데이터소스 병합
│
├── notebooks/                          # 단계별 분석 노트북
│   ├── 01_데이터전처리.ipynb           # TPS + SHA 데이터 병합 및 피처 생성
│   ├── 02_EDA_STB장애분석.ipynb        # 셋톱박스 장애 이탈 상관관계 EDA
│   ├── 03_알고리즘선정.ipynb           # LR / RF / XGBoost 비교 실험
│   ├── 04_XGBoost_RFECV_고도화.ipynb   # RFECV 피처 선택 (36→17개)
│   ├── 05_Walk_Forward_시계열검증.ipynb # 6-Fold 시계열 검증 구현
│   ├── 06_Pain_Score.ipynb             # DuckDB SQL Pain Score 산출
│   ├── 06b_Pain_Score_시각화.ipynb     # Pain Score 구간별 이탈률 시각화
│   ├── 07_앙상블_Soft_Voting_Optuna.ipynb  # 앙상블 + Optuna 가중치 최적화
│   └── 08_성능검증.ipynb               # Hold-out OOT 최종 성능 평가
│
├── 과정/                               # 탐색 및 실험 과정 기록
│   ├── 데이터분석/                     # 시청 패턴·VOD·넷플릭스 탐색
│   ├── REFCV/2차/                      # RFECV 성능 향상 실험 과정
│   └── 머신러닝/                       # Walk-Forward 초기 실험
│
└── images/                             # 분석 결과 차트
    ├── shap_summary.png                # SHAP 피처 중요도 요약
    ├── calibration_curve.png           # Platt Scaling 보정 전후
    ├── optuna_result.png               # Optuna 200 trials 결과
    └── final_comparison.png            # 단일 모델 vs 앙상블 성능 비교
```

---

## 📊 Streamlit 대시보드 (`app.py`)

실제 모델 결과 파일(Parquet)을 업로드하면 아래 기능을 제공합니다.

| 기능 | 설명 |
|---|---|
| **위험군 분포 시각화** | 🔴 초고위험 / 🟠 고위험 / 🟡 위험 / 🟢 안정군 비율 |
| **고객별 이탈 확률** | sha2_hash 기준 이탈 확률 조회 |
| **Pain Score 현황** | 불만 지수 구간별 고객 분포 |
| **3모델 교차 분석** | Soft Voting vs Hard Voting 결과 비교 |
| **CSV 다운로드** | 위험군별 고객 리스트 추출 |

---

## ⚙️ 실행 방법

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. Streamlit 대시보드 실행
streamlit run app.py
```

> ⚠️ **데이터 보안 안내**
> `dataset/` 폴더 내 실제 데이터 파일은 개인정보 보호 및 보안상 이유로 포함되지 않습니다.  
> 앱 실행 시 사이드바에서 결과 Parquet 파일을 직접 업로드하여 사용하세요.

---

## 🛠 기술 스택

| 분류 | 도구 |
|---|---|
| **언어** | Python 3.10+ |
| **ML 모델** | XGBoost 2.0, scikit-learn |
| **하이퍼파라미터 최적화** | Optuna 3.4 (TPE Sampler, 200 trials) |
| **SQL 엔진** | DuckDB (Pain Score 집계) |
| **데이터 처리** | pandas, numpy, pyarrow |
| **시각화** | matplotlib, seaborn, SHAP |
| **대시보드** | Streamlit |

---

## 📈 분석 인사이트

1. **Pain Score의 압도적 예측력**: STB 장애 + VOC 이력만으로 이탈률 25.1% 구간 생성 → Optuna가 가중치 0.73 부여
2. **Model 1 vs Model 2의 정보 중복**: 취향/요금제(M1)와 시청데이터(M2)는 유사한 신호 → M2 가중치 0.015로 수렴
3. **Hard Voting의 초정밀 탐지**: 세 모델이 모두 동의한 469명에서 Precision 19.83% — 집중 리텐션 캠페인에 적합
4. **Platt Scaling의 중요성**: 다운샘플링 학습 시 확률 과대추정 문제를 사후 보정으로 해결 (ECE 99.9% 개선)
