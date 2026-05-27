# LG HelloVision 고객 이탈 예측 앙상블 모델

> LG HelloVision 케이블TV 가입자 190만 명 대상 해지 예측 시스템

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **목표** | 고위험 해지 고객 사전 식별 → 선제적 리텐션 대응 |
| **데이터** | 2023년 2월~11월 가입자 행동 데이터 (취향/요금제, 시청, VOC) |
| **검증 방식** | Walk-Forward Expanding Window (OOT: 2023년 11월) |
| **최종 성능** | Precision 7.02% · Lift 5.06x (상위 5%, Optuna 최적화 후) |

## 모델 구성

```
[Model 1] XGBoost + RFECV     취향 및 요금제 피처 36→17개 선택
[Model 2] XGBoost (Walk-Fwd)  시청 관련 행동 데이터
[Pain Score] DuckDB SQL 규칙  STB장애 + VOC전체 + VOC해지불만
        ↓ Optuna 가중치 최적화 (200 trials)
[Soft Voting]  Model1×0.256 + Model2×0.015 + PainScore×0.730
[Hard Voting]  3모델 초고위험군 교집합 → 469명, Precision 19.83%
```

## 디렉토리 구조

```
├── app.py                          # Streamlit 대시보드 (메인)
├── requirements.txt
├── .gitignore
│
├── src/                            # 공통 유틸 및 설정
│   ├── config.py
│   ├── common_utils.py
│   ├── data_processing.py
│   ├── data_processing_sha.py
│   └── data_merge.py
│
├── notebooks/                      # 단계별 분석 노트북
│   ├── 01_데이터전처리.ipynb
│   ├── 02_EDA_STB장애분석.ipynb
│   ├── 03_알고리즘선정.ipynb
│   ├── 04_XGBoost_RFECV_고도화.ipynb
│   ├── 05_Walk_Forward_시계열검증.ipynb
│   ├── 06_Pain_Score.ipynb
│   ├── 07_앙상블_Soft_Voting_Optuna.ipynb
│   └── 08_성능검증.ipynb
│
├── portfolio/                      # 포트폴리오 문서
│   ├── 포트폴리오_고객이탈예측_앙상블모델.md
│   ├── portfolio_page2.html        # 시계열 검증 & 앙상블 설계
│   ├── portfolio_page3.html        # 성과 & 비즈니스 임팩트
│   └── portfolio_page4_analysis.html  # 심층 분석 (Optuna/Calibration)
│
└── images/                         # 분석 결과 차트
    ├── optuna_result.png           # Optuna 가중치 최적화 결과
    ├── calibration_curve.png       # Platt Scaling 보정 전후
    ├── shap_summary.png            # Feature Importance
    └── final_comparison.png        # 전체 성능 비교
```

## 주요 기법

- **시계열 검증**: Walk-Forward Expanding Window (Fold 1~6, AUC 0.722~0.734)
- **피처 선택**: RFECV (36→17개, Recall 0.85→0.91)
- **가중치 최적화**: Optuna 200 trials, Precision@Top5% 목적함수
- **확률 보정**: Platt Scaling (ECE 0.3835→0.0004, 99.9% 개선)
- **규칙 기반 점수**: DuckDB SQL Pain Score (이탈율 1.4%→25.1%)

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

> ⚠️ `data/` 폴더 내 실제 데이터 파일은 보안상 포함되지 않습니다.
> 데이터 경로는 `app.py` 내 `MODEL1_PATH`, `MODEL2_PATH`, `PAIN_PATH` 변수에서 설정하세요.
