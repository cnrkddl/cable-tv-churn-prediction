import pandas as pd
import numpy as np
import streamlit as st
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# =============================
# 기본 설정
# 텍스트로만 되어 있는 위험군을, 시각적으로 더 직관적으로 보이게 이모지로 매핑
# 칼럼명 하드코딩하지 않고, 후보 리스트로 관리
# =============================
st.set_page_config(page_title="해지 위험 고객 분석 리포트", layout="wide")

RED = "🔴 초고위험군"
ORANGE = "🟠 고위험군"
YELLOW = "🟡 위험군"
GREEN = "🟢 안정군"
RISK_ORDER = [RED, ORANGE, YELLOW, GREEN] #리스크 등급의 명칭과 우선 순위 선정 -> 비즈니스 중요도 순 

ID_CANDIDATES = ["sha2_hash", "sha2", "sha_hash", "sha", "고객ID", "customer_id"]
RISK_CANDIDATES = ["risk_group", "risk", "segment", "등급", "위험군"]
PROB_CANDIDATES = ["churn_probability", "prob", "proba"]
PAIN_CANDIDATES = ["pain_score", "pain_stage", "painstage"]


# =============================
# 유틸
# =============================
#데이터 표준화
def find_col(df, candidates): #대소문자 구분 없이 후보 리스트에서 컬럼명 찾아주는 함수
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def normalize(df, kind): #찾아낸 칼럼을 customer_id, risk_group, churn_probability, pain_score로 표준화하는 함수
    df = df.copy()

    id_col = find_col(df, ID_CANDIDATES)
    risk_col = find_col(df, RISK_CANDIDATES)

    if id_col is None or risk_col is None:
        raise ValueError("ID 또는 risk_group 컬럼을 찾을 수 없습니다.")
    
    # 원본 데이터의 '안정집단' 텍스트를 '안정군'으로 치환하여 매핑 호환성 확보
    temp_series = df[risk_col].astype(str).str.strip()
    temp_series = temp_series.replace("안정집단", "안정군")
    temp_series = temp_series.replace("🟢 안정집단", "🟢 안정군")

    df["customer_id"] = df[id_col].astype(str)
    df["risk_group"] = pd.Categorical( 
            temp_series,
            categories=RISK_ORDER,
            ordered=True
        )

    df["is_high_risk"] = df["risk_group"].isin([RED, ORANGE])
    df["is_top_risk"] = df["risk_group"] == RED

    if kind == "model":
        prob_col = find_col(df, PROB_CANDIDATES)
        if prob_col is None:
            raise ValueError("churn_probability 컬럼이 없습니다.")
        df["churn_probability"] = pd.to_numeric(df[prob_col], errors="coerce")

    if kind == "pain":
        pain_col = find_col(df, PAIN_CANDIDATES)
        if pain_col is None:
            raise ValueError("pain_score 컬럼이 없습니다.")
        df["pain_score"] = pd.to_numeric(df[pain_col], errors="coerce")

    return df


@st.cache_data #데이터 로딩 속도를 획기적으로 개선
def load_path(path: str, kind: str): #파일 존재 여부 확인부터 확장자(CSV/Parquet)에 따른 처리까지 수행
    """
    path: 'data/xxx.parquet' 같은 문자열 경로
    kind: 'model' | 'pain'
    """
    if not path or not isinstance(path, str):
        return None

    path = path.strip()
    if not os.path.exists(path):
        return None

    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    elif path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError("지원하지 않는 확장자입니다. (csv/parquet만 가능)")

    return normalize(df, kind)



def risk_dist(df): #데이터에 특정 등급이 없더라도 '0건'으로 표시
    out = (
        df["risk_group"]
        .value_counts()
        .reindex(RISK_ORDER)
        .fillna(0)
        .astype(int)
        .reset_index(name="고객 수") # count -> 고객 수
    )
    out["비율 (%)"] = (out["고객 수"] / len(df) * 100).round(0) # ratio(%) -> 비율 (%)
    out = out.rename(columns={"risk_group": "위험 등급"}) # risk_group -> 위험 등급
    return out

# 불만지수 탭 전용: 소수점 두자리 ratio(%) 그대로
def risk_dist_raw(df):
    out = (
        df["risk_group"]
        .value_counts()
        .reindex(RISK_ORDER)
        .fillna(0)
        .astype(int)
        .reset_index(name="고객 수") # count -> 고객 수
    )

    # 🔥 pain_score 매핑 추가
    pain_map = {
        RED: 3,
        ORANGE: 2,
        YELLOW: 1,
        GREEN: 0
    }

    out["불만 점수"] = out["risk_group"].map(pain_map) # pain_score -> 불만 점수

    out["비율 (%)"] = (out["고객 수"] / len(df) * 100).round(2) # ratio(%) -> 비율 (%)

    # 컬럼 순서 정리 및 한글화
    out = out.rename(columns={"risk_group": "위험 등급"})
    out = out[["위험 등급", "불만 점수", "고객 수", "비율 (%)"]]

    return out


def present(df, cols, sort_cols, asc, TOP_N):
    # 출력용 칼럼명 매핑 사전
    rename_dict = {
        "customer_id": "고객 ID",
        "risk_group": "위험 등급",
        "churn_probability": "이탈 예측치",
        "pain_score": "불만 점수",
        "sha2_hash": "고객 ID"
    }
    
    view = df.rename(columns=rename_dict)
    view = view.loc[:, ~view.columns.duplicated()]
    
    # 정렬 기준도 변경된 칼럼명에 맞춰 적용
    sort_cols_ko = [rename_dict.get(c, c) for c in sort_cols]
    view = view.sort_values(sort_cols_ko, ascending=asc)

    # 요청된 칼럼 리스트를 한글로 변환
    cols_ko = [rename_dict.get(c, c) for c in cols]
    
    return view[cols_ko].head(TOP_N)




def to_download_view(df):
    view = df.rename(columns={
        "customer_id": "고객 ID",
        "risk_group": "위험 등급",
        "churn_probability": "이탈 예측치",
        "pain_score": "불만 점수"
    })
    view = view.loc[:, ~view.columns.duplicated()]

    # 다운로드 코드에서 "sha2_hash"를 쓰고 있으니 동일하게 호환되게 하나 더 만들어줌(안전)
    view["sha2_hash"] = view["고객 ID"]
    return view

# =============================
# Platt Scaling 보정 유틸
# =============================
@st.cache_resource
def build_platt_scaler(prob_series: pd.Series, label_series: pd.Series):
    """
    Model 1 Platt Scaling 보정기를 학습합니다.
    50%를 calibration set으로 사용해 과대추정 문제를 해결합니다.
    (ECE 0.3835 → 0.0004, 99.9% 개선)
    """
    X = prob_series.values.reshape(-1, 1)
    y = label_series.values
    cal_idx, _ = train_test_split(
        np.arange(len(X)), test_size=0.5, random_state=42,
        stratify=y
    )
    platt = LogisticRegression(C=1.0, max_iter=1000)
    platt.fit(X[cal_idx], y[cal_idx])
    return platt


@st.cache_data
def load_labels(cancel_path: str, target_mt: int = 202311):
    """실제 해지 레이블을 로드합니다."""
    if not os.path.exists(cancel_path):
        return None
    cancel = pd.read_csv(cancel_path)
    cancel_mt = cancel[cancel["p_mt"] == target_mt].copy()
    status_cnt = (
        cancel_mt.groupby("sha2_hash")["cancel_yn"]
        .nunique()
        .reset_index(name="n")
    )
    valid_ids = set(status_cnt[status_cnt["n"] == 1]["sha2_hash"])
    clean = cancel_mt[cancel_mt["sha2_hash"].isin(valid_ids)]
    label_df = clean.drop_duplicates("sha2_hash")[["sha2_hash", "cancel_yn"]].copy()
    label_df["label"] = (label_df["cancel_yn"] == "해지").astype(int)
    label_df["sha2_hash"] = label_df["sha2_hash"].astype(str)
    return label_df


# =============================
# UI
# =============================
st.title("해지 위험 고객 분석 리포트")
st.caption("파일 업로드 여부에 따라 각 모델 결과가 활성화됩니다.")

#----고정 경로 (코드 내부에서 직접 관리)--
MODEL1_PATH  = "data/XGB_Segment_Preference_Model_risk_groups_20260212-101726.parquet"
MODEL2_PATH  = "data/churn_risk_targets_final.parquet"
PAIN_PATH    = "data/sha_202311_pain_3col.parquet"
CANCEL_PATH  = "../../dataset/sha_tps/sha_tps_cancel_202311_to_202312.csv"
TOP_N = 100000

# Optuna 최적 가중치 (200 trials, Precision@Top5% 목적함수)
# 기존: Model1=0.4, Model2=0.4, Pain=0.2
# 최적: Model1=0.256, Model2=0.015, Pain=0.730
W1, W2, W3 = 0.256, 0.015, 0.730

# ---- Load ----
m1   = load_path(MODEL1_PATH, "model")
m2   = load_path(MODEL2_PATH, "model")
pain = load_path(PAIN_PATH, "pain")
labels = load_labels(CANCEL_PATH)


# ---- Tab labels (상태 표시) ----

t1 = "✅ 통합 결과" if all(x is not None for x in [m1, m2, pain]) else "⚪ 통합 결과"
t2 = "✅ 취향 및 요금제 기반" if m1 is not None else "⚪ 취향 및 요금제 기반"
t3 = "✅ 시청 관련 데이터 기반" if m2 is not None else "⚪ 시청 관련 데이터 기반"
t4 = "✅ 불만 지수 기반" if pain is not None else "⚪ 불만 지수 기반"
tab1, tab2, tab3, tab4 = st.tabs([t1, t2, t3, t4])


# =============================
# Tab 1 : 통합 결과 (Hard Voting + Soft Voting)
# =============================
with tab1:
    if not all(x is not None for x in [m1, m2, pain]):
        st.info("통합 결과는 취향 및 요금제 기반 · 시청 관련 데이터 기반 · 불만지수 파일이 모두 필요합니다.")
    else:
        # ── 공통 병합 ──────────────────────────────────────
        merged = (
            m1[["customer_id", "risk_group", "churn_probability"]]
              .rename(columns={"risk_group": "risk_m1", "churn_probability": "prob_m1"})
            .merge(
                m2[["customer_id", "risk_group", "churn_probability"]]
                  .rename(columns={"risk_group": "risk_m2", "churn_probability": "prob_m2"}),
                on="customer_id", how="inner"
            )
            .merge(
                pain[["customer_id", "risk_group", "pain_score"]]
                  .rename(columns={"risk_group": "risk_pain"}),
                on="customer_id", how="inner"
            )
        )

        def to_code(x):
            if x == RED:    return "초"
            if x == ORANGE: return "고"
            if x == YELLOW: return "위"
            return "안"

        merged["red_cnt"] = (
            (merged["risk_m1"] == RED).astype(int)
            + (merged["risk_m2"] == RED).astype(int)
            + (merged["risk_pain"] == RED).astype(int)
        )
        merged["orange_cnt"] = (
            (merged["risk_m1"] == ORANGE).astype(int)
            + (merged["risk_m2"] == ORANGE).astype(int)
            + (merged["risk_pain"] == ORANGE).astype(int)
        )

        # Hard Voting 그룹
        hv_top_df  = merged[merged["red_cnt"] == 3].copy()
        hv_high_df = merged[
            (merged["red_cnt"] + merged["orange_cnt"] == 3)
            & (merged["red_cnt"] != 3)
        ].copy()

        def make_hv_view(df):
            out = df.rename(columns={
                "customer_id": "고객 ID",
                "risk_m1":     "취향 및 요금제 기반",
                "risk_m2":     "시청 관련 데이터 기반",
                "risk_pain":   "불만 지수 기반"
            }).reset_index(drop=True)
            return out[["고객 ID", "취향 및 요금제 기반", "시청 관련 데이터 기반", "불만 지수 기반"]]

        hv_top_view  = make_hv_view(hv_top_df)
        hv_high_view = make_hv_view(hv_high_df)

        # ── Soft Voting (Optuna 최적 가중치) ───────────────
        # Model 1 Platt Scaling 보정 (과대추정 해결: ECE 0.3835 → 0.0004)
        sv_df = merged[["customer_id", "prob_m1", "prob_m2", "pain_score"]].copy()
        sv_df["pain_norm"] = sv_df["pain_score"] / sv_df["pain_score"].max()

        if labels is not None:
            lbl = labels.rename(columns={"sha2_hash": "customer_id"})
            sv_lbl = sv_df.merge(lbl[["customer_id", "label"]], on="customer_id", how="inner")
            if len(sv_lbl) > 100:
                platt = build_platt_scaler(sv_lbl["prob_m1"], sv_lbl["label"])
                sv_df["prob_m1_cal"] = platt.predict_proba(sv_df[["prob_m1"]])[:, 1]
            else:
                sv_df["prob_m1_cal"] = sv_df["prob_m1"]  # 레이블 부족 시 원본 사용
        else:
            sv_df["prob_m1_cal"] = sv_df["prob_m1"]

        # Optuna 최적 가중치: Model1=0.256, Model2=0.015, PainScore=0.730
        sv_df["meta_score"] = (
            W1 * sv_df["prob_m1_cal"] +
            W2 * sv_df["prob_m2"] +
            W3 * sv_df["pain_norm"]
        )

        top5_cut  = sv_df["meta_score"].quantile(0.95)
        top3_cut  = sv_df["meta_score"].quantile(0.97)
        top1_cut  = sv_df["meta_score"].quantile(0.99)

        sv_top5_df = sv_df[sv_df["meta_score"] >= top5_cut].sort_values("meta_score", ascending=False).copy()
        sv_top1_df = sv_df[sv_df["meta_score"] >= top1_cut].sort_values("meta_score", ascending=False).copy()

        def make_sv_view(df):
            out = df.rename(columns={
                "customer_id":  "고객 ID",
                "meta_score":   "앙상블 점수",
                "prob_m1_cal":  "Model1 보정확률",
                "prob_m2":      "Model2 확률",
                "pain_norm":    "Pain Score (정규화)"
            }).reset_index(drop=True)
            cols = ["고객 ID", "앙상블 점수", "Model1 보정확률", "Model2 확률", "Pain Score (정규화)"]
            return out[[c for c in cols if c in out.columns]]

        sv_top5_view = make_sv_view(sv_top5_df)
        sv_top1_view = make_sv_view(sv_top1_df)

        # ── KPI 요약 ─────────────────────────────────────
        st.markdown("#### 방식별 타겟 규모 비교")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Hard Voting — 초고위험군 (3교집합)", f"{len(hv_top_view):,}명")
        k2.metric("Hard Voting — 고위험군 (2+교집합)", f"{len(hv_high_view):,}명")
        k3.metric("Soft Voting — 상위 1% (Optuna)", f"{len(sv_top1_view):,}명")
        k4.metric("Soft Voting — 상위 5% (Optuna)", f"{len(sv_top5_view):,}명")

        st.caption(
            "Soft Voting 가중치: Model1(보정)=0.256 · Model2=0.015 · PainScore=0.730  |  "
            "Optuna 200 trials, Precision@Top5% 최적화  |  Precision 5.32%→7.02%, Lift 3.83x→5.06x"
        )
        st.divider()

        # ── 내부 탭: Hard Voting / Soft Voting ────────────
        subtab_hv, subtab_sv = st.tabs([
            "🔵 Hard Voting (3교집합)",
            "🟣 Soft Voting — Optuna 최적 가중치"
        ])

        # ── Hard Voting 탭 ──────────────────────────────
        with subtab_hv:
            st.markdown("**Hard Voting**: 3개 모델 모두 초고위험군(🔴) 또는 고위험군(🟠)으로 분류된 교집합")
            st.markdown(f"- Precision 19.83% · Lift 14.19x (상위 0.02%, n=469)")

            inner1, inner2 = st.tabs(["🔴 통합 초고위험군 (3×🔴)", "🟠 통합 고위험군 (🔴🔴+🟠)"])
            with inner1:
                st.dataframe(hv_top_view.head(TOP_N), use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 초고위험군 다운로드 (CSV)",
                    data=hv_top_view.to_csv(index=False).encode("utf-8-sig"),
                    file_name="hard_voting_top_risk.csv",
                    mime="text/csv",
                    key="dl_hv_top"
                )
            with inner2:
                st.dataframe(hv_high_view.head(TOP_N), use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 고위험군 다운로드 (CSV)",
                    data=hv_high_view.to_csv(index=False).encode("utf-8-sig"),
                    file_name="hard_voting_high_risk.csv",
                    mime="text/csv",
                    key="dl_hv_high"
                )

        # ── Soft Voting 탭 ──────────────────────────────
        with subtab_sv:
            st.markdown(
                "**Soft Voting (Optuna)**: 3개 모델 점수의 가중합 기준 상위 고객 추출  \n"
                "Model1 Platt Scaling 보정 적용 (ECE 0.3835→0.0004)  \n"
                "가중치: **Model1=0.256**, Model2=0.015, **PainScore=0.730**"
            )

            inner3, inner4 = st.tabs(["🔝 상위 1%", "🔝 상위 5%"])
            with inner3:
                st.markdown(f"**상위 1%** — {len(sv_top1_view):,}명")
                st.dataframe(sv_top1_view.head(TOP_N), use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 상위 1% 다운로드 (CSV)",
                    data=sv_top1_view.to_csv(index=False).encode("utf-8-sig"),
                    file_name="soft_voting_top1pct.csv",
                    mime="text/csv",
                    key="dl_sv_top1"
                )
            with inner4:
                st.markdown(f"**상위 5%** — {len(sv_top5_view):,}명  |  Precision 7.02%, Lift 5.06x")
                st.dataframe(sv_top5_view.head(TOP_N), use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 상위 5% 다운로드 (CSV)",
                    data=sv_top5_view.to_csv(index=False).encode("utf-8-sig"),
                    file_name="soft_voting_top5pct.csv",
                    mime="text/csv",
                    key="dl_sv_top5"
                )


# =============================
# Tab 2 : Model 1
# =============================
with tab2:
    if m1 is None: #파일 누락 시 안내문구 표시
        st.info("취향 및 요금제 기반 파일을 업로드하면 결과가 표시됩니다.")
    else:
        st.subheader("취향 및 요금제 기반 결과")
        st.dataframe(risk_dist(m1), use_container_width=True, hide_index=True)
        table = present( #가장 위험한 고객부터 정렬하여 표로 출력
            m1,
            ["customer_id", "risk_group", "churn_probability"],
            ["risk_group", "churn_probability"],
            [True, False],
            TOP_N
        )
        st.dataframe(table, use_container_width=True, hide_index=True)
            # ---- 다운로드: 초고위험군 / 고위험군 ----
        dl_view = to_download_view(m1)

        top_risk = dl_view[dl_view["위험 등급"] == RED][["고객 ID", "위험 등급", "이탈 예측치"]]
        high_risk = dl_view[dl_view["위험 등급"].isin([RED, ORANGE])][["고객 ID", "위험 등급", "이탈 예측치"]]

        with st.expander("📥 다운로드", expanded=False):
            c1, c2 = st.columns(2)

            with c1:
                st.download_button(
                    "🔴 초고위험군 CSV",
                    data=top_risk.to_csv(index=False).encode("utf-8-sig"),
                    file_name="model1_top_risk.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_model1_top"   
                )

            with c2:
                st.download_button(
                    "🔴+🟠 고위험군 CSV",
                    data=high_risk.to_csv(index=False).encode("utf-8-sig"),
                    file_name="model1_high_risk.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_model1_high" 
                )



# =============================
# Tab 3 : Model 2
# =============================
with tab3:
    if m2 is None:
        st.info("시청 관련 데이터 기반 파일을 업로드하면 결과가 표시됩니다.")
    else:
        st.subheader("시청 관련 데이터 기반 결과")
        st.dataframe(risk_dist(m2), use_container_width=True, hide_index=True)
        table = present(
            m2,
            ["customer_id", "risk_group", "churn_probability"],
            ["risk_group", "churn_probability"],
            [True, False],
            TOP_N
        )
        st.dataframe(table, use_container_width=True, hide_index=True)

        dl_view = to_download_view(m2)

        top_risk = dl_view[dl_view["위험 등급"] == RED][["고객 ID", "위험 등급", "이탈 예측치"]]
        high_risk = dl_view[dl_view["위험 등급"].isin([RED, ORANGE])][["고객 ID", "위험 등급", "이탈 예측치"]]

        with st.expander("📥 다운로드", expanded=False):
            c1, c2 = st.columns(2)

            with c1:
                st.download_button(
                    "🔴 초고위험군 CSV",
                    data=top_risk.to_csv(index=False).encode("utf-8-sig"),
                    file_name="model2_top_risk.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_model2_top"  
                )

            with c2:
                st.download_button(
                    "🔴+🟠 고위험군 CSV",
                    data=high_risk.to_csv(index=False).encode("utf-8-sig"),
                    file_name="model2_high_risk.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_model2_high"  
                )


        


# =============================
# Tab 4 : Pain
# =============================
with tab4:
    if pain is None:
        st.info("불만지수 파일을 업로드하면 결과가 표시됩니다.")
    else:
        st.subheader("불만지수 결과")
        st.dataframe(risk_dist_raw(pain), use_container_width=True, hide_index=True) #불만 지수 데이터의 요약표 생성
        table = present( #고객별 pain_score를 기준정렬 -> 이탈 가능성이 높은 '불만 고객'을 우선 배치
            pain,
            ["customer_id", "risk_group", "pain_score"],
            ["risk_group", "pain_score"],
            [True, False],
            TOP_N
        ).reset_index(drop=True) 
        st.dataframe(table, use_container_width=True, hide_index=True)

        dl_view = to_download_view(pain)
        # 다운로드용 데이터 필터링 (불만 지수 기반)
        top_risk = dl_view[dl_view["위험 등급"] == RED][["고객 ID", "위험 등급", "불만 점수"]]
        high_risk = dl_view[dl_view["위험 등급"].isin([RED, ORANGE])][["고객 ID", "위험 등급", "불만 점수"]]


        with st.expander("📥 다운로드", expanded=False):
            c1, c2 = st.columns(2)

            with c1:
                st.download_button(
                    "🔴 초고위험군 CSV",
                    data=top_risk.to_csv(index=False).encode("utf-8-sig"),
                    file_name="pain_top_risk.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_pain_top"
                )

            with c2:
                st.download_button(
                    "🔴+🟠 고위험군 CSV",
                    data=high_risk.to_csv(index=False).encode("utf-8-sig"),
                    file_name="pain_high_risk.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_pain_high"
                )

#streamlit run app.py