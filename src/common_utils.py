import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats
import src.config as config_module

# config 파일에서 설정 및 컬럼 설명 가져오기
from src.config import logger, DEFAULT_COLUMN_RULES, COLUMN_DESC, init_settings

try:
    from IPython.display import display
except ImportError:
    display = print

# 환경설정 자동 적용
init_settings()


# -----------------------------------------------------------
# [기능 1] 데이터 로드 및 구조 확인
# -----------------------------------------------------------
def load_parquet(file_path, cols=None, lang="eng"):
    """
    Parquet 파일 로드 함수 (다국어/대소문자 지원)

    Args:
        file_path (str): 파일 경로
        cols (list): 로드할 컬럼 리스트 (한글, 영문, 대소문자 혼용 가능)
        lang (str): 결과 DataFrame의 컬럼 언어 설정 ('eng' 또는 'kor')
                    - 'eng': 영문 대문자로 통일 (예: AGE_GRP10)
                    - 'kor': 한글 설명으로 통일 (예: 연령대)
    """
    if not os.path.exists(file_path):
        log(f"❌ 파일 없음: {file_path}")
        return pd.DataFrame()

    try:
        final_cols = None

        # 실제 파일의 컬럼명 확인을 위한 스키마 로드
        import pyarrow.parquet as pq

        schema = pq.read_schema(file_path)
        actual_cols = (
            schema.names
        )  # 파일에 저장된 실제 컬럼명 리스트 (대부분 대문자라고 가정)

        # -------------------------------------------------------
        # 1. 입력 매핑 테이블 생성 (사용자 입력 -> 실제 파일 컬럼)
        # -------------------------------------------------------
        input_to_actual = {}

        # (1) 실제 컬럼명 매핑 (대소문자 무시)
        # 예: 'age_grp10' -> 'AGE_GRP10', 'AGE_GRP10' -> 'AGE_GRP10'
        for col in actual_cols:
            input_to_actual[col] = col
            input_to_actual[col.upper()] = col
            input_to_actual[col.lower()] = col

        # (2) 한글 컬럼명 매핑 (COLUMN_DESC 활용)
        # 예: '연령대' -> 'AGE_GRP10'
        # COLUMN_DESC의 Key(영문)가 실제 컬럼과 매칭되는지 확인
        eng_to_kor = {}  # 나중에 출력 변환용
        kor_to_eng = {}  # 입력 해석용

        for key, val in COLUMN_DESC.items():
            # config의 키(key)가 실제 파일에 있는지 확인 (대소문자 무시)
            upper_key = key.upper()
            if upper_key in input_to_actual:
                real_col = input_to_actual[upper_key]
                input_to_actual[val] = real_col  # 한글 -> 실제컬럼
                eng_to_kor[real_col] = val  # 실제컬럼 -> 한글
                kor_to_eng[val] = real_col

        # -------------------------------------------------------
        # 2. 로드할 컬럼 선별
        # -------------------------------------------------------
        if cols:
            load_set = set()
            for user_col in cols:
                user_col_str = str(user_col).strip()

                # 입력값이 매핑 테이블에 있으면 실제 컬럼명 추가
                if user_col_str in input_to_actual:
                    load_set.add(input_to_actual[user_col_str])
                # 혹시 매핑엔 없지만 대소문자 변환 시도해서 있으면 추가
                elif user_col_str.upper() in input_to_actual:
                    load_set.add(input_to_actual[user_col_str.upper()])
                else:
                    log(f"⚠️ 경고: '{user_col}' 컬럼을 찾을 수 없어 제외합니다.")

            if not load_set:
                log("❌ 유효한 컬럼이 하나도 없습니다. 전체 데이터를 로드합니다.")
                final_cols = None
            else:
                final_cols = list(load_set)

        # -------------------------------------------------------
        # 3. 데이터 로드
        # -------------------------------------------------------
        df = pd.read_parquet(file_path, columns=final_cols)

        # -------------------------------------------------------
        # 4. 언어 변환 (lang 옵션 적용)
        # -------------------------------------------------------
        rename_map = {}

        if lang == "kor":
            # 실제 영문 컬럼 -> 한글 (COLUMN_DESC에 있는 경우만)
            for col in df.columns:
                if col in eng_to_kor:
                    rename_map[col] = eng_to_kor[col]
            if rename_map:
                df = df.rename(columns=rename_map)
                log(f"🔤 컬럼명 변환 (To Korean): {len(rename_map)}개 변환됨")

        elif lang == "eng":
            # 모든 컬럼을 대문자로 통일 (표준화)
            rename_map = {col: col.upper() for col in df.columns if col != col.upper()}
            if rename_map:
                df = df.rename(columns=rename_map)
                # log(f"🔤 컬럼명 변환 (To English Upper): {len(rename_map)}개 변환됨")

        log(f"📂 로드 완료: {len(df):,} Rows (Selected {len(df.columns)} Columns)")
        return df

    except Exception as e:
        log(f"❌ 로드 실패: {e}")
        return pd.DataFrame()


def i_view(df, sample_n=5):
    """[Info View] 데이터 구조 확인"""
    if df.empty:
        print("📭 Empty DataFrame")
        return

    print(f"\n📊 [Structure] Shape: {df.shape}")
    display(df.head(sample_n))

    summary = pd.DataFrame(
        {"Dtype": df.dtypes, "Null_Count": df.isnull().sum(), "N_Unique": df.nunique()}
    )
    print("\n🔍 [Quality Summary]")
    display(summary)


def s_view(df):
    """
    데이터프레임의 현재 컬럼명(한글/영문)에 맞춰 설명을 출력
    """
    if df.empty:
        print("📭 Empty DataFrame")
        return

    print(f"\n📋 [Column Dictionary] Total {len(df.columns)} Columns")
    print("=" * 70)
    print(f"{'Current Name':<30} | {'Description / Original Key'}")
    print("-" * 70)

    # 설명을 찾기 위한 역매핑 (Korean -> English Key)
    kor_to_eng = {v: k for k, v in COLUMN_DESC.items()}

    for col in df.columns:
        desc = "-"

        # 1. 컬럼명이 한글인 경우 -> 원래 영문 키를 찾음
        if col in kor_to_eng:
            desc = f"Original: {kor_to_eng[col]}"

        # 2. 컬럼명이 영문인 경우 (대문자 변환 고려) -> 한글 설명을 찾음
        else:
            # COLUMN_DESC에서 대소문자 무시하고 찾기
            for k, v in COLUMN_DESC.items():
                if k.upper() == col.upper():
                    desc = v
                    break

        print(f"{col:<30} | {desc}")
    print("=" * 70)


# -----------------------------------------------------------
# [기능 2] 데이터 전처리 (추출, 파생변수)
# -----------------------------------------------------------
# ==========================================================
# [기능 2.5] 데이터 전처리 (Preprocessing) - Final
# ==========================================================
# ==========================================================
# [기능 4] 데이터셋 분할 (Split) - Train/Test & Downsampling
# ==========================================================
def split_oot_dataset(
    df,
    train_months,
    test_months,
    target_col,
    date_col="P_MT",
    id_col="SHA2_HASH",
    sampling_ratio=None,
):
    """
    OOT(Out-of-Time) 방식 데이터 분할 및 다운샘플링 함수

    Args:
        df (pd.DataFrame): 전처리가 완료된 데이터프레임
        train_months (list): 학습 월 리스트 (예: ['202302', ...])
        test_months (list): 평가 월 리스트 (예: ['202311', ...])
        target_col (str): 타겟 컬럼명 (예: 'derived_cancel_yn')
        date_col (str): 기준년월 컬럼명 (기본값: 'P_MT')
        id_col (str): 고객ID 컬럼명 (기본값: 'SHA2_HASH')
        sampling_ratio (float): 다운샘플링 비율 (유지:해지). None이면 수행 안 함.

    Returns:
        X_train, y_train, X_test, y_test
    """
    log("🚀 데이터셋 분할(split_oot_dataset) 시작...")

    # 0. 컬럼명 대소문자 호환 처리 (p_mt -> P_MT 등)
    def find_col(name):
        for c in df.columns:
            if c.upper() == name.upper():
                return c
        return name

    real_date_col = find_col(date_col)
    real_id_col = find_col(id_col)

    # 1. Train Set 생성
    # 날짜 컬럼을 문자열로 변환하여 리스트와 비교
    train_mask = df[real_date_col].astype(str).isin([str(m) for m in train_months])
    df_train = df[train_mask].copy()

    # 2. Test Set 생성
    test_mask = df[real_date_col].astype(str).isin([str(m) for m in test_months])
    df_test = df[test_mask].copy()

    log(
        f"📊 분할 결과: Train {len(df_train):,}건 (기간: {train_months[0]}~{train_months[-1]}), Test {len(df_test):,}건"
    )

    # 3. 다운샘플링 (Train 데이터에만 적용)
    if sampling_ratio is not None and not df_train.empty:
        # 타겟값(0, 1) 분리
        major = df_train[df_train[target_col] == 0]  # 유지
        minor = df_train[df_train[target_col] == 1]  # 해지

        # 해지 데이터가 있을 경우에만 수행
        if len(minor) > 0:
            target_n = int(len(minor) * sampling_ratio)

            # 유지 데이터가 타겟 수보다 많을 때만 줄임
            if len(major) > target_n:
                major_sampled = major.sample(n=target_n, random_state=42)
                df_train = pd.concat([major_sampled, minor])
                # 셔플 (섞기)
                df_train = df_train.sample(frac=1, random_state=42).reset_index(
                    drop=True
                )

                log(
                    f"✂️ 다운샘플링 적용 (비율 1:{sampling_ratio}): Train 데이터가 {len(df_train):,}건으로 조정됨"
                )
            else:
                log("⚠️ 다운샘플링 미적용: 유지 데이터가 비율보다 적음")

    # 4. X, y 분리 및 불필요 컬럼(ID, 날짜, 타겟) 제거
    # 학습에 방해되는 식별자 컬럼들 제거 리스트
    drop_cols = [real_date_col, target_col]
    if real_id_col in df_train.columns:
        drop_cols.append(real_id_col)

    # 실제 존재하는 컬럼만 골라서 삭제
    train_drop = [c for c in drop_cols if c in df_train.columns]
    test_drop = [c for c in drop_cols if c in df_test.columns]

    X_train = df_train.drop(columns=train_drop)
    y_train = df_train[target_col]

    X_test = df_test.drop(columns=test_drop)
    y_test = df_test[target_col]

    return X_train, y_train, X_test, y_test


def data_preprocess(df, rules=None):
    """
    [전처리 전용 함수]
    Config에 정의된 규칙(DEFAULT_COLUMN_RULES)을 기반으로
    1. 값 매핑 (Mapping)
    2. 결측치 처리 (Fillna)
    3. 데이터 타입 변환 (Type Casting)
    4. 컬럼명 변경 (Renaming)
    을 수행합니다.
    """
    if df.empty:
        return df

    # 규칙이 없으면 기본 규칙 로드
    if rules is None:
        rules = config_module.DEFAULT_COLUMN_RULES

    log("🚀 데이터 전처리(data_preprocess) 시작...")

    # 변경할 컬럼명 저장소
    rename_map = {}

    # 규칙 반복 적용
    for rule_key, rule_info in rules.items():
        # -------------------------------------------------
        # 1. 실제 컬럼 찾기
        # -------------------------------------------------
        target_col = None

        # (1) Rule Key가 컬럼에 그대로 있는 경우
        if rule_key in df.columns:
            target_col = rule_key
        # (2) 한글 설명으로 되어 있는 경우
        elif (
            rule_key in config_module.COLUMN_DESC
            and config_module.COLUMN_DESC[rule_key] in df.columns
        ):
            target_col = config_module.COLUMN_DESC[rule_key]
        # (3) 대문자로 있는 경우
        elif rule_key.upper() in df.columns:
            target_col = rule_key.upper()

        if target_col is None:
            continue

        # -------------------------------------------------
        # 2. 값 매핑 (Mapping)
        # -------------------------------------------------
        if rule_info.get("mapping"):
            df[target_col] = df[target_col].map(rule_info["mapping"])

        # -------------------------------------------------
        # 3. 결측치 채우기 (Fillna)
        # -------------------------------------------------
        fill_val = rule_info.get("fillna")
        if fill_val is not None:
            df[target_col] = df[target_col].fillna(fill_val)

        # -------------------------------------------------
        # 4. 타입 변환 (Type Casting)
        # -------------------------------------------------
        dtype = rule_info.get("dtype")
        if dtype:
            try:
                # int 변환 시 NaN이 있으면 pass (나중에 dropna로 처리)
                if dtype == int and df[target_col].isnull().any():
                    pass
                else:
                    df[target_col] = df[target_col].astype(dtype)
            except Exception:
                pass

        # -------------------------------------------------
        # 5. 이름 변경 준비 (Renaming)
        # -------------------------------------------------
        new_name = rule_info.get("new_name")
        if new_name and target_col != new_name:
            rename_map[target_col] = new_name

    # -------------------------------------------------
    # 6. 컬럼명 일괄 변경 적용
    # -------------------------------------------------
    if rename_map:
        df = df.rename(columns=rename_map)
        log(f"✨ 전처리 완료: {len(rename_map)}개 컬럼 변환 및 정리됨")

    return df


def get_data_set(df, id_col, date_col, target_month=None, dedup=True, date_op="eq"):
    """데이터 필터링 및 중복제거"""
    if df.empty:
        return df

    # 컬럼명이 대문자나 한글로 바뀌었을 수 있으므로 찾아야 함
    def find_col_in_df(target_name):
        # 1. 그대로 존재
        if target_name in df.columns:
            return target_name
        # 2. 대문자로 존재
        if target_name.upper() in df.columns:
            return target_name.upper()
        # 3. 한글/영문 매핑 확인
        # target_name이 한글인 경우 -> 영문 키 찾기
        kor_to_eng = {v: k for k, v in COLUMN_DESC.items()}
        if target_name in kor_to_eng:
            eng = kor_to_eng[target_name]
            if eng in df.columns:
                return eng
            if eng.upper() in df.columns:
                return eng.upper()

        # target_name이 영문인 경우 -> 한글 값 찾기
        if target_name in COLUMN_DESC:
            kor = COLUMN_DESC[target_name]
            if kor in df.columns:
                return kor

        # COLUMN_DESC 키 중에 대소문자 무시하고 찾기
        for k, v in COLUMN_DESC.items():
            if k.upper() == target_name.upper():
                if v in df.columns:
                    return v

        return target_name  # 못 찾으면 그대로 반환 (에러 발생 가능)

    real_id = find_col_in_df(id_col)
    real_date = find_col_in_df(date_col)

    # 1. 날짜 필터링
    if target_month is not None:
        original_len = len(df)
        try:
            if pd.api.types.is_integer_dtype(df[real_date]):
                target_month = int(target_month)
            else:
                target_month = str(target_month)
        except:
            pass

        if date_op == "eq":
            df = df[df[real_date] == target_month].copy()
            op_str = "=="
        elif date_op == "lte":
            df = df[df[real_date] <= target_month].copy()
            op_str = "<="

        log(
            f"📅 기간 필터링 ({real_date} {op_str} {target_month}): {original_len:,} -> {len(df):,} 행"
        )
        if df.empty:
            return df

    # 2. 정렬 (ID ASC, Date DESC)
    real_cancel = find_col_in_df("cancel_yn")

    sort_cols = [real_id, real_date]
    asc_order = [True, False]

    if real_cancel in df.columns:
        sort_cols.append(real_cancel)
        asc_order.append(False)

    df = df.sort_values(by=sort_cols, ascending=asc_order)

    # 3. 중복 제거
    if dedup:
        before_len = len(df)
        df = df.drop_duplicates(subset=[real_id], keep="first").reset_index(drop=True)
        log(f"✂️ 중복제거 완료: {before_len:,} -> {len(df):,} 행")
    else:
        df = df.reset_index(drop=True)
        log(f"📚 중복제거 미수행 (단순 정렬): {len(df):,} 행")

    return df


def create_derived_variable(df, src_col, mapping=None, new_name=None, fillna_val=None):
    """
    파생변수 생성 함수 (한글/영문 자동 감지)
    - 원본 컬럼이 한글로 되어 있으면, 파생변수도 한글명(config 매핑)으로 생성
    - [NEW] src_col이 리스트일 경우 일괄 반복 처리
    """
    if df.empty:
        return df

    # =========================================================
    # [수정] 리스트 입력 지원 (재귀 호출)
    # =========================================================
    if isinstance(src_col, list):
        for col in src_col:
            # 리스트 내 각 컬럼에 대해 함수를 반복 호출합니다.
            # mapping, new_name 등은 개별 적용이 어려우므로 보통 None으로 두어 Config 규칙을 따르게 합니다.
            df = create_derived_variable(df, col, mapping, new_name, fillna_val)
        return df

    # =========================================================
    # [기존 로직] 단일 컬럼 처리
    # =========================================================

    # [중요] 최신 설정 참조 (Reload 대응)
    DEFAULT_COLUMN_RULES = config_module.DEFAULT_COLUMN_RULES
    COLUMN_DESC = config_module.COLUMN_DESC

    # 1. 실제 컬럼 찾기
    real_src_col = None
    if src_col in df.columns:
        real_src_col = src_col
    elif src_col.upper() in df.columns:
        real_src_col = src_col.upper()
    else:
        kor_to_eng = {v: k for k, v in COLUMN_DESC.items()}
        if src_col in kor_to_eng:
            eng_key = kor_to_eng[src_col]
            if eng_key in df.columns:
                real_src_col = eng_key
            elif eng_key.upper() in df.columns:
                real_src_col = eng_key.upper()
        elif src_col in COLUMN_DESC:
            kor_key = COLUMN_DESC[src_col]
            if kor_key in df.columns:
                real_src_col = kor_key
        else:
            for k, v in COLUMN_DESC.items():
                if k.upper() == src_col.upper():
                    if v in df.columns:
                        real_src_col = v
                        break

    if not real_src_col:
        # log(f"⚠️ 컬럼을 찾을 수 없음: {src_col}") # 필요시 주석 해제
        return df

    # 2. 규칙 찾기 (English Key 기준)
    config = {}
    rule_key = src_col

    # real_src_col이 한글이면 영문 키로 변환하여 규칙 검색
    is_kor_mode = False
    kor_to_eng = {v: k for k, v in COLUMN_DESC.items()}

    if real_src_col in kor_to_eng:
        rule_key = kor_to_eng[real_src_col]
        is_kor_mode = True  # 소스 컬럼이 한글이면 한글 모드 ON
    elif real_src_col in COLUMN_DESC:
        rule_key = real_src_col

    if rule_key in DEFAULT_COLUMN_RULES:
        config = DEFAULT_COLUMN_RULES[rule_key]
    elif rule_key.upper() in DEFAULT_COLUMN_RULES:
        config = DEFAULT_COLUMN_RULES[rule_key.upper()]
    elif rule_key.lower() in DEFAULT_COLUMN_RULES:
        config = DEFAULT_COLUMN_RULES[rule_key.lower()]

    final_map = mapping if mapping else config.get("mapping")

    # 3. 새 컬럼명 결정 (한글 모드 반영)
    if new_name:
        final_name = new_name
    elif config.get("new_name"):
        temp_name = config.get("new_name")
        # [NEW] 한글 모드이고, 해당 파생변수의 한글 매핑이 있으면 변환
        if is_kor_mode and temp_name in COLUMN_DESC:
            final_name = COLUMN_DESC[temp_name]
        else:
            final_name = temp_name
    else:
        final_name = f"{real_src_col}_new"

    final_fill = fillna_val if fillna_val is not None else config.get("fillna", -1)
    target_dtype = config.get("dtype", int)

    # 4. 변환 적용
    if final_map:
        log(f"🔢 파생변수 생성(Mapping): {real_src_col} -> {final_name}")
        df[final_name] = df[real_src_col].map(final_map)
    else:
        if real_src_col != final_name:
            log(f"🔢 파생변수 생성(Copy): {real_src_col} -> {final_name}")
            df[final_name] = df[real_src_col]

    df[final_name] = df[final_name].fillna(final_fill)
    try:
        df[final_name] = df[final_name].astype(target_dtype)
    except:
        pass

    return df


def delete_source_columns(df, target_cols):
    """
    [기능] 원본 컬럼 삭제 함수 (리스트 지원 & 한글/영문 자동 인식)
    - 입력된 컬럼명(target_cols)이 한글이든 영문이든, 현재 DataFrame에 존재하는 실제 이름을 찾아 삭제합니다.
    """
    if df.empty or not target_cols:
        return df

    # [중요] 최신 설정 참조
    COLUMN_DESC = config_module.COLUMN_DESC

    # 입력이 문자열 하나라면 리스트로 변환
    if isinstance(target_cols, str):
        target_cols = [target_cols]

    # 삭제할 실제 컬럼명들을 담을 리스트
    cols_to_drop = []

    # 매핑용 딕셔너리 (Korean -> English)
    kor_to_eng = {v: k for k, v in COLUMN_DESC.items()}

    for col in target_cols:
        found_col = None

        # 1. 현재 이름 그대로 존재
        if col in df.columns:
            found_col = col

        # 2. 대문자로 존재
        elif col.upper() in df.columns:
            found_col = col.upper()

        # 3. 매핑 확인 (한글 입력 -> 영문 컬럼 찾기)
        elif col in kor_to_eng:
            eng_key = kor_to_eng[col]
            if eng_key in df.columns:
                found_col = eng_key
            elif eng_key.upper() in df.columns:
                found_col = eng_key.upper()

        # 4. 매핑 확인 (영문 입력 -> 한글 컬럼 찾기)
        elif col in COLUMN_DESC:
            kor_key = COLUMN_DESC[col]
            if kor_key in df.columns:
                found_col = kor_key

        # 5. Config의 Key(대소문자 무시)로 한글 값 찾기
        else:
            for k, v in COLUMN_DESC.items():
                if k.upper() == col.upper():
                    if v in df.columns:
                        found_col = v
                        break

        # 실제 존재하는 컬럼이면 삭제 리스트에 추가
        if found_col:
            cols_to_drop.append(found_col)

    # 중복 제거 후 삭제 수행
    cols_to_drop = list(set(cols_to_drop))

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        log(f"🗑️ 원본 컬럼 {len(cols_to_drop)}개 삭제 완료")
    else:
        log("⚠️ 삭제할 컬럼을 찾지 못했습니다 (이미 삭제되었거나 이름이 다름)")

    return df


# -----------------------------------------------------------
# [기능 3] 통계 분석 및 시각화
# -----------------------------------------------------------
def g_view(df, x_col, y_col, figsize=(10, 6), rot=0, palette=None):
    """자동 시각화 및 검정"""

    # 컬럼 찾기 헬퍼
    def resolve_col(c):
        if c in df.columns:
            return c
        if c.upper() in df.columns:
            return c.upper()
        # 한글/영문 교차 확인
        kor_to_eng = {v: k for k, v in COLUMN_DESC.items()}
        # c가 한글이면 -> 영문으로 변환해서 확인 (DF가 영문일 때)
        if c in kor_to_eng:
            eng = kor_to_eng[c]
            if eng in df.columns:
                return eng
            if eng.upper() in df.columns:
                return eng.upper()
        # c가 영문이면 -> 한글로 변환해서 확인 (DF가 한글일 때)
        for k, v in COLUMN_DESC.items():
            if k.upper() == c.upper():
                if v in df.columns:
                    return v
        return c

    real_x = resolve_col(x_col)
    real_y = resolve_col(y_col)

    if df.empty or real_x not in df.columns or real_y not in df.columns:
        # print(f"Skipping visualization: {x_col} or {y_col} not found.")
        return

    print(f"\n📊 [Auto Analysis] '{real_x}' vs '{real_y}'")
    print("-" * 60)

    def is_categorical(col):
        if not pd.api.types.is_numeric_dtype(df[col]):
            return True
        if df[col].nunique() < 20:
            return True
        return False

    is_x_cat = is_categorical(real_x)
    is_y_cat = is_categorical(real_y)
    if palette is None:
        palette = "coolwarm" if is_x_cat and is_y_cat else "Set2"

    # Case 1: 범주 vs 범주
    if is_x_cat and is_y_cat:
        print("💡 Type: Categorical vs Categorical (Chi-Square Test)")
        cross_tab = pd.crosstab(df[real_x], df[real_y])
        try:
            chi2, p, _, _ = stats.chi2_contingency(cross_tab)
            res = "Significant" if p < 0.05 else "Not Significant"
            print(f"🧪 P-value: {p:.4e} ({res})")
        except:
            print("🧪 검정 실패")

        cross_tab_prop = pd.crosstab(df[real_x], df[real_y], normalize="index")
        cross_tab_prop.plot(
            kind="bar",
            stacked=True,
            figsize=figsize,
            colormap=palette,
            alpha=0.85,
            rot=rot,
        )
        plt.title(f"[{real_x}] -> [{real_y}] 비율")
        plt.show()

    # Case 2: 범주 vs 수치
    elif is_x_cat and not is_y_cat:
        print("💡 Type: Categorical vs Numerical (ANOVA / T-test)")
        groups = [g[real_y].dropna().values for _, g in df.groupby(real_x)]
        try:
            if len(groups) == 2:
                _, p = stats.ttest_ind(*groups)
            else:
                _, p = stats.f_oneway(*groups)
            res = "Significant" if p < 0.05 else "Not Significant"
            print(f"🧪 P-value: {p:.4e} ({res})")
        except:
            pass

        plt.figure(figsize=figsize)
        sns.boxplot(x=real_x, y=real_y, data=df, palette=palette)
        plt.title(f"[{real_x}]별 [{real_y}] 분포")
        plt.xticks(rotation=rot)
        plt.show()

    # Case 3: 수치 vs 범주 (Swap)
    elif not is_x_cat and is_y_cat:
        g_view(df, y_col, x_col, figsize, rot, palette)

    # Case 4: 수치 vs 수치
    else:
        print("💡 Type: Numerical vs Numerical (Correlation)")
        df_sub = df[[real_x, real_y]].dropna()
        if len(df_sub) > 1:
            corr, p = stats.pearsonr(df_sub[real_x], df_sub[real_y])
            print(f"🧪 Pearson Corr: {corr:.4f} (P-value: {p:.4e})")
        plt.figure(figsize=figsize)
        sns.regplot(
            x=real_x,
            y=real_y,
            data=df_sub,
            scatter_kws={"alpha": 0.3},
            line_kws={"color": "red"},
        )
        plt.title(f"[{real_x}] vs [{real_y}]")
        plt.show()


# -----------------------------------------------------------
# [기능 4] 유틸리티
# -----------------------------------------------------------
def log(msg):
    """config의 logger 사용"""
    logger.info(msg)


view = i_view
