import pandas as pd
import os
import sys
import glob

# -----------------------------------------------------------
# 1. 환경 설정 및 공통 모듈 로드
# -----------------------------------------------------------
# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config as cfg
from src import common_utils as utils  # [NEW] 공통 함수 모듈 임포트


# -----------------------------------------------------------
# [기능 1] 공통 ID 추출 함수 (PyArrow 유지)
# -----------------------------------------------------------
def get_common_ids(tps_files, sha_files, id_col="sha2_hash"):
    utils.log("🔍 [Step 1] 공통 고객 ID(sha2_hash) 추출 시작...")

    ids_tps = set()
    ids_sha = set()

    utils.log("   >> TPS 파일 ID 스캔 중...")
    for f in tps_files:
        try:
            # 여긴 nrows가 없으므로 PyArrow 사용 가능 (속도 UP)
            df = pd.read_csv(f, usecols=[id_col], engine="pyarrow")
            ids_tps.update(df[id_col].astype(str).unique())
        except Exception as e:
            utils.log(f"⚠️ TPS 파일 읽기 실패 ({os.path.basename(f)}): {e}")

    utils.log("   >> SHA 파일 ID 스캔 중...")
    for f in sha_files:
        try:
            df = pd.read_csv(f, usecols=[id_col], engine="pyarrow")
            ids_sha.update(df[id_col].astype(str).unique())
        except Exception as e:
            utils.log(f"⚠️ SHA 파일 읽기 실패 ({os.path.basename(f)}): {e}")

    common_ids = ids_tps.intersection(ids_sha)
    utils.log(f"✅ 공통 고객 ID 추출 완료: {len(common_ids):,} 명")
    return common_ids


# -----------------------------------------------------------
# [기능 2] 데이터 처리 함수 (PyArrow 유지)
# -----------------------------------------------------------
def process_sha_data(sha_files, valid_ids, output_name, id_col="sha2_hash"):
    utils.log(
        f"🚜 [Step 2] SHA 데이터 처리 시작 (Load -> Filter -> Sort -> Dedup -> Save)"
    )

    merged_chunk = []

    # 저장 경로 설정
    processed_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "10_processed"
    )
    output_path = os.path.join(processed_dir, f"{output_name}.parquet")

    # [NEW] 공통 함수: 폴더가 없으면 자동 생성
    utils.ensure_dir(output_path)

    for f in sha_files:
        filename = os.path.basename(f)
        utils.log(f"   >> 파일 로드 중: {filename}")
        try:
            # 전체 읽기: PyArrow 필수 (가장 시간 많이 걸리는 곳)
            df = pd.read_csv(f, engine="pyarrow")

            if id_col in df.columns:
                df[id_col] = df[id_col].astype(str)
                filtered_df = df[df[id_col].isin(valid_ids)].copy()
            else:
                utils.log(f"⚠️ 경고: {id_col} 컬럼 없음. 건너뜀.")
                continue

            # [NEW] 공통 함수: Float 타입 최적화 (메모리 절약)
            filtered_df = utils.optimize_float_types(filtered_df)

            merged_chunk.append(filtered_df)

        except Exception as e:
            utils.log(f"⚠️ 파일 처리 중 오류 ({filename}): {e}")

    if merged_chunk:
        utils.log("   >> 전체 데이터프레임 병합 중...")
        final_df = pd.concat(merged_chunk, ignore_index=True)

        sort_cols = ["sha2_hash", "p_mt", "cancel_yn"]
        missing_cols = [c for c in sort_cols if c not in final_df.columns]

        if not missing_cols:
            utils.log(
                "   >> 🌪️ 데이터 정렬 중 (sha2_hash:ASC, p_mt:ASC, cancel_yn:Y->N)..."
            )
            final_df = final_df.sort_values(
                by=["sha2_hash", "p_mt", "cancel_yn"], ascending=[True, True, False]
            )

            before_dedup = len(final_df)
            utils.log(f"   >> ✂️ 중복 제거 전: {before_dedup:,} 행")

            final_df = final_df.drop_duplicates(
                subset=["sha2_hash", "p_mt"], keep="first"
            )

            after_dedup = len(final_df)
            utils.log(
                f"   >> ✅ 중복 제거 완료: {after_dedup:,} 행 (삭제: {before_dedup - after_dedup:,} 건)"
            )

        else:
            utils.log(f"⚠️ 필수 컬럼 누락으로 정렬/중복제거 불가: {missing_cols}")

        utils.log(f"💾 Parquet 저장 중... ({output_path})")
        final_df.to_parquet(
            output_path, engine="pyarrow", compression="gzip", index=False
        )
        utils.log(f"🎉 작업 완료! 최종 행 수: {len(final_df):,} 행")

    else:
        utils.log("❌ 병합할 데이터가 없습니다.")


# -----------------------------------------------------------
# 3. 메인 실행부
# -----------------------------------------------------------
def main():
    project_root = os.path.dirname(os.path.dirname(__file__))
    tps_pattern = os.path.join(
        project_root,
        "data",
        "00_raw",
        "250102_3기_데이터2_해지스코어(해당월 추가)",
        "TPS_cancel*.csv",
    )
    sha_pattern = os.path.join(
        project_root, "data", "00_raw", "250207_3기_데이터(추가2)", "sha_tps*.csv"
    )

    tps_files = glob.glob(tps_pattern)
    sha_files = glob.glob(sha_pattern)

    utils.log("=======================================================")
    utils.log("🚀 [SHA 데이터 정제 프로세스 V5 (Refactored)] 시작")
    utils.log("=======================================================")

    if not tps_files or not sha_files:
        utils.log("❌ 파일을 찾을 수 없습니다.")
        sys.exit(1)

    # [NEW] 공통 함수: 헤더 검증 (실패 시 False 반환)
    if not utils.validate_header_consistency(tps_files, "TPS Group"):
        sys.exit(1)
    if not utils.validate_header_consistency(sha_files, "SHA Group"):
        sys.exit(1)

    # 데이터 처리는 PyArrow 엔진 사용 (속도 확보)
    common_ids = get_common_ids(tps_files, sha_files)

    if common_ids:
        process_sha_data(sha_files, common_ids, "df_sha")
    else:
        utils.log("❌ 공통 ID 없음. 종료.")


if __name__ == "__main__":
    main()
