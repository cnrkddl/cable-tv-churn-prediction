import pandas as pd
import os
import sys
import glob

# 설정 파일 불러오기
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import config as cfg


def get_common_ids(file_list_A, file_list_B, id_col="sha2_hash"):
    """
    두 파일 리스트에서 'sha2_hash'만 읽어와서 교집합(공통 ID)을 구함
    """
    print("🔍 [Step 1] 공통 고객 ID 추출 중...")

    ids_A = set()
    ids_B = set()

    # 그룹 A (TPS) ID 수집
    for f in file_list_A:
        print(f"   - Reading ID from: {os.path.basename(f)}")
        # usecols로 ID만 읽으므로 메모리를 거의 안 씀
        df = pd.read_csv(f, usecols=[id_col])
        ids_A.update(df[id_col].astype(str).unique())

    # 그룹 B (SHA) ID 수집
    for f in file_list_B:
        print(f"   - Reading ID from: {os.path.basename(f)}")
        df = pd.read_csv(f, usecols=[id_col])
        ids_B.update(df[id_col].astype(str).unique())

    # 교집합 구하기
    common_ids = ids_A.intersection(ids_B)
    print(f"✅ 공통 고객 수: {len(common_ids):,} 명")
    return common_ids


def filter_and_save(file_list, valid_ids, output_path, id_col="고객ID"):
    """
    CSV를 읽어서 valid_ids에 있는 행만 남기고 Parquet로 저장
    """
    print(f"🚜 [Step 2] 데이터 필터링 및 병합 저장 -> {output_path}")

    merged_df_list = []

    for f in file_list:
        print(f"   - Processing: {os.path.basename(f)}")
        try:
            # 1. CSV 읽기
            df = pd.read_csv(f)

            # 2. ID 문자열 변환 (안전장치)
            df[id_col] = df[id_col].astype(str)

            # 3. 공통 ID만 남기기 (Filtering)
            filtered_df = df[df[id_col].isin(valid_ids)]

            # 4. 리스트에 담기
            merged_df_list.append(filtered_df)

        except Exception as e:
            print(f"⚠️ 파일 처리 중 오류 발생 ({f}): {e}")

    # 5. 하나로 합치기
    if merged_df_list:
        final_df = pd.concat(merged_df_list, ignore_index=True)

        # 6. 파케이로 저장 (압축)
        final_df.to_parquet(
            output_path, engine="pyarrow", compression="gzip", index=False
        )
        print(f"💾 저장 완료: {output_path} ({len(final_df):,} rows)")
        return final_df
    else:
        return pd.DataFrame()


def main():
    # 1. 파일 목록 가져오기 (glob 사용)
    # 실제 파일 경로에 맞게 패턴 수정 필요 (*.csv)
    tps_files = glob.glob(os.path.join(cfg.RAW_DATA_PATH, "TPS_*.csv"))
    sha_files = glob.glob(os.path.join(cfg.RAW_DATA_PATH, "SHA_*.csv"))

    if not tps_files or not sha_files:
        print("❌ 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return

    # 2. 공통 ID 찾기 (살생부 작성)
    valid_ids = get_common_ids(tps_files, sha_files)

    # 3. TPS 데이터 처리 (필터링 후 저장)
    tps_out = os.path.join(cfg.PROCESSED_DATA_PATH, "TPS_Filtered_Combined.parquet")
    df_tps = filter_and_save(tps_files, valid_ids, tps_out)

    # 4. SHA 데이터 처리 (필터링 후 저장)
    sha_out = os.path.join(cfg.PROCESSED_DATA_PATH, "SHA_Filtered_Combined.parquet")
    df_sha = filter_and_save(sha_files, valid_ids, sha_out)

    # 5. (선택사항) 최종적으로 TPS와 SHA를 옆으로 합치고 싶다면? (Merge)
    # 데이터가 너무 크면 이 단계는 생략하고, 분석할 때 따로 불러서 합치는 게 나을 수 있음
    print("🔄 [Step 3] 최종 TPS + SHA 병합 시도...")
    try:
        final_merge = pd.merge(df_tps, df_sha, on="고객ID", how="inner")
        final_out = os.path.join(cfg.PROCESSED_DATA_PATH, "Master_Final_22M.parquet")
        final_merge.to_parquet(final_out, compression="gzip")
        print(f"🎉 전체 병합 완료! 최종 파일: {final_out}")
    except Exception as e:
        print(f"⚠️ 메모리 부족으로 최종 병합 실패 (개별 파일은 저장됨): {e}")


if __name__ == "__main__":
    main()
