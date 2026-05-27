import pandas as pd
import os
import sys

# 프로젝트 루트 경로 설정 (src 모듈을 불러오기 위함)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config as cfg


def process_and_save():
    print("🚀 데이터 전처리 및 변환 시작...")

    # 1. 파일 경로 설정
    # (data/0_raw 폴더 안에 'raw_data.csv'가 있다고 가정)
    # input_path = os.path.join(cfg.RAW_DATA_PATH, "241224_3기_추가데이터1", "*_VOD.csv")
    # input_path = os.path.join(cfg.RAW_DATA_PATH, "241224_3기_추가데이터1", "vod_mart_data.csv")
    # input_path = os.path.join(cfg.RAW_DATA_PATH, "250102_3기_데이터2_해지스코어(해당월 추가)", "*.csv")
    input_path = os.path.join(cfg.RAW_DATA_PATH, "250207_3기_데이터(추가2)", "*.csv")
    # output_path = os.path.join(cfg.PROCESSED_DATA_PATH, "VOD_DATA.parquet")
    # output_path = os.path.join(cfg.PROCESSED_DATA_PATH, "MART_DATA.parquet")
    # output_path = os.path.join(cfg.PROCESSED_DATA_PATH, "TPS_DATA.parquet")
    output_path = os.path.join(cfg.PROCESSED_DATA_PATH, "SHA_DATA.parquet")

    # 2. 데이터 로드 (필요한 경우 chunksize 사용 고려)
    if not os.path.exists(input_path):
        print(f"❌ 원본 파일을 찾을 수 없습니다: {input_path}")
        return

    print(f"📂 원본 로드 중: {input_path}")

    # 📝 [메모리 절약 팁]
    # 전체 컬럼이 너무 많다면, 여기서 필요한 컬럼만 지정해서 읽으세요.
    # use_cols = ['ID', 'DATE', 'VIEW_TIME', 'CHURN', ...]
    # df = pd.read_csv(input_path, usecols=use_cols)

    # 일단 CSV라고 가정하고 읽기 (파일 형식에 따라 read_excel 등 변경)
    df = pd.read_csv(input_path)

    # Parquet로 저장 (압축 사용)
    print(f"💾 Parquet 저장 중: {output_path}")
    df.to_parquet(output_path, engine="pyarrow", index=False, compression="gzip")

    print("✅ 변환 완료! 성공적으로 저장되었습니다.")


if __name__ == "__main__":
    process_and_save()
