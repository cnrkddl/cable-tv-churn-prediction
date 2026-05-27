# -*- coding: utf-8 -*-
import os
import sys
import platform
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc
from loguru import logger

# ==========================================================
# 1. 로거 설정 (Loguru)
# ==========================================================
logger.remove()
LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
)
os.makedirs(LOG_PATH, exist_ok=True)

logger.add(
    sys.stderr,
    format="<green>[{time:HH:mm:ss}]</green> <level>{message}</level>",
    level="INFO",
)
logger.add(
    os.path.join(LOG_PATH, "process_{time:YYYY-MM-DD}.log"),
    rotation="10 MB",
    retention="10 days",
    encoding="utf-8",
    level="INFO",
)


# ==========================================================
# 2. 데이터 변환 규칙 (Column Rules Registry)
# ==========================================================
DEFAULT_COLUMN_RULES = {
    # ------------------------------------------------------
    # [Target] 타겟 변수 (int 유지)
    # ------------------------------------------------------
    "cancel_yn": {
        "new_name": "derived_cancel_yn",
        "mapping": {"해지": 1, "유지": 0, "Y": 1, "N": 0, 1: 1, 0: 0},
        "fillna": None,  # 타겟 없는 행은 삭제
        "dtype": int,
    },
    # ------------------------------------------------------
    # [Category] 범주형 변수 -> dtype: str
    # ------------------------------------------------------
    "AGE_GRP10": {
        "new_name": "DERIVED_AGE_GRP10",
        "mapping": {
            "10대미만": "00",
            "10대": "10",
            "20대": "20",
            "30대": "30",
            "40대": "40",
            "50대": "50",
            "60대": "60",
            "70대": "70",
            "80대": "80",
            "90대이상": "90",
        },
        "fillna": None,
        "dtype": str,
    },
    "SVC_USE_DAYS_GRP": {
        "new_name": "DERIVED_SVC_USE_DAYS_GRP",
        "mapping": {
            "3개월미만": "0_3개월미만",
            "3~12개월": "1_3to12개월",
            "12~24개월": "2_12to24개월",
            "24~36개월": "3_24to36개월",
            "36개월이상": "4_36개월이상",
        },
        "fillna": None,
        "dtype": str,
    },
    "MEDIA_NM_GRP": {
        "new_name": "DERIVED_MEDIA_NM_GRP",
        # [수정] 알수없음 -> None (삭제 대상)
        "mapping": {"HD": "HD", "UHD": "UHD", "알수없음": None},
        "fillna": None,
        "dtype": str,
    },
    "PROD_NM_GRP": {
        "new_name": "DERIVED_PROD_NM_GRP",
        # [수정] 알수없음 -> None (삭제 대상)
        "mapping": {
            "이코노미/베이직": "1_이코노미_베이직",
            "스탠다드": "2_스탠다드",
            "프리미엄": "3_프리미엄",
            "VIP/플래티넘": "4_VIP_플래티넘",
            "알수없음": None,  # 삭제 대상
        },
        "fillna": None,
        "dtype": str,
    },
    "AGMT_KIND_NM": {
        "new_name": "DERIVED_AGMT_KIND_NM",
        # [전략] 영문 변환 대신 '한글 그대로' 사용 (직관성 확보)
        # 키(Key)와 값(Value)을 똑같이 두되, 제거할 대상만 None으로 설정
        "mapping": {
            "신규": "신규",
            "약정승계": "약정승계",
            "재약정": "재약정",
            "약정갱신": "약정갱신",
            "약정연장": "약정연장",
            "약정축소": "약정축소",
            "정보없음": None,  # 삭제 대상
        },
        "fillna": None,
        "dtype": str,
    },
    "SCRB_PATH_NM_GRP": {
        "new_name": "DERIVED_SCRB_PATH_NM_GRP",
        "mapping": {
            "현장경로현장경로": "현장경로현장경로",
            "I/B": "I/B",
            "O/B": "O/B",
            "일반상담": "일반상담",
            "직영몰": "직영몰",
            "임직원": "임직원",
            "전략채널": "전략채널",
            "렌탈제휴": "렌탈제휴",
            "기타": "기타",
        },
        "fillna": None,
        "dtype": str,
    },
    "AGMT_END_SEG": {
        "new_name": "DERIVED_AGMT_END_SEG",
        "mapping": {
            "약정만료전 12개월이상": "13_약정만료전_12개월이상",
            "약정만료전 9~12개월": "12_약정만료전_9~12개월",
            "약정만료전 6~9개월": "11_약정만료전_6~9개월",
            "약정만료전 3~6개월": "10_약정만료전_3~6개월",
            "약정만료전 2~3개월": "09_약정만료전_2~3개월",
            "약정만료전 1~2개월": "08_약정만료전_1~2개월",
            "약정만료전 1개월": "07_약정만료전_1개월",
            "약정만료 1개월": "06_약정만료_1개월",
            "약정만료후 1개월~2개월": "05_약정만료후_1개월~2개월",
            "약정만료후 2개월~3개월": "04_약정만료후_2개월~3개월",
            "약정만료후 3~6개월": "03_약정만료후_3~6개월",
            "약정만료후 6~9개월": "02_약정만료후_6~9개월",
            "약정만료후 9~12개월": "01_약정만료후_9~12개월",
            "약정만료후 12개월이상": "00_약정만료후_12개월이상",
        },
        "fillna": None,
        "dtype": str,
    },
    "CH_LAST_DAYS_BF_GRP": {
        "new_name": "DERIVED_CH_LAST_DAYS_BF_GRP",
        "mapping": {
            "일주일내": "0_일주일내",
            "일주일전": "1_일주일전",
            "2주일전": "2_2주일전",
            "3주일전": "3_3주일전",
            "4주일전": "4_4주일전",
            "3개월내없음": "5_3개월내없음",
        },
        "fillna": None,
        "dtype": str,
    },
    "CH_FAV_RNK1": {
        "new_name": "DERIVED_CH_FAV_RNK1",
        "fillna": None,
        "dtype": str,
    },
    "EMAIL_RECV_CLS_NM": {
        "new_name": "DERIVED_EMAIL_RECV_CLS_NM",
        "mapping": {"수신": "1", "광고거부": "0", "전체거부": "0", "미응답": "0"},
        "fillna": None,
        "dtype": str,
    },
    "SMS_SEND_CLS_NM": {
        "new_name": "DERIVED_SMS_SEND_CLS_NM",
        "mapping": {"수신": "1", "광고거부": "0", "전체거부": "0", "미응답": "0"},
        "fillna": None,
        "dtype": str,
    },
    # [Binary Y/N 변수]
    "PROD_OLD_YN": {
        "new_name": "DERIVED_PROD_OLD_YN",
        "mapping": {"Y": "1", "N": "0"},
        "fillna": None,
        "dtype": str,
    },
    "PROD_ONE_PLUS_YN": {
        "new_name": "DERIVED_PROD_ONE_PLUS_YN",
        "mapping": {"Y": "1", "N": "0"},
        "fillna": None,
        "dtype": str,
    },
    "STB_RES_1M_YN": {
        "new_name": "DERIVED_STB_RES_1M_YN",
        "mapping": {"Y": "1", "N": "0"},
        "fillna": None,
        "dtype": str,
    },
    "BUNDLE_YN": {
        "new_name": "DERIVED_BUNDLE_YN",
        "mapping": {"Y": "1", "N": "0"},
        "fillna": None,
        "dtype": str,
    },
    "DIGITAL_GIGA_YN": {
        "new_name": "DERIVED_DIGITAL_GIGA_YN",
        "mapping": {"Y": "1", "N": "0"},
        "fillna": None,
        "dtype": str,
    },
    "DIGITAL_ALOG_YN": {
        "new_name": "DERIVED_DIGITAL_ALOG_YN",
        "mapping": {"Y": "1", "N": "0"},
        "fillna": None,
        "dtype": str,
    },
    "VOC_TOTAL_MONTH1_YN": {
        "new_name": "DERIVED_VOC_TOTAL_MONTH1_YN",
        "mapping": {"Y": "1", "N": "0"},
        "fillna": None,
        "dtype": str,
    },
    "VOC_STOP_CANCEL_MONTH1_YN": {
        "new_name": "DERIVED_VOC_STOP_CANCEL_MONTH1_YN",
        "mapping": {"Y": "1", "N": "0"},
        "fillna": None,
        "dtype": str,
    },
    "NFX_USE_YN": {
        "new_name": "DERIVED_NFX_USE_YN",
        "mapping": {"Y": "1", "N": "0"},
        "fillna": None,
        "dtype": str,
    },
    "YTB_USE_YN": {
        "new_name": "DERIVED_YTB_USE_YN",
        "mapping": {"Y": "1", "N": "0"},
        "fillna": None,
        "dtype": str,
    },
    # ------------------------------------------------------
    # [Numerical] 수치형 변수 -> dtype: float
    # ------------------------------------------------------
    "SVOD_SCRB_CNT_GRP": {
        "new_name": "DERIVED_SVOD_SCRB_CNT_GRP",
        "mapping": {"0건": 0.0, "1건": 1.0, "2건": 2.0, "3건 이상": 3.0},
        "fillna": None,
        "dtype": float,
    },
    "PAID_CHNL_CNT_GRP": {
        "new_name": "DERIVED_PAID_CHNL_CNT_GRP",
        "mapping": {"0건": 0.0, "1건": 1.0, "2건": 2.0, "3건이상": 3.0},
        "fillna": None,
        "dtype": float,
    },
    "INHOME_RATE": {
        "new_name": "DERIVED_INHOME_RATE",
        "mapping": {
            "알수없음": None,
            "0.0": "0.0",
            "10.0": "10.0",
            "20.0": "20.0",
            "30.0": "30.0",
            "40.0": "40.0",
            "50.0": "50.0",
            "60.0": "60.0",
            "70.0": "70.0",
            "80.0": "80.0",
            "90.0": "90.0",
            "100.0": "100.0",
        },  # [수정] 알수없음 -> None (삭제)
        "fillna": None,
        "dtype": float,
    },
    "TOTAL_USED_DAYS": {
        "new_name": "DERIVED_TOTAL_USED_DAYS",
        "fillna": None,
        "dtype": float,
    },
    "TV_SCRB": {"new_name": "DERIVED_TV_SCRB", "fillna": None, "dtype": float},
    "ANALOG_SCRB": {"new_name": "DERIVED_ANALOG_SCRB", "fillna": None, "dtype": float},
    "DIGITAL_SCRB": {
        "new_name": "DERIVED_DIGITAL_SCRB",
        "fillna": None,
        "dtype": float,
    },
    "TOTAL_INTERNET_SCRB": {
        "new_name": "DERIVED_TOTAL_INTERNET_SCRB",
        "fillna": None,
        "dtype": float,
    },
    "GIGA_INTERNET_SCRB": {
        "new_name": "DERIVED_GIGA_INTERNET_SCRB",
        "fillna": None,
        "dtype": float,
    },
    "TV_I_CNT": {"new_name": "DERIVED_TV_I_CNT", "fillna": None, "dtype": float},
    "CH_HH_AVG_MONTH1": {
        "new_name": "DERIVED_CH_HH_AVG_MONTH1",
        "fillna": None,
        "dtype": float,
    },
    "CH_25_RATIO_MONTH1": {
        "new_name": "DERIVED_CH_25_RATIO_MONTH1",
        "fillna": None,
        "dtype": float,
    },
    "CH_25_RATIO_MEAN_3MM": {
        "new_name": "DERIVED_CH_25_RATIO_MEAN_3MM",
        "fillna": None,
        "dtype": float,
    },
    "KIDS_USE_PV_MONTH1": {
        "new_name": "DERIVED_KIDS_USE_PV_MONTH1",
        "fillna": None,
        "dtype": float,
    },
    "AGMT_END_YMD": {
        "new_name": "DERIVED_AGMT_END_YMD",
        "fillna": None,
        "dtype": float,
    },
}

# DEFAULT_COLUMN_RULES = {
#     "cancel_yn": {
#         "new_name": "derived_cancel_yn",
#         "mapping": {"해지": 1, "유지": 0},
#         "fillna": -1,
#         "dtype": int,
#     },
#     "AGE_GRP10": {
#         "new_name": "DERIVED_AGE_GRP10",
#         "mapping": {
#             "10대미만": 0,
#             "10대": 1,
#             "20대": 2,
#             "30대": 3,
#             "40대": 4,
#             "50대": 5,
#             "60대": 6,
#             "70대": 7,
#             "80대": 8,
#             "90대이상": 9,
#         },
#         "fillna": -1,
#         "dtype": int,
#     },
#     "STR_RES_YN_NM": {
#         "new_name": "str_res_yn",
#         "mapping": {"Y": 1, "N": 0},
#         "fillna": 0,
#         "dtype": int,
#     },
#     "SCRB_PATH_NM_GRP": {
#         "new_name": "DERIVED_SCRB_PATH_NM_GRP",
#         "mapping": {
#             "현장경로현장경로": 0,
#             "I/B": 1,
#             "O/B": 2,
#             "일반상담": 3,
#             "직영몰": 4,
#             "임직원": 5,
#             "전략채널": 6,
#             "렌탈제휴": 7,
#             "기타": 8,
#         },
#         "fillna": -1,
#         "dtype": int,
#     },
#     "SVC_USE_DAYS_GRP": {
#         "new_name": "DERIVED_SVC_USE_DAYS_GRP",
#         "mapping": {
#             "3개월미만": 0,
#             "3~12개월": 1,
#             "12~24개월": 2,
#             "24~36개월": 3,
#             "36개월이상": 4,
#         },
#         "fillna": -1,
#         "dtype": int,
#     },
#     "AGMT_KIND_NM": {
#         "new_name": "DERIVED_AGMT_KIND_NM",
#         "mapping": {
#             "신규": 0,
#             "약정승계": 1,
#             "재약정": 2,
#             "약정갱신": 3,
#             "약정연장": 4,
#             "약정축소": 5,
#             "정보없음": 6,
#         },
#         "fillna": -1,
#         "dtype": int,
#     },
#     "AS_MT_CND_NM": {
#         "new_name": "contract_status_code",
#         "mapping": {
#             "무약정": 0,
#             "약정기간내": 1,
#             "약정만료(재약정X)": 2,
#             "약정만료(재약정O)": 3,
#             "약정만료전해지": 3,
#         },
#         "fillna": 0,
#         "dtype": int,
#     },
#     "AS_MT_T_RM_GRP": {
#         "new_name": "contract_remain_code",
#         "mapping": {
#             "약정만료(재약정X)": 0,
#             "약정만료(재약정O)": 0,
#             "약정만료전해지": 0,
#             "약정기간내(0~3개월)": 1,
#             "약정기간내(3~6개월)": 2,
#             "약정기간내(6~12개월)": 3,
#             "약정기간내(12~24개월)": 4,
#             "약정기간내(24~36개월)": 5,
#             "약정기간내(36개월이상)": 6,
#         },
#         "fillna": 0,
#         "dtype": int,
#     },
#     "AGMT_END_SEG": {
#         "new_name": "DERIVED_AGMT_END_SEG",
#         "mapping": {
#             "약정만료전 12개월이상": 0,
#             "약정만료전 9~12개월": 1,
#             "약정만료전 6~9개월": 2,
#             "약정만료전 3~6개월": 3,
#             "약정만료전 2~3개월": 4,
#             "약정만료전 1~2개월": 5,
#             "약정만료전 1개월": 6,
#             "약정만료 1개월": 7,
#             "약정만료후 1개월~2개월": 8,
#             "약정만료후 2개월~3개월": 9,
#             "약정만료후 3~6개월": 10,
#             "약정만료후 6~9개월": 11,
#             "약정만료후 9~12개월": 12,
#             "약정만료후 12개월이상": 13,
#         },
#         "fillna": -1,
#         "dtype": int,
#     },
#     "AGMT_END_YMD": {
#         "new_name": "DERIVED_AGMT_END_YMD",
#         "fillna": -1,
#         "dtype": int,
#     },
#     "PROD_NM_GRP": {
#         "new_name": "DERIVED_PROD_NM_GRP",
#         "mapping": {
#             "알수없음": 0,
#             "이코노미/베이직": 1,
#             "스탠다드": 2,
#             "프리미엄": 3,
#             "VIP/플래티넘": 4,
#         },
#         "fillna": -1,
#         "dtype": int,
#     },
#     "MEDIA_NM_GRP": {
#         "new_name": "DERIVED_MEDIA_NM_GRP",
#         "mapping": {"알수없음": 0, "HD": 1, "UHD": 2},
#         "fillna": -1,
#         "dtype": int,
#     },
#     "PROD_OLD_YN": {
#         "new_name": "DERIVED_PROD_OLD_YN",
#         "mapping": {"Y": 1, "N": 0},
#         "fillna": -1,
#         "dtype": int,
#     },
#     "PROD_ONE_PLUS_YN": {
#         "new_name": "DERIVED_PROD_ONE_PLUS_YN",
#         "mapping": {"Y": 1, "N": 0},
#         "fillna": -1,
#         "dtype": int,
#     },
#     "PROD_DIF_P_LS_YN": {
#         "new_name": "price_up_yn",
#         "mapping": {"Y": 1, "N": 0},
#         "fillna": 0,
#         "dtype": int,
#     },
#     "STB_RES_1M_YN": {
#         "new_name": "DERIVED_STB_RES_1M_YN",
#         "mapping": {"Y": 1, "N": 0},
#         "fillna": -1,
#         "dtype": int,
#     },
#     "BUNDLE_YN": {
#         "new_name": "DERIVED_BUNDLE_YN",
#         "mapping": {"Y": 1, "N": 0},
#         "fillna": -1,
#         "dtype": int,
#     },
#     "PROD_SCBT_CNT_GRP": {
#         "new_name": "bundle_cnt_code",
#         "mapping": {"0건": 0, "1건": 1, "2건": 2, "3건이상": 3},
#         "fillna": 0,
#         "dtype": int,
#     },
#     "DIGITAL_GIGA_YN": {
#         "new_name": "DERIVED_DIGITAL_GIGA_YN",
#         "mapping": {"Y": 1, "N": 0},
#         "fillna": -1,
#         "dtype": int,
#     },
#     "DIGITAL_ALOG_YN": {
#         "new_name": "DERIVED_DIGITAL_ALOG_YN",
#         "mapping": {"Y": 1, "N": 0},
#         "fillna": -1,
#         "dtype": int,
#     },
#     "TV_I_CNT": {
#         "new_name": "DERIVED_TV_I_CNT",
#         "fillna": -1,
#         "dtype": int,
#     },
#     "TV_SCRB": {
#         "new_name": "DERIVED_TV_SCRB",
#         "fillna": -1,
#         "dtype": int,
#     },
#     "ANALOG_SCRB": {
#         "new_name": "DERIVED_ANALOG_SCRB",
#         "fillna": -1,
#         "dtype": int,
#     },
#     "DIGITAL_SCRB": {
#         "new_name": "DERIVED_DIGITAL_SCRB",
#         "fillna": -1,
#         "dtype": int,
#     },
#     "TOTAL_INTERNET_SCRB": {
#         "new_name": "DERIVED_TOTAL_INTERNET_SCRB",
#         "fillna": -1,
#         "dtype": int,
#     },
#     "GIGA_INTERNET_SCRB": {
#         "new_name": "DERIVED_GIGA_INTERNET_SCRB",
#         "fillna": -1,
#         "dtype": int,
#     },
#     "INHOME_RATE": {
#         "new_name": "DERIVED_INHOME_RATE",
#         "mapping": {
#             "0.0": 0.0,
#             "10.0": 1.0,
#             "20.0": 2.0,
#             "30.0": 3.0,
#             "40.0": 4.0,
#             "50.0": 5.0,
#             "60.0": 6.0,
#             "70.0": 7.0,
#             "80.0": 8.0,
#             "90.0": 9.0,
#             "100.0": 10.0,
#             "알수없음": 0.0,
#         },
#         "fillna": 0,
#         "dtype": int,
#     },
#     "CH_LAST_DAYS_BF_GRP": {
#         "new_name": "DERIVED_CH_LAST_DAYS_BF_GRP",
#         "mapping": {
#             "3개월내없음": 0,
#             "4주일전": 1,
#             "3주일전": 2,
#             "2주일전": 3,
#             "일주일전": 4,
#             "일주일내": 5,
#         },
#         "fillna": -1,
#         "dtype": int,
#     },
#     "PAID_CHNL_CNT_GRP": {
#         "new_name": "DERIVED_PAID_CHNL_CNT_GRP",
#         "mapping": {"0건": 0, "1건": 1, "2건": 2, "3건이상": 3},
#         "fillna": -1,
#         "dtype": int,
#     },
#     "SVOD_SCRB_CNT_GRP": {
#         "new_name": "DERIVED_SVOD_SCRB_CNT_GRP",
#         "mapping": {"0건": 0, "1건": 1, "2건": 2, "3건 이상": 3},
#         "fillna": -1,
#         "dtype": int,
#     },
#     "KIDS_USE_PV_MONTH1": {
#         "new_name": "DERIVED_KIDS_USE_PV_MONTH1",
#         "fillna": -1,
#         "dtype": int,
#     },
#     "NFX_USE_YN": {
#         "new_name": "DERIVED_NFX_USE_YN",
#         "mapping": {"Y": 1, "N": 0},
#         "fillna": -1,
#         "dtype": int,
#     },
#     "YTB_USE_YN": {
#         "new_name": "DERIVED_YTB_USE_YN",
#         "mapping": {"Y": 1, "N": 0},
#         "fillna": -1,
#         "dtype": int,
#     },
#     "VOC_TOTAL_MONTH1_YN": {
#         "new_name": "DERIVED_VOC_TOTAL_MONTH1_YN",
#         "mapping": {"Y": 1, "N": 0},
#         "fillna": -1,
#         "dtype": int,
#     },
#     "VOC_STOP_CANCEL_MONTH1_YN": {
#         "new_name": "DERIVED_VOC_STOP_CANCEL_MONTH1_YN",
#         "mapping": {"Y": 1, "N": 0},
#         "fillna": -1,
#         "dtype": int,
#     },
#     "EMAIL_RECV_CLS_NM": {
#         "new_name": "DERIVED_EMAIL_RECV_CLS_NM",
#         "mapping": {"광고거부": 0, "미응답": 0, "수신": 1, "전체거부": 0},
#         "fillna": -1,
#         "dtype": int,
#     },
#     "SMS_SEND_CLS_NM": {
#         "new_name": "DERIVED_SMS_SEND_CLS_NM",
#         "mapping": {"광고거부": 0, "미응답": 0, "수신": 1, "전체거부": 0},
#         "fillna": -1,
#         "dtype": int,
#     },
# }


# ==========================================================
# 3. 컬럼 설명 매핑 (English -> Korean)
# ==========================================================
COLUMN_DESC = {
    # --- 원본 컬럼 ---
    "sha2_hash": "고객ID",
    "SVC_USE_DAYS_GRP": "서비스 이용기간",
    "MEDIA_NM_GRP": "상품 매체명",
    "PROD_NM_GRP": "상품명",
    "PROD_OLD_YN": "구)상품 이용 유무",
    "PROD_ONE_PLUS_YN": "추가 이용 유무",
    "AGMT_KIND_NM": "약정 종류",
    "STB_RES_1M_YN": "셋탑박스 휴면 유무",
    "SVOD_SCRB_CNT_GRP": "월정액 가입 수",
    "PAID_CHNL_CNT_GRP": "유료채널 가입 수",
    "SCRB_PATH_NM_GRP": "유치경로",
    "INHOME_RATE": "집돌이 지수",
    "AGMT_END_SEG": "약정 종료일(구간)",
    "AGMT_END_YMD": "약정 종료일",
    "TOTAL_USED_DAYS": "총 사용일수",
    "TV_SCRB": "TV사용 댓수",
    "ANALOG_SCRB": "아날로그 이용 댓수",
    "DIGITAL_SCRB": "디지털 이용 댓수",
    "TOTAL_INTERNET_SCRB": "인터넷 이용 댓수",
    "GIGA_INTERNET_SCRB": "기가인터넷 이용 댓수",
    "BUNDLE_YN": "번들 유무",
    "DIGITAL_GIGA_YN": "디지털&기가 결합 유무",
    "DIGITAL_ALOG_YN": "디지털&아날로그 결합 유무",
    "TV_I_CNT": "TV, 인터넷 전체 이용댓수",
    "CH_LAST_DAYS_BF_GRP": "최근 시청일(구간)",
    "VOC_TOTAL_MONTH1_YN": "1개월내 VOC 인입 유무",
    "VOC_STOP_CANCEL_MONTH1_YN": "1개월내 해지VOC 인입유무",
    "AGE_GRP10": "연령 구간",
    "EMAIL_RECV_CLS_NM": "이메일 수신 유무",
    "SMS_SEND_CLS_NM": "SMS 수신 유무",
    "CH_HH_AVG_MONTH1": "1개월 평균 채널 시청시간",
    "CH_25_RATIO_MONTH1": "1개월 간 지역채널 시청률",
    "CH_25_RATIO_MEAN_3MM": "3개월 간 지역채널 시청률",
    "CH_FAV_RNK1": "선호채널 랭크",
    "KIDS_USE_PV_MONTH1": "1개월 키즈 진입횟수",
    "NFX_USE_YN": "넷플릭스사용여부",
    "YTB_USE_YN": "유튜브사용여부",
    "p_mt": "기준년월",
    "cancel_yn": "해지여부",
    # --- [NEW] 파생변수 매핑 (Derived Columns) ---
    "derived_cancel_yn": "파생_해지여부",
    "DERIVED_AGE_GRP10": "파생_연령 구간",
    "DERIVED_SCRB_PATH_NM_GRP": "파생_유치경로",
    "DERIVED_SVC_USE_DAYS_GRP": "파생_서비스 이용기간",
    "DERIVED_AGMT_KIND_NM": "파생_약정 종류",
    "DERIVED_AGMT_END_SEG": "파생_약정 종료일(구간)",
    "DERIVED_AGMT_END_YMD": "파생_약정 종료일",
    "DERIVED_PROD_NM_GRP": "파생_상품명",
    "DERIVED_MEDIA_NM_GRP": "파생_상품 매체명",
    "DERIVED_PROD_OLD_YN": "파생_구)상품 이용 유무",
    "DERIVED_PROD_ONE_PLUS_YN": "파생_추가 이용 유무",
    "DERIVED_STB_RES_1M_YN": "파생_셋탑박스 휴면 유무",
    "DERIVED_BUNDLE_YN": "파생_번들 유무",
    "DERIVED_DIGITAL_GIGA_YN": "파생_디지털&기가 결합 유무",
    "DERIVED_DIGITAL_ALOG_YN": "파생_디지털&아날로그 결합 유무",
    "DERIVED_TV_I_CNT": "파생_TV, 인터넷 전체 이용댓수",
    "DERIVED_TV_SCRB": "파생_TV사용 댓수",
    "DERIVED_ANALOG_SCRB": "파생_아날로그 이용 댓수",
    "DERIVED_DIGITAL_SCRB": "파생_디지털 이용 댓수",
    "DERIVED_TOTAL_INTERNET_SCRB": "파생_인터넷 이용 댓수",
    "DERIVED_GIGA_INTERNET_SCRB": "파생_기가인터넷 이용 댓수",
    "DERIVED_INHOME_RATE": "파생_집돌이 지수",
    "DERIVED_CH_LAST_DAYS_BF_GRP": "파생_최근 시청일(구간)",
    "DERIVED_PAID_CHNL_CNT_GRP": "파생_유료채널 가입 수",
    "DERIVED_SVOD_SCRB_CNT_GRP": "파생_월정액 가입 수",
    "DERIVED_KIDS_USE_PV_MONTH1": "파생_1개월 키즈 진입횟수",
    "DERIVED_NFX_USE_YN": "파생_넷플릭스사용여부",
    "DERIVED_YTB_USE_YN": "파생_유튜브사용여부",
    "DERIVED_VOC_TOTAL_MONTH1_YN": "파생_1개월내 VOC 인입 유무",
    "DERIVED_VOC_STOP_CANCEL_MONTH1_YN": "파생_1개월내 해지VOC 인입유무",
    "DERIVED_EMAIL_RECV_CLS_NM": "파생_이메일 수신 유무",
    "DERIVED_SMS_SEND_CLS_NM": "파생_SMS 수신 유무",
}


# ==========================================================
# 4. 환경 초기화 함수
# ==========================================================
def init_settings():
    """프로젝트 공통 환경 설정"""
    warnings.filterwarnings("ignore")
    sns.set_style("whitegrid")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.figsize"] = (14, 8)
    plt.rcParams["font.size"] = 12

    system_name = platform.system()
    try:
        if system_name == "Windows":
            font_path = r"c:/Windows/Fonts/malgun.ttf"
            if os.path.exists(font_path):
                font_name = font_manager.FontProperties(fname=font_path).get_name()
                plt.rc("font", family=font_name)
            else:
                plt.rc("font", family="Malgun Gothic")
        elif system_name == "Darwin":
            plt.rc("font", family="AppleGothic")
        else:
            plt.rc("font", family="NanumGothic")

        logger.info(f"✅ [Config] 환경 설정 완료 (OS: {system_name})")
    except Exception as e:
        logger.error(f"⚠️ [Config] 폰트 설정 오류: {e}")
