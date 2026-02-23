from pathlib import Path

# ====== 研报关键参数 ======
INDEX_CODE = "000905.SH"       # 中证500（CSI 500）
REB_FREQ_TRADE_DAYS = 20       # 每 20 个交易日调仓

START_DATE = "20100104"
END_DATE   = "20250430"

# 留出至少一个调仓周期的缓冲，避免前视偏差和样本污染。
TRAIN_START = "20100104"
TRAIN_END   = "20170505"
TEST_START  = "20170606"
TEST_END    = "20190325"
BT_START    = "20190423"
BT_END      = "20250430"


# ====== 数据目录 ======
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RAW_DIR.mkdir(parents=True, exist_ok=True)   # 创建 raw 文件夹
INTERIM_DIR.mkdir(parents=True, exist_ok=True)   # 创建 interim 文件夹
MODEL_DIR.mkdir(parents=True, exist_ok=True)   # 创建 models 文件夹
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)   # 创建 outputs 文件夹


# ====== RandomForest hyperparameter tuning ======
RF_PARAM_DIST = {
    "n_estimators": [200, 400, 600],
    "max_depth": [3, 4, 5, 6],
    "min_samples_leaf": [10, 20, 50, 100],
    "min_samples_split": [2, 5, 10, 20, 40],
    "max_features": ["sqrt", 0.3, 0.5, 0.8],
    "bootstrap": [True],
}

TS_CV_SPLITS = 4
RANDOMSEARCH_N_ITER = 10
RANDOMSEARCH_SCORING = "accuracy"   # 或 "f1", "roc_auc"
RANDOMSEARCH_N_JOBS = -1
RANDOMSEARCH_RANDOM_STATE = 42