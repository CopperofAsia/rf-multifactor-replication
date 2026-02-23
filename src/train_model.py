import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

from src.config import INTERIM_DIR, RAW_DIR, MODEL_DIR
from src.config import TRAIN_START, TRAIN_END, TEST_START, TEST_END, BT_START, BT_END
from src.config import (
    RF_PARAM_DIST, TS_CV_SPLITS,
    RANDOMSEARCH_N_ITER, RANDOMSEARCH_SCORING,
    RANDOMSEARCH_N_JOBS, RANDOMSEARCH_RANDOM_STATE
)
from src.preprocess import preprocess_factors


FACTOR_COLS = ["mom_20d", "mom_60d", "ep", "bp", "turnover_20d"]


def _fallback_time_split(df: pd.DataFrame):
    """
    如果研报日期不覆盖当前样本，则按trade_date切分 60/20/20
    Returns:
        train_mask (pd.DataFrame)
        test_mask (pd.DataFrame)
        bt_mask (pd.DataFrame)
        (start_date, train_end, test_end, end_date) (tuple)
    """
    dates = sorted(df["trade_date"].unique())  # 拿到去重后按升序排列的交易日集合
    n = len(dates)
    d_train_end = dates[int(n * 0.6) - 1]
    d_test_end = dates[int(n * 0.8) - 1]
    train_mask = df["trade_date"] <= d_train_end
    test_mask = (df["trade_date"] > d_train_end) & (df["trade_date"] <= d_test_end)
    bt_mask = df["trade_date"] > d_test_end
    return train_mask, test_mask, bt_mask, (dates[0], d_train_end, d_test_end, dates[-1])

def make_time_series_cv_splits(df: pd.DataFrame, n_splits: int):
    """
    Build TimeSeries CV splits by unique trade_date, returning 
    list of (train_idx, val_idx), where idx are row indices for df 
    (df must be sorted by trade_date).
    """
    dates = np.array(sorted(df["trade_date"].unique()))
    if len(dates) < (n_splits + 1):
        raise ValueError(f"Not enough unique trade_date={len(dates)} for n_splits={n_splits}")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    
    # 构建一个“日期 → 行号”的映射字典
    date_to_rows = {d: df.index[df["trade_date"] == d].to_numpy() for d in dates}

    for train_date_idx, val_date_idx in tscv.split(dates):
        train_dates = dates[train_date_idx]
        val_dates = dates[val_date_idx]

        train_rows = np.concatenate([date_to_rows[d] for d in train_dates])
        val_rows = np.concatenate([date_to_rows[d] for d in val_dates])

        splits.append((train_rows, val_rows))
    # 返回的是 list[tuple]，每一个 tuple 含有每一轮的训练集行号（np.array）和验证集行号(np.array)
    return splits

def main():
    df = pd.read_parquet(INTERIM_DIR / "panel_with_labels.parquet")
    df = df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    # 只保留有 label 的样本（监督学习必须）
    df_ml = df[df["label"].notna()].copy()  

    # === 预处理（横截面winsorize/median-fill/zscore）===
    df_ml = preprocess_factors(df_ml, FACTOR_COLS)

    # === 按日期切分 ===
    train_mask = (df_ml["trade_date"] >= TRAIN_START) & (df_ml["trade_date"] <= TRAIN_END)
    test_mask  = (df_ml["trade_date"] >= TEST_START) & (df_ml["trade_date"] <= TEST_END)
    bt_mask    = (df_ml["trade_date"] >= BT_START) & (df_ml["trade_date"] <= BT_END)

    # 如果切分失败（样本不覆盖），启用 fallback
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        train_mask, test_mask, bt_mask, span = _fallback_time_split(df_ml)
        print("[INFO] Fallback time split used (60/20/20 by trade_date).")
        print("       span:", span)

    train_df = df_ml[train_mask].copy()
    test_df  = df_ml[test_mask].copy()
    
    train_df = train_df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    test_df  = test_df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


    # === 模型：随机森林 ===
    X_train = train_df[FACTOR_COLS].values
    y_train = train_df["label"].astype(int).values

    X_test = test_df[FACTOR_COLS].values
    y_test = test_df["label"].astype(int).values

    # == 时间序列切分（按trade_date）==
    cv_splits = make_time_series_cv_splits(train_df, n_splits=TS_CV_SPLITS)

    # == 随机搜索交叉验证选超参 ==
    base_rf = RandomForestClassifier(
        random_state=RANDOMSEARCH_RANDOM_STATE,
        n_jobs=-1
    )

    rs = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=RF_PARAM_DIST,
        n_iter=RANDOMSEARCH_N_ITER,
        scoring=RANDOMSEARCH_SCORING,
        cv=cv_splits,
        n_jobs=RANDOMSEARCH_N_JOBS,
        verbose=2,   # 打印每个参数组合的训练进度
        random_state=RANDOMSEARCH_RANDOM_STATE,
        refit=True,   # 自动在“全训练集”上用best params重训
    )

    rs.fit(X_train, y_train)

    print("\n=== RandomizedSearch Best Params ===")
    print(rs.best_params_)
    print("best_cv_score:", rs.best_score_)

    best_model = rs.best_estimator_

    # === 评估（用 best_model）===
    pred_train = best_model.predict(X_train)
    pred_test  = best_model.predict(X_test)

    print("\n=== Accuracy ===")
    print("train_acc:", accuracy_score(y_train, pred_train))
    print("test_acc :", accuracy_score(y_test, pred_test))

    print("\n=== Confusion Matrix (test) ===")
    print(confusion_matrix(y_test, pred_test))

    print("\n=== Classification Report (test) ===")
    print(classification_report(y_test, pred_test, digits=4))

    imp = pd.Series(best_model.feature_importances_, index=FACTOR_COLS).sort_values(ascending=False)
    print("\n=== Feature Importances (best model) ===")
    print(imp)

    # 保存搜索结果
    cv_out = MODEL_DIR / "rf_randomsearch_cv_results.csv"
    pd.DataFrame(rs.cv_results_).to_csv(cv_out, index=False)
    print("\nSaved CV results:", cv_out)
    

    # === 为回测重训：用 train+test 全部样本训练一个“最终模型” ===
    final_df = pd.concat([train_df, test_df], ignore_index=True)
    X_final = final_df[FACTOR_COLS].values
    y_final = final_df["label"].astype(int).values

    # 在参数前使用 ** 表示将一个字典解包为关键字参数传入函数
    rf_final = RandomForestClassifier(**rs.best_params_, random_state=42, n_jobs=-1)
    rf_final.fit(X_final, y_final)

    out_model = MODEL_DIR / "rf_model_final.joblib"
    joblib.dump(
        {"model": rf_final, "factor_cols": FACTOR_COLS, "best_params": rs.best_params_},
        out_model
    )
    print("\nSaved final model:", out_model)


if __name__ == "__main__":
    main()
