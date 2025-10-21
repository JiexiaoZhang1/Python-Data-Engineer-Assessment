import json
import sys
from pathlib import Path
import pandas as pd

INPUT = "output_clean_date_technical.json"
OUTPUT = "dataset_quarterly.csv"

# ---- period 映射（兼容 pandas.NA）----
def period_to_range(calendar_year, period):
    if pd.isna(calendar_year) or pd.isna(period):
        return None, None
    y = int(str(calendar_year))
    p = str(period).upper()
    if p == "Q1":
        return f"{y}-01-01", f"{y}-03-31"
    if p == "Q2":
        return f"{y}-04-01", f"{y}-06-30"
    if p == "Q3":
        return f"{y}-07-01", f"{y}-09-30"
    if p == "Q4":
        return f"{y}-10-01", f"{y}-12-31"
    if p in ("FY", "ANNUAL", "YEAR"):
        return f"{y}-01-01", f"{y}-12-31"
    return None, None

def norm_date(x):
    if pd.isna(x) or x is None or x == "":
        return pd.NA
    try:
        return pd.to_datetime(str(x)).strftime("%Y-%m-%d")
    except Exception:
        return pd.NA

def to_df(records, source_suffix: str):
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    for k in ["symbol", "date", "calendarYear", "period"]:
        if k not in df.columns:
            df[k] = pd.NA
    df["date"] = df["date"].apply(norm_date)
    df.attrs["source_suffix"] = source_suffix
    return df

def extract_historical_price_full(obj) -> pd.DataFrame:
    rows = []
    def push_one(one):
        sym = one.get("symbol")
        hist = one.get("historical") or []
        for r in hist:
            rr = dict(r)
            rr.setdefault("symbol", sym)
            rows.append(rr)
    if isinstance(obj, list):
        for one in obj:
            if isinstance(one, dict):
                push_one(one)
    elif isinstance(obj, dict):
        push_one(obj)
    return to_df(rows, "_historical")

def safe_outer_merge(left: pd.DataFrame, right: pd.DataFrame, suffix_right: str):
    if right.empty:
        return left
    key_cols = ["symbol", "date"]
    overlap = [c for c in right.columns if c in left.columns and c not in key_cols]
    if overlap:
        right = right.rename(columns={c: f"{c}{suffix_right}" for c in overlap})
    return pd.merge(left, right, on=key_cols, how="outer")

def main():
    input_path = Path(INPUT if len(sys.argv) == 1 else sys.argv[1])
    if not input_path.exists():
        raise FileNotFoundError(f"找不到文件：{input_path}")
    data = json.loads(input_path.read_text(encoding="utf-8"))

    # 1) historical
    df_hist = extract_historical_price_full(data.get("historicalPriceFull"))
    if df_hist.empty:
        for k, v in data.items():
            if "historicalpricefull" in str(k).lower():
                df_hist = extract_historical_price_full(v)
                break
    if not df_hist.empty:
        df_hist["date"] = df_hist["date"].apply(norm_date)
        df_hist = df_hist.dropna(subset=["symbol", "date"])

    # 2) 其他数组
    sections = [
        ("financialGrowth", "_financialGrowth"),
        ("ratios", "_ratios"),
        ("cashFlowStatementGrowth", "_cashFlowStatementGrowth"),
        ("incomeStatementGrowth", "_incomeStatementGrowth"),
        ("balanceSheetStatementGrowth", "_balanceSheetStatementGrowth"),
    ]
    merged = df_hist.copy() if not df_hist.empty else pd.DataFrame(columns=["symbol","date"])
    for key, suf in sections:
        df = to_df(data.get(key, []), suf)
        if df.empty:
            continue
        df["date"] = df["date"].apply(norm_date)
        df = df.dropna(subset=["symbol", "date"])
        merged = safe_outer_merge(merged, df, df.attrs.get("source_suffix", "_r"))

    # 3) calendarYear/period & 区间
    for k in ["calendarYear", "period"]:
        if k not in merged.columns:
            merged[k] = pd.NA
    ps, pe = zip(*merged.apply(lambda r: period_to_range(r["calendarYear"], r["period"]), axis=1))
    merged["period_start"] = list(ps)
    merged["period_end"] = list(pe)

    # 4) 去重 & 列序
    merged["date"] = merged["date"].apply(norm_date)
    merged = merged.dropna(subset=["symbol", "date"]).drop_duplicates(subset=["symbol","date"], keep="first")

    ohlcv_cols = [
        "date","symbol","open","high","low","close","adjClose","volume",
        "unadjustedVolume","change","changePercent","vwap","label","changeOverTime"
    ]
    key_and_period = ["calendarYear","period","period_start","period_end"]
    front = [c for c in ohlcv_cols if c in merged.columns] + [c for c in key_and_period if c in merged.columns]
    rest = [c for c in merged.columns if c not in front]
    merged = merged[front + rest]

    # 5) 输出 & 质检
    print(f"[INFO] 形状: {merged.shape}")
    print(f"[QC] 重复键数量: {merged.duplicated(subset=['symbol','date']).sum()}")
    print("[QC] 缺失率Top10:")
    print(merged.isna().mean().sort_values(ascending=False).head(10))

    merged.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"[DONE] 导出: {OUTPUT}")

if __name__ == "__main__":
    main()