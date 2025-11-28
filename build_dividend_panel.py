#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quarterly dividend-oriented panel builder.

The script ingests Finviz/Polygon style Excel exports stored under downloads/<TICKER>/
and produces a quarterly panel with derived ratios, QoQ changes, lags, and DPS
change classes suitable for downstream classification models.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import linregress
except Exception:  # pragma: no cover - scipy optional
    linregress = None

# file name patterns (quarterly)
BALANCE_PATTERN = "*_Quarterly_balance_sheet_statement_*.xlsx"
INCOME_PATTERN = "*_Quarterly_income_statements_*.xlsx"
CASH_FLOW_PATTERN = "*_Quarterly_cash_flow_statements_*.xlsx"
ENTERPRISE_PATTERN = "*_Quarterly_enterprise_values_*.xlsx"
DIVIDENDS_PATTERN = "*_Company_dividends_*.xlsx"
ESG_RISK_PATTERN = "*_ESG_Risk_Ratings_*.xlsx"
ESG_SCORE_PATTERN = "*_ESG_Score_*.xlsx"
DAILY_PRICE_PATTERN = "*_Historical_Daily_Prices_*.xlsx"
MONTHLY_PRICE_PATTERN = "*_Historical_Monthly_Prices_*.xlsx"

BALANCE_SHEETS = ("BS", "Sheet1", "Balance Sheet")
INCOME_SHEETS = ("IS", "Income Statement", "Sheet1", "BS")
CASH_FLOW_SHEETS = ("CF", "Cash Flow", "Sheet1", "BS")
ENTERPRISE_SHEETS = ("Sheet1", "BS", "data")

MANDATORY_FEATURES = [
    "DPS",
    "total_assets",
    "total_equity",
    "total_debt",
    "market_cap",
    "enterprise_value",
    "log_assets",
    "log_market_cap",
    "debt_to_assets",
    "debt_to_equity",
    "net_debt",
    "net_debt_to_assets",
    "cash_and_equivalents",
    "cash_to_assets",
    "current_assets",
    "current_liabilities",
    "current_ratio",
    "asset_growth",
    "ESGScore",
]

DERIVATIVE_FEATURES = [
    "DPS",
    "total_dividends",
    "total_assets",
    "total_equity",
    "total_debt",
    "market_cap",
    "enterprise_value",
    "log_assets",
    "log_market_cap",
    "debt_to_assets",
    "debt_to_equity",
    "net_debt",
    "net_debt_to_assets",
    "cash_and_equivalents",
    "cash_to_assets",
    "current_assets",
    "current_liabilities",
    "current_ratio",
    "asset_growth",  # already a change metric
    "net_income",
    "sales",
    "number_of_shares",
    "stock_return_q",
    "stock_vol_q",
    "beta_to_sp500",
    "beta_to_ust10y",
    "ESGScore",
    "environmentalScore",
    "socialScore",
    "governanceScore",
    "ESGRiskRating",
    "delta_ESG",
]

BASE_OUTPUT_COLUMNS = [
    "firm_id",
    "symbol",
    "period",
    "period_str",
    "fiscal_date",
    "DPS",
    "total_dividends",
    "total_assets",
    "total_equity",
    "total_debt",
    "market_cap",
    "enterprise_value",
    "log_assets",
    "log_market_cap",
    "debt_to_assets",
    "debt_to_equity",
    "net_debt",
    "net_debt_to_assets",
    "cash_and_equivalents",
    "cash_to_assets",
    "current_assets",
    "current_liabilities",
    "current_ratio",
    "asset_growth",
    "stock_return_q",
    "stock_vol_q",
    "beta_to_sp500",
    "beta_to_ust10y",
    "ESGScore",
    "environmentalScore",
    "socialScore",
    "governanceScore",
    "ESGRiskRating",
    "industryRank",
    "delta_ESG",
]


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _normalize_metric_label(label: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(label).lower())


def _clean_period_label(value: object) -> Optional[int]:
    text = str(value).strip().lower()
    if not text:
        return None
    match = re.search(r"q([1-4])", text)
    if match:
        return int(match.group(1))
    if text in {"1", "2", "3", "4"}:
        return int(text)
    return None


def _to_quarter_period(date_val: object, period_label: object, calendar_year: object) -> Optional[pd.Period]:
    date = pd.to_datetime(date_val, errors="coerce")
    if pd.notna(date):
        return date.to_period("Q")
    quarter_num = _clean_period_label(period_label)
    year = pd.to_numeric(calendar_year, errors="coerce")
    if pd.notna(year) and quarter_num:
        try:
            return pd.Period(f"{int(year)}Q{int(quarter_num)}")
        except Exception:
            return None
    return None


def _read_statement_table(path: Path, sheet_candidates: Iterable[str]) -> pd.DataFrame:
    for sheet in sheet_candidates:
        try:
            df = pd.read_excel(path, sheet_name=sheet)
        except ValueError:
            continue
        df = _standardize_columns(df)
        if not df.empty:
            return df
    logging.warning("Sheets %s not found or empty in %s", sheet_candidates, path.name)
    return pd.DataFrame()


def _read_table_with_candidates(path: Path, sheet_candidates: Iterable[str]) -> pd.DataFrame:
    for sheet in sheet_candidates:
        try:
            df = pd.read_excel(path, sheet_name=sheet)
        except ValueError:
            continue
        df = _standardize_columns(df)
        if not df.empty:
            return df
    logging.warning("No usable sheet found in %s among %s", path.name, sheet_candidates)
    return pd.DataFrame()


def _log_frame_overview(label: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        logging.debug("%s: empty", label)
        return
    preview_cols = ", ".join(map(str, frame.columns))
    logging.debug("%s: shape=%s columns=%s", label, frame.shape, preview_cols)


def safe_log(series: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    mask = series > 0
    out.loc[mask] = np.log(series.loc[mask])
    return out


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = numerator.astype("float64")
    denominator = denominator.astype("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    zero_mask = denominator == 0
    result[zero_mask | denominator.isna()] = np.nan
    return result


def find_first_matching(directory: Path, pattern: str) -> Optional[Path]:
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def load_quarterly_statements(
    path: Optional[Path],
    sheet_candidates: Iterable[str],
    metric_filter: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    df = _read_statement_table(path, sheet_candidates)
    if df.empty:
        return pd.DataFrame()

    metric_col = df.columns[0]
    df = df.rename(columns={metric_col: "metric"}).dropna(subset=["metric"])
    df["metric"] = df["metric"].astype(str)
    df = df.set_index("metric")

    fiscal_dates = pd.to_datetime(df.loc["date"], errors="coerce") if "date" in df.index else None
    period_labels = df.loc["period"] if "period" in df.index else None
    calendar_years = pd.to_numeric(df.loc["calendarYear"], errors="coerce") if "calendarYear" in df.index else None

    drop_idx = [idx for idx in ["calendarYear", "date", "period"] if idx in df.index]
    data = df.drop(index=drop_idx)

    if metric_filter:
        allowed = {_normalize_metric_label(metric) for metric in metric_filter}
        keep_rows = [idx for idx in data.index if _normalize_metric_label(idx) in allowed]
        data = data.loc[keep_rows]

    records: List[Dict[str, object]] = []
    for col in data.columns:
        record: Dict[str, object] = {
            "fiscal_date": fiscal_dates[col] if fiscal_dates is not None else pd.NaT,
            "period_label": period_labels[col] if period_labels is not None else None,
            "calendar_year": calendar_years[col] if calendar_years is not None else np.nan,
            "period_key": col,
        }
        record.update(data[col].to_dict())
        records.append(record)

    tidy = pd.DataFrame(records)
    tidy["fiscal_date"] = pd.to_datetime(tidy["fiscal_date"], errors="coerce")
    tidy["period"] = tidy.apply(
        lambda row: _to_quarter_period(row["fiscal_date"], row["period_label"], row["calendar_year"]), axis=1
    )
    tidy = tidy.dropna(subset=["period"])
    tidy["period"] = tidy["period"].astype("period[Q]")
    tidy = tidy.sort_values(["period", "fiscal_date"]).reset_index(drop=True)
    return tidy


def _collapse_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    dup_cols = [col for col in df.columns if col.endswith("_dup")]
    for dup in dup_cols:
        base_name = dup[:-4]
        if base_name in df.columns:
            df[base_name] = df[base_name].combine_first(df[dup])
            df = df.drop(columns=[dup])
        else:
            df = df.rename(columns={dup: base_name})
    if "fiscal_date_y" in df.columns:
        df["fiscal_date"] = df.get("fiscal_date", pd.Series(dtype="datetime64[ns]")).combine_first(df["fiscal_date_y"])
        df = df.drop(columns=["fiscal_date_y"])
    if "fiscal_date_x" in df.columns:
        df = df.rename(columns={"fiscal_date_x": "fiscal_date"})
    return df


def merge_statement_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    merged = frames[0].copy()
    for nxt in frames[1:]:
        merged = merged.merge(nxt, on="period", how="outer", suffixes=("", "_dup"))
        merged = _collapse_duplicate_columns(merged)
    merged = merged.sort_values(["period", "fiscal_date"])
    return merged


def load_enterprise_values(path: Optional[Path]) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    df = _read_table_with_candidates(path, ENTERPRISE_SHEETS)
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["period"] = df["date"].dt.to_period("Q")
    agg_dict = {
        "numberOfShares": "last",
        "marketCapitalization": "last",
        "stockPrice": "last",
        "enterpriseValue": "last",
        "addTotalDebt": "last",
        "minusCashAndCashEquivalents": "last",
        "symbol": "last",
    }
    agg_available = {col: func for col, func in agg_dict.items() if col in df.columns}
    grouped = df.sort_values("date").groupby("period", as_index=False)
    aggregated = grouped.agg(agg_available) if agg_available else pd.DataFrame(columns=["period"])
    if "symbol" not in aggregated.columns:
        aggregated["symbol"] = np.nan
    return aggregated


def load_dividends(path: Optional[Path]) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    df = pd.read_excel(path)
    df = _standardize_columns(df)
    if df.empty:
        return pd.DataFrame()
    date_cols = [col for col in ["paymentDate", "recordDate", "declarationDate", "date"] if col in df.columns]
    for col in date_cols:
        candidate = pd.to_datetime(df[col], errors="coerce")
        if candidate.notna().any():
            df["dividend_date"] = candidate
            break
    if "dividend_date" not in df.columns:
        df["dividend_date"] = pd.NaT
    df["period"] = pd.to_datetime(df["dividend_date"], errors="coerce").dt.to_period("Q")

    value_col = None
    for col in df.columns:
        label = _normalize_metric_label(col)
        if "value" in label or "dividend" in label:
            value_col = col
            break
    if value_col is None:
        logging.warning("Dividends file %s missing value column; skipping.", path)
        return pd.DataFrame()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    grouped = (
        df.dropna(subset=["period"])
        .groupby("period", as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "DPS"})
    )
    if "symbol" in df.columns and df["symbol"].notna().any():
        grouped["symbol"] = df["symbol"].dropna().iloc[-1]
    else:
        grouped["symbol"] = np.nan
    return grouped


def load_esg_risk(path: Optional[Path]) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    df = pd.read_excel(path, sheet_name="Sheet1")
    df = _standardize_columns(df)
    if df.empty:
        return pd.DataFrame()
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["risk_date"] = pd.to_datetime(df["year"].astype("Int64").astype(str) + "-12-31", errors="coerce")
    elif "date" in df.columns:
        df["risk_date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        logging.warning("ESG risk file %s missing 'year'/'date' columns; skipping.", path.name)
        return pd.DataFrame()
    keep = ["risk_date", "ESGRiskRating", "industryRank", "industry", "companyName", "symbol"]
    keep_cols = [col for col in keep if col in df.columns]
    risk = df[keep_cols].dropna(subset=["risk_date"]).drop_duplicates("risk_date", keep="last").copy()
    if "symbol" not in risk.columns:
        risk["symbol"] = np.nan
    return risk


def load_esg_scores(path: Optional[Path]) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    df = pd.read_excel(path, sheet_name="Sheet1")
    df = _standardize_columns(df)
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    keep_cols = [
        "date",
        "symbol",
        "ESGScore",
        "environmentalScore",
        "socialScore",
        "governanceScore",
    ]
    keep_present = [col for col in keep_cols if col in df.columns]
    score = df[keep_present].sort_values("date").dropna(subset=["date"]).copy()
    if "symbol" not in score.columns:
        score["symbol"] = np.nan
    return score


def load_prices(path: Optional[Path]) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    df = pd.read_excel(path, sheet_name="Sheet1")
    df = _standardize_columns(df)
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    for col in ["price", "SP500", "UST10Y"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_beta(y: pd.Series, x: pd.Series) -> float:
    mask = y.notna() & x.notna()
    if mask.sum() < 10:
        return np.nan
    y_valid = y[mask]
    x_valid = x[mask]
    if linregress is not None:
        result = linregress(x_valid, y_valid)
        return result.slope
    x_centered = x_valid - x_valid.mean()
    denom = np.sum(x_centered**2)
    if denom == 0:
        return np.nan
    return float(np.sum((y_valid - y_valid.mean()) * x_centered) / denom)


def compute_price_features(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df.empty:
        return pd.DataFrame()
    if "price" not in price_df.columns:
        logging.warning("Price data missing 'price' column; skipping price features.")
        return pd.DataFrame()

    work = price_df.copy()
    work["stock_ret"] = np.log(work["price"]).diff()
    work["sp_ret"] = np.log(work["SP500"]).diff() if "SP500" in work.columns else np.nan
    if "UST10Y" in work.columns:
        work["ust_ret"] = work["UST10Y"].pct_change()
    else:
        work["ust_ret"] = np.nan

    work = work.replace([np.inf, -np.inf], np.nan)
    work["period"] = work["date"].dt.to_period("Q")
    work = work.dropna(subset=["period"])

    records: List[Dict[str, object]] = []
    for period, group in work.groupby("period"):
        group = group.sort_values("date")
        if group.empty:
            continue
        price_first = group["price"].dropna().iloc[0] if group["price"].notna().any() else np.nan
        price_last = group["price"].dropna().iloc[-1] if group["price"].notna().any() else np.nan
        total_return = np.log(price_last) - np.log(price_first) if pd.notna(price_first) and pd.notna(price_last) else np.nan
        rec: Dict[str, object] = {
            "period": period,
            "stock_return_q": total_return,
            "stock_vol_q": group["stock_ret"].std(ddof=0),
        }
        if "sp_ret" in group.columns:
            rec["beta_to_sp500"] = compute_beta(group["stock_ret"], group["sp_ret"])
        if "ust_ret" in group.columns:
            rec["beta_to_ust10y"] = compute_beta(group["stock_ret"], group["ust_ret"])
        records.append(rec)

    return pd.DataFrame(records)


def merge_on_period(left: pd.DataFrame, right: pd.DataFrame, how: str = "left") -> pd.DataFrame:
    if right.empty:
        return left
    merged = left.merge(right, on="period", how=how, suffixes=("", "_dup"))
    merged = _collapse_duplicate_columns(merged)
    return merged


def infer_symbol(values: Iterable[object], fallback: str) -> str:
    for val in values:
        if isinstance(val, str) and val.strip():
            return val.strip().upper()
    return fallback.upper()


def compute_growth(series: pd.Series) -> pd.Series:
    prev = series.shift(1)
    return safe_divide(series - prev, prev.abs())


def compute_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    rename_rules = {
        "totalassets": "total_assets",
        "totalstockholdersequity": "total_equity",
        "totalequity": "total_equity",
        "totaldebt": "total_debt",
        "netdebt": "net_debt",
        "cashandcashequivalents": "cash_and_equivalents",
        "cashandshortterminvestments": "cash_and_equivalents",
        "totalcurrentassets": "current_assets",
        "currentassets": "current_assets",
        "totalcurrentliabilities": "current_liabilities",
        "currentliabilities": "current_liabilities",
        "netincome": "net_income",
        "netincomeloss": "net_income",
        "netincomeapplicabletocommonshares": "net_income",
        "revenue": "sales",
        "revenues": "sales",
        "totalrevenue": "sales",
        "numberofshares": "number_of_shares",
        "marketcapitalization": "market_cap",
        "enterprisevalue": "enterprise_value",
    }
    original_columns = list(work.columns)
    column_lookup = {_normalize_metric_label(col): col for col in original_columns}
    for source_key, target in rename_rules.items():
        actual_col = column_lookup.get(source_key)
        if not actual_col:
            continue
        new_values = pd.to_numeric(work[actual_col], errors="coerce")
        if target in work.columns:
            work[target] = pd.to_numeric(work[target], errors="coerce")
            work[target] = work[target].combine_first(new_values)
        else:
            work[target] = new_values

    numeric_targets = [
        "total_assets",
        "total_equity",
        "total_debt",
        "net_debt",
        "cash_and_equivalents",
        "current_assets",
        "current_liabilities",
        "net_income",
        "sales",
        "number_of_shares",
        "market_cap",
        "enterprise_value",
        "DPS",
    ]
    for col in numeric_targets:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        else:
            work[col] = np.nan

    if "net_debt" not in work or work["net_debt"].isna().all():
        work["net_debt"] = work["total_debt"] - work["cash_and_equivalents"]

    work["total_dividends"] = work["DPS"] * work["number_of_shares"]

    work["log_assets"] = safe_log(work["total_assets"])
    work["log_market_cap"] = safe_log(work["market_cap"])
    work["debt_to_assets"] = safe_divide(work["total_debt"], work["total_assets"])
    work["debt_to_equity"] = safe_divide(work["total_debt"], work["total_equity"])
    work["net_debt_to_assets"] = safe_divide(work["net_debt"], work["total_assets"])
    work["cash_to_assets"] = safe_divide(work["cash_and_equivalents"], work["total_assets"])
    work["current_ratio"] = safe_divide(work["current_assets"], work["current_liabilities"])

    work = work.sort_values(["firm_id", "period"])
    work["asset_growth"] = work.groupby("firm_id", group_keys=False)["total_assets"].apply(compute_growth)
    work["delta_ESG"] = work.groupby("firm_id")["ESGScore"].diff() if "ESGScore" in work.columns else np.nan

    return work


def build_features_for_firm(firm_dir: Path) -> pd.DataFrame:
    symbol_guess = firm_dir.name.upper()

    balance_df = load_quarterly_statements(
        find_first_matching(firm_dir, BALANCE_PATTERN), sheet_candidates=BALANCE_SHEETS
    )
    income_df = load_quarterly_statements(
        find_first_matching(firm_dir, INCOME_PATTERN), sheet_candidates=INCOME_SHEETS
    )
    cash_df = load_quarterly_statements(
        find_first_matching(firm_dir, CASH_FLOW_PATTERN), sheet_candidates=CASH_FLOW_SHEETS
    )
    _log_frame_overview(f"{firm_dir.name} balance", balance_df)
    _log_frame_overview(f"{firm_dir.name} income", income_df)
    _log_frame_overview(f"{firm_dir.name} cashflow", cash_df)

    statements = merge_statement_frames([balance_df, income_df, cash_df])
    if statements.empty:
        logging.warning("Skipping %s (no quarterly statements found)", firm_dir.name)
        return pd.DataFrame()

    statements["symbol"] = symbol_guess
    statements["firm_id"] = symbol_guess

    enterprise = load_enterprise_values(find_first_matching(firm_dir, ENTERPRISE_PATTERN))
    _log_frame_overview(f"{firm_dir.name} enterprise", enterprise)
    if not enterprise.empty:
        symbol_guess = infer_symbol(enterprise["symbol"].dropna().unique().tolist(), symbol_guess)
        statements = merge_on_period(statements, enterprise, how="left")

    dividends = load_dividends(find_first_matching(firm_dir, DIVIDENDS_PATTERN))
    _log_frame_overview(f"{firm_dir.name} dividends", dividends)
    if not dividends.empty:
        statements = merge_on_period(statements, dividends, how="left")

    esg_risk = load_esg_risk(find_first_matching(firm_dir, ESG_RISK_PATTERN))
    _log_frame_overview(f"{firm_dir.name} esg_risk", esg_risk)
    if not esg_risk.empty:
        statements["fiscal_date"] = pd.to_datetime(statements["fiscal_date"], errors="coerce")
        esg_risk["risk_date"] = pd.to_datetime(esg_risk["risk_date"], errors="coerce")
        esg_risk = esg_risk.dropna(subset=["risk_date"])
        if esg_risk.empty or not statements["fiscal_date"].notna().any():
            pass
        else:
            statements["fiscal_date"] = statements["fiscal_date"].astype("datetime64[ns]")
            esg_risk["risk_date"] = esg_risk["risk_date"].astype("datetime64[ns]")
            esg_risk = esg_risk.sort_values("risk_date")
            statements = statements.sort_values("fiscal_date")
            try:
                aligned = pd.merge_asof(
                    statements,
                    esg_risk,
                    left_on="fiscal_date",
                    right_on="risk_date",
                    by="symbol",
                    direction="backward",
                )
                aligned = aligned.drop(columns=["risk_date"])
                statements = aligned
            except Exception as exc:  # pragma: no cover - defensive
                logging.warning("ESG risk merge failed for %s: %s", firm_dir.name, exc)

    esg_score = load_esg_scores(find_first_matching(firm_dir, ESG_SCORE_PATTERN))
    _log_frame_overview(f"{firm_dir.name} esg_score", esg_score)
    if not esg_score.empty:
        esg_score["symbol"] = esg_score["symbol"].fillna(symbol_guess)
        statements["fiscal_date"] = pd.to_datetime(statements["fiscal_date"], errors="coerce")
        esg_score["date"] = pd.to_datetime(esg_score["date"], errors="coerce")
        statements = statements.sort_values("fiscal_date")
        esg_score = esg_score.sort_values("date")
        aligned = pd.merge_asof(
            statements,
            esg_score,
            left_on="fiscal_date",
            right_on="date",
            by="symbol",
            direction="backward",
        )
        aligned = aligned.drop(columns=["date"])
        statements = aligned

    price_path = find_first_matching(firm_dir, DAILY_PRICE_PATTERN) or find_first_matching(firm_dir, MONTHLY_PRICE_PATTERN)
    price_df = load_prices(price_path)
    _log_frame_overview(f"{firm_dir.name} prices", price_df)
    price_features = compute_price_features(price_df)
    _log_frame_overview(f"{firm_dir.name} price_features", price_features)
    if not price_features.empty:
        statements = merge_on_period(statements, price_features, how="left")

    statements["symbol"] = symbol_guess
    statements["firm_id"] = symbol_guess
    statements = statements.sort_values(["period", "fiscal_date"]).reset_index(drop=True)
    statements = compute_financial_ratios(statements)
    return statements


def build_panel_dataset(base_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir():
            continue
        logging.info("Processing %s", entry.name)
        firm_df = build_features_for_firm(entry)
        if firm_df.empty:
            continue
        frames.append(firm_df)
    if not frames:
        logging.warning("No firm-level data assembled from %s", base_dir)
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True, sort=False)
    panel = panel.sort_values(["firm_id", "period"]).reset_index(drop=True)
    panel["period"] = panel["period"].astype("period[Q]")
    panel["period_str"] = panel["period"].astype(str)

    for col in BASE_OUTPUT_COLUMNS:
        if col not in panel.columns:
            panel[col] = np.nan
    panel = panel.loc[:, list(dict.fromkeys(BASE_OUTPUT_COLUMNS + [c for c in panel.columns if c not in BASE_OUTPUT_COLUMNS]))]
    return panel


def enforce_mandatory(df: pd.DataFrame) -> pd.DataFrame:
    work = df.replace([np.inf, -np.inf], np.nan).copy()
    for col in MANDATORY_FEATURES:
        if col not in work.columns:
            work[col] = np.nan
    clean = work.dropna(subset=MANDATORY_FEATURES)
    return clean


def add_changes_and_lags(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    change_like: Sequence[str] = ("asset_growth",),
    group_col: str = "firm_id",
    period_col: str = "period",
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    work = df.copy()
    work = work.sort_values([group_col, period_col])
    diff_cols: List[str] = []
    lag_cols: List[str] = []

    for feature in feature_names:
        if feature not in work.columns:
            continue
        if feature.endswith("_diff") or feature.endswith("_lag1"):
            continue
        series = pd.to_numeric(work[feature], errors="coerce")
        if feature in change_like:
            pass
        else:
            diff_col = f"{feature}_diff"
            work[diff_col] = series.groupby(work[group_col]).diff()
            diff_cols.append(diff_col)
        lag_col = f"{feature}_lag1"
        work[lag_col] = series.groupby(work[group_col]).shift(1)
        lag_cols.append(lag_col)

    work["DPS_pct_change"] = work.groupby(group_col)["DPS"].pct_change(fill_method=None)
    work["DPS_change_class"] = work["DPS_pct_change"].apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0) if pd.notna(x) else np.nan
    )
    return work, diff_cols, lag_cols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build quarterly dividend panel dataset.")
    parser.add_argument("--base_dir", required=True, help="Directory containing downloaded ticker folders.")
    parser.add_argument("--out_excel", default="panel_dividend_dataset.xlsx", help="Output Excel path (.xlsx).")
    parser.add_argument(
        "--premodel_path",
        default="premodel.xlsx",
        help="Path to save cleaned quarterly dataset with diffs/lags/target.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")

    base_dir = Path(args.base_dir).expanduser()
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    panel = build_panel_dataset(base_dir)
    if panel.empty:
        logging.warning("No data assembled. Outputs will not be written.")
        return

    panel.to_excel(Path(args.out_excel).expanduser(), index=False)

    # clean mandatory fields, then derive diffs/lags/targets
    cleaned = enforce_mandatory(panel)
    cleaned = cleaned.sort_values(["firm_id", "period"]).reset_index(drop=True)
    cleaned, diff_cols, lag_cols = add_changes_and_lags(cleaned, DERIVATIVE_FEATURES)

    premodel_path = Path(args.premodel_path).expanduser()
    cleaned.to_excel(premodel_path, index=False)

    n_firms = cleaned["firm_id"].nunique(dropna=True)
    n_periods = cleaned["period"].nunique(dropna=True)
    logging.info(
        "Panel shape: %s rows x %s cols | firms=%s | quarters=%s | diff_cols=%s | lag_cols=%s",
        *cleaned.shape,
        n_firms,
        n_periods,
        len(diff_cols),
        len(lag_cols),
    )
    logging.info("Preview:\n%s", cleaned.head().to_string())


if __name__ == "__main__":
    main()
