from __future__ import annotations

import numpy as np
import pandas as pd

BURN_IN = 120
SMA_FAST = 20
SMA_SLOW = 60
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_N = 14
ATR_N = 14
BB_N = 20
BB_K = 2.0
VOL_N = 20
MFI_N = 14
HIGHVOL_LOOKBACK = 252
HIGHVOL_Q = 0.80


def normalize_symbol(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )


def add_trading_day_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    volume = pd.to_numeric(out["Volume"], errors="coerce")
    close = pd.to_numeric(out["Close"], errors="coerce")
    adj_close = pd.to_numeric(out["AdjustedClose"], errors="coerce")

    out["TradingDay"] = (
        volume.gt(0)
        & close.gt(0)
        & adj_close.gt(0)
    )
    return out


def prepare_single_stock(
    df: pd.DataFrame,
    symbol: str,
    *,
    trading_only: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Symbol"] = normalize_symbol(out["Symbol"])
    symbol = str(symbol).zfill(6)

    out = (
        out.loc[out["Symbol"].eq(symbol)]
        .sort_values("Date")
        .drop_duplicates(subset=["Date", "Symbol"], keep="last")
        .reset_index(drop=True)
    )

    if out.empty:
        raise ValueError(f"No rows found for Symbol={symbol}")

    out = add_trading_day_flag(out)

    if trading_only:
        out = out.loc[out["TradingDay"]].copy().reset_index(drop=True)

    return out


def add_adjusted_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    close = pd.to_numeric(out["Close"], errors="coerce")
    adj_close = pd.to_numeric(out["AdjustedClose"], errors="coerce")

    valid = close.gt(0) & adj_close.gt(0)
    factor = pd.Series(np.nan, index=out.index, dtype="float64")
    factor.loc[valid] = adj_close.loc[valid] / close.loc[valid]

    out["AdjustmentFactor_Lab"] = factor
    out["AdjOpen"] = pd.to_numeric(out["Open"], errors="coerce") * factor
    out["AdjHigh"] = pd.to_numeric(out["High"], errors="coerce") * factor
    out["AdjLow"] = pd.to_numeric(out["Low"], errors="coerce") * factor
    out["AdjVolume"] = pd.to_numeric(out["Volume"], errors="coerce") / factor

    return out


def ema(s: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").ewm(
        span=span,
        adjust=False,
        min_periods=span,
    ).mean()


def wilder_smooth(s: pd.Series, n: int) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.ewm(
        alpha=1.0 / n,
        adjust=False,
        min_periods=n,
    ).mean()


def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    px = pd.to_numeric(out["AdjustedClose"], errors="coerce")

    out["SMA20"] = px.rolling(SMA_FAST, min_periods=SMA_FAST).mean()
    out["SMA60"] = px.rolling(SMA_SLOW, min_periods=SMA_SLOW).mean()

    ema_fast = ema(px, MACD_FAST)
    ema_slow = ema(px, MACD_SLOW)

    out["MACD"] = ema_fast - ema_slow
    out["MACD_Signal"] = ema(out["MACD"], MACD_SIGNAL)
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]
    return out


def add_rsi(df: pd.DataFrame, n: int = RSI_N) -> pd.DataFrame:
    out = df.copy()
    px = pd.to_numeric(out["AdjustedClose"], errors="coerce")
    delta = px.diff()

    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = wilder_smooth(gain, n)
    avg_loss = wilder_smooth(loss, n)

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)

    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), np.nan)

    out[f"RSI{n}"] = rsi
    return out


def add_atr(df: pd.DataFrame, n: int = ATR_N) -> pd.DataFrame:
    out = df.copy()

    high = pd.to_numeric(out["AdjHigh"], errors="coerce")
    low = pd.to_numeric(out["AdjLow"], errors="coerce")
    prev_close = pd.to_numeric(out["AdjustedClose"], errors="coerce").shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out["TR"] = tr
    out[f"ATR{n}"] = wilder_smooth(tr, n)
    out[f"ATRPercent{n}"] = (
        out[f"ATR{n}"]
        / pd.to_numeric(out["AdjustedClose"], errors="coerce")
    )
    return out


def add_bollinger(df: pd.DataFrame, n: int = BB_N, k: float = BB_K) -> pd.DataFrame:
    out = df.copy()
    px = pd.to_numeric(out["AdjustedClose"], errors="coerce")

    ma = px.rolling(n, min_periods=n).mean()
    sd = px.rolling(n, min_periods=n).std(ddof=0)

    out[f"BB_Mid{n}"] = ma
    out[f"BB_Upper{n}"] = ma + k * sd
    out[f"BB_Lower{n}"] = ma - k * sd
    out[f"BB_Width{n}"] = (2.0 * k * sd) / ma
    return out


def add_volume_ratio(df: pd.DataFrame, n: int = VOL_N) -> pd.DataFrame:
    out = df.copy()

    adj_vol = pd.to_numeric(out["AdjVolume"], errors="coerce")
    trading_value = pd.to_numeric(out["TradingValue"], errors="coerce")

    out[f"AdjVolume_MA{n}"] = adj_vol.rolling(n, min_periods=n).mean()
    out[f"VolumeRatio{n}"] = adj_vol / out[f"AdjVolume_MA{n}"]

    out[f"TradingValue_MA{n}"] = trading_value.rolling(
        n, min_periods=n
    ).mean()
    out[f"TradingValueRatio{n}"] = (
        trading_value / out[f"TradingValue_MA{n}"]
    )
    return out


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    px = pd.to_numeric(out["AdjustedClose"], errors="coerce")
    vol = pd.to_numeric(out["AdjVolume"], errors="coerce").fillna(0.0)

    direction = np.sign(px.diff()).fillna(0.0)

    signed = pd.Series(
        np.where(
            direction > 0,
            vol,
            np.where(direction < 0, -vol, 0.0),
        ),
        index=out.index,
        dtype="float64",
    )

    out["OBV"] = signed.cumsum()
    return out


def add_mfi(df: pd.DataFrame, n: int = MFI_N) -> pd.DataFrame:
    out = df.copy()

    high = pd.to_numeric(out["AdjHigh"], errors="coerce")
    low = pd.to_numeric(out["AdjLow"], errors="coerce")
    close = pd.to_numeric(out["AdjustedClose"], errors="coerce")
    vol = pd.to_numeric(out["AdjVolume"], errors="coerce")

    tp = (high + low + close) / 3.0
    raw_flow = tp * vol
    delta = tp.diff()

    pos_flow = raw_flow.where(delta > 0, 0.0)
    neg_flow = raw_flow.where(delta < 0, 0.0)

    pos_sum = pos_flow.rolling(n, min_periods=n).sum()
    neg_sum = neg_flow.rolling(n, min_periods=n).sum()

    ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    mfi = 100.0 - 100.0 / (1.0 + ratio)

    mfi = mfi.mask((neg_sum == 0) & (pos_sum > 0), 100.0)
    mfi = mfi.mask((neg_sum == 0) & (pos_sum == 0), np.nan)

    out["TypicalPrice"] = tp
    out[f"MFI{n}"] = mfi
    return out


def add_highvol_gate(
    df: pd.DataFrame,
    lookback: int = HIGHVOL_LOOKBACK,
    q: float = HIGHVOL_Q,
) -> pd.DataFrame:
    out = df.copy()

    col = f"ATRPercent{ATR_N}"
    threshold = out[col].rolling(
        lookback,
        min_periods=lookback,
    ).quantile(q)

    out["HighVolThreshold"] = threshold
    out["HighVol"] = (
        out[col].gt(threshold) & threshold.notna()
    ).astype(int)

    return out


def add_composite_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["TrendSignal"] = (out["SMA20"] > out["SMA60"]).astype(int)
    out["MomentumSignal"] = (out["MACD"] > out["MACD_Signal"]).astype(int)
    out["VolumeSignal"] = (
        (out["VolumeRatio20"] > 1.2)
        & (out["MFI14"] > 50)
    ).astype(int)

    out["Score"] = (
        out["TrendSignal"]
        + out["MomentumSignal"]
        + out["VolumeSignal"]
    )

    out["RSIEntryFilter"] = (out["RSI14"] < 70).astype(int)

    out["EntryCondition_COMP"] = (
        (out["Score"] >= 2)
        & (out["RSIEntryFilter"] == 1)
    ).astype(int)

    out["EntryCondition_COMP_RM"] = (
        (out["Score"] >= 2)
        & (out["RSIEntryFilter"] == 1)
        & (out["HighVol"] == 0)
    ).astype(int)

    out["ExitCondition"] = (out["Score"] < 2).astype(int)
    return out


def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if "TradingDay" in df.columns and not df["TradingDay"].all():
        raise ValueError(
            "build_indicators expects trading-day-only data. "
            "Use prepare_single_stock(..., trading_only=True)."
        )

    out = add_adjusted_ohlcv(df)
    out = add_trend_indicators(out)
    out = add_rsi(out)
    out = add_atr(out)
    out = add_bollinger(out)
    out = add_volume_ratio(out)
    out = add_obv(out)
    out = add_mfi(out)
    out = add_highvol_gate(out)
    out = add_composite_signals(out)
    return out


def correlation_inputs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SMA_Gap"] = out["SMA20"] / out["SMA60"] - 1.0
    out["ATR_Pct"] = out["ATR14"] / out["AdjustedClose"]

    return out[
        [
            "SMA_Gap",
            "MACD_Hist",
            "RSI14",
            "VolumeRatio20",
            "MFI14",
            "ATR_Pct",
        ]
    ]


def public_indicator_sanity(df: pd.DataFrame) -> dict:
    required = [
        "AdjustedClose",
        "AdjOpen",
        "AdjHigh",
        "AdjLow",
        "AdjVolume",
        "RSI14",
        "ATR14",
        "MFI14",
        "VolumeRatio20",
    ]

    missing = [c for c in required if c not in df.columns]

    return {
        "missing_required_columns": missing,
        "n_rows": int(len(df)),
        "all_rows_are_trading_days": bool(
            df["TradingDay"].all()
        ) if "TradingDay" in df.columns else None,
        "invalid_adjustment_factor_rows": int(
            (~pd.to_numeric(
                df["AdjustmentFactor_Lab"],
                errors="coerce",
            ).gt(0)).sum()
        ) if "AdjustmentFactor_Lab" in df.columns else None,
        "rsi_range_ok": bool(
            df["RSI14"].dropna().between(0, 100).all()
        ) if "RSI14" in df.columns else False,
        "mfi_range_ok": bool(
            df["MFI14"].dropna().between(0, 100).all()
        ) if "MFI14" in df.columns else False,
        "atr_nonnegative": bool(
            (df["ATR14"].dropna() >= 0).all()
        ) if "ATR14" in df.columns else False,
    }
