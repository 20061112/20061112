import pandas as pd
import numpy as np
from labels import load_labels

LAB_START, LAB_END = pd.Timestamp("2026-10-05 08:00"), pd.Timestamp("2026-10-08 10:35")


def recall(sig, slack_before=1, slack_after=3):
    """標記窗口 [a-slack_before, b+slack_after] 內有同方向訊號即算命中。"""
    hits, matched = [], set()
    for a, b, s in load_labels():
        m = sig[(sig.side == s) & (sig.time >= a - pd.Timedelta(minutes=slack_before)) &
                (sig.time <= b + pd.Timedelta(minutes=slack_after))]
        hits.append(len(m) > 0)
        matched.update(m.index)
    in_period = sig[(sig.time >= LAB_START) & (sig.time <= LAB_END)]
    return np.mean(hits), hits, len(in_period), len(matched & set(in_period.index))


def outcomes(df, sig, stop_buf=0.3, horizon=30):
    """以確認棒收盤進場，停損設在極值外 stop_buf×ATR，追蹤 horizon 根內 MFE/MAE（以 R 計）。"""
    pos = df.index.get_indexer(sig.time)
    H, L, C = df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy()
    res = []
    for (i, r) in zip(pos, sig.itertuples()):
        stop = r.ext - r.side * stop_buf * r.atr
        R = abs(r.entry - stop)
        hit = {1: np.nan, 2: np.nan}
        out = "open"
        for j in range(i + 1, min(i + 1 + horizon, len(df))):
            adverse = (stop >= L[j]) if r.side == 1 else (stop <= H[j])
            fav = ((H[j] - r.entry) if r.side == 1 else (r.entry - L[j])) / R
            if adverse:
                out = "stop"; break
            if fav >= 2:
                out = "2R"; break
        fut = slice(i + 1, min(i + 1 + horizon, len(df)))
        mfe = ((H[fut].max() - r.entry) if r.side == 1 else (r.entry - L[fut].min())) / R
        res.append(dict(R_usd=R, R_atr=R / r.atr, outcome=out, mfe_R=mfe))
    return pd.concat([sig.reset_index(drop=True), pd.DataFrame(res)], axis=1)


def fwd_returns(df, sig, horizons=(3, 5, 10, 20, 30)):
    """確認棒收盤後 h 根的順向報酬（ATR 倍數，正 = 往反轉方向走）。"""
    pos = df.index.get_indexer(sig.time)
    C = df.close.to_numpy()
    out = {}
    for h in horizons:
        j = np.minimum(pos + h, len(df) - 1)
        out[f"f{h}"] = sig.side.to_numpy() * (C[j] - C[pos]) / sig.atr.to_numpy()
    return pd.DataFrame(out, index=sig.index)


def summarize_fwd(F):
    m = F.mean(); se = F.std() / np.sqrt(len(F))
    return pd.DataFrame({"mean_atr": m, "t": m / se, "p_pos": (F > 0).mean()}).round(3)
