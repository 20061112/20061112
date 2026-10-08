"""重現本次研究的主要數字：python3 run_research.py"""
import numpy as np
import pandas as pd
from load import load
from features import add_features
from detector import detect, Params
from candidates import enrich, add_context, LOOSE
from evaluate import fwd_returns, summarize_fwd, outcomes, LAB_START, LAB_END
from labels import load_labels

LABEL_LIKE = dict(er=0.5, run20=5.5, wick=0.3, conf_atr=0.7)


def label_like(X):
    return (X.er >= LABEL_LIKE["er"]) & (X.run20 >= LABEL_LIKE["run20"]) & \
           (X.wick >= LABEL_LIKE["wick"]) & (X.conf_atr >= LABEL_LIKE["conf_atr"])


def mark_labels(S):
    lab = np.zeros(len(S), bool)
    for a, b, s in load_labels():
        lab |= ((S.side == s) & (S.time >= a - pd.Timedelta(minutes=1)) &
                (S.time <= b + pd.Timedelta(minutes=3))).to_numpy()
    return lab


def main():
    raw = load("data/XAUUSD_M1_full.csv")
    df = add_features(raw)

    print("== 1. 標記期間 (10/5~10/8)：寬鬆偵測器 + 篩選後的覆蓋率 ==")
    f = df.loc["2026-10-03":]
    S = detect(f, LOOSE)
    S = S[(S.time >= LAB_START) & (S.time <= LAB_END)].reset_index(drop=True)
    S = add_context(f, enrich(f, S))
    lab = mark_labels(S)
    for name, m in [("寬鬆候選", np.ones(len(S), bool)),
                    ("er>=.3 & run20>=3", ((S.er >= .3) & (S.run20 >= 3)).to_numpy()),
                    ("er>=.4 & run20>=4.5 & wick>=.3", ((S.er >= .4) & (S.run20 >= 4.5) & (S.wick >= .3)).to_numpy()),
                    ("label-like", label_like(S).to_numpy())]:
        hit = sum(any(((S.side == s) & (S.time >= a - pd.Timedelta(minutes=1)) &
                       (S.time <= b + pd.Timedelta(minutes=3))).to_numpy() & m) for a, b, s in load_labels())
        print(f"  {name:32s} 命中 {hit}/31  訊號數 {m.sum()}")

    F = fwd_returns(f, S)
    print("\n== 2. 事後挑選偏差：同期間 f10（ATR 倍數）==")
    print(f"  被標記       n={lab.sum():3d}  f10={F.f10[lab].mean():+.2f}")
    print(f"  未標記       n={(~lab).sum():3d}  f10={F.f10[~lab].mean():+.2f}")
    m = label_like(S).to_numpy() & ~lab
    print(f"  長得一樣但未標記 n={m.sum():3d}  f10={F.f10[m].mean():+.2f}")

    print("\n== 3. 樣本外 (6/29~10/3) M1 ==")
    fo = df.loc[:"2026-10-04"]
    O = enrich(fo, detect(fo, LOOSE)).dropna(subset=["x240"]).reset_index(drop=True)
    Fo = fwd_returns(fo, O)
    print("  全部候選"); print(summarize_fwd(Fo).T.to_string())
    print("  label-like"); print(summarize_fwd(Fo[label_like(O)]).T.to_string())
    R = outcomes(fo, O)
    print("  2R vs 1R 停損（隨機漫步基準約 33% : 67%）:", R.outcome.value_counts(normalize=True).round(3).to_dict())

    print("\n== 4. 樣本外 M5 ==")
    m5 = raw.resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                   "tickvol": "sum", "spread": "mean"}).dropna()
    m5 = add_features(m5).loc[:"2026-10-04"]
    s5 = enrich(m5, detect(m5, Params(swing_len=10, run_len=12, run_atr=2.0, max_wait=3, confirm_atr=0.5)))
    s5 = s5.dropna(subset=["x240"]).reset_index(drop=True)
    F5 = fwd_returns(m5, s5, horizons=(1, 2, 4, 6, 12))
    print("  全部候選"); print(summarize_fwd(F5).T.to_string())
    print("  label-like"); print(summarize_fwd(F5[label_like(s5)]).T.to_string())


if __name__ == "__main__":
    main()
