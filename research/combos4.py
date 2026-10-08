"""型態 + ER 比值 / 布林帶寬 / 外軌收盤 的組合，逐段結果與交易數（M15，突破進場，2R）。"""
import itertools
import numpy as np
import pandas as pd
from backtest import stats

T = pd.read_pickle("cand4_m15.pkl")
FOLDS = [("2~3月", "2026-02-01", "2026-04-01"), ("4~5月", "2026-04-01", "2026-06-01"),
         ("6~7月", "2026-06-01", "2026-08-01"), ("8~10月", "2026-08-01", "2026-10-09")]
DAYS = T.setup.dt.normalize().nunique()


def nonoverlap(D):
    D = D.sort_values("entry_time")
    keep, busy = [], pd.Timestamp.min
    for r in D.itertuples():
        if r.entry_time > busy:
            keep.append(r.Index); busy = r.exit_time
    return D.loc[keep]


def report(name, m, y="pnl_R"):
    D = T[m]
    row = {"規則": name}
    for fn, a, z in FOLDS:
        x = D[(D.entry_time >= a) & (D.entry_time < z)]
        row[fn] = f"{x[y].mean():+.2f}"
    s = stats(D.rename(columns={y: "pnl_R"}) if y != "pnl_R" else D)
    no = stats(nonoverlap(D))
    row.update(訊號數=s["n"], 每日=round(s["n"] / DAYS, 2), 期望=round(s["expR"], 3), t=round(s["t"], 2), PF=round(s["pf"], 2),
               不重疊n=no["n"], 不重疊期望=round(no["expR"], 3), 不重疊t=round(no["t"], 2))
    return row


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    B = T.er5_20 >= 2.0
    C = T.bbw_pct <= 0.5
    D = T.close_outside == 1
    rules = {"全部型態": T.index == T.index, "B: ER5/ER20>=2": B, "C: 帶寬百分位<=50%": C, "D: 收在外軌外": D,
             "B+C": B & C, "B+D": B & D, "C+D": C & D, "B+C+D": B & C & D,
             "三選二": (B.astype(int) + C.astype(int) + D.astype(int)) >= 2,
             "B 或 D": B | D}
    print(pd.DataFrame([report(k, v) for k, v in rules.items()]).to_string(index=False))
    print("\n門檻敏感度（B+C）")
    rows = []
    for er, bw in itertools.product((1.25, 1.5, 2.0, 3.0), (0.3, 0.4, 0.5, 0.6, 0.7)):
        rows.append(report(f"ER比>={er} 帶寬<={bw}", (T.er5_20 >= er) & (T.bbw_pct <= bw)))
    print(pd.DataFrame(rows).to_string(index=False))
