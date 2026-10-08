"""幾個常見的反轉假說，直接當成規則跑 8 個月 M15，逐段報告（不調參）。"""
import numpy as np
import pandas as pd
from feature_stability import T, FOLDS
from backtest import stats

HYP = {
    "基準：全部候選": lambda D: np.ones(len(D), bool),
    "H1 衝刺竭盡：極值收在布林外軌外 + 最後一段加速": lambda D: (D.close_outside == 1) & (D.roc_decel > 0),
    "H2 擠壓後擴張：近 20 根有擠壓 + 帶寬正在擴張": lambda D: (D.squeeze_recent == 1) & (D.bbw_chg3 > 0),
    "H2b 帶寬已在高檔（擴張末端）": lambda D: D.bbw_pct > 0.8,
    "H3 掃前日高/低點後收回": lambda D: D.dist_pd < 0,
    "H4 RSI14 背離": lambda D: D.rsi_div > 5,
    "H5 Hull MA 剛轉向": lambda D: D.hma_turn == 1,
    "H6 量能高潮 (極值 K 棒量 > 1.5 倍)": lambda D: D.tv_e > 1.5,
    "H1+H3": lambda D: (D.close_outside == 1) & (D.roc_decel > 0) & (D.dist_pd < 0),
}


def nonoverlap(D):
    D = D.sort_values("entry_time")
    keep, busy = [], pd.Timestamp.min
    for r in D.itertuples():
        if r.entry_time > busy:
            keep.append(r.Index); busy = r.exit_time
    return D.loc[keep]


if __name__ == "__main__":
    rows = []
    for name, fn in HYP.items():
        D = nonoverlap(T[np.asarray(fn(T))])
        r = {"假說": name}
        for fname, a, z in FOLDS:
            x = D[(D.entry_time >= a) & (D.entry_time < z)]
            r[fname] = f"{x.pnl_R.mean():+.2f} ({len(x)})"
        s = stats(D)
        r.update(全期=f"{s['expR']:+.3f}", t=round(s["t"], 2), PF=round(s["pf"], 2), n=s["n"])
        rows.append(r)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string(index=False))
