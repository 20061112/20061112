"""第五十一部分補充：ATR熵變化的參數敏感度（實際下單 A；門檻 = 訓練期分位數）。
熵視窗 W（M15 根數）× 變化間隔 Lag × 門檻分位 Q；規則：熵變化 > 門檻 → 不做 / 減半。"""
import numpy as np
import pandas as pd
import part51_pf_states as S

T = S.T
k = np.maximum(S.pid[T.i.values] - 1, 0)
trm = T.train.values
R = T.R.values
lines = ["ATR熵變化 參數敏感度（A 實際下單，RR 3）：格內 = 規則「不做」後的 年獲利/含浮虧MDD（美元）｜訓練比值｜驗證比值；基準 1.59｜3.6｜5.0"]
base_tr15 = S.tr15
for W in (10, 20, 30, 40):
    ent = S.norm_entropy_rows(S.rolling_windows(base_tr15, W))
    for lag in (5, 10, 20, 30):
        chg = ent - np.r_[np.full(lag, np.nan), ent[:-lag]]
        v = chg[k]
        cells = []
        for qq in (0.7, 0.8, 0.9):
            c = np.nanquantile(v[trm], qq)
            w = np.where(v > c, 0.0, 1.0)
            pf_bad = S.pf(R[~trm & (v > c)]); pf_rest = S.pf(R[~trm & (v <= c)])
            x = T.copy(); x["net"] = x.net * w
            mm = S.P.minute_mdd(x)
            usd = (T.net.values * 2 * w)
            yrs = (T.進場時間.max() - T.進場時間.min()).days / 365

            def rat(m):
                s_ = pd.Series(usd[m]).groupby(T.date.values[m]).sum(); e = s_.cumsum(); return usd[m].sum() / (e.cummax() - e).max()
            cells.append(f"Q{int(qq * 100)}: {usd.sum() / yrs / mm:4.2f}|{rat(trm):3.1f}|{rat(~trm):3.1f}（驗證PF 壞{pf_bad:.2f}/其餘{pf_rest:.2f}）")
        lines.append(f"  W={W:2d} lag={lag:2d}  " + "  ".join(cells))
        print(lines[-1], flush=True)
open("results_part51_entropy_robust.txt", "w").write("\n".join(lines) + "\n")
