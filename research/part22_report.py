"""第二十二部分報表：python3 part22_report.py > results_part22.txt（先跑 part22_divergence.py 的 build 會較快，這裡直接重算）。"""
import pickle
import os
import numpy as np
import pandas as pd
from part22_divergence import get_frames, build, nonoverlap, DIVS, TRIG, SPLIT

PKL = "part22_events.pkl"


def tstat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return x.mean() / x.std(ddof=1) * np.sqrt(len(x)) if len(x) > 2 else np.nan


def fmt(r):
    if len(r) < 5:
        return f"n={len(r):4d}      -        "
    return f"n={len(r):4d} {r.mean():+.3f}R t={tstat(r):+5.2f}"


def per_year(T):
    g = T.groupby(T.entry_time.dt.year).pnl_R
    return " ".join(f"{y}:{v.mean():+.2f}" for y, v in g)


def load_all():
    if os.path.exists(PKL):
        return pickle.load(open(PKL, "rb"))
    out = {tf: build(df) for tf, df in get_frames().items()}
    pickle.dump(out, open(PKL, "wb"))
    return out


def main():
    data = load_all()
    for tf, (E, T) in data.items():
        print(f"\n{'=' * 100}\n## {tf}  事件期間 {E.time.min():%Y-%m-%d} ~ {E.time.max():%Y-%m-%d}")
        E["res"] = E.time < SPLIT
        T["res"] = T.entry_time < SPLIT
        nb = (E.cond == "base").sum()

        print("\n### 1. 事件數、出現比例、之後有沒有進場訊號（WAIT=4 根內觸發的比例）")
        print(f"{'條件':7s} {'事件':>6s} {'/base':>6s} " + " ".join(f"{t:>6s}" for t in TRIG))
        for cnd in ["base"] + DIVS:
            e = E[E.cond == cnd]
            conv = [(T[(T.cond == cnd) & (T.trig == tg)].shape[0]) / max(len(e), 1) for tg in TRIG]
            print(f"{cnd:7s} {len(e):6d} {len(e) / nb:6.2f} " + " ".join(f"{x:6.2f}" for x in conv))

        print("\n### 2. 訊號棒收盤後的前瞻報酬（反轉方向為正，ATR 單位，已扣點差；事件可重疊）")
        print(f"{'條件':7s} {'研究期 f4':>18s} {'研究期 f8':>18s} {'研究期 f16':>18s} | {'驗證期 f8':>18s} {'驗證期 f16':>18s}")
        for cnd in ["base"] + DIVS:
            e = E[E.cond == cnd]
            a, b = e[e.res], e[~e.res]
            cell = lambda x: f"{x.mean():+.3f} (t{tstat(x):+5.1f})"
            print(f"{cnd:7s} {cell(a.f4):>18s} {cell(a.f8):>18s} {cell(a.f16):>18s} | {cell(b.f8):>18s} {cell(b.f16):>18s}")

        print("\n### 3. 進場規則回測（單一持倉、不重疊；2R 停利、停損 = 極值外 0.1 ATR、至少 0.5 ATR）")
        for tg in TRIG:
            print(f"\n  [{tg}]   研究期 2023~24            驗證期 2025~26/10        逐年平均R")
            for cnd in ["base"] + DIVS:
                x = nonoverlap(T[(T.cond == cnd) & (T.trig == tg)])
                print(f"  {cnd:7s} {fmt(x[x.res].pnl_R):28s} {fmt(x[~x.res].pnl_R):28s} {per_year(x)}")

        # 波動狀態條件：只用研究期決定，驗證期照搬
        print("\n### 4. 波動狀態條件（T1 進場，事件可重疊；研究期 | 驗證期）")
        T1 = T[T.trig == "T1"].copy()
        T1["vol_lvl_b"] = pd.cut(T1.vol_lvl, [0, .8, 1.25, 99], labels=["低", "中", "高"])
        T1["vol_chg_b"] = pd.cut(T1.vol_chg, [0, .85, 1.15, 99], labels=["收縮", "持平", "擴張"])
        T1["bbw_b"] = pd.cut(T1.bbw_pct, [0, .33, .67, 1.01], labels=["窄", "中", "寬"])
        for var, labs in (("vol_lvl_b", ["低", "中", "高"]), ("vol_chg_b", ["收縮", "持平", "擴張"]), ("bbw_b", ["窄", "中", "寬"])):
            print(f"\n  {var}")
            for cnd in ["base"] + DIVS:
                x = T1[T1.cond == cnd]
                cells = []
                for lb in labs:
                    a, b = x[(x[var] == lb) & x.res].pnl_R, x[(x[var] == lb) & ~x.res].pnl_R
                    cells.append(f"{lb}:{a.mean():+.2f}({len(a)})|{b.mean():+.2f}({len(b)})")
                print(f"  {cnd:7s} " + "  ".join(cells))

        print("\n### 5. 同時背離的指標數（ndiv，第一根突破棒；T1；可重疊）與推進/速度合併")
        B = T1[T1.cond == "base"]
        for k in range(6):
            x = B[B.ndiv == k]
            print(f"  ndiv={k}: 研究期 {fmt(x[x.res].pnl_R):28s} 驗證期 {fmt(x[~x.res].pnl_R)}")
        Eb = E[E.cond == "base"].set_index(["side", "P1"])
        for cnd in ("push", "speed"):
            Ec = E[E.cond == cnd].set_index(["side", "P1"])
            print(f"  base 事件中，同一個第二推之後出現 {cnd} 背離的比例：{Eb.index.isin(Ec.index).mean():.2f}")
        x = T1[(T1.cond == "push") & (T1.ndiv >= 4)]
        print(f"  push 背離 + >=4 個指標背離 : 研究期 {fmt(x[x.res].pnl_R):28s} 驗證期 {fmt(x[~x.res].pnl_R)}")
        x = T1[(T1.cond == "speed") & (T1.ndiv >= 4)]
        print(f"  speed背離 + >=4 個指標背離 : 研究期 {fmt(x[x.res].pnl_R):28s} 驗證期 {fmt(x[~x.res].pnl_R)}")

        x = nonoverlap(T[(T.cond == "base") & (T.trig == "T1")])
        print(f"\n  base T1 的停損 R 中位數 {x.R_atr.median():.2f} ATR；停損出場 {(x.exit == 'stop').mean():.2f}、停利 {(x.exit == 'tp').mean():.2f}、時間 {(x.exit == 'time').mean():.2f}")

        print("\n### 6. 研究期挑組合 → 驗證期照搬（背離 × 進場 × 波動狀態，不重疊交易）")
        T2 = T[T.trig != "T0"].copy()
        T2["vol_lvl_b"] = pd.cut(T2.vol_lvl, [0, .8, 1.25, 99], labels=["低", "中", "高"]).astype(str)
        T2["vol_chg_b"] = pd.cut(T2.vol_chg, [0, .85, 1.15, 99], labels=["收縮", "持平", "擴張"]).astype(str)
        T2["bbw_b"] = pd.cut(T2.bbw_pct, [0, .33, .67, 1.01], labels=["窄", "中", "寬"]).astype(str)
        T2["all"] = "全部"
        nmin = {"M5": 300, "M15": 120, "H1": 40}[tf]
        res = []
        for (cnd, tg), g in T2.groupby(["cond", "trig"]):
            for var in ("all", "vol_lvl_b", "vol_chg_b", "bbw_b"):
                for lb, gg in g.groupby(var):
                    x = nonoverlap(gg)
                    a, b = x[x.res].pnl_R, x[~x.res].pnl_R
                    if len(a) >= nmin:
                        res.append((cnd, tg, var, lb, len(a), a.mean(), tstat(a), len(b), b.mean(), tstat(b)))
        R_ = pd.DataFrame(res, columns=["cond", "trig", "var", "lvl", "n_a", "exp_a", "t_a", "n_b", "exp_b", "t_b"])
        print(f"  共 {len(R_)} 個組合；研究期與驗證期期望值的相關係數 {R_.exp_a.corr(R_.exp_b):+.2f}；"
              f"研究期為正 {(R_.exp_a > 0).mean():.0%}、驗證期為正 {(R_.exp_b > 0).mean():.0%}")
        top = R_.sort_values("t_a", ascending=False).head(10)
        print("  研究期 t 值前 10 名 → 驗證期：")
        for r in top.itertuples():
            print(f"    {r.cond:6s} {r.trig} {r.var:9s} {r.lvl:4s}  研究期 n={r.n_a:4d} {r.exp_a:+.3f}R t={r.t_a:+.2f}   驗證期 n={r.n_b:4d} {r.exp_b:+.3f}R t={r.t_b:+.2f}")
        print(f"  前 10 名驗證期平均 {top.exp_b.mean():+.3f}R；全部組合驗證期平均 {R_.exp_b.mean():+.3f}R")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
