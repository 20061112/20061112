"""第三十三部分：「某一邊最近 120 個交易日累計 R < 0 → 那一邊今天不開」的詳細統計與回撤。
套用在 L1 核心 + 反向尾巴（主要）與版本 A（凍結門檻），並和不加過濾比較。
PF 以 R 計；帳戶換算 = 1 萬美元、每筆固定風險 r%、不複利。
"""
import numpy as np
import pandas as pd
from load import load
from part29_dir_filter import BASES, OOS_END, pf
from part32_side_gate import perf_gate, DAYS

out = None


def P(s=""):
    print(s)
    if out:
        out.write(s + "\n")
        out.flush()


m1 = load("data/XAUUSD_M1_2023_2026.csv")
tm = m1.index.values
H, L = m1.high.to_numpy(), m1.low.to_numpy()


def curves(T):
    """逐分鐘浮動損益（最差價），回傳 平倉權益、含浮動權益、同時持倉數、帳戶浮動。"""
    open_arr = np.zeros(len(tm))
    cnt = np.zeros(len(tm) + 1)
    for r in T.itertuples():
        a = np.searchsorted(tm, np.datetime64(r.entry_time), side="right")
        b = np.searchsorted(tm, np.datetime64(r.exit_time), side="left")
        if b > a:
            open_arr[a:b] += np.maximum((r.side * ((L if r.side == 1 else H)[a:b] - r.lvl) - r.spread) / r.risk, -1.0)
            cnt[a] += 1
            cnt[b] -= 1
    n_all = np.cumsum(cnt)[:-1]
    act = n_all > 0
    open_R = pd.Series(open_arr[act], index=tm[act])
    n_open = pd.Series(n_all[act], index=tm[act])
    real = T.groupby("exit_time").R.sum().sort_index()
    ix = open_R.index.union(real.index)
    eq_mtm = real.reindex(ix, fill_value=0).cumsum() + open_R.reindex(ix, fill_value=0)
    return real.cumsum(), eq_mtm, n_open, open_R


def episodes(eq, k=5):
    """前 k 大回撤：開始（前高）、谷底、回到前高的時間、深度、天數。"""
    peak = eq.cummax()
    dd = eq - peak
    res, used = [], pd.Series(False, index=eq.index)
    for _ in range(k):
        d = dd[~used]
        if d.empty or d.min() >= 0:
            break
        t_low = d.idxmin()
        start = eq[:t_low][eq[:t_low] >= peak[t_low]].index[-1]
        after = eq[t_low:]
        rec = after[after >= peak[t_low]]
        end = rec.index[0] if len(rec) else None
        res.append((start, t_low, end, dd[t_low]))
        used[(eq.index >= start) & (eq.index <= (end if end is not None else eq.index[-1]))] = True
    return res


def detailed(T, name):
    T = T.sort_values("entry_time").reset_index(drop=True)
    eq_cl, eq_mtm, n_open, open_R = curves(T)
    dd_cl, dd_mtm = eq_cl - eq_cl.cummax(), eq_mtm - eq_mtm.cummax()
    R = T.R
    w, l = R[R > 0], R[R <= 0]
    days = DAYS[(DAYS >= T.day.min()) & (DAYS <= T.day.max())]
    daily = pd.Series(0.0, index=days).add(T.groupby("day").R.sum(), fill_value=0)
    lose = (R <= 0).astype(int)
    top = T.groupby("day").R.sum().sort_values(ascending=False)
    hi = eq_cl.cummax()
    longest = pd.Series(eq_cl.index[eq_cl >= hi]).diff().dt.days.max()
    P(f"\n\n# {name}")
    P(f"期間 {T.day.min().date()} ~ {T.day.max().date()}")
    S = {
        "交易筆數": f"{len(T):,}（多 {(T.side == 1).sum():,} / 空 {(T.side == -1).sum():,}），每交易日 {len(T) / len(daily):.2f} 筆，有單日 {T.day.nunique() / len(daily):.0%}",
        "勝率 / 賺賠比": f"{100 * (R > 0).mean():.1f}% / {w.mean() / -l.mean():.2f}（平均 {w.mean():+.2f}R / {l.mean():+.2f}R）",
        "每筆期望": f"{R.mean():+.3f}R（中位數 {R.median():+.2f}R）",
        "PF（以 R 計）": f"{pf(R):.2f}（多 {pf(R[T.side == 1]):.2f} / 空 {pf(R[T.side == -1]):.2f}）",
        "總損益": f"{R.sum():+.0f}R",
        "最大回撤 平倉 / 含浮動": f"{dd_cl.min():.0f}R（{dd_cl.idxmin():%Y-%m-%d}） / {dd_mtm.min():.0f}R（{dd_mtm.idxmin():%Y-%m-%d %H:%M}）",
        "總損益 / 含浮動回撤": f"{R.sum() / -dd_mtm.min():.1f}",
        "最長沒創新高": f"{longest:.0f} 天",
        "日 Sharpe（年化）": f"{daily.mean() / daily.std() * np.sqrt(252):.2f}",
        "正報酬日 / 正報酬月": f"{(daily > 0).mean():.0%} / {(T.groupby(T.day.dt.to_period('M')).R.sum() > 0).mean():.0%}",
        "最差 / 最好單日": f"{daily.min():+.1f}R / {daily.max():+.1f}R",
        "最長連虧": f"{lose.groupby((lose != lose.shift()).cumsum()).sum().max()} 筆",
        "獲利集中": f"最好 10 天占 {top.head(10).sum() / R.sum():.0%}；拿掉最好 10 天 {top.iloc[10:].sum():+.0f}R、20 天 {top.iloc[20:].sum():+.0f}R",
        "同時持倉 最多 / 平均": f"{int(n_open.max())} / {n_open.mean():.1f} 筆；帳戶同時浮虧最大 {open_R.min():.1f}R",
    }
    for k, v in S.items():
        P(f"  {k}：{v}")
    P("\n  ## 前 5 大回撤（平倉權益）")
    for s, lo, e, d in episodes(eq_cl):
        P(f"    {d:6.0f}R：{s:%Y-%m-%d} 高點 → {lo:%Y-%m-%d} 谷底（{(lo - s).days} 天）→ "
          + (f"{e:%Y-%m-%d} 回到高點（共 {(e - s).days} 天）" if e is not None else "尚未回到高點"))
    P("\n  ## 帳戶換算（1 萬美元、不複利）")
    for r in (0.1, 0.15, 0.2, 0.25):
        P(f"    每筆 {r}%：總報酬 {R.sum() * r:+.0f}%；最大回撤 平倉 {dd_cl.min() * r:.1f}% / 含浮動 {dd_mtm.min() * r:.1f}%；"
          f"同時最大浮虧 {open_R.min() * r:.1f}%；年化約 {R.sum() * r / (len(daily) / 252):+.0f}%")
    yr = T.groupby(T.day.dt.year)
    tb = pd.DataFrame({"筆數": yr.size(), "勝率%": yr.R.apply(lambda x: 100 * (x > 0).mean()), "每筆R": yr.R.mean(), "總R": yr.R.sum(),
                       "PF": yr.R.apply(pf), "多筆": yr.side.apply(lambda s: (s == 1).sum()), "空筆": yr.side.apply(lambda s: (s == -1).sum()),
                       "多PF": yr.apply(lambda g: pf(g[g.side == 1].R), include_groups=False),
                       "空PF": yr.apply(lambda g: pf(g[g.side == -1].R), include_groups=False),
                       "年內最大回撤R": yr.apply(lambda g: ((lambda e: (e - e.cummax()).min())(g.sort_values("exit_time").R.cumsum())), include_groups=False)})
    P("\n  ## 逐年\n" + tb.round(2).to_string())
    mon = T.groupby(T.day.dt.to_period("M")).R.sum()
    P(f"\n  ## 月份：{len(mon)} 個月中 {(mon > 0).sum()} 個月賺；最差 {mon.min():+.0f}R（{mon.idxmin()}），最好 {mon.max():+.0f}R（{mon.idxmax()}）")
    P("  " + "  ".join(f"{k}:{v:+.0f}" for k, v in mon.items()))
    return T, eq_cl, eq_mtm, dd_mtm


def gate_state(T, N=120):
    """每天多 / 空兩邊是否開放。"""
    st = {}
    for s in (1, -1):
        daily = T[T.side == s].groupby("day").R.sum().reindex(DAYS, fill_value=0)
        st[s] = ~(daily.rolling(N, min_periods=N).sum().shift(1) < 0)
    return pd.DataFrame({"多開": st[1], "空開": st[-1]})


if __name__ == "__main__":
    out = open("results_part33.txt", "w")
    P("第三十三部分：120 天側邊開關（某一邊最近 120 個交易日累計 R < 0 → 那一邊今天不開）")
    res = {}
    for name in ("L1 核心 + 反向尾巴（前推）", "版本 A（凍結門檻）"):
        T0 = BASES[name]
        G = perf_gate(T0, 120, 0)
        res[name] = (detailed(G, f"{name} + 120 天開關"), detailed(T0, f"{name}（對照：不加開關）"))
        st = gate_state(T0).loc[T0.day.min():T0.day.max()]
        P(f"\n  ## {name}：開關狀態（每年兩邊開放的天數比例）")
        for y, g in st.groupby(st.index.year):
            P(f"    {y}: 多單開放 {g['多開'].mean():.0%}，空單開放 {g['空開'].mean():.0%}")
        ch = st.astype(int).diff().abs().sum()
        P(f"    開關切換次數：多 {int(ch['多開'])} 次，空 {int(ch['空開'])} 次")
        G.drop(columns=["ym"], errors="ignore").to_csv(f"part33_{'L1tail' if 'L1' in name else 'A'}_gate120_trades.csv", index=False, float_format="%.4f")


    def chart():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(15, 8), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
        for j, (name, ((G, ecg, emg, ddg), (T0, ec0, em0, dd0))) in enumerate(res.items()):
            lab = "L1 core + tail" if "L1" in name else "Version A (frozen)"
            a = ax[0, j]
            a.plot(em0.resample("1h").last().dropna(), color="#999999", lw=1, label="no gate")
            a.plot(emg.resample("1h").last().dropna(), color="#3b6ea8", lw=1.2, label="120-day side gate")
            for s, c, l in ((1, "#2e8b57", "gate: long side"), (-1, "#c0392b", "gate: short side")):
                x = G[G.side == s].sort_values("exit_time")
                a.plot(x.exit_time, x.R.cumsum(), color=c, lw=.8, alpha=.7, label=l)
            a.axvline(pd.Timestamp("2026-01-01"), color="k", ls="--", lw=.8)
            a.set_title(f"{lab}: cumulative R incl. open positions"); a.legend(loc="upper left", fontsize=8); a.grid(alpha=.3)
            b = ax[1, j]
            b.fill_between(dd0.resample("1h").min().dropna().index, dd0.resample("1h").min().dropna().values, 0, color="#999999", alpha=.4, label="no gate")
            b.fill_between(ddg.resample("1h").min().dropna().index, ddg.resample("1h").min().dropna().values, 0, color="#c0392b", alpha=.5, label="120-day gate")
            b.set_ylabel("drawdown incl. open (R)"); b.legend(loc="lower left", fontsize=8); b.grid(alpha=.3)
        plt.tight_layout(); plt.savefig("part33_gate120.png", dpi=105)


    chart()
    out.close()
