"""第五十六部分（應用 A）：用預測波動 σ̂ 取代原本的停損距離 / ATR 停損 / 固定美元停損，進場完全不變。

問題：進場訊號不變、出場時間不變，只換「停損距離」與「部位大小」（固定每筆風險 B = 100 美元），
      預測波動 σ̂（第五十五部分的 har_rel，逐月 walk-forward）能否讓風險更一致、回撤更小、風險調整後報酬更好？
進場來源（都只用事前資訊；停損 / 部位在進場前一根 K 棒收盤時決定）
  1. 版本 A（IB 突破）：part53_trades_2025_2026.csv，M5 2025/8 ~ 2026/10（原停損 = IB 另一側，當天收盤出場）
  2. OU 趨勢：part49_trades.csv（M15 2023/4 ~ 2026/10；M5 2025/8 ~ 2026/10），原停損 = ATR 型，時間 / 週五出場
  3. 隨機進場（虛無對照）：M15 2023~2026，每天 3 個隨機時點、隨機方向、持有 16 根（4 小時）
停損版本（中位停損距離都校準成與原策略相同 —— 用前 40% 的交易估計倍數 k，之後固定）
  ORIG   原本的停損距離，部位 = B / 停損
  ATR    k_A × ATR14
  SIG    k_S × σ̂（未來 h 根的預測波動，美元）
  VT     原停損不變，部位改成 B / (k_S × σ̂)（只用 σ̂ 定部位）
  FIX    固定美元停損（隨機進場沒有原停損時用）
停損成交：碰到停損價成交；若開盤已跳過停損，以開盤價成交（跳空滑價）。點差每筆扣一次。
"""
import numpy as np
import pandas as pd
from volfc import har_rel

B = 100.0
OUT = "results_part56.txt"
RNG = np.random.default_rng(56)


def locate(idx, times, on_open):
    """回傳每筆進場對應的 K 棒位置：若 on_open，該根開盤即進場（從該根開始檢查停損），否則從下一根開始。"""
    pos = idx.searchsorted(times, side="right") - 1
    start = pos if on_open else pos + 1
    return pos, start


def simulate(bars, T, D, size):
    """T: DataFrame(entry_px, side, start, exit_i, spread)；D: 停損距離（美元）；size: 盎司。"""
    o, h, l, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    out = np.full(len(T), np.nan)
    stopped = np.zeros(len(T), bool)
    for n, (e, s, st, xi, sp) in enumerate(zip(T.entry_px, T.side, T.start, T.exit_i, T.spread)):
        d = D[n]
        if not (np.isfinite(d) and d > 0 and np.isfinite(size[n])):
            continue
        stop = e - s * d
        px = c[xi]
        for j in range(st, xi + 1):
            if s > 0 and l[j] <= stop:
                px = min(o[j], stop)
                stopped[n] = True
                break
            if s < 0 and h[j] >= stop:
                px = max(o[j], stop)
                stopped[n] = True
                break
        out[n] = size[n] * ((px - e) * s - sp)
    return out, stopped


def stats(pnl, stopped, times, sig_ref, label, pr, ratio=None):
    ok = np.isfinite(pnl)
    p = pnl[ok]
    t = times[ok]
    days = pd.Series(p, index=pd.DatetimeIndex(t).normalize()).groupby(level=0).sum()
    alld = pd.date_range(days.index.min(), days.index.max(), freq="B")
    days = days.reindex(alld, fill_value=0.0)
    eq = days.cumsum()
    dd = (eq - eq.cummax()).min()
    sh = days.mean() / days.std() * np.sqrt(252)
    gp, gl = p[p > 0].sum(), -p[p < 0].sum()
    roll = days.rolling(60).std().dropna()
    w1 = np.sort(p)[:max(1, len(p) // 100)].mean()
    # 停損率依 σ̂ 五分位（進場時的預測波動 / 價格）
    q = pd.qcut(sig_ref[ok], 5, labels=False, duplicates="drop")
    sr = pd.Series(stopped[ok]).groupby(q).mean().to_numpy()
    pr(f"  {label:6s} n={ok.sum():5d} 總 {p.sum():+9.0f} 每筆 {p.mean():+6.2f} PF {gp / gl:4.2f} 勝 {np.mean(p > 0):.0%} 停損率 {stopped[ok].mean():.0%} | "
       f"每筆 std {p.std():5.1f} 最差 {p.min():+6.0f} 最差1% {w1:+6.0f} | 日 Sharpe {sh:+5.2f} DD {dd:+7.0f} 最差日 {days.min():+6.0f} "
       f"報酬/DD {p.sum() / -dd if dd < 0 else np.nan:+5.2f} | 60日風險 CV {roll.std() / roll.mean():.2f} | 停損率 σ̂ 五分位 " + " ".join(f"{x:.2f}" for x in sr)
       + ("" if ratio is None else " | σ̂/ATR 五分位 " + " ".join(f"{x:.2f}" for x in pd.Series(stopped[ok]).groupby(pd.qcut(ratio[ok], 5, labels=False, duplicates="drop")).mean().to_numpy())))
    return days


def boot_sharpe_diff(d1, d2, nb=2000):
    """日損益配對（同一天）區塊 bootstrap（20 天區塊）：Sharpe(d1) − Sharpe(d2) 的 90% 區間。"""
    a = pd.concat([d1, d2], axis=1).fillna(0.0).to_numpy()
    n = len(a)
    blk = 20
    res = []
    for _ in range(nb):
        st = RNG.integers(0, n - blk, n // blk + 1)
        ix = (st[:, None] + np.arange(blk)).ravel()[:n]
        x = a[ix]
        s = x.mean(0) / x.std(0)
        res.append((s[0] - s[1]) * np.sqrt(252))
    res = np.array(res)
    return np.percentile(res, [5, 50, 95]), np.mean(res > 0)


def run_set(name, bars, T, variants, pr, years=True):
    """T 需要欄位：time, entry_px, side, start, exit_i, spread, D0（可 NaN）, pos。"""
    sig = bars.sig.to_numpy()[T.pos - 1]
    atr = bars.atr.to_numpy()[T.pos - 1]
    keep = np.isfinite(sig) & np.isfinite(atr) & (T.exit_i.to_numpy() >= T.start.to_numpy())
    T = T[keep].reset_index(drop=True)
    sig, atr = sig[keep], atr[keep]
    n0 = int(len(T) * 0.4)
    D0 = T.D0.to_numpy()
    ref = D0 if np.isfinite(D0).all() else np.full(len(T), np.nan)
    if np.isfinite(ref).all():
        kA = np.median(ref[:n0] / atr[:n0])
        kS = np.median(ref[:n0] / sig[:n0])
    else:
        kS = 1.5
        kA = np.median(kS * sig[:n0] / atr[:n0])
    fix = np.median(kS * sig[:n0])
    pr(f"\n=== {name}：{len(T)} 筆，{T.time.min()} ~ {T.time.max()}；k_ATR = {kA:.2f}，k_σ̂ = {kS:.2f}"
       + (f"，中位原停損 {np.median(ref):.2f} 美元" if np.isfinite(ref).all() else f"，固定停損 {fix:.2f} 美元") + " ===")
    Ds = {"ORIG": (ref, B / ref), "ATR": (kA * atr, B / (kA * atr)), "SIG": (kS * sig, B / (kS * sig)),
          "VT": (ref, B / (kS * sig)), "FIX": (np.full(len(T), fix), np.full(len(T), B / fix))}
    sref = sig / T.entry_px.to_numpy()
    daily = {}
    res = {}
    for v in variants:
        D, size = Ds[v]
        pnl, st = simulate(bars, T, D, size)
        daily[v] = stats(pnl, st, T.time.to_numpy(), sref, v, pr, ratio=sig / atr)
        res[v] = pnl
    if years:
        yr = pd.DatetimeIndex(T.time).year
        pr("  逐年總損益：" + "  ".join(f"{y}: " + " / ".join(f"{v} {np.nansum(res[v][yr == y]):+.0f}" for v in variants)
                                  for y in sorted(set(yr))))
    pr("  停損寬度掃描（相對校準倍數 m）：ATR vs SIG 的 日 Sharpe / DD / 停損率")
    for m in (0.5, 0.75, 1.0, 1.5, 2.0):
        line = f"    m={m:4.2f}"
        for v, base_d in (("ATR", kA * atr), ("SIG", kS * sig)):
            D = m * base_d
            pnl, st = simulate(bars, T, D, B / D)
            ok = np.isfinite(pnl)
            d = pd.Series(pnl[ok], index=pd.DatetimeIndex(T.time.to_numpy()[ok]).normalize()).groupby(level=0).sum()
            d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="B"), fill_value=0.0)
            eq = d.cumsum()
            line += f" | {v} Sh {d.mean() / d.std() * np.sqrt(252):+5.2f} DD {(eq - eq.cummax()).min():+7.0f} 停損 {st[ok].mean():.0%}"
        pr(line)
    base = [v for v in variants if v != "SIG"]
    for v in base:
        ci, p = boot_sharpe_diff(daily["SIG"], daily[v])
        pr(f"  Sharpe(SIG) − Sharpe({v})：中位 {ci[1]:+.2f}，90% 區間 [{ci[0]:+.2f}, {ci[2]:+.2f}]，>0 的比例 {p:.0%}")
    return res, T


def main():
    out = open(OUT, "w")

    def pr(*a):
        print(*a, file=out, flush=True)
    pr("第五十六部分（應用 A）：σ̂ 停損 / 部位 vs 原停損、ATR、固定停損；每筆風險 B = 100 美元，進場與出場時間不變")
    pr("欄位：每筆 std = 每筆損益標準差（越接近一致越好）；60日風險 CV = 滾動 60 天日損益標準差的變異係數（越小 = 風險越穩定）；")
    pr("      停損率 σ̂ 五分位 = 依進場時預測波動分五組的停損比例（越平 = 停損距離與行情匹配）")

    m5, _ = har_rel("data/XAUUSD_M5_2025_2026.csv", 48)          # 4 小時
    m15, _ = har_rel("data/XAUUSD_M15_2023_2026.csv", 16)        # 4 小時

    # 1. 版本 A
    a = pd.read_csv("part53_trades_2025_2026.csv", parse_dates=["entry_time", "exit_time"])
    pos, start = locate(m5.index, a.entry_time, on_open=False)
    xi = m5.index.searchsorted(a.exit_time, side="right") - 1
    T = pd.DataFrame(dict(time=a.entry_time, entry_px=a.lvl, side=a.side, start=start, exit_i=xi,
                          spread=a.spread, D0=a.risk, pos=pos))
    run_set("版本 A（IB 突破，M5）", m5, T, ["ORIG", "ATR", "SIG", "VT"], pr)

    # 2. OU 趨勢
    o = pd.read_csv("part49_trades.csv", parse_dates=["entry", "exit"])
    o = o[o.cfg.isin(["15min|100h", "15min|70h", "15min|130h", "15min|200h", "5min|100h", "5min|70h", "5min|130h", "5min|200h"])]
    for tf, bars in (("15min", m15), ("5min", m5)):
        s = o[o.tf == tf].sort_values("entry")
        pos, start = locate(bars.index, s.entry, on_open=True)
        xi = bars.index.searchsorted(s.exit, side="left") - 1
        side = np.where(s.side == "多", 1, -1)
        T = pd.DataFrame(dict(time=s.entry.to_numpy(), entry_px=s.entry_px.to_numpy(), side=side, start=pos, exit_i=xi,
                              spread=s.spread_usd.to_numpy(), D0=(s.entry_px - s.stop_px).abs().to_numpy(), pos=pos))
        run_set(f"OU 趨勢（{tf}，8 組裡的 {tf} 4 組）", bars, T, ["ORIG", "ATR", "SIG", "VT"], pr)

    # 3. 隨機進場（虛無對照）
    idx = m15.index
    days = pd.Series(idx.normalize()).unique()
    rows = []
    for d in days:
        cand = np.flatnonzero((idx.normalize() == d) & (idx.hour >= 2) & (idx.hour <= 19))
        if len(cand) < 10:
            continue
        for i in RNG.choice(cand, 3, replace=False):
            if i + 17 >= len(idx) or m15.gapid.iloc[i + 17] != m15.gapid.iloc[i]:
                continue
            rows.append((idx[i + 1], m15.open.iloc[i + 1], RNG.choice([-1, 1]), i + 1, i + 16, m15.spread.iloc[i + 1] * 0.01, np.nan, i + 1))
    T = pd.DataFrame(rows, columns=["time", "entry_px", "side", "start", "exit_i", "spread", "D0", "pos"]).sort_values("time").reset_index(drop=True)
    run_set("隨機進場（M15，持有 4 小時；虛無對照）", m15, T, ["FIX", "ATR", "SIG"], pr)
    out.close()
    print(open(OUT).read())


if __name__ == "__main__":
    main()
