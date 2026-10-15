"""第二十二部分：CCT 矩形設定（影片策略重現）— H4 定方向、M15 找弱勢、M1 矩形進場。

規則（空單；多單鏡像）：
  步驟 1（H4 方向 K 棒）
    方向棒 D：高點 > 前一根高點（先掃前高），收盤 < 前一根低點（吞沒、收破前低）
    之後在「下一根 H4」（K 根）內找回檔做空；以下一根的開盤價 O 為參考線
    第一目標 = D 的低點（結構目標）；價格突破 D 的高點 → 方向失效
  步驟 2（M15 弱勢）
    參考高點 = 之前 40 根內的 M15 擺動高點（左右各 2 根的 fractal），之後沒有收盤站上
    弱勢棒 W：掃過參考高點，且收盤 < 被掃到的最上面那個高點（未能收在其上）
    另外要求 W 的高點 > O（回檔到開盤價上方 = 溢價區），且 < D 的高點
  步驟 3（M1 進場）
    矩形 = W 的上影線：[max(開, 收), 高點]
    W 收盤後，M1 收盤跌破矩形下緣 → 以該根收盤進場；進場前 M1 先突破矩形上緣 → 取消
    停損 = 矩形上緣；停利 = D 低點（結構）或固定 nR
  成本：每筆扣進場當下點差（MT5 K 棒為 bid，spread 單位 0.01 美元）
  同一根 M1 同時碰停損與停利 → 算停損；超過 hold 分鐘以收盤平倉
  一根方向棒最多一筆交易（被取消的矩形可換下一根弱勢棒）
"""
import sys
import numpy as np
import pandas as pd
from load import load

M1_PATH = "data/XAUUSD_M1_2023_2026.csv"


def resample(m1, rule):
    return m1.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "spread": "max"}).dropna()


def fractals(h, l, k=2):
    n = len(h)
    sh = np.zeros(n, bool); sl = np.zeros(n, bool)
    for i in range(k, n - k):
        win_h = h[i - k:i + k + 1]; win_l = l[i - k:i + k + 1]
        sh[i] = h[i] == win_h.max() and (win_h == h[i]).sum() == 1
        sl[i] = l[i] == win_l.min() and (win_l == l[i]).sum() == 1
    return sh, sl


class Data:
    def __init__(self, path=M1_PATH):
        m1 = load(path)
        self.m1 = m1
        self.t1 = m1.index.values
        self.o1, self.h1, self.l1, self.c1 = (m1[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        self.sp1 = m1.spread.to_numpy(float) * 0.01
        self.m15 = resample(m1, "15min")
        self.h4 = resample(m1, "4h")
        m = self.m15
        self.t15 = m.index.values
        self.o15, self.h15, self.l15, self.c15 = (m[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        self.sh15, self.sl15 = fractals(self.h15, self.l15)
        pc = np.r_[self.c15[0], self.c15[:-1]]
        tr = np.maximum(self.h15 - self.l15, np.maximum(abs(self.h15 - pc), abs(self.l15 - pc)))
        self.atr15 = pd.Series(tr).rolling(14, min_periods=1).mean().to_numpy()
        h = self.h4
        self.t4 = h.index.values
        self.o4, self.h4h, self.l4, self.c4 = (h[k].to_numpy(float) for k in ("open", "high", "low", "close"))


def direction_candles(d: Data, loc=0):
    """回傳 (i, dir)：dir=-1 看跌方向棒、+1 看漲方向棒。loc>0：方向棒極值須為過去 loc 根 H4 的最高/最低（技巧 4：位置）。"""
    o, h, l, c = d.o4, d.h4h, d.l4, d.c4
    out = []
    for i in range(max(1, loc), len(c) - 1):
        if h[i] > h[i - 1] and c[i] < l[i - 1] and (loc == 0 or h[i] >= h[i - loc:i + 1].max()):
            out.append((i, -1))
        elif l[i] < l[i - 1] and c[i] > h[i - 1] and (loc == 0 or l[i] <= l[i - loc:i + 1].min()):
            out.append((i, +1))
    return out


def simulate(d: Data, k0, side, entry, sl, tp, hold):
    """從 M1 第 k0 根之後開始模擬。回傳 (出場索引, 出場價, 原因)。"""
    end = min(k0 + 1 + hold, len(d.c1))
    hi, lo = d.h1[k0 + 1:end], d.l1[k0 + 1:end]
    if side < 0:
        hit_sl = hi >= sl; hit_tp = lo <= tp
    else:
        hit_sl = lo <= sl; hit_tp = hi >= tp
    a = np.argmax(hit_sl) if hit_sl.any() else len(hi)
    b = np.argmax(hit_tp) if hit_tp.any() else len(hi)
    if a == len(hi) and b == len(hi):
        return end - 1, d.c1[end - 1], "time"
    if a <= b:
        return k0 + 1 + a, sl, "sl"
    return k0 + 1 + b, tp, "tp"


def run(d: Data, K=1, loc=0, premium=True, tp_mode="struct", min_rr=1.0, wait_min=120, hold=48 * 60,
        need_bias=True, tol=0.0):
    """tp_mode: 'struct'（方向棒極值）或數字（固定 nR）。need_bias=False：舊版兩時框（不看 H4，任何弱勢都做，固定 R）。
    tol：弱勢棒收盤可超出被掃高點 tol×ATR15（影片的「幾乎形成弱勢也算」）。"""
    trades = []
    if need_bias:
        windows = []
        for i, dr in direction_candles(d, loc):
            if i + 1 + K > len(d.t4):
                continue
            ws = d.t4[i + 1]
            we = d.t4[i + 1 + K] if i + 1 + K < len(d.t4) else d.t4[-1] + np.timedelta64(4, "h")
            windows.append((ws, we, dr, d.o4[i + 1], d.h4h[i], d.l4[i], d.t4[i]))
    else:
        # 不看方向：整段時間以 H4 為單位切窗，兩個方向都找
        windows = [(d.t4[j], d.t4[j + 1], dr, np.nan, np.nan, np.nan, d.t4[j])
                   for j in range(len(d.t4) - 1) for dr in (-1, 1)]

    for ws, we, dr, O, Dh, Dl, Dt in windows:
        a = np.searchsorted(d.t15, ws); b = np.searchsorted(d.t15, we)
        if b - a < 2:
            continue
        side = dr
        for j in range(a, b):
            # 方向失效：已突破方向棒極值
            if need_bias and ((side < 0 and d.h15[j] > Dh) or (side > 0 and d.l15[j] < Dl)):
                break
            lo_p = max(j - 40, 2)
            if side < 0:
                # 未被收盤站上的擺動高點中，被 W 掃到的「最上面那個」；W 須收在它之下（容忍 tol×ATR）
                swept = [d.h15[p] for p in range(lo_p, j - 2)
                         if d.sh15[p] and d.h15[j] > d.h15[p] and not (d.c15[p + 1:j] > d.h15[p]).any()]
                if not swept or not d.c15[j] < max(swept) + tol * d.atr15[j]:
                    continue
                if premium and need_bias and not d.h15[j] > O:
                    continue
                top, bot = d.h15[j], max(d.o15[j], d.c15[j])
            else:
                swept = [d.l15[p] for p in range(lo_p, j - 2)
                         if d.sl15[p] and d.l15[j] < d.l15[p] and not (d.c15[p + 1:j] < d.l15[p]).any()]
                if not swept or not d.c15[j] > min(swept) - tol * d.atr15[j]:
                    continue
                if premium and need_bias and not d.l15[j] < O:
                    continue
                top, bot = d.l15[j], min(d.o15[j], d.c15[j])   # top = 停損端
            if abs(top - bot) <= 0:
                continue
            # 步驟 3：M1 進場
            k = np.searchsorted(d.t1, d.t15[j] + np.timedelta64(15, "m"))
            kend = min(k + wait_min, len(d.c1) - 1)
            entered = None
            for kk in range(k, kend):
                if (side < 0 and d.h1[kk] > top) or (side > 0 and d.l1[kk] < top):
                    break                                    # 矩形被突破 → 取消
                if (side < 0 and d.c1[kk] < bot) or (side > 0 and d.c1[kk] > bot):
                    entered = kk
                    break
            if entered is None:
                continue
            entry = d.c1[entered]
            risk = abs(top - entry)
            if risk <= 0:
                continue
            if tp_mode == "struct":
                tp = Dl if side < 0 else Dh
                rr = (entry - tp) / risk if side < 0 else (tp - entry) / risk
                if rr < min_rr:
                    break                                    # 目標太近，這根方向棒放棄
            else:
                rr = float(tp_mode)
                tp = entry - rr * risk if side < 0 else entry + rr * risk
            ke, px, why = simulate(d, entered, side, entry, top, tp, hold)
            cost = d.sp1[entered]
            pnl = (px - entry) * side - cost
            trades.append(dict(dir_time=pd.Timestamp(Dt), side=side, weak_time=pd.Timestamp(d.t15[j]),
                               entry_time=pd.Timestamp(d.t1[entered]), entry=entry, sl=top, tp=tp,
                               risk=risk, rr_target=rr, exit_time=pd.Timestamp(d.t1[ke]), exit=px,
                               reason=why, cost=cost, pnl=pnl, pnl_R=pnl / risk))
            break                                            # 一個窗最多一筆
    T = pd.DataFrame(trades)
    if need_bias is False and len(T):
        # 舊版兩時框：同一時間只持有一筆
        T = T.sort_values("entry_time")
        keep, last_exit = [], pd.Timestamp(0)
        for r in T.itertuples():
            if r.entry_time >= last_exit:
                keep.append(r.Index); last_exit = r.exit_time
        T = T.loc[keep]
    return T.reset_index(drop=True)


def summary(T):
    if len(T) == 0:
        return dict(n=0)
    r = T.pnl_R
    w = r[r > 0].sum(); l = -r[r < 0].sum()
    return dict(n=len(T), win=round((r > 0).mean(), 3), avgR=round(r.mean(), 3),
                t=round(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))), 2) if len(r) > 1 else np.nan,
                PF=round(w / l, 2) if l > 0 else np.inf, sumR=round(r.sum(), 1),
                tp=round((T.reason == "tp").mean(), 3), avg_rr=round(T.rr_target.mean(), 2),
                cost_R=round((T.cost / T.risk).median(), 3))


def by_year(T):
    if len(T) == 0:
        return ""
    g = T.groupby(T.entry_time.dt.year).pnl_R
    return "  ".join(f"{y}: n={len(s)} avg={s.mean():+.2f}R sum={s.sum():+.1f}" for y, s in g)


if __name__ == "__main__":
    d = Data()
    print(f"M1 {d.m1.index[0]} ~ {d.m1.index[-1]}  M15 {len(d.t15)}  H4 {len(d.t4)}")
    dc = direction_candles(d)
    print(f"H4 方向棒：{len(dc)} 根（看跌 {sum(1 for _, x in dc if x < 0)}、看漲 {sum(1 for _, x in dc if x > 0)}）\n")

    variants = [
        ("基準：K=1、溢價區、結構目標 RR≥1", dict()),
        ("基準但不要求溢價區", dict(premium=False)),
        ("K=2（允許第二根 H4 才回檔）", dict(K=2)),
        ("K=3", dict(K=3)),
        ("結構目標 RR≥2", dict(min_rr=2.0)),
        ("結構目標 RR≥3", dict(min_rr=3.0)),
        ("固定 1R", dict(tp_mode=1)),
        ("固定 2R", dict(tp_mode=2)),
        ("固定 3R", dict(tp_mode=3)),
        ("固定 4R（影片宣稱 1:4）", dict(tp_mode=4)),
        ("容忍 0.1 ATR（幾乎弱勢也算）", dict(tol=0.1)),
        ("容忍 0.1 ATR + K=2 + 固定 4R", dict(tol=0.1, K=2, tp_mode=4)),
        ("技巧4 位置：方向棒為 6 根 H4 極值", dict(loc=6)),
        ("技巧4 位置：方向棒為 20 根 H4 極值", dict(loc=20)),
        ("技巧4 位置 20 + 固定 4R", dict(loc=20, tp_mode=4)),
        ("舊版兩時框（不看 H4）固定 2R", dict(need_bias=False, tp_mode=2)),
        ("舊版兩時框（不看 H4）固定 4R", dict(need_bias=False, tp_mode=4)),
    ]
    res = {}
    for name, kw in variants:
        T = run(d, **kw)
        res[name] = T
        print(f"[{name}] {summary(T)}")
        print(f"    {by_year(T)}")
        if len(T):
            print(f"    多單 avg={T[T.side > 0].pnl_R.mean():+.3f}R n={(T.side > 0).sum()}  "
                  f"空單 avg={T[T.side < 0].pnl_R.mean():+.3f}R n={(T.side < 0).sum()}")
    base = res["基準：K=1、溢價區、結構目標 RR≥1"]
    base.to_csv("part22_trades_base.csv", index=False)
    res["固定 4R（影片宣稱 1:4）"].to_csv("part22_trades_4R.csv", index=False)

    # 影片中的例子：黃金 2026/6/10（影片時間比本資料伺服器時間早約 7 小時）
    print("\n影片例子（方向棒 = 2026/6/10 08:00 看跌 H4；弱勢棒 = 16:45 M15 高 4186.29、收 4179.65）：")
    for name, kw in [("嚴格（K=2、tol=0）", dict(K=2, tp_mode=4)), ("寬鬆（K=2、tol=0.1）", dict(K=2, tol=0.1, tp_mode=4))]:
        T = run(d, **kw)
        ex = T[(T.dir_time >= "2026-06-10") & (T.dir_time < "2026-06-11")]
        print(f"  [{name}]")
        print(ex[["dir_time", "side", "weak_time", "entry_time", "entry", "sl", "tp", "reason", "exit_time", "pnl_R"]]
              .to_string() if len(ex) else "    無交易")
