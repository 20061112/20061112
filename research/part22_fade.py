"""第二十二部分 B：ρ 高時反向做短爆發，有沒有反彈力？

規則：
  1. 事件流（step_k / bar_k，見 hawkes.py），在每根 M1 收盤更新遞迴 Hawkes 的 ρ（online EM）。
  2. 爆發：過去 W 分鐘的事件淨值 net = Σ方向，z = net / √(事件數)；z 第一次 ≥ z0 且 |net| ≥ 3。
  3. ρ 高 / 低：當下 ρ 落在「過去 20 個交易日每根 K 棒 ρ」的上 / 下 1/3（只用過去資料）。
  4. 反向進場：下一根開盤進，持有 H 分鐘後收盤出，扣訊號棒點差。同一天內距離上一次爆發至少 60 分鐘。
報酬單位 = 前一天 M1 平均高低差（「M1 ATR」，2026 年約 2~4 美元）。
對照：同一批爆發不分 ρ（全部）、只看 ρ 低、以及最簡單的「事件反向比例」取代 Hawkes ρ。
切分：2023/6~2024/12 樣本內（選參數），2025/1~2026/10 樣本外。
"""
import itertools
import numpy as np
import pandas as pd
from load import load
from hawkes import trading_clock, day_scale, make_events, online_em, online_reversal, to_bars

KINDS = ["step_1", "step_2", "step_4", "bar_1", "bar_2"]
WS = [15, 30, 60]
Z0 = [2.0, 3.0]
BETAS = [0.05, 0.1, 0.3]
HLS = [120.0, 480.0, 1440.0]          # 遺忘半衰期（交易分鐘）：2 小時 / 8 小時 / 1 天
HS = [15, 30, 60]
COOL = 60
LOOKBACK = 20
OOS = pd.Timestamp("2025-01-01")


def rolling_within_day(x, day_id, W):
    cs = np.cumsum(x)
    lag = np.r_[np.zeros(W), cs[:-W]]
    ok = np.r_[np.zeros(W, bool), day_id[W:] == day_id[:-W]]
    return np.where(ok, cs - lag, np.nan)


def trailing_terciles(v, day_id, nd, step=5):
    """每一天的 1/3、2/3 門檻，只用前 LOOKBACK 天的 K 棒值。"""
    lo = np.full(nd, np.nan); hi = np.full(nd, np.nan)
    starts = np.searchsorted(day_id, np.arange(nd + 1))
    for d in range(LOOKBACK, nd):
        w = v[starts[d - LOOKBACK]:starts[d]:step]
        w = w[np.isfinite(w)]
        if len(w) > 100:
            lo[d], hi[d] = np.quantile(w, [1 / 3, 2 / 3])
    return lo[day_id], hi[day_id]


def signals(z, net, day_id, minute, z0, nbar, maxH):
    """爆發第一次成立的 K 棒（含冷卻），回傳 index 與方向。"""
    burst = (np.abs(z) >= z0) & (np.abs(net) >= 3)
    prev = np.r_[False, burst[:-1]] & np.r_[False, day_id[1:] == day_id[:-1]]
    cand = np.nonzero(burst & ~prev & (minute + maxH + 1 < 1380))[0]
    out = []; last = -10 ** 9; lastd = -1
    for i in cand:
        if i + maxH + 1 >= nbar or day_id[i + maxH + 1] != day_id[i]:
            continue
        if day_id[i] == lastd and i - last < COOL:
            continue
        out.append(i); last = i; lastd = day_id[i]
    out = np.array(out, dtype=np.int64)
    return out, -np.sign(net[out])                  # 反向


def cl_t(r, d):
    """以交易日為群集的 t 值。"""
    if len(r) < 5:
        return np.nan
    e = r - r.mean()
    s = pd.Series(e).groupby(d).sum().values
    se = np.sqrt((s ** 2).sum()) / len(r)
    return r.mean() / se if se > 0 else np.nan


def main():
    m = load("data/XAUUSD_M1_2023_2026.csv")
    day_id, minute, t0, days = trading_clock(m)
    nd = len(days); nbar = len(m)
    sc = day_scale(m, day_id)
    scb = sc[day_id]
    o = m.open.values; c = m.close.values; sp = m.spread.values * 0.01
    is_oos = (days[day_id] >= OOS)
    valid_day = day_id >= LOOKBACK
    rows = []
    sig_store = {}
    for kind in KINDS:
        T, S, B = make_events(m, day_id, t0, sc, kind)
        T = T.astype(np.float64); ed = day_id[B]
        net_bar = np.bincount(B, S, nbar); cnt_bar = np.bincount(B, None, nbar)
        # 指標：Hawkes ρ（β × 半衰期）與反向比例（半衰期）
        inds = {}
        for beta, hl in itertools.product(BETAS, HLS):
            rho = online_em(T, S, ed, beta, hl)[0]
            inds[f"hawkes b={beta} hl={int(hl)}"] = to_bars(np.log(rho), B, nbar)
        for hl in HLS:
            inds[f"rev hl={int(hl)}"] = to_bars(online_reversal(T, S, ed, hl), B, nbar)
        terc = {k: trailing_terciles(v, day_id, nd) for k, v in inds.items()}
        for W, z0 in itertools.product(WS, Z0):
            net = rolling_within_day(net_bar, day_id, W)
            cnt = rolling_within_day(cnt_bar, day_id, W)
            z = net / np.sqrt(np.maximum(cnt, 1))
            idx, dirn = signals(np.nan_to_num(z), np.nan_to_num(net), day_id, minute, z0, nbar, max(HS))
            idx_ok = valid_day[idx]
            idx, dirn = idx[idx_ok], dirn[idx_ok]
            entry = o[idx + 1]
            for H in HS:
                gross = dirn * (c[idx + H] - entry) / scb[idx]
                r = gross - sp[idx] / scb[idx]
                for name, v in inds.items():
                    lo, hi = terc[name]
                    x = v[idx]
                    g_hi = x >= hi[idx]; g_lo = x <= lo[idx]
                    for per, pm in (("IS", ~is_oos[idx]), ("OOS", is_oos[idx])):
                        d = day_id[idx]
                        a, h_, l_ = pm, pm & g_hi, pm & g_lo
                        rows.append(dict(kind=kind, W=W, z0=z0, H=H, ind=name, per=per,
                                         n=a.sum(), all=r[a].mean(), gross=gross[a].mean(), t_all=cl_t(r[a], d[a]),
                                         n_hi=h_.sum(), hi=r[h_].mean(), t_hi=cl_t(r[h_], d[h_]),
                                         n_lo=l_.sum(), lo=r[l_].mean(), t_lo=cl_t(r[l_], d[l_]),
                                         diff=r[h_].mean() - r[l_].mean(),
                                         t_diff=(r[h_].mean() - r[l_].mean()) /
                                         np.sqrt((r[h_].std() ** 2 / max(h_.sum(), 1)) + (r[l_].std() ** 2 / max(l_.sum(), 1)))))
                sig_store[(kind, W, z0, H)] = (idx, r, gross, {k: (inds[k][idx], terc[k][0][idx], terc[k][1][idx]) for k in inds})
        print(kind, "done", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv("part22_fade_grid.csv", index=False)
    pd.to_pickle(sig_store, "part22_signals.pkl")
    return D


if __name__ == "__main__":
    main()
