"""第二十三部分：同一套 Hawkes 量測，換成不同策略用法。

第二十二部分只測了「ρ 高時反向做爆發」。這裡把 Hawkes 的各種輸出拿來做別的策略：
  S1 順勢跟爆發（第二十二部分樣本內爆發偏延續），用 ρ / 分支比 n / 當下激發度 λ/μ 篩選
  S2 強度不平衡 x = (λ↑−λ↓)/(λ↑+λ↓)：Hawkes 對「下一個事件往哪走」的直接預測（對稱 + 非對稱模型）
  S3 慢速方向偏差 b = log(α↑↑/α↓↓)：上漲比較會自我強化，還是下跌比較會？
  S4 13:00 盤型切換：前半天 ρ 高 → 後半天反向前半天；ρ 低 → 順著前半天
  S5 當作既有策略的篩選器：LONBRK（倫敦開盤突破亞洲區間）進場時的 ρ / n
共同設定：2023/6~2024/12 樣本內、2025/1~2026/10 樣本外；都扣點差；報酬單位 = 前一天 M1 平均高低差（S4 為前半天區間、S5 為 R）。
"""
import itertools
import numpy as np
import pandas as pd
from load import load
from hawkes import (trading_clock, day_scale, make_events, online_em, online_em_asym, intensity_at_bars,
                    batch_em_days, to_bars)
from part22_fade import rolling_within_day, trailing_terciles, signals, cl_t, OOS, LOOKBACK

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 400)
HS = [15, 30, 60]


def line(r, d):
    return f"n={len(r):5d} 平均 {np.mean(r):+.3f} t={cl_t(r, d):+.2f}" if len(r) >= 5 else f"n={len(r):5d}"


def setup():
    m = load("data/XAUUSD_M1_2023_2026.csv")
    day_id, minute, t0, days = trading_clock(m)
    sc = day_scale(m, day_id)
    E = dict(m=m, day_id=day_id, minute=minute, t0=t0, days=days, nd=len(days), nbar=len(m), sc=sc)
    E["scb"] = sc[day_id]
    E["oos"] = days[day_id] >= OOS
    E["year"] = days.year.values[day_id]
    return E


def fwd(E, idx, H):
    o, c = E["m"].open.values, E["m"].close.values
    return (c[idx + H] - o[idx + 1]) / E["scb"][idx]


def cost(E, idx):
    return E["m"].spread.values[idx] * 0.01 / E["scb"][idx]


# ------------------------------------------------------------------ S1
def s1_follow(E, ev, rows):
    print("\n" + "=" * 100)
    print("S1. 順勢跟爆發（z ≥ 2），依 Hawkes 指標的過去 20 天 1/3 分組；數字 = 扣點差後平均（t）")
    print("=" * 100)
    day_id, minute, nbar = E["day_id"], E["minute"], E["nbar"]
    for kind in ("step_1", "step_2"):
        T, S, B, ed = ev[kind]
        net_bar = np.bincount(B, S, nbar); cnt_bar = np.bincount(B, None, nbar)
        inds = {}
        for beta, hl in itertools.product((0.05, 0.1), (480.0, 1440.0)):
            rho, ns, nc, mu = online_em(T, S, ed, beta, hl)
            inds[f"ρ b={beta} hl={int(hl)}"] = to_bars(np.log(rho), B, nbar)
            inds[f"n b={beta} hl={int(hl)}"] = to_bars(ns + nc, B, nbar)
        lu, ld, st = intensity_at_bars(T, online_em_asym(T, S, ed, 0.1, 480.0), B, E["t0"] + 1.0, 0.1)
        inds["λ/μ 當下激發度"] = np.log((lu + ld) / (st[:, 2] + st[:, 3]))
        terc = {k: trailing_terciles(v, day_id, E["nd"]) for k, v in inds.items()}
        for W in (15, 30, 60):
            net = rolling_within_day(net_bar, day_id, W); cnt = rolling_within_day(cnt_bar, day_id, W)
            z = np.nan_to_num(net / np.sqrt(np.maximum(cnt, 1)))
            idx, dirn = signals(z, np.nan_to_num(net), day_id, minute, 2.0, nbar, max(HS))
            idx = idx[day_id[idx] >= LOOKBACK]; follow = np.sign(np.nan_to_num(net))[idx]
            for H in HS:
                r = follow * fwd(E, idx, H) - cost(E, idx)
                d = day_id[idx]; oos = E["oos"][idx]
                for name, v in inds.items():
                    lo, hi = terc[name]
                    gh, gl = v[idx] >= hi[idx], v[idx] <= lo[idx]
                    row = dict(strategy="S1 順勢跟爆發", kind=kind, W=W, H=H, ind=name)
                    for per, pm in (("IS", ~oos), ("OOS", oos)):
                        row[f"all_{per}"] = r[pm].mean(); row[f"t_all_{per}"] = cl_t(r[pm], d[pm])
                        row[f"hi_{per}"] = r[pm & gh].mean(); row[f"lo_{per}"] = r[pm & gl].mean()
                        row[f"t_hi_{per}"] = cl_t(r[pm & gh], d[pm & gh]); row[f"t_lo_{per}"] = cl_t(r[pm & gl], d[pm & gl])
                        row[f"n_{per}"] = pm.sum()
                    rows.append(row)
    D = pd.DataFrame([r for r in rows if r["strategy"] == "S1 順勢跟爆發"])
    print("全部爆發順勢做（不分組，各 W/H）：")
    print(D.drop_duplicates(["kind", "W", "H"])[["kind", "W", "H", "n_IS", "all_IS", "t_all_IS", "n_OOS", "all_OOS", "t_all_OOS"]]
          .round(3).to_string(index=False))
    print("\n各指標的高 1/3 / 低 1/3（平均跨事件 × W × H）：")
    print(D.groupby("ind")[["hi_IS", "lo_IS", "hi_OOS", "lo_OOS"]].mean().round(3).to_string())
    best = D.assign(sc=np.maximum(D.t_hi_IS, D.t_lo_IS)).sort_values("sc", ascending=False).head(8)
    print("\n樣本內最好的 8 組（取高或低組中 t 較大者）→ 樣本外：")
    print(best[["kind", "W", "H", "ind", "hi_IS", "t_hi_IS", "lo_IS", "t_lo_IS", "hi_OOS", "t_hi_OOS", "lo_OOS", "t_lo_OOS"]]
          .round(3).to_string(index=False))


# ------------------------------------------------------------------ S2
def s2_imbalance(E, ev, rows):
    print("\n" + "=" * 100)
    print("S2. 強度不平衡 x = (λ↑−λ↓)/(λ↑+λ↓) → 往 x 的方向做（x < 0 代表模型預期反轉）")
    print("    IC = 每 5 分鐘抽樣，x 與未來 H 分鐘報酬的 Spearman；交易 = |x| 超過過去 20 天 90 百分位才進，持有 H")
    print("=" * 100)
    day_id, minute, nbar = E["day_id"], E["minute"], E["nbar"]
    samp = (minute % 5 == 4) & (minute < 1380 - 61) & (day_id >= LOOKBACK)
    samp &= np.r_[day_id[61:] == day_id[:-61], np.zeros(61, bool)]
    out = []
    for kind in ("step_1", "step_2", "step_4"):
        T, S, B, ed = ev[kind]
        for model, beta, hl in [("非對稱", 0.1, 480.0), ("非對稱", 0.05, 1440.0), ("非對稱", 0.3, 480.0)]:
            st = online_em_asym(T, S, ed, beta, hl)
            lu, ld, _ = intensity_at_bars(T, st, B, E["t0"] + 1.0, beta)
            x = (lu - ld) / (lu + ld)
            ax = np.abs(x)
            # 90 百分位門檻（只用過去 20 天）
            starts = np.searchsorted(day_id, np.arange(E["nd"] + 1))
            q = np.full(E["nd"], np.nan)
            for d in range(LOOKBACK, E["nd"]):
                w = ax[starts[d - LOOKBACK]:starts[d]:3]; w = w[np.isfinite(w)]
                if len(w) > 100:
                    q[d] = np.quantile(w, 0.9)
            thr = q[day_id]
            for H in HS:
                idx = np.nonzero(samp)[0]
                f = fwd(E, idx, H)
                ok = np.isfinite(x[idx])
                ic = {}
                for per, pm in (("IS", ~E["oos"][idx]), ("OOS", E["oos"][idx])):
                    s_ = ok & pm
                    ic[per] = pd.Series(x[idx][s_]).rank().corr(pd.Series(f[s_]).rank())
                # 交易：|x| 第一次超過門檻，冷卻 H
                trig = (ax >= thr) & np.isfinite(thr)
                prev = np.r_[False, trig[:-1]] & np.r_[False, day_id[1:] == day_id[:-1]]
                cand = np.nonzero(trig & ~prev & (minute + H + 1 < 1380))[0]
                sel = []; last = -10 ** 9
                for i in cand:
                    if i + H + 1 < nbar and day_id[i + H + 1] == day_id[i] and i - last >= H:
                        sel.append(i); last = i
                sel = np.array(sel, dtype=np.int64)
                g = np.sign(x[sel]) * fwd(E, sel, H)
                c = cost(E, sel)
                row = dict(strategy="S2 強度不平衡", kind=kind, model=model, beta=beta, hl=int(hl), H=H,
                           IC_IS=ic["IS"], IC_OOS=ic["OOS"])
                for per, pm in (("IS", ~E["oos"][sel]), ("OOS", E["oos"][sel])):
                    row[f"n_{per}"] = pm.sum()
                    row[f"gross_{per}"] = g[pm].mean()
                    row[f"net_{per}"] = (g - c)[pm].mean(); row[f"t_{per}"] = cl_t((g - c)[pm], day_id[sel][pm])
                    row[f"rev_net_{per}"] = (-g - c)[pm].mean()
                out.append(row); rows.append(row)
    D = pd.DataFrame(out)
    print(D[["kind", "beta", "hl", "H", "IC_IS", "IC_OOS", "n_IS", "gross_IS", "net_IS", "t_IS", "rev_net_IS",
             "n_OOS", "gross_OOS", "net_OOS", "t_OOS", "rev_net_OOS"]].round(3).to_string(index=False))
    return D


# ------------------------------------------------------------------ S3
def s3_bias(E, ev, rows):
    print("\n" + "=" * 100)
    print("S3. 慢速方向偏差 b = log(α↑↑/α↓↓)（非對稱模型、長半衰期）：b > 0 = 上漲比較會自我強化 → 做多")
    print("    (a) IC：每 5 分鐘抽樣 b 與未來 60 / 240 分鐘報酬；(b) 每天 13:00 依 sign(b) 進場、22:00 出場")
    print("=" * 100)
    day_id, minute = E["day_id"], E["minute"]
    m = E["m"]; o, c = m.open.values, m.close.values
    at13 = np.nonzero(minute == 720)[0]; at22 = np.nonzero(minute == 1260)[0]
    d13 = pd.Series(at13, index=day_id[at13]); d22 = pd.Series(at22, index=day_id[at22])
    both = d13.index.intersection(d22.index); both = both[both >= LOOKBACK]
    out = []
    for kind in ("step_1", "step_2", "step_4"):
        T, S, B, ed = ev[kind]
        for beta, hl in [(0.1, 1440.0), (0.1, 4320.0), (0.05, 4320.0)]:
            st = online_em_asym(T, S, ed, beta, hl)
            _, _, stb = intensity_at_bars(T, st, B, E["t0"] + 1.0, beta)
            b = np.log(stb[:, 4] / stb[:, 7])
            row = dict(strategy="S3 方向偏差", kind=kind, beta=beta, hl=int(hl))
            for Hh in (60, 240):
                samp = np.nonzero((minute % 5 == 4) & (minute < 1380 - Hh - 1) & (day_id >= LOOKBACK))[0]
                samp = samp[samp + Hh + 1 < len(day_id)]
                samp = samp[day_id[samp + Hh] == day_id[samp]]
                f = (c[samp + Hh] - o[samp + 1])
                for per, pm in (("IS", ~E["oos"][samp]), ("OOS", E["oos"][samp])):
                    row[f"IC{Hh}_{per}"] = pd.Series(b[samp][pm]).rank().corr(pd.Series(f[pm]).rank())
            i13 = d13[both].values - 1; i22 = d22[both].values
            r = np.sign(b[i13]) * (c[i22] - o[i13 + 1]) / E["scb"][i13] - cost(E, i13)
            oo = E["oos"][i13]
            for per, pm in (("IS", ~oo), ("OOS", oo)):
                row[f"day_{per}"] = r[pm].mean(); row[f"t_day_{per}"] = r[pm].mean() / r[pm].std() * np.sqrt(pm.sum())
            row["frac_long"] = (b[i13] > 0).mean()
            out.append(row); rows.append(row)
    D = pd.DataFrame(out)
    print(D.drop(columns="strategy").round(3).to_string(index=False))


# ------------------------------------------------------------------ S4
def s4_switch(E, ev, rows):
    print("\n" + "=" * 100)
    print("S4. 13:00 盤型切換：前半天（01:00~13:00）batch ρ 高 1/3 → 後半天反向前半天；低 1/3 → 順勢；13:00 進、22:00 出")
    print("    數字 = 「順著前半天方向」的報酬 / 前半天區間（扣點差）；切換策略 = 低組順勢 + 高組反向")
    print("=" * 100)
    day_id, minute, nd = E["day_id"], E["minute"], E["nd"]
    m = E["m"]; o, c, h, l = m.open.values, m.close.values, m.high.values, m.low.values
    D = pd.DataFrame({"d": day_id, "mi": minute, "o": o, "c": c, "h": h, "l": l})
    A = D[D.mi < 720].groupby("d").agg(o=("o", "first"), c=("c", "last"), h=("h", "max"), l=("l", "min"))
    at13 = pd.Series(np.nonzero(minute == 720)[0], index=day_id[minute == 720])
    at22 = pd.Series(np.nonzero(minute == 1260)[0], index=day_id[minute == 1260])
    days_ok = A.index.intersection(at13.index).intersection(at22.index)
    days_ok = days_ok[days_ok >= 60]
    A = A.loc[days_ok]
    i13 = at13[days_ok].values; i22 = at22[days_ok].values
    dirA = np.sign(A.c - A.o).values
    rngA = (A.h - A.l).values
    cont = (dirA * (c[i22] - o[i13]) - m.spread.values[i13] * 0.01) / rngA
    rev = (-dirA * (c[i22] - o[i13]) - m.spread.values[i13] * 0.01) / rngA
    oos = E["days"][days_ok] >= OOS
    out = []
    for kind, beta in itertools.product(("step_1", "step_2"), (0.05, 0.1)):
        T, S, B, ed = ev[kind]
        R = batch_em_days(T, S, ed, nd, 0.0, 720.0, beta, 80)
        lr = np.log(R[:, 2] / R[:, 1])
        s = pd.Series(lr)
        lo = s.shift(1).rolling(60, min_periods=40).quantile(1 / 3).values
        hi = s.shift(1).rolling(60, min_periods=40).quantile(2 / 3).values
        v = lr[days_ok]; gl = v <= lo[days_ok]; gh = v >= hi[days_ok]
        row = dict(strategy="S4 盤型切換", kind=kind, beta=beta)
        for per, pm in (("IS", ~oos), ("OOS", oos)):
            row[f"cont_all_{per}"] = cont[pm].mean()
            row[f"cont_lo_{per}"] = cont[pm & gl].mean(); row[f"cont_hi_{per}"] = cont[pm & gh].mean()
            sw = np.r_[cont[pm & gl], rev[pm & gh]]
            row[f"switch_{per}"] = sw.mean(); row[f"t_switch_{per}"] = sw.mean() / sw.std() * np.sqrt(len(sw))
            row[f"n_{per}"] = len(sw)
        out.append(row); rows.append(row)
    print(pd.DataFrame(out).drop(columns="strategy").round(3).to_string(index=False))


# ------------------------------------------------------------------ S5
def s5_filter(E, ev, rows):
    print("\n" + "=" * 100)
    print("S5. 當篩選器：LONBRK 樣本外交易（oos_lonbrk_trades.csv，取 2023/6 以後）依進場前一分鐘的指標 1/3 分組，單位 R")
    print("=" * 100)
    L = pd.read_csv("oos_lonbrk_trades.csv", parse_dates=["entry_time"])
    L = L[L.entry_time >= "2023-06-01"]
    pos = E["m"].index.get_indexer(L.entry_time - pd.Timedelta(minutes=1))
    L = L[pos >= 0]; pos = pos[pos >= 0]
    oos = (L.entry_time >= OOS).values
    out = []
    for kind in ("step_1", "step_2"):
        T, S, B, ed = ev[kind]
        for beta, hl in [(0.05, 480.0), (0.1, 1440.0)]:
            rho, ns, nc, _ = online_em(T, S, ed, beta, hl)
            for nm, val in (("ρ", np.log(rho)), ("n", ns + nc)):
                v = to_bars(val, B, E["nbar"])
                lo, hi = trailing_terciles(v, E["day_id"], E["nd"])
                x = v[pos]; gh = x >= hi[pos]; gl = x <= lo[pos]
                row = dict(strategy="S5 LONBRK 篩選", kind=kind, beta=beta, hl=int(hl), ind=nm)
                for per, pm in (("IS", ~oos), ("OOS", oos)):
                    r = L.pnl_R.values
                    row[f"all_{per}"] = r[pm].mean(); row[f"hi_{per}"] = r[pm & gh].mean(); row[f"lo_{per}"] = r[pm & gl].mean()
                    row[f"n_hi_{per}"] = (pm & gh).sum(); row[f"n_lo_{per}"] = (pm & gl).sum()
                out.append(row); rows.append(row)
    print(pd.DataFrame(out).drop(columns="strategy").round(3).to_string(index=False))


def main():
    E = setup()
    ev = {}
    for kind in ("step_1", "step_2", "step_4"):
        T, S, B = make_events(E["m"], E["day_id"], E["t0"], E["sc"], kind)
        ev[kind] = (T.astype(np.float64), S, B, E["day_id"][B])
    rows = []
    s1_follow(E, ev, rows)
    s2_imbalance(E, ev, rows)
    s3_bias(E, ev, rows)
    s4_switch(E, ev, rows)
    s5_filter(E, ev, rows)
    pd.DataFrame(rows).to_csv("part23_grid.csv", index=False)


if __name__ == "__main__":
    main()
