"""第二十七部分：M15 前一根高/低點突破的逐筆交易明細（以 1 盎司計的美元損益）。

規則（同 part27_wide_brackets.py）：M15 K 棒開盤在前一根高點下方，期間內第一次碰到前高 → 在前高做多；前低對稱做空。
停損 = 點位反側 s × ATR15（ATR15 = 前 20 根 M15 波幅平均），停利 = t × 停損；沒碰到就當天最後一根收盤出場。
同一根 M1 的先後順序用 K 棒顏色推估。點差以進場那根 M1 的點差扣一次。
損益單位：美元 / 盎司（= 下 0.01 手的美元損益；1 標準手 = 100 盎司）。
輸出：part27_m15_trades_s{s}_t{t}.csv、results_part27_m15.txt
"""
import numpy as np
import pandas as pd
import part27_wide_brackets as W

H, Lo, C, SP, dayv, day_end = W.H, W.Lo, W.C, W.SP, W.dayv, W.day_end
IDX = W.B.d.index
CASES = [(2.0, 3.0), (3.0, 2.0)]


def sim_detail(i, entry, dir_, stop, tgt):
    """回傳 (出場索引, 出場價, 原因)。"""
    pts = W.path(i)
    for a in range(3):
        lo_, hi_ = min(pts[a], pts[a + 1]), max(pts[a], pts[a + 1])
        if lo_ <= entry <= hi_:
            r = W.walk(pts[a + 1:], dir_, stop, tgt)
            if r == "S":
                return i, stop, "停損"
            if r == "T":
                return i, tgt, "停利"
            break
    else:
        return None
    last = day_end[i]
    if last <= i:
        return i, C[i], "收盤"
    hh, ll = H[i + 1:last + 1], Lo[i + 1:last + 1]
    hs, ht = ((ll <= stop), (hh >= tgt)) if dir_ > 0 else ((hh >= stop), (ll <= tgt))
    js = int(np.argmax(hs)) if hs.any() else 10 ** 9
    jt = int(np.argmax(ht)) if ht.any() else 10 ** 9
    if js == jt == 10 ** 9:
        return last, C[last], "收盤"
    if js == jt:
        r = W.walk(W.path(i + 1 + js), dir_, stop, tgt)
        return (i + 1 + js, stop, "停損") if r == "S" else (i + 1 + js, tgt, "停利")
    return (i + 1 + js, stop, "停損") if js < jt else (i + 1 + jt, tgt, "停利")


def build(s, t, ev):
    rows = []
    for i, L, side, a in ev:
        risk = s * a
        stop, tgt = L - side * risk, L + side * t * risk
        r = sim_detail(i, L, side, stop, tgt)
        if r is None:
            continue
        j, px, why = r
        gross = (px - L) * side
        sp = SP[i]
        rows.append(dict(進場時間=IDX[i], 方向="多" if side > 0 else "空", 進場價=round(L, 2),
                         停損價=round(stop, 2), 停利價=round(tgt, 2), 出場時間=IDX[j], 出場價=round(px, 2),
                         出場原因=why, 持有分鐘=int((IDX[j] - IDX[i]).total_seconds() // 60),
                         停損距離=round(risk, 2), 毛利_每盎司=round(gross, 2), 點差=round(sp, 2),
                         淨利_每盎司=round(gross - sp, 2), R=round((gross - sp) / risk, 3),
                         前日在50日均線上=bool(W.B.UP50.iat[i]) if W.B.UP50.iat[i] == W.B.UP50.iat[i] else None))
    return pd.DataFrame(rows)


def summary(T, s, t):
    out = []
    n = T.淨利_每盎司
    d = T.groupby(T.進場時間.dt.date).淨利_每盎司.sum()
    eq = d.cumsum()
    win, loss = n[n > 0], n[n <= 0]
    out.append(f"\n=== M15 突破，停損 {s} ATR15、停利 {t} 倍（單位：美元/盎司，已扣點差）===")
    out.append(f"  筆數 {len(T)}（{T.進場時間.dt.date.nunique()} 個交易日，每天 {len(T) / T.進場時間.dt.date.nunique():.1f} 筆）")
    out.append(f"  停損距離中位數 {T.停損距離.median():.2f}，點差平均 {T.點差.mean():.2f}")
    out.append(f"  每筆淨利 {n.mean():+.3f}（毛利 {T.毛利_每盎司.mean():+.3f}），中位數 {n.median():+.2f}，總計 {n.sum():+,.0f}")
    out.append(f"  勝率 {(n > 0).mean():.1%}，平均賺 {win.mean():+.2f}，平均賠 {loss.mean():+.2f}，PF {win.sum() / -loss.sum():.3f}")
    out.append(f"  出場：停利 {(T.出場原因 == '停利').mean():.0%}、停損 {(T.出場原因 == '停損').mean():.0%}、收盤 {(T.出場原因 == '收盤').mean():.0%}；持有中位數 {T.持有分鐘.median():.0f} 分")
    out.append(f"  每日合計：平均 {d.mean():+.1f}，正報酬日 {(d > 0).mean():.0%}，最差日 {d.min():+.0f}，最好日 {d.max():+.0f}，最大回撤 {(eq.cummax() - eq).max():,.0f}")
    out.append(f"  多：{len(T[T.方向 == '多'])} 筆 每筆 {n[T.方向 == '多'].mean():+.3f}；空：{len(T[T.方向 == '空'])} 筆 每筆 {n[T.方向 == '空'].mean():+.3f}")
    y = T.groupby(T.進場時間.dt.year).agg(筆數=("淨利_每盎司", "size"), 每筆=("淨利_每盎司", "mean"), 總計=("淨利_每盎司", "sum"),
                                       停損距離=("停損距離", "median"), 點差=("點差", "mean"))
    out.append("  逐年：")
    for k, v in y.iterrows():
        out.append(f"    {k}: {int(v.筆數):6d} 筆，每筆 {v.每筆:+.3f}，總計 {v.總計:+9,.0f}，停損距離中位數 {v.停損距離:.2f}，點差 {v.點差:.2f}")
    m = T.groupby(T.進場時間.dt.to_period("M")).淨利_每盎司.sum()
    out.append(f"  逐月：正報酬 {(m > 0).sum()}/{len(m)} 個月；最差月 {m.min():+,.0f}（{m.idxmin()}），最好月 {m.max():+,.0f}（{m.idxmax()}）")
    out.append("  滑價敏感度（每筆再多扣 X 美元）：" + "，".join(f"{x}: {n.mean() - x:+.3f}" for x in (0.05, 0.1, 0.2, 0.3)))
    # 同時持倉
    ev = pd.concat([pd.Series(1, T.進場時間), pd.Series(-1, T.出場時間)]).sort_index()
    out.append(f"  同時持倉：最多 {ev.cumsum().max()} 筆")
    return out


if __name__ == "__main__":
    ev = W.events(15)
    lines = ["第二十七部分：M15 前一根 K 棒高/低點突破 —— 逐筆交易（XAUUSD 2023/6 ~ 2026/10）"]
    for s, t in CASES:
        T = build(s, t, ev)
        T.to_csv(f"part27_m15_trades_s{s:g}_t{t:g}.csv", index=False, encoding="utf-8-sig")
        lines += summary(T, s, t)
    print("\n".join(lines))
    open("results_part27_m15.txt", "w").write("\n".join(lines) + "\n")
