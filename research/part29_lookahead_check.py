"""第二十九部分 (3)：未來函數檢查 → results_part29_lookahead.txt

1. 截斷測試：把 M1 資料在多個時間點 T 截斷，重新計算訊號、方向、停損距離、H1/H4 ER；
   截斷前已收完的 K 棒，結果必須與用完整資料算的完全一樣（有偷看未來就會不同）。
2. 打亂未來測試：把 T 之後的價格換成隨機漫步，T 之前的訊號必須不變。
3. 加碼單同一根 K 棒的順序：原版在「觸發加碼那根」不檢查加碼單停損（偏樂觀）；
   保守版：觸發那根的低點（做多）若也碰到原進場價，就當作加碼單同一根停損出場。比較差異。
"""
import numpy as np
import pandas as pd
from part23_ma_cycle import load_m1
from part25_exits import prep, entries
from part25_ext import tf_er_done
from part29_optimize import LEGS
from part28_final import nights
import part27_sizing

EXIT = {"A": (1.5, 80), "B": (2, 20), "C": (1.5, 40)}


def signals(m1, leg):
    tf, L, M, S, a, cool = LEGS[leg]
    B = prep(m1, tf)
    B["h1"], B["h4"] = tf_er_done(m1, B, "1h"), tf_er_done(m1, B, "4h")
    t, side = entries(B, L, M, S, a, cool)
    step = B["idx"][1] - B["idx"][0]
    rows = []
    for k, s in zip(t, side):
        ext = B["l"][k - M + 1:k + 1].min() if s == 1 else B["h"][k - M + 1:k + 1].max()
        rows.append(dict(time=B["idx"][k], close_time=B["idx"][k] + step, side=int(s),
                         stop=round(ext - s * 0.1 * B["A"][k], 4), h1=round(float(B["h1"][k]), 6),
                         h4=round(float(B["h4"][k]), 6)))
    return pd.DataFrame(rows), B


def compare(full, part, T):
    """比較 close_time <= T 之前的訊號；回傳 (筆數, 不一致數)。最後一個 cool 期間內的訊號因冷卻期以外都應一致。"""
    f = full[full.close_time <= T].reset_index(drop=True)
    p = part[part.close_time <= T].reset_index(drop=True)
    # 截斷版最後幾根 K 棒在 entries() 裡會被 t < len-2 排除，所以只比到截斷前 1 小時
    lim = T - pd.Timedelta("1h")
    f, p = f[f.close_time <= lim], p[p.close_time <= lim]
    if len(f) != len(p):
        return len(f), abs(len(f) - len(p)) + 10 ** 6
    bad = (f[["time", "side", "stop", "h1", "h4"]].values != p[["time", "side", "stop", "h1", "h4"]].values).any(axis=1).sum()
    return len(f), int(bad)


def main():
    m1 = load_m1()
    out = []
    P = out.append
    rng = np.random.default_rng(7)
    cuts = [pd.Timestamp(x) for x in ("2026-02-17 13:37", "2026-03-25 08:05", "2026-05-14 19:52", "2026-06-30 02:11",
                                      "2026-08-06 15:44", "2026-09-18 10:23")]
    P("=== 1. 截斷測試（截斷前 1 小時以前的訊號是否與完整資料一致）===")
    full = {leg: signals(m1, leg)[0] for leg in EXIT}
    for T in cuts:
        line = [str(T)]
        for leg in EXIT:
            part, _ = signals(m1[m1.index < T], leg)
            n, bad = compare(full[leg], part, T)
            line.append(f"{leg}: {n} 筆 / 不一致 {bad}")
        P("  ".join(line))

    P("\n=== 2. 打亂未來測試（T 之後的價格換成隨機漫步）===")
    for T in cuts[1::2]:
        fake = m1.copy()
        fut = fake.index >= T
        steps = rng.normal(0, 0.5, fut.sum()).cumsum()
        base = fake.close[~fut].iloc[-1]
        for col in ("open", "high", "low", "close"):
            fake.loc[fut, col] = base + steps
        fake.loc[fut, "high"] += 0.3
        fake.loc[fut, "low"] -= 0.3
        line = [str(T)]
        for leg in EXIT:
            part, _ = signals(fake, leg)
            n, bad = compare(full[leg], part, T)
            line.append(f"{leg}: {n} 筆 / 不一致 {bad}")
        P("  ".join(line))

    P("\n=== 3. 加碼單同一根 K 棒的處理（整合版：A/B/C、週五平倉、上限 $100、掉期 0.7、+1R 且同向加 1）===")
    orig_sim = part27_sizing.sim

    def conservative_sim(B, k, s, M, L, **kw):
        r = orig_sim(B, k, s, M, L, **kw)
        if not r or not r["added"]:
            return r
        # 找出觸發加碼那根，若同一根也碰到原進場價 → 加碼單以原進場價出場
        o, h, l = B["o"], B["h"], B["l"]
        e = k + 1
        ep = o[e]
        R = r["Rusd"]
        for j in range(e, r["j"] + 1):
            if (h[j] >= ep + R) if s == 1 else (l[j] <= ep - R):
                if (l[j] <= ep) if s == 1 else (h[j] >= ep):
                    cost = B["spr"][e]
                    r = dict(r, add_usd=-R - cost, addR=(-R - cost) / R)
                break
        return r

    res = {}
    for lab, fn in (("原版", orig_sim), ("保守版", conservative_sim)):
        rows = []
        for leg, (tp, hold) in EXIT.items():
            _, B = signals(m1, leg)
            tf, L, M, S, a, cool = LEGS[leg]
            step = B["idx"][1] - B["idx"][0]
            t, side = entries(B, L, M, S, a, cool)
            for k, s in zip(t, side):
                ts = B["idx"][k] + step
                if ts.dayofweek == 0 or (ts.dayofweek == 4 and ts.hour >= 20):
                    continue
                r = fn(B, k, s, M, hold // 2, tp=tp, cap_usd=100, cap_mode="cap", flat="week")
                if not r:
                    continue
                sw = 0.7 * nights(B["idx"], k + 1, r["j"])
                htf = (s * B["h1"][k] > 0) and (s * B["h4"][k] > 0)
                add = (r["addR"] - sw / r["Rusd"]) if (r["added"] and htf) else 0.0
                rows.append(dict(R=r["R"] - sw / r["Rusd"] + add, addv=add, added=r["added"] and htf))
        D = pd.DataFrame(rows)
        v = D.R.to_numpy()
        res[lab] = D
        P(f"{lab}：{len(v)} 筆、加碼 {int(D.added.sum())} 次、加碼單平均 {D.addv[D.added].mean():+.3f}R、"
          f"總 R {v.sum():.1f}、PF {v[v > 0].sum() / -v[v < 0].sum():.2f}")
    open("results_part29_lookahead.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
