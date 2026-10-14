"""第二十三部分：AMT 參考價的「品質」與「用法」— 共用結構。

在 part22 的 session 表上，為每個 session 補上兩組參考剖面（只用過去的資料）：
  D_*  前一個交易日（broker 01:00~23:59）的 30 分 TPO 剖面
  S_*  前一個 session 的 30 分 TPO 剖面（LDN ← ASIA、NY ← LDN、ASIA ← 前一天 NY）
每組：高 / 低 / POC / VAH / VAL，以及高低點的品質：
  ntop_h / ntop_l ：有幾個 30 分時段的高（低）點落在極值 Q_BAND×ATR 以內
                   ≥ 2 → poor high / low（平頂、拍賣沒有完成）
  tail_h / tail_l ：極值那一端「只有 1 個 TPO 的單印尾巴」長度（ATR）
                   ntop = 1 且 tail ≥ TAIL_EXCESS → excess（有尾巴、拍賣已完成）
"""
import pickle
import numpy as np
import pandas as pd
from part22_core import load_m1, arrays, tick_for, SESS_ORDER
from tpo import poc_va

Q_BAND = 0.02          # poor：極值 0.02 ATR 內有 ≥ 2 個時段
TAIL_EXCESS = 0.05     # excess：單印尾巴 ≥ 0.05 ATR（事前設定，另做 0.03 / 0.10 敏感度）


def seg_profile(H, L, a, b, atr):
    """M1 第 a ~ b 根（含）的 30 分 TPO 剖面與極值品質。時段以 a 為起點每 30 根切。"""
    h, l = H[a:b + 1], L[a:b + 1]
    tick = tick_for((h.max() + l.min()) / 2)
    lo = np.floor(l.min() / tick) * tick
    n = int(round((np.floor(h.max() / tick) * tick - lo) / tick)) + 1
    counts = np.zeros(n, int)
    ph, pl = [], []
    for s in range(0, len(h), 30):
        hh, ll = h[s:s + 30].max(), l[s:s + 30].min()
        ph.append(hh); pl.append(ll)
        i0 = int(round((np.floor(ll / tick) * tick - lo) / tick))
        i1 = int(round((np.floor(hh / tick) * tick - lo) / tick))
        counts[i0:i1 + 1] += 1
    prices = lo + tick * np.arange(n)
    poc, vah, val = poc_va(prices, counts)
    ph, pl = np.array(ph), np.array(pl)
    H_, L_ = h.max(), l.min()
    top = np.argmax(counts[::-1] > 1)            # 從最上面往下數有幾列是單印
    bot = np.argmax(counts > 1)
    return dict(h=H_, l=L_, poc=poc + tick / 2, vah=vah + tick, val=val,
                ntop_h=int((ph >= H_ - Q_BAND * atr).sum()), ntop_l=int((pl <= L_ + Q_BAND * atr).sum()),
                tail_h=top * tick / atr, tail_l=bot * tick / atr)


def quality(ntop, tail, thr=TAIL_EXCESS):
    if ntop >= 2:
        return "poor"
    return "excess" if tail >= thr else "normal"


def build_refs():
    d = load_m1()
    H, L, C, O, SP = arrays(d)
    _, S = pickle.load(open("part22_sessions.pkl", "rb"))
    day_idx = d.groupby("day").apply(lambda g: (d.index.get_loc(g.index[0]), d.index.get_loc(g.index[-1])))
    days = list(day_idx.index)
    pos = {dd: k for k, dd in enumerate(days)}
    rows = []
    prev = None
    for r in S.itertuples():
        out = dict(day=r.day, sess=r.sess)
        k = pos[pd.Timestamp(r.day)]
        if k > 0:
            a, b = day_idx.iloc[k - 1]
            if b - a > 1000:
                out.update({"D_" + kk: v for kk, v in seg_profile(H, L, a, b, r.atr).items()})
        # 前一個 session：必須緊接著（中間沒有缺 session）
        if prev is not None and 0 < r.i0 - prev.i1 <= 400:
            out.update({"S_" + kk: v for kk, v in seg_profile(H, L, prev.i0, prev.i1, r.atr).items()})
        rows.append(out)
        prev = r
    R = pd.DataFrame(rows)
    S = S.merge(R, on=["day", "sess"])
    S = S[S.D_h.notna() & S.S_h.notna()].reset_index(drop=True)
    pickle.dump(S, open("part23_refs.pkl", "wb"))
    return d, S


if __name__ == "__main__":
    d, S = build_refs()
    print(len(S), "sessions")
    for p in ("D", "S"):
        for e in ("h", "l"):
            q = [quality(n, t) for n, t in zip(S[f"{p}_ntop_{e}"], S[f"{p}_tail_{e}"])]
            print(p, e, pd.Series(q).value_counts(normalize=True).round(2).to_dict(),
                  "tail 分位", S[f"{p}_tail_{e}"].quantile([.25, .5, .75]).round(3).tolist(),
                  "ntop", S[f"{p}_ntop_{e}"].value_counts(normalize=True).sort_index().round(2).head(4).to_dict())
