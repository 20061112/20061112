"""Part 23：(1) POC 改做逆勢/穿越方向？ (2) 一堆 FVG / IFVG 時只做最上面或最下面那個？

沿用 part22 的偵測、觸碰、M1 模擬與對照組（隨機平移的假區域）。

(1) 穿越方向（rev）：價格碰到區域時不做反彈，改做「穿過去」的方向。
    以停損單在同一價位成交，停損 1 ATR、停利 1 ATR (r1) / 2 ATR (r2)，最多 48 根，扣點差。
    成交那一分鐘只檢查停損（保守）。所有價位都做一次，看 POC 是否特別。

(2) FVG 堆疊：同方向 FVG 連續出現（中間沒有反向 FVG 成立）視為同一組。
    在第一次觸碰當下，只用「觸碰前已成立」的同組成員排序（沒有未來函數）：
      outer = 離價格最遠的那個（多方 FVG 的最下面 / 空方 FVG 的最上面，通常是這段走勢的起點）
      inner = 離價格最近的那個（多方最上面 / 空方最下面）
      middle、single（組裡只有一個）
    對照組：假區域沿用母區域的分組標籤。IFVG 用同樣方法分組。
"""
import sys
import numpy as np
import pandas as pd
import part22_smc as P
from load import load

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
COLS = ["N", "b1", "b1_pl", "b1_t", "r2", "r2_pl", "r2_t"]


def stack_ids(zones):
    """zones: 已按成立時間排序的同類價位。回傳每個的 stack id。"""
    sid, cur, last_d = [], -1, 0
    for z in zones:
        if z[1] != last_d:
            cur += 1
            last_d = z[1]
        sid.append(cur)
    return np.array(sid)


def position(ev_real, zones, sid):
    """ev_real: 真實事件（含 zid, t=觸碰 bar）。回傳 zid -> (位置, 組大小)。"""
    lo = np.array([z[2] for z in zones])
    hi = np.array([z[3] for z in zones])
    ci = np.array([z[4] for z in zones])
    d = np.array([z[1] for z in zones])
    # 每組成員索引
    groups = pd.Series(np.arange(len(zones))).groupby(sid).apply(np.array).to_dict()
    out = {}
    for zid, j in zip(ev_real.zid.to_numpy(), ev_real.t.to_numpy()):
        mem = groups[sid[zid]]
        mem = mem[ci[mem] < j]
        n = len(mem)
        if n <= 1:
            out[zid] = ("single", n)
            continue
        if d[zid] == 1:
            outer = lo[zid] <= lo[mem].min()
            inner = hi[zid] >= hi[mem].max()
        else:
            outer = hi[zid] >= hi[mem].max()
            inner = lo[zid] <= lo[mem].min()
        out[zid] = ("outer" if outer else "inner" if inner else "middle", n)
    return out


def main():
    m1 = load("data/XAUUSD_M1_2023_2026.csv")
    m1a = tuple(m1[k].to_numpy(float) for k in ("open", "high", "low", "close")) + \
        (m1.spread.to_numpy(float) * 0.01,)
    rev_all, stack_all = [], []
    for rule, tf_min in (("15min", 15), ("1h", 60)):
        df = P.resample(m1, rule)
        zones, A, tr = P.detect(df)
        zones += P.poc_zones(m1, df, A)
        P.A_glob = A
        c = df.close.to_numpy(float)
        m1_start = m1.index.searchsorted(df.index)
        args = (df, A, tr, m1a, m1_start, tf_min)

        # (1) 穿越方向
        pl = P.make_placebos(zones, c)
        ev = pd.DataFrame(P.run(zones, *args, 0, rev=True) + P.run(pl, *args, 1, rev=True))
        ev["tf"] = rule
        ev["hour"] = df.index[ev.t].hour
        rev_all.append(ev)

        # (2) FVG / IFVG 堆疊
        for kind in ("FVG", "IFVG"):
            zk = sorted([z for z in zones if z[0] == kind], key=lambda z: z[4])
            zk = [z + (n,) for n, z in enumerate(zk)]
            sid = stack_ids(zk)
            real = pd.DataFrame(P.run(zk, *args, 0))
            pos = position(real, zk, sid)
            plz = P.make_placebos([zk[z] for z in pos], c)
            e = pd.concat([real, pd.DataFrame(P.run(plz, *args, 1))], ignore_index=True)
            e = e[e.zid.isin(pos)]
            e["pos"] = e.zid.map(lambda z: pos[z][0])
            e["n"] = e.zid.map(lambda z: pos[z][1])
            e["tf"] = rule
            stack_all.append(e)
        print(rule, "done", file=sys.stderr)

    rev = pd.concat(rev_all, ignore_index=True)
    st = pd.concat(stack_all, ignore_index=True)
    rev.to_pickle("part23_rev.pkl")
    st.to_pickle("part23_stack.pkl")

    print("說明：b1 = 先往交易方向 1 ATR 的機率；r2 = 2R 交易每筆平均 R（扣點差）；_pl = 隨機價位對照；t = 真實-對照的 t 值")
    for tf in ("15min", "1h"):
        e = rev[rev.tf == tf]
        print(f"\n===== (1) {tf}：碰到後做『穿越』方向（停損 1 ATR / 停利 2 ATR）=====")
        print(P.summarize(e, "kind")[COLS].round(3).to_string())
    p = rev[rev.kind == "POC"].copy()
    p["half"] = np.where(p.t < p.groupby("tf").t.transform("median"), "前半", "後半")
    print("\n----- POC 穿越：逐年 / 前後半 -----")
    print(P.summarize(p, ["tf", "year"])[COLS].round(3).to_string())
    print(P.summarize(p, ["tf", "half"])[COLS].round(3).to_string())
    p["sess"] = pd.cut(p.hour, [-1, 8, 14, 19, 24], labels=["亞洲(0-8)", "倫敦(9-14)", "紐約(15-19)", "尾盤(20-23)"])
    print(P.summarize(p, ["tf", "sess"])[COLS].round(3).to_string())
    p["agebin"] = pd.cut(p.age, [0, 8, 32, 96, 1001])
    print(P.summarize(p, ["tf", "agebin"])[COLS].round(3).to_string())

    for tf in ("15min", "1h"):
        for kind in ("FVG", "IFVG"):
            e = st[(st.tf == tf) & (st.kind == kind)]
            print(f"\n===== (2) {tf} {kind}：依同組位置（反彈方向，停損在區域外，2R）=====")
            print(P.summarize(e, "pos").reindex(["outer", "middle", "inner", "single"])[COLS].round(3).to_string())
            e2 = e[e.pos.isin(["outer", "inner"])].copy()
            e2["nbin"] = pd.cut(e2.n, [1, 2, 3, 99], labels=["2個", "3個", "4個以上"])
            print(P.summarize(e2, ["pos", "nbin"])[COLS].round(3).to_string())
            s = P.summarize(e, ["pos", "year"])
            print("逐年 b1 真實-對照：")
            print((s.b1 - s.b1_pl).unstack().round(3).to_string())


if __name__ == "__main__":
    main()
