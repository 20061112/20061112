"""黃金諧波型態抓取：加特利 / 蝙蝠 / 蝴蝶 / 螃蟹 / 鯊魚 / ABCD

流程：
1. ZigZag（ATR 倍數反轉門檻）找轉折點，高低交替。
2. 取連續 5 個轉折當 X-A-B-C-D（ABCD 取 4 個），算比例：
     AB/XA、BC/AB、CD/BC、AD/XA（鯊魚另看 D 對 XC 的回撤）
3. 每個比例落在規格區間（再加容許誤差 TOL）才算成立。
4. 另外列出「進行中」：X-A-B-C 已確認、D 尚未出現，直接投影各型態的 D 潛在反轉區 (PRZ)。

用法：python3 harmonic.py [TF] [ZigZag ATR倍數] [顯示天數]
      例：python3 harmonic.py H1 3 60
"""
import sys
import numpy as np
import pandas as pd
from load import load

TOL = 0.03  # 比例容許誤差（絕對值）

# 規格（Scott Carney 標準；None = 不檢查）
# ab: B 對 XA 回撤；bc: C 對 AB 回撤；cd: CD 對 BC 延伸；ad: D 對 XA 回撤/延伸
SPECS = {
    "加特利 Gartley":   dict(ab=(0.618, 0.618), bc=(0.382, 0.886), cd=(1.13, 1.618), ad=(0.786, 0.786)),
    "蝙蝠 Bat":         dict(ab=(0.382, 0.500), bc=(0.382, 0.886), cd=(1.618, 2.618), ad=(0.886, 0.886)),
    "蝴蝶 Butterfly":   dict(ab=(0.786, 0.786), bc=(0.382, 0.886), cd=(1.618, 2.618), ad=(1.27, 1.618)),
    "螃蟹 Crab":        dict(ab=(0.382, 0.618), bc=(0.382, 0.886), cd=(2.24, 3.618), ad=(1.618, 1.618)),
    # 鯊魚：C 超過 A（BC 是 AB 的 1.13~1.618 延伸），D 在 XC 的 0.886~1.13
    "鯊魚 Shark":       dict(ab=(0.382, 0.618), bc=(1.13, 1.618), cd=(1.618, 2.24), xc=(0.886, 1.13)),
}
ABCD_SPEC = dict(bc=(0.382, 0.886), cd=(1.13, 2.618), ab_cd=(0.9, 1.1))  # ab_cd = CD/AB


def atr(df, n=14):
    pc = df["close"].shift()
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - pc).abs(), (df["low"] - pc).abs()))
    return tr.rolling(n).mean()


def zigzag(df, k):
    """ATR 倍數 ZigZag。回傳 [(bar_idx, price, +1高/-1低), ...]，最後一點是未確認的當前極值。"""
    h, l, a = df["high"].values, df["low"].values, atr(df).bfill().values
    piv = []
    d = 0
    ext_i, ext_p = 0, h[0]
    lo_i, lo_p, hi_i, hi_p = 0, l[0], 0, h[0]
    for i in range(1, len(df)):
        th = k * a[i]
        if d == 0:
            if h[i] > hi_p: hi_i, hi_p = i, h[i]
            if l[i] < lo_p: lo_i, lo_p = i, l[i]
            if hi_p - lo_p > th:
                if hi_i > lo_i:
                    piv.append((lo_i, lo_p, -1)); d, ext_i, ext_p = 1, hi_i, hi_p
                else:
                    piv.append((hi_i, hi_p, 1)); d, ext_i, ext_p = -1, lo_i, lo_p
        elif d == 1:
            if h[i] > ext_p: ext_i, ext_p = i, h[i]
            elif ext_p - l[i] > th:
                piv.append((ext_i, ext_p, 1)); d, ext_i, ext_p = -1, i, l[i]
        else:
            if l[i] < ext_p: ext_i, ext_p = i, l[i]
            elif h[i] - ext_p > th:
                piv.append((ext_i, ext_p, -1)); d, ext_i, ext_p = 1, i, h[i]
    piv.append((ext_i, ext_p, d))  # 未確認
    return piv


def inr(v, rng):
    return rng[0] - TOL <= v <= rng[1] + TOL


def dist(v, rng):
    return 0.0 if rng[0] <= v <= rng[1] else min(abs(v - rng[0]), abs(v - rng[1]))


def ratios(X, A, B, C, D):
    xa, ab, bc, cd = abs(A - X), abs(B - A), abs(C - B), abs(D - C)
    return dict(ab=ab / xa, bc=bc / ab, cd=cd / bc, ad=abs(A - D) / xa,
                xc=abs(C - D) / abs(C - X))


def match5(r):
    out = []
    for name, sp in SPECS.items():
        if all(inr(r[k], v) for k, v in sp.items()):
            out.append((name, sum(dist(r[k], v) for k, v in sp.items())))
    return out


def fmt_t(t):
    return t.strftime("%m/%d %H:%M")


def scan(df, k, days):
    piv = zigzag(df, k)
    conf = piv[:-1]
    t = df.index
    start = t[-1] - pd.Timedelta(days=days)
    rows = []
    # 完成型態（D 為已確認轉折）
    for j in range(4, len(conf)):
        X, A, B, C, D = conf[j - 4:j + 1]
        if t[D[0]] < start:
            continue
        r = ratios(X[1], A[1], B[1], C[1], D[1])
        side = "多(看漲)" if D[2] == -1 else "空(看跌)"
        for name, sc in match5(r):
            rows.append(dict(型態=name, 方向=side, X=f"{fmt_t(t[X[0]])} {X[1]:.2f}",
                             A=f"{A[1]:.2f}", B=f"{B[1]:.2f}", C=f"{C[1]:.2f}",
                             D=f"{fmt_t(t[D[0]])} {D[1]:.2f}",
                             AB_XA=r["ab"], BC_AB=r["bc"], CD_BC=r["cd"],
                             AD_XA=r["ad"], D_XC=r["xc"], 誤差=sc))
    # ABCD
    for j in range(3, len(conf)):
        A, B, C, D = conf[j - 3:j + 1]
        if t[D[0]] < start:
            continue
        ab, bc, cd = abs(B[1] - A[1]), abs(C[1] - B[1]), abs(D[1] - C[1])
        r = dict(bc=bc / ab, cd=cd / bc, ab_cd=cd / ab)
        if all(inr(r[k], v) for k, v in ABCD_SPEC.items()):
            sc = sum(dist(r[k], v) for k, v in ABCD_SPEC.items())
            side = "多(看漲)" if D[2] == -1 else "空(看跌)"
            rows.append(dict(型態="ABCD", 方向=side, X="-", A=f"{fmt_t(t[A[0]])} {A[1]:.2f}",
                             B=f"{B[1]:.2f}", C=f"{C[1]:.2f}",
                             D=f"{fmt_t(t[D[0]])} {D[1]:.2f}",
                             AB_XA=np.nan, BC_AB=r["bc"], CD_BC=r["cd"],
                             AD_XA=np.nan, D_XC=r["ab_cd"], 誤差=sc))
    res = pd.DataFrame(rows)
    if len(res):
        res["_t"] = res["D"].str[:11]
        res = res.sort_values("_t", kind="stable").drop(columns="_t")
    return piv, res


def forming(df, piv):
    """以最後 4 個已確認轉折當 X-A-B-C，投影各型態 D 的 PRZ。"""
    conf = piv[:-1]
    X, A, B, C = conf[-4:]
    t = df.index
    xa, ab, bc = abs(A[1] - X[1]), abs(B[1] - A[1]), abs(C[1] - B[1])
    sgn = -1 if A[2] == 1 else 1  # D 方向：A 是高點 → D 往下（看漲型態）
    side = "看漲(D 為低點)" if A[2] == 1 else "看跌(D 為高點)"
    lines = [f"X {fmt_t(t[X[0]])} {X[1]:.2f} | A {fmt_t(t[A[0]])} {A[1]:.2f} | "
             f"B {fmt_t(t[B[0]])} {B[1]:.2f} | C {fmt_t(t[C[0]])} {C[1]:.2f}  → {side}",
             f"AB/XA = {ab/xa:.3f}   BC/AB = {bc/ab:.3f}   目前價 {df['close'].iloc[-1]:.2f}"
             f"（未確認極值 {piv[-1][1]:.2f}）"]
    for name, sp in SPECS.items():
        ok_ab, ok_bc = inr(ab / xa, sp["ab"]), inr(bc / ab, sp["bc"])
        if "ad" in sp:
            d1 = A[1] + sgn * sp["ad"][0] * xa
            d2 = A[1] + sgn * sp["ad"][1] * xa
        else:  # 鯊魚用 XC
            xc = C[1] - X[1]
            d1, d2 = C[1] - sp["xc"][0] * xc, C[1] - sp["xc"][1] * xc
        e1 = C[1] + sgn * sp["cd"][0] * bc
        e2 = C[1] + sgn * sp["cd"][1] * bc
        flag = "✔" if ok_ab and ok_bc else "✘"
        lines.append(f"  {flag} {name:<14} XA/XC 定位 D = {min(d1,d2):.2f}~{max(d1,d2):.2f}"
                     f"   BC 延伸 D = {min(e1,e2):.2f}~{max(e1,e2):.2f}"
                     f"   (AB {'OK' if ok_ab else 'NG'}, BC {'OK' if ok_bc else 'NG'})")
    # ABCD 用最後 3 點 A-B-C
    A2, B2, C2 = conf[-3:]
    ab2, bc2 = abs(B2[1] - A2[1]), abs(C2[1] - B2[1])
    s2 = 1 if C2[2] == 1 else -1
    d_eq = C2[1] - s2 * ab2
    lines.append(f"  {'✔' if inr(bc2/ab2, ABCD_SPEC['bc']) else '✘'} ABCD           "
                 f"A {A2[1]:.2f} B {B2[1]:.2f} C {C2[1]:.2f}  BC/AB={bc2/ab2:.3f}  "
                 f"AB=CD 的 D = {d_eq:.2f}   1.27~1.618 BC = "
                 f"{C2[1]-s2*1.27*bc2:.2f}~{C2[1]-s2*1.618*bc2:.2f}")
    return "\n".join(lines)


def print_specs():
    print("=== 使用的比例規格（容許誤差 ±%.2f）===" % TOL)
    print(f"{'型態':<16}{'B: AB/XA':<14}{'C: BC/AB':<14}{'CD/BC':<14}{'D: AD/XA'}")
    for name, sp in SPECS.items():
        f = lambda r: f"{r[0]}" if r[0] == r[1] else f"{r[0]}~{r[1]}"
        last = f(sp["ad"]) if "ad" in sp else f"D 在 XC 的 {f(sp['xc'])}"
        print(f"{name:<14}{f(sp['ab']):<14}{f(sp['bc']):<14}{f(sp['cd']):<14}{last}")
    s = ABCD_SPEC
    print(f"{'ABCD':<16}{'-':<14}{s['bc'][0]}~{s['bc'][1]:<9}{s['cd'][0]}~{s['cd'][1]:<9}"
          f"CD/AB {s['ab_cd'][0]}~{s['ab_cd'][1]}")
    print()


def main():
    tf = sys.argv[1] if len(sys.argv) > 1 else "H1"
    k = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    df = load("data/XAUUSD_M15_2023_2026.csv")
    rule = {"M15": None, "H1": "1h", "H4": "4h", "D1": "1D"}[tf]
    if rule:
        df = df.resample(rule).agg(dict(open="first", high="max", low="min", close="last",
                                        tickvol="sum", spread="mean")).dropna()
    print_specs()
    piv, res = scan(df, k, days)
    print(f"=== {tf}  ZigZag {k}×ATR14  最近 {days} 天  資料到 {df.index[-1]}（伺服器時間）===")
    print(f"轉折點數：{len(piv)}，完成型態：{len(res)}\n")
    if len(res):
        with pd.option_context("display.width", 250, "display.max_columns", 30,
                               "display.max_rows", 500, "display.float_format", "{:.3f}".format):
            print(res.to_string(index=False))
        print("\n（ABCD 列的 D_XC 欄位 = CD/AB）")
    print("\n=== 進行中（D 尚未完成）的潛在反轉區 ===")
    print(forming(df, piv))


if __name__ == "__main__":
    main()
