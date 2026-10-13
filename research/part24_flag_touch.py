"""第二十四部分補充：第二十二部分的 M15 旗面改用「盤中觸及」判斷先突破哪一邊（與實際停損 / 停利成交方式相同）。"""
import numpy as np
import pandas as pd
from load import load
m = load("data/XAUUSD_M15_2023_2026.csv")
h,l,c = m.high.values,m.low.values,m.close.values
F = pd.read_csv("part22_formations_M15.csv")
from part22_flag import _atr
atr=_atr(h,l,c,96)
rows=[]
for t0,d,eu,ba in zip(F.t,F.dir,F.exp_up,F.box_atr):
    box=ba*atr[t0]; c0=c[t0]*d; lo=c0-eu*box; hi=lo+box
    fav = (h if d==1 else -l)[t0+1:t0+49]; adv=(l if d==1 else -h)[t0+1:t0+49]
    a=np.flatnonzero(fav>hi); b=np.flatnonzero(adv<lo)
    a=a[0] if len(a) else 10**6; b=b[0] if len(b) else 10**6
    if a==b: continue
    rows.append((pd.Timestamp(m.index[t0]).year,float(a<b),eu))
X=pd.DataFrame(rows,columns=['y','up','exp']); X['ex']=X.up-X.exp
print("flag touch overall", round(X.ex.mean(),3), "t", round(X.ex.mean()/X.ex.std()*np.sqrt(len(X)),2), len(X))
print(X.groupby('y').ex.mean().round(3).to_dict())
print(X.groupby(pd.cut(X.exp,[0,.2,.4,.6,.8,1]),observed=True).ex.mean().round(3).tolist())
