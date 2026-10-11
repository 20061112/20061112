import sys; pass
import pandas as pd
from harmonic_bt import run, summary
pd.set_option("display.width",250); pd.set_option("display.float_format","{:.3f}".format)
rows=[]
for tf in ["M15","H1"]:
    for k in [1.5,2,2.5,3]:
        tr=run(tf,k)
        s=summary(tr); s.insert(0,"k",k); s.insert(0,"TF",tf); rows.append(s)
        h=len(tr)//2; tr=tr.sort_values("t_entry")
        print(tf,k,"前半 %.3f 後半 %.3f"%(tr.R.iloc[:h].mean(),tr.R.iloc[h:].mean()))
print(pd.concat(rows)[["TF","k","型態","筆數","勝率","TP2全中","直接止損","平均R","t值","總R"]].to_string(index=False))
