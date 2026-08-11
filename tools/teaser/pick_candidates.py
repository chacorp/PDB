import json, numpy as np
d=json.load(open('/tmp/mf12_rom_screen.json'))
rows=d['rows']
meta=[(r[0],r[1]) for r in rows]; arr=np.array([r[2:] for r in rows])
dw,g,dj=arr[:,0],arr[:,1],arr[:,2]
def fnum(name): return int(name.split('-')[-1].split('.')[0])
def dedupe(idx,gapf=8):
    out=[]
    for i in idx:
        if all(meta[i][0]!=meta[j][0] or abs(fnum(meta[i][1])-fnum(meta[j][1]))>=gapf for j in out):
            out.append(i)
    return out
# pool A: wide & closed (low gap); per-clip top then global
A=np.where((dw>np.quantile(dw,0.90))&(g<np.quantile(g,0.40)))[0]
A=dedupe(sorted(A,key=lambda i:-dw[i]))[:12]
# pool B: open smile — wide-ish AND gap high
B=np.where((dw>np.quantile(dw,0.80))&(g>np.quantile(g,0.92)))[0]
B=dedupe(sorted(B,key=lambda i:-(dw[i]+g[i])))[:6]
sel={'A':[[meta[i][0],meta[i][1],float(dw[i]),float(g[i]),float(dj[i])] for i in A],
     'B':[[meta[i][0],meta[i][1],float(dw[i]),float(g[i]),float(dj[i])] for i in B]}
json.dump(sel,open('/tmp/mf12_candidates.json','w'),indent=1)
for k in ('A','B'):
    print('== pool',k)
    for c,f,w,gg,j in sel[k]: print('%-40s %-9s dw %+0.4f g %+0.4f dj %+0.4f'%(c,f,w,gg,j))
