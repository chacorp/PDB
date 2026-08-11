import numpy as np, glob, os, pickle, json
from scipy.spatial import cKDTree
R='/data/sihun/multiface_align/ROM/test/vertices_npy/m--20190828--1318--002645310--GHS'
tpl=pickle.load(open('/source/inyup/NeuralFacialAnimation/utils/templates/mf_templates.pkl','rb'))
ids=[k for k in tpl.keys() if k!='face']
neu=np.asarray(tpl[ids[12]],dtype=np.float64)
V=len(neu); diag=np.linalg.norm(neu.max(0)-neu.min(0))
def frames(clip): return sorted(glob.glob(f'{R}/{clip}/*.npy'))
def motion_map(clips,step=4):
    m=np.zeros(V)
    for c in clips:
        for f in frames(c)[::step]: m=np.maximum(m,np.linalg.norm(np.load(f)-neu,axis=1))
    return m
MP=motion_map(['EXP_lip001','EXP_lip002','EXP_lip003','EXP_cheek001','EXP_cheek002'])
mouth=np.where(MP>np.quantile(MP,0.95))[0]
# open-mouth reference frame = jaw001 frame with max mean mouth motion
jf=frames('EXP_jaw001'); best=None;bv=-1
for f in jf[::2]:
    v=np.load(f); s=np.linalg.norm((v-neu)[mouth],axis=1).mean()
    if s>bv: bv=s; best=f
vo=np.load(best)
dy=(vo-neu)[mouth,1]
lower=mouth[dy<np.quantile(dy,0.35)]   # moved down on jaw open
upper=mouth[dy>np.quantile(dy,0.65)]
# central band by x
def central(idx,frac=0.5):
    x=neu[idx,0]; xc=np.median(neu[mouth,0]); xr=neu[mouth,0].max()-neu[mouth,0].min()
    return idx[np.abs(x-xc)<frac*0.5*xr]
lc=central(lower); uc=central(upper)
xs=neu[mouth,0]; xr=xs.max()-xs.min()
Lcorner=mouth[xs<xs.min()+0.08*xr]; Rcorner=mouth[xs>xs.max()-0.08*xr]
chin_pool=mouth[dy<np.quantile(dy,0.15)]
print('mouth',len(mouth),'lower',len(lower),'upper',len(upper),'lc/uc',len(lc),len(uc),'corners',len(Lcorner),len(Rcorner))
def gap(v):
    d,_=cKDTree(v[uc]).query(v[lc])
    return np.quantile(d,0.1)
def width(v): return np.linalg.norm(v[Lcorner].mean(0)-v[Rcorner].mean(0))
w_neu=width(neu); g_neu=gap(neu)
print('neutral w %.4f g %.5f diag %.3f open-ref %s g_open %.4f'%(w_neu,g_neu,diag,os.path.basename(os.path.dirname(best)),gap(vo)))
rows=[]
for c in sorted(os.listdir(R)):
    if not os.path.isdir(f'{R}/{c}'): continue
    for f in frames(c):
        v=np.load(f)
        rows.append((c,os.path.basename(f),(width(v)-w_neu)/diag,(gap(v)-g_neu)/diag,
                     (neu[chin_pool,1]-v[chin_pool,1]).mean()/diag))
arr=np.array([r[2:] for r in rows]); meta=[(r[0],r[1]) for r in rows]
json.dump({'rows':[list(m)+list(map(float,a)) for m,a in zip(meta,arr)],
 'sets':{'lc':lc.tolist(),'uc':uc.tolist(),'Lc':Lcorner.tolist(),'Rc':Rcorner.tolist(),'chin':chin_pool.tolist(),'mouth':mouth.tolist()}},
 open('/tmp/mf12_rom_screen.json','w'))
dw,g,dj=arr[:,0],arr[:,1],arr[:,2]
print('frames',len(rows),'dw range %.4f..%.4f g range %.4f..%.4f'%(dw.min(),dw.max(),g.min(),g.max()))
hi=dw>np.quantile(dw,0.97)
A=hi&(g<np.quantile(g,0.40)); B=hi&(g>np.quantile(g,0.90))
def show(mask,tag,n=12):
    idx=np.where(mask)[0]; idx=idx[np.argsort(-dw[idx])][:n]
    print('== pool',tag)
    for i in idx: print('%-40s %-9s dw %+0.4f g %+0.4f dj %+0.4f'%(meta[i][0],meta[i][1],dw[i],g[i],dj[i]))
show(A,'A closed-grin'); show(B,'B open-smile')
