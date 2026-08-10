# Pseudo clips for BIWI ids without motion data (M1, M2): shared-topology
# displacement transfer from clip-bearing ids (M3/M4/M5 test clips e37-e40).
# frame_tgt = tpl_tgt + (frame_src - tpl_src)
import numpy as np, pickle, os, glob
root='/data/sihun/BIWI_align_deci'
tpl=pickle.load(open(f'{root}/templates_align_deci.pkl','rb'))
out='biwi_pseudo_clips'
SRC=[('M3',['e37','e38','e39','e40']),('M4',['e37','e38','e39','e40']),('M5',['e37','e38','e39','e40'])]
for tgt in ['M1','M2']:
    t_t=np.asarray(tpl[tgt],dtype=np.float64)
    n=0
    for sid,es in SRC:
        t_s=np.asarray(tpl[sid],dtype=np.float64)
        for e in es:
            d=None
            for mode in ('test','train','val'):
                c=f'{root}/{mode}/vertices_npy/{sid}_{e}'
                if os.path.isdir(c): d=c; break
            if d is None: continue
            od=f'{out}/{tgt}_p{sid}{e}'; os.makedirs(od,exist_ok=True)
            for fp in sorted(glob.glob(d+'/*.npy')):
                fr=np.load(fp)
                np.save(f'{od}/'+os.path.basename(fp),(t_t+(fr-t_s)).astype(np.float32))
            n+=1
    print(tgt,'clips',n,flush=True)
print('PSEUDO_DONE')
