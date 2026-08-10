"""TTA v3: full-sequence pool + clip-level holdout + mouth-directed gradients.
- ALL frames of ALL clips (no subsampling). Holdout = whole clips (~10%).
- Per-vertex loss weight: 1 + 3 * (|GT disp| / p95) clipped to [1,4]  -> gradients
  concentrate on high-motion (mouth) vertices instead of static skin area.
- Frame sampling: 50% uniform + 50% proportional to current per-frame error
  (hard-example mining; scores refreshed online from batch losses + periodic sweep).
Free vars (networks frozen): W logits, bind pose, per-joint s_rot / s_trn.
ckpt combo700_best_import (=2026-07-09-13-46-51 best@570).
Out: <repo>/tta_proto_v3/{tag}/
"""
import torch, numpy as np, json, os, sys, yaml, glob, pickle, igl
from types import SimpleNamespace
R="/source/inyup/NeuralFacialAnimation"; sys.path.insert(0,R); os.chdir(R)
from utils.rig_loader import load_rig
from models.hierarchical_lbs import HierarchicalLBS_FullPred
from dataloader_CBD import procrustes_LDM
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix

def _pvn(v,f):
    vv=np.ascontiguousarray(v,dtype=np.float64); ff=np.ascontiguousarray(f,dtype=np.int64)
    try:
        return igl.per_vertex_normals(vv,ff).astype(np.float32)
    except TypeError:
        print("PVN DEBUG:",type(v),getattr(v,"shape",None),getattr(v,"dtype",None),
              "->",vv.shape,vv.dtype,vv.flags["C_CONTIGUOUS"],"|",ff.shape,ff.dtype,flush=True)
        raise

CKPT_DIR=os.environ.get("TTA_CKPT","ckpts_hlbs/xcycle670_best_import")
OUT=os.path.join(R,os.environ.get("TTA_OUT","tta_unified"))
ITERS=int(os.environ.get("TTA_ITERS","300")); VIZ_EVERY=10; BATCH=8; LR=5e-3
NEU_W=float(os.environ.get("TTA_NEU_W","0.25"))
TR_W=float(os.environ.get("TTA_TR_W","1.0"))      # static-region W trust region
TR_BIND=float(os.environ.get("TTA_TR_BIND","10.0"))  # bind drift penalty
TR_S=float(os.environ.get("TTA_TR_S","0.1"))      # transform-scale trust
RNEU_W=float(os.environ.get("TTA_RNEU_W","0.5"))  # retarget-neutral anchor (ICT mean src)
dev="cuda"; SEED=42

o=SimpleNamespace(**yaml.safe_load(open(f"{CKPT_DIR}/train_opts.yml"))); g=lambda k,d=None: getattr(o,k,d)
rig=load_rig(o.rig_path); aj=json.load(open(o.active_joints_json))
face_idx=aj['face_joint_idx']; base_idx=aj['base_joint_idx']
sd=torch.load(f"{CKPT_DIR}/model_hlbs_best.pth",map_location='cpu',weights_only=False)
if 'face_joint_idx' in sd: face_idx=sd['face_joint_idx'].tolist()
helper_idx=sd['helper_joint_idx_buf'].tolist() if 'helper_joint_idx_buf' in sd else []
m=HierarchicalLBS_FullPred(rig=rig,topology=g('topo_key','ict'),in_dim_exp=12,hid_dim=o.hid_dim,num_layers=o.num_layers,device=dev,use_joint_trans=o.use_joint_trans,smooth_W=g('smooth_delta_W',0),smooth_W_alpha=g('smooth_delta_W_alpha',0.5),dfn_skin=g('dfn_skin',False),dfn_bind=g('dfn_bind',False),dfn_exp=g('dfn_exp',False),nfs_feat_dim=(int(g('nfs_feat_dim',256) or 256) if o.nfs_feat_dir else 0),nfs_proj_dim=int(g('nfs_proj_dim',0) or 0),nfs_concat=g('nfs_concat',False),adain_pos_norm=g('adain_pos_norm',False),freeze_bind_pose=g('freeze_bind_pose',False),use_gmm_hybrid=g('use_gmm_hybrid',False),init_log_sigma=g('init_log_sigma',-1.2),gmm_mode=g('gmm_mode','additive'),residual_scale=g('residual_scale',2.0),bind_pose_mode=g('bind_pose_mode','net'),face_joint_idx=face_idx,base_joint_idx=base_idx,face_mask_r0=g('face_mask_r0',1.0),face_mask_r1=g('face_mask_r1',2.25),helper_joint_idx=helper_idx,bind_pose_base_residual=bool(g('bind_pose_base_residual',0))).to(dev)
m.load_state_dict(sd,strict=False); m.eval()
for p in m.parameters(): p.requires_grad_(False)
names=rig.joint_names; J=m.num_joints
jaw_j=[i for i,n in enumerate(names) if "Jaw" in n]
print(f"model ready J={J}",flush=True)

# ---------- renderer ----------
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, MeshRenderer, MeshRasterizer, HardPhongShader,
    DirectionalLights, TexturesVertex)
import imageio.v2 as imageio
IMS=256
Rr,Tt=look_at_view_transform(dist=2.4,elev=0,azim=0)
cams=FoVPerspectiveCameras(device=dev,R=Rr,T=Tt,fov=30)
lights=DirectionalLights(device=dev,direction=[[0,0.2,1]],
    ambient_color=[[0.45,0.45,0.48]],diffuse_color=[[0.55,0.55,0.6]],specular_color=[[0.05,0.05,0.05]])
rast=RasterizationSettings(image_size=IMS,blur_radius=0.0,faces_per_pixel=1)
renderer=MeshRenderer(rasterizer=MeshRasterizer(cameras=cams,raster_settings=rast),
                      shader=HardPhongShader(device=dev,cameras=cams,lights=lights))
def render_v(v,faces_t,center,scale):
    vv=(torch.as_tensor(v,dtype=torch.float32,device=dev)-center)/scale
    mesh=Meshes(verts=[vv],faces=[faces_t],
                textures=TexturesVertex(torch.tensor([[0.78,0.85,0.95]],device=dev).expand(1,vv.shape[0],3)))
    img=renderer(mesh)[0,...,:3].clamp(0,1).cpu().numpy()
    return (img*255).astype(np.uint8)

# ---------- data ----------
_DEFAULT=["biwi:M4","biwi:M5","coma:FaceTalk_170809_00138_TA","coma:FaceTalk_170908_03277_TA",
          "newtopo:flame_mean","newtopo:gnm_mean","newtopo:gnm_full"]
TARGETS=[tuple(s.split(":",1)) for s in (sys.argv[1:] or _DEFAULT)]

def _auto_align(clips,tpl,fallback):
    """Unified rigid-canonicalization: measure per-frame rigid offset vs template
    on sample frames; enable per-frame Procrustes only if non-negligible."""
    fr=[p for v in clips.values() for p in v[:2]][:6]
    angs=[];tns=[]
    for p in fr:
        try: arr=np.load(p)
        except Exception: return fallback
        if not (isinstance(arr,np.ndarray) and arr.ndim==2 and arr.shape[-1]==3):
            return fallback
        Ra,ta,_=procrustes_LDM(arr.astype(np.float32),tpl)
        angs.append(float(np.degrees(np.arccos(np.clip((np.trace(Ra)-1)/2,-1,1)))))
        tns.append(float(np.linalg.norm(ta)))
    if not angs: return fallback
    scale=float(np.abs(tpl-tpl.mean(0)).max())
    on=(np.median(angs)>0.5) or (np.median(tns)>2e-3*scale)
    lab="ON" if on else "OFF"
    print(f"[auto-align] median rot {np.median(angs):.2f}deg |t| {np.median(tns):.4f} -> {lab}",flush=True)
    return on

def _resolve(sub):
    for root in ("/source/inyup/tta_data","/data/sihun","/data/sihun/data/sihun"):
        p=os.path.join(root,sub)
        if os.path.exists(p): return p
    raise FileNotFoundError(sub)

def load_target(kind,idn):
    """Return tpl, faces, clips: {clip_name: [frame npy paths]}"""
    clips={}
    if kind=="mf":
        _mtpl=pickle.load(open("utils/templates/mf_templates.pkl","rb"))
        _ids=[k for k in _mtpl.keys() if k!="face"]
        if idn.isdigit(): idn=_ids[int(idn)]
        faces=np.asarray(_mtpl["face"]); tpl=np.asarray(_mtpl[idn],dtype=np.float32)
        import glob as _g
        for cat in ("SEN","ROM"):
            for mode in ("test","train","val"):
                base=f"/data/sihun/multiface_align/{cat}/{mode}/vertices_npy/{idn}"
                for p in sorted(_g.glob(os.path.join(base,"*"))):
                    if os.path.isdir(p):
                        fr=sorted(_g.glob(os.path.join(p,"*.npy")))
                        if fr: clips[f"{cat}_{os.path.basename(p)}"]=fr
        return tpl,faces,clips,True,idn
    if kind=="newtopo":
        base="new_topo_meshes"
        tpl=[]; faces=[]
        for ln in open(f"{base}/{idn}.obj"):
            if ln.startswith("v "): tpl.append([float(x) for x in ln.split()[1:4]])
            elif ln.startswith("f "): faces.append([int(x.split("/")[0])-1 for x in ln.split()[1:4]])
        tpl=np.asarray(tpl,dtype=np.float32); faces=np.asarray(faces,dtype=np.int64)
        import glob as _g
        for p in sorted(_g.glob(os.path.join(base,f"{idn}_clips","*"))):
            if os.path.isdir(p):
                fr=sorted(_g.glob(os.path.join(p,"*.npy")))
                if fr: clips[os.path.basename(p)]=fr
        return tpl,faces,clips,False
    if kind=="biwi":
        mesh=pickle.load(open(_resolve("BIWI_align_deci/templates_align_deci.pkl"),"rb"))
        faces=np.asarray(mesh["face"]); tpl=np.asarray(mesh[idn],dtype=np.float32)
        for mode in ("test","train","val"):
            try: base=_resolve(f"BIWI_align_deci/{mode}/vertices_npy")
            except FileNotFoundError: continue
            for p in sorted(glob.glob(os.path.join(base,f"{idn}_*"))):
                if "_pca" in os.path.basename(p): continue
                cn=os.path.basename(p).replace(".npy","")
                if os.path.isdir(p): clips[cn]=sorted(glob.glob(os.path.join(p,"*.npy")))
                else: clips.setdefault(cn,[]).append(p)
        # ids without released motion data (e.g. M1/M2): synthesized clips via
        # shared-topology displacement transfer (see tools/clips/gen_biwi_pseudo_clips.py)
        for p in sorted(glob.glob(os.path.join("biwi_pseudo_clips",f"{idn}_*"))):
            if os.path.isdir(p):
                fr=sorted(glob.glob(os.path.join(p,"*.npy")))
                if fr: clips[os.path.basename(p)]=fr
        align=False
    else:
        mesh=pickle.load(open(_resolve("VOCA-COMA/voca_templates.pkl"),"rb"))
        faces=np.asarray(mesh["face"]); tpl=np.asarray(mesh[idn],dtype=np.float32)
        for mode in ("test","train","val"):
            try: base=_resolve(f"VOCA-COMA/COMA/{mode}/{idn}/vertices_npy")
            except FileNotFoundError: continue
            for p in sorted(glob.glob(os.path.join(base,"*"))):
                if os.path.isdir(p): clips[os.path.basename(p)]=sorted(glob.glob(os.path.join(p,"*.npy")))
        align=True
    clips={k:v for k,v in clips.items() if len(v)>0}
    return tpl,faces,clips,align

rng=np.random.RandomState(SEED)
os.makedirs(OUT,exist_ok=True)
orig_W=m._get_skinning_weights; orig_B=m._get_bind_pose
orig_rot6d=m._rot6d; orig_pose=m.lbs_pose_model

class PoseWrap(torch.nn.Module):
    def __init__(self,orig,s_trn,J):
        super().__init__(); self.orig=orig; self.s_trn=s_trn; self.J=J
    def forward(self,z,**kw):
        out=self.orig(z,**kw)
        B_=out.shape[0]; J=self.J
        rot=out[...,:J*6]
        trn=out[...,J*6:].reshape(B_,1,J,3)*self.s_trn[None,None,:,None]
        return torch.cat([rot,trn.reshape(B_,1,J*3)],dim=-1)

for kind,idn in TARGETS:
    tag=f"{kind}_{idn}"
    odir=f"{OUT}/{tag}"; os.makedirs(odir,exist_ok=True)
    _lt=load_target(kind,idn)
    if len(_lt)==5: tpl,faces,clips,align,idn=_lt; tag=f"{kind}_{idn}"
    else: tpl,faces,clips,align=_lt
    align=_auto_align(clips,tpl,align)
    cnames=sorted(clips.keys())
    n_hold_clips=max(1,len(cnames)//10)
    hold_clips=set(rng.choice(cnames,n_hold_clips,replace=False).tolist())
    faces_np=faces.astype(np.int32)
    faces_t=torch.tensor(faces.astype(np.int64),device=dev)
    neu_n=_pvn(tpl,faces_np)
    sv=torch.tensor(tpl,device=dev)[None]; sn=torch.tensor(neu_n,device=dev)[None]
    dis_neu=torch.cat([torch.zeros_like(sv),sn,sv,sn],dim=-1)  # neutral-consistency anchor input
    ft=torch.tensor(np.load(f"diff3f_feat_raw/{idn}_nfs_feat.npy").astype(np.float32),device=dev)[None]
    center=torch.tensor(tpl.mean(0),device=dev); scale=float(np.abs(tpl-tpl.mean(0)).max())

    def _load_frames(fp):
        arr=np.load(fp)
        if not isinstance(arr,np.ndarray):          # NpzFile: pick the vertex array
            pick=None
            for k in arr.keys():
                a=arr[k]
                if getattr(a,"ndim",0)>=2 and a.shape[-1]==3: pick=a; break
            if pick is None: return []
            arr=pick
        arr=np.asarray(arr,dtype=np.float32)
        if arr.ndim==2 and arr.shape[-1]==3: return [arr]
        if arr.ndim==3 and arr.shape[-1]==3: return [arr[k] for k in range(arr.shape[0])]
        return []                                    # not vertex data (times etc.) -> skip

    dis=[]; gts=[]; wgt=[]; idx_opt=[]; idx_hold=[]
    for cn in cnames:
        frames_cn=[]
        for fp in clips[cn]:
            frames_cn.extend(_load_frames(fp))
        for gt in frames_cn:
            if align:
                Ra,ta,_=procrustes_LDM(gt,tpl); gt=gt@Ra.T+ta
            gt_n=_pvn(gt,faces_np)
            disp=np.linalg.norm(gt-tpl,axis=-1)
            p95=max(float(np.quantile(disp,0.95)),1e-6)
            w=np.clip(1.0+3.0*disp/p95,1.0,4.0).astype(np.float32)   # mouth-directed vertex weights
            i=len(gts)
            dis.append(torch.tensor(np.concatenate([gt-tpl,gt_n,tpl,neu_n],-1)))
            gts.append(torch.tensor(gt)); wgt.append(torch.tensor(w))
            (idx_hold if cn in hold_clips else idx_opt).append(i)
    n=len(gts)
    print(f"[{tag}] clips={len(cnames)} (hold {sorted(hold_clips)}) frames={n} opt={len(idx_opt)} hold={len(idx_hold)} V={tpl.shape[0]}",flush=True)
    # per-vertex motion magnitude over all frames (for static trust region)
    _mv=np.zeros(tpl.shape[0],dtype=np.float32); _mc=0
    for _g in gts[::max(1,len(gts)//60)]:
        _mv+=np.linalg.norm(_g.numpy()-tpl,axis=-1); _mc+=1
    _mv/=max(_mc,1); _mv=np.clip(_mv/max(float(np.quantile(_mv,0.95)),1e-6),0,1)
    static_w=torch.tensor(1.0-_mv,device=dev)                    # 1=static, 0=mover
    # retarget-neutral: transforms from a canonical neutral source (ICT mean)
    from utils.remesh_utils import ICT_face_model as _ICT
    _ict=_ICT(base_dir=R)
    _iv=np.asarray(_ict.neutral_verts,dtype=np.float32); _if=_ict.faces.astype(np.int64)
    _in=_pvn(_iv,_if)
    with torch.no_grad():
        _src=torch.cat([torch.tensor(_iv,device=dev)[None],torch.tensor(_in,device=dev)[None]],dim=-1)
        _di=torch.cat([torch.zeros(1,_iv.shape[0],3,device=dev),torch.tensor(_in,device=dev)[None],_src],dim=-1)
        _z=m.lbs_exp_z_model(_di); _zf=_z.squeeze(1) if _z.dim()==3 else _z
        _po=orig_pose(_zf.unsqueeze(1)).squeeze(1)
        _r6=_po[:,:J*6].reshape(J,6); _lt=_po[:,J*6:].reshape(1,J,3,1)
        _lR=orig_rot6d(_r6).reshape(1,J,3,3)
        _top=torch.cat([_lR,_lt],dim=-1)
        _bot=torch.cat([torch.zeros(1,J,1,3,device=dev),torch.ones(1,J,1,1,device=dev)],dim=-1)
        T_srcneu=m._chain_hierarchy(torch.cat([_top,_bot],dim=-2)).detach()  # [1,J,4,4]

    # ---- capture + free vars ----
    cap={}
    def capW(*a,**k):
        W,dW=orig_W(*a,**k); cap['W']=W.detach(); cap['dW']=dW; return W,dW
    def capB(*a,**k):
        Bi,jp=orig_B(*a,**k); cap['jp']=jp.detach(); return Bi,jp
    m._get_skinning_weights=capW; m._get_bind_pose=capB
    with torch.no_grad():
        m(sv,dis[0][None].to(dev),source_normal=sn,nfs_feat=ft)
    W0=cap['W']; dW0=cap['dW']; jp0=cap['jp']
    _prev=f"{odir}/optimized.npz"
    if os.path.isfile(_prev):
        _pz=np.load(_prev)
        W_logit=torch.log(torch.tensor(_pz["W"],device=dev).clamp_min(1e-8)).requires_grad_(True)
        bind_var=torch.tensor(_pz["bind"],device=dev).requires_grad_(True)
        s_rot=torch.tensor(_pz["s_rot"],device=dev).requires_grad_(True)
        s_trn=torch.tensor(_pz["s_trn"],device=dev).requires_grad_(True)
        os.replace(_prev,f"{odir}/optimized_prev.npz")
        print(f"[{tag}] resumed from optimized.npz (backed up as optimized_prev.npz)",flush=True)
    else:
        W_logit=torch.log(W0.clamp_min(1e-8)).clone().requires_grad_(True)
        bind_var=jp0.clone().requires_grad_(True)
        s_rot=torch.ones(J,device=dev,requires_grad=True)
        s_trn=torch.ones(J,device=dev,requires_grad=True)
    m._get_skinning_weights=lambda *a,**k:(torch.softmax(W_logit,dim=-1),dW0)
    m._get_bind_pose=lambda *a,**k:(m._build_B_inv(bind_var),bind_var)
    def rot6d_scaled(x):
        Rm=orig_rot6d(x); aa=matrix_to_axis_angle(Rm).reshape(-1,J,3)*s_rot[None,:,None]
        return axis_angle_to_matrix(aa.reshape(-1,3))
    m._rot6d=rot6d_scaled
    m.lbs_pose_model=PoseWrap(orig_pose,s_trn,J)
    _groups=[{"params":[W_logit,bind_var],"lr":LR}]
    if int(os.environ.get("TTA_OPT_SCALES","1")):
        _groups.append({"params":[s_rot,s_trn],"lr":1e-2})
    opt=torch.optim.Adam(_groups)

    def frame_rmse(i):
        with torch.no_grad():
            pred=m(sv,dis[i][None].to(dev),source_normal=sn,nfs_feat=ft)
            return float(np.sqrt(((pred[0]-gts[i].to(dev))**2).sum(-1).mean().item()))
    def eval_set(idxs,cap_n=60):
        sub=idxs if len(idxs)<=cap_n else [idxs[k] for k in np.linspace(0,len(idxs)-1,cap_n).astype(int)]
        return float(np.mean([frame_rmse(i) for i in sub]))

    # initial per-frame error sweep over opt pool (for hard-example sampling + viz pick)
    err=np.zeros(n,dtype=np.float32)
    sweep=[idx_opt[k] for k in np.linspace(0,len(idx_opt)-1,min(150,len(idx_opt))).astype(int)]
    for i in sweep: err[i]=frame_rmse(i)
    med=np.median(err[err>0]); err[err==0]=med   # unswept frames get median score
    hard=[i for i in np.argsort(-err) if i in set(idx_opt)][:2]
    rand=rng.choice([i for i in idx_opt if i not in hard],2,replace=False).tolist()
    viz_idx=sorted(hard+rand)
    print(f"[{tag}] viz frames {viz_idx} (2 hardest + 2 random)",flush=True)

    def viz(it):
        rows_gt=[]; rows_pd=[]
        with torch.no_grad():
            for i in viz_idx:
                pred=m(sv,dis[i][None].to(dev),source_normal=sn,nfs_feat=ft)[0]
                rows_gt.append(render_v(gts[i].numpy(),faces_t,center,scale))
                rows_pd.append(render_v(pred,faces_t,center,scale))
        grid=np.concatenate([np.concatenate(rows_gt,1),np.concatenate(rows_pd,1)],0)
        imageio.imwrite(f"{odir}/it{it:03d}.png",grid)

    r0o,r0h=eval_set(idx_opt),eval_set(idx_hold)
    print(f"[{tag}] it000 RMSE opt={r0o:.5f} hold={r0h:.5f} (resume point)",flush=True)
    viz(0)
    opt_arr=np.array(idx_opt)
    best_hold=r0h; best_state=None; stall=0
    for it in range(1,ITERS+1):
        # 50% uniform + 50% error-proportional (clipped) sampling
        e=np.clip(err[opt_arr],np.quantile(err[opt_arr],0.1),np.quantile(err[opt_arr],0.9))
        p=0.5/len(opt_arr)+0.5*e/e.sum()
        p=p/p.sum()
        bidx=rng.choice(opt_arr,BATCH,replace=False,p=p)
        opt.zero_grad()
        loss=0.0
        for i in bidx:
            pred=m(sv,dis[i][None].to(dev),source_normal=sn,nfs_feat=ft)
            per_v=((pred[0]-gts[i].to(dev))**2).sum(-1)
            loss=loss+(per_v*wgt[i].to(dev)).mean()
            err[i]=float(np.sqrt(per_v.mean().item()))   # online score refresh
        loss=loss/BATCH
        if TR_W>0:
            _Wcur=torch.softmax(W_logit,dim=-1)
            loss=loss+TR_W*(static_w[None,:,None]*(_Wcur-W0).abs()).sum(-1).mean()
        if TR_BIND>0:
            loss=loss+TR_BIND*((bind_var-jp0)**2).sum(-1).mean()
        if TR_S>0:
            loss=loss+TR_S*(((s_rot-1.0)**2).mean()+((s_trn-1.0)**2).mean())
        if RNEU_W>0:
            _Binv=m._build_B_inv(bind_var)
            _G=torch.matmul(T_srcneu.reshape(J,4,4),_Binv.reshape(J,4,4)).reshape(1,J,4,4)
            _vh=torch.cat([sv,torch.ones(1,sv.shape[1],1,device=dev)],dim=-1)
            _vp=torch.einsum("bjkl,bnl->bnjk",_G[:,:,:3,:],_vh)
            _rv=torch.einsum("bnj,bnjk->bnk",torch.softmax(W_logit,dim=-1),_vp)
            loss=loss+RNEU_W*((_rv-sv)**2).sum(-1).mean()
        if NEU_W>0:
            pred_neu=m(sv,dis_neu,source_normal=sn,nfs_feat=ft)
            loss=loss+NEU_W*((pred_neu[0]-sv[0])**2).sum(-1).mean()
        loss.backward(); opt.step()
        if it%VIZ_EVERY==0:
            ro,rh=eval_set(idx_opt),eval_set(idx_hold)
            js=" ".join(f"{names[j].replace('hybrid_jnt_','')}:{float(s_rot[j]):.2f}" for j in jaw_j[:3])
            with torch.no_grad():
                _pn=m(sv,dis_neu,source_normal=sn,nfs_feat=ft)
                nrm=float(np.sqrt(((_pn[0]-sv[0])**2).sum(-1).mean().item()))
            print(f"[{tag}] it{it:03d} loss={float(loss):.3e} RMSE opt={ro:.5f} hold={rh:.5f} neu={nrm:.5f} | s_rot jaw {js} | mean {float(s_rot.mean()):.3f}",flush=True)
            viz(it)
            if rh<best_hold-1e-6:
                best_hold=rh; stall=0
                best_state={k:v.detach().clone() for k,v in
                            [("W_logit",W_logit),("bind",bind_var),("s_rot",s_rot),("s_trn",s_trn)]}
            else:
                stall+=1
                if stall>=8:
                    print(f"[{tag}] early stop at it{it} (hold plateau, best={best_hold:.5f})",flush=True)
                    break
    m._get_skinning_weights=orig_W; m._get_bind_pose=orig_B
    m._rot6d=orig_rot6d; m.lbs_pose_model=orig_pose
    if best_state is not None:
        W_logit=best_state["W_logit"]; bind_var=best_state["bind"]
        s_rot=best_state["s_rot"]; s_trn=best_state["s_trn"]
        print(f"[{tag}] saving best-by-hold state ({best_hold:.5f})",flush=True)
    np.savez(f"{odir}/optimized.npz",W=torch.softmax(W_logit,-1).detach().cpu().numpy(),
             bind=bind_var.detach().cpu().numpy(),
             s_rot=s_rot.detach().cpu().numpy(),s_trn=s_trn.detach().cpu().numpy(),
             hold_clips=np.array(sorted(hold_clips)))
print("DONE",flush=True)
