import random
import numpy as np
import torch
# import meshplot as mp

from pytorch3d.ops import cot_laplacian
from pytorch3d.loss import point_mesh_face_distance, chamfer_distance, mesh_laplacian_smoothing, mesh_edge_loss
from pytorch3d.structures import Meshes, Pointclouds
# from chamferdist import ChamferDistance
from utils.remesh_utils import ICT_face_model


from utils.exp_utils import *

# for visualization
from matplotrender import *

def get_random_data(id_vert, vn_list, batch_size, device, USE_ROTATE, USE_SCALE, USE_NORMALIZE=False):
    if USE_NORMALIZE:
        id_vert_no_R_no_S = (id_vert / torch.linalg.norm(id_vert.float(), axis=-1, keepdims=True)).to(device)
    else:
        id_vert_no_R_no_S = id_vert.to(device)
    
    if USE_ROTATE:
        batch_R = []
        for _ in range(batch_size):
            batch_R.append(random_rotation_matrix())
        batch_R = np.array(batch_R)
        batch_R = torch.tensor(batch_R).float().to(device)
        
        id_vert_R = torch.einsum('bck,bnk->bnc', batch_R.permute(0,2,1), id_vert_no_R_no_S)
        vn_list_ = torch.einsum('bck,bnk->bnc', batch_R.permute(0,2,1), vn_list.to(device))
    else:
        id_vert_R = id_vert_no_R_no_S
        vn_list_ = vn_list.to(device)
    
    
    if USE_SCALE:
        batch_S = []
        for _ in range(batch_size):
            # scale range in 0.5 ~ 1.5
            tmp_S = np.eye(3) * (np.random.rand(1)+0.5)
            batch_S.append(tmp_S)
        batch_S = np.array(batch_S)
        batch_S = torch.tensor(batch_S).float().to(device)
        
        batch_S_inv = batch_S.to(device).clone()
        batch_S_inv[:,[0,1,2],[0,1,2]] = 1/batch_S_inv[:,[0,1,2],[0,1,2]]
        
        id_vert_ = torch.einsum('bck,bnk->bnc', batch_S, id_vert_R)
    else:
        id_vert_ = id_vert_R    
            
    ## mean to the zero
    id_vert_ = id_vert_ - id_vert_.mean(-2, keepdims=True)
    
    return id_vert_, vn_list_, batch_R, batch_S, batch_S_inv, id_vert_R, id_vert_no_R_no_S


def main(model_type, logdir):
    device='cuda'
    batch_size = 8
    
    USE_STD = False
    USE_NORMALIZE = False
    USE_ROTATE = True
    USE_SCALE = True

    MODEL_TYPE = model_type
    
    #### prepare ICT data ##########################################
    with torch.no_grad():
        ict = ICT_face_model()
        
        
        ict_std = np.load('/source/sihun/NFS/utils/ict/standardization.npy', allow_pickle=True).item()
        ict_decimate = np.load('utils/ict/ICT_decimate.npz')
        ict_iden_vecs = np.load('/source/sihun/NFS/ict_face_pt/random_identity_vecs.npy')
        
        id_code = ict_iden_vecs[:batch_size]
        id_disp = ict.get_id_disp(id_code)
        
        # ID_code = torch.tensor(id_code).float()
        id_vert = torch.tensor(id_disp+ict.neutral_verts[None]).float()
        
        
        ict_v = ict.neutral_verts
        ict_f = ict.faces
        if USE_STD:
            ict_v = ict.neutral_verts[ict_std['v_idx']]
            ict_f = ict_std['new_f']
            id_vert = id_vert[:, ict_std['v_idx']]
        
        base_verts = torch.tensor(ict.neutral_verts).float()[None]
        base_faces = torch.tensor(ict.faces).long()
        base_vn = compute_vertex_normals(ict.neutral_verts, ict.faces)
        base_vn = torch.tensor(base_vn).float()[None]
    
        ## decimate
        # gt_vert = torch.tensor(ict.neutral_verts[ict_decimate['v_idx']]).float()
        # gt_faces = torch.tensor(ict_decimate['new_f']).long()
        # gt_vn_full = compute_vertex_normals(ict.neutral_verts, ict.faces)
        # gt_vn = torch.tensor(gt_vn_full[ict_decimate['v_idx']]).float()
        
        # print('loaded ict mesh:')
        # print("\t", id_vert.shape)
        # print("\t", base_verts.shape)
            
        ## vertex normal
        vn_list=[]
        for id_v in id_vert:
            _vn = compute_vertex_normals(id_v.numpy(), ict_f)
            vn_list.append(_vn)
        vn_list = np.array(vn_list)
        vn_list = torch.tensor(vn_list).float()
    ################################################################
    
    
    #### positional encoding #######################################
    with torch.no_grad():
        # PE = torch.tensor(get_colors(ict_v))[None].float()
        # PE = PE.repeat(batch_size, 1, 1)
        # PE = (PE / torch.linalg.norm(PE, axis=-1, keepdims=True))
        
        PE_list = []
        dfn_info_list = []
        for i in range(batch_size):
            pth_file = f'/data/sihun/ICT-audio2face/precompute-synth-fullhead/{i:03d}_diff3f.pth'
            fm_feat = torch.load(pth_file)
            PE_list.append(fm_feat[None])
            pkl_file = f'/data/sihun/ICT-audio2face/precompute-synth-fullhead/{i:03d}_dfn_info.pkl'
            with open(pkl_file, 'rb') as f:
                pkl_dfn_info = pickle.load(f)
            dfn_info_list.append(pkl_dfn_info)
        
        PE = torch.vstack(PE_list).float().to(device)
        N_PE = PE.shape[-1]
    ################################################################
    
    
    
    
    #### model #####################################################
    # N_layer=8
    N_layer=8
    # in_dim = 3+3+N_PE
    if MODEL_TYPE=='mlp':
        model_r_g = Model(3+3+N_PE, 6, mode='6D', num_layers=N_layer, use_residual=True, out_type='global').to(device)
        model_r_l = Model(3+3+N_PE, 6, mode='6D', num_layers=N_layer, use_residual=True).to(device)
        model_s   = Model(3+3+N_PE, 1, mode='scale', num_layers=N_layer, use_residual=True).to(device)
    if MODEL_TYPE=='mk1':
        model_r_g = Model(3+3+N_PE, 6, mode='6D', num_layers=N_layer, use_residual=True, use_adain=True, out_type='global').to(device)
        model_r_l = Model(3+3+N_PE, 6, mode='6D', num_layers=N_layer, use_residual=True, use_adain=True).to(device)
        model_s   = Model(3+3+N_PE, 1, mode='scale', num_layers=N_layer, use_residual=True, use_adain=True).to(device)
    if MODEL_TYPE=='mk2':
        model_r_g = Model_mk2(3+3+N_PE, 6, mode='6D', num_layers=N_layer, use_residual=True, use_adain=True, use_to_out=False, out_type='global').to(device)
        model_r_l = Model_mk2(3+3+N_PE, 6, mode='6D', num_layers=N_layer, use_residual=True, use_adain=True, use_to_out=False).to(device)
        model_s   = Model_mk2(3+3+N_PE, 1, mode='scale', num_layers=N_layer, use_residual=True, use_adain=True, use_to_out=False).to(device)
    elif MODEL_TYPE=='mk2_2':
        model_r = Model_mk2_2(3+3+N_PE, 6, mode='6D', num_layers=N_layer, use_residual=True, use_adain=True, use_to_out=False, out_type='global').to(device)
        model_s = Model_mk2(3+3+N_PE, 1, mode='scale', num_layers=N_layer, use_residual=True, use_adain=True, use_to_out=False).to(device)
    elif MODEL_TYPE=='mk3':
        model_r_g = Model_mk3(3+3+N_PE, 6, mode='6D').to(device)
        model_r_l = Model_mk3_2(3+3+N_PE, 6, mode='6D').to(device)
        model_s   = Model_mk3_2(3+3+N_PE, 1, mode='scale').to(device)
    elif MODEL_TYPE=='mk4':
        model_r_g = Model_mk4(3+3+N_PE, 6, mode='6D', num_layers=N_layer, outputs_at='global_mean').to(device)
        model_r_l = Model_mk4(3+3+N_PE, 6, mode='6D', num_layers=N_layer, outputs_at='vertices').to(device)
        model_s = Model_mk4(3+3+N_PE, 1, mode='scale', num_layers=N_layer, outputs_at='vertices').to(device)
    ################################################################


    
    #### optimizer #################################################
    SIZE=2
    # lr = 1e-3
    lr = 2e-4
    if MODEL_TYPE=='mk2_2':
        optimizer = torch.optim.Adam([
            *model_r.parameters(), 
            *model_s.parameters()
        ], lr=lr, betas=(0.9, 0.999))
    else:
        optimizer = torch.optim.Adam([
            *model_r_g.parameters(), 
            *model_r_l.parameters(), 
            *model_s.parameters()
        ], lr=lr, betas=(0.9, 0.999))
    ################################################################
    
    
    criterion = nn.MSELoss()
    
    
    
    
    
    ### precomputes ################################################
    with torch.no_grad():
        N = base_verts.shape[1]
        GT = base_verts.repeat(batch_size, 1, 1).to(device)
        # zero center
        GT = GT - GT.mean(-2, keepdims=True)
        
        p3d_mesh_1 = Meshes(verts=[gt for gt in GT], faces=[base_faces.to(device)]*batch_size)
        # p3d_mesh_1_verts = p3d_mesh_1.verts_packed()
        # p3d_mesh_1_faces = p3d_mesh_1.faces_packed()
        # # cot_laplacian
        # L_gt, inv_areas = cot_laplacian(p3d_mesh_1_verts, p3d_mesh_1_faces)
        # norm_w = torch.sparse.sum(L_gt, dim=1).to_dense().view(-1, 1)
        # idx = norm_w > 0
        # norm_w[idx] = 1.0 / norm_w[idx]
        # LV_gt = L_gt.mm(p3d_mesh_1_verts) * norm_w - p3d_mesh_1_verts        
        
        L_gt = p3d_mesh_1.laplacian_packed()
        LV_gt = L_gt.mm(p3d_mesh_1.verts_packed())
        
        if USE_NORMALIZE:
            base_verts = base_verts / torch.linalg.norm(base_verts, axis=-1, keepdims=True)
            
        if MODEL_TYPE=='mk4':
            batch_mass, batch_L, batch_evals, batch_evecs, batch_grad_X, batch_grad_Y, batch_faces = load_batch_dfn_ino(dfn_info_list, device)
    ################################################################
    
    if MODEL_TYPE=='mk2_2':
        model_r.train()
        model_s.train()
    else:
        model_r_g.train()
        model_r_l.train()
        model_s.train()

    BEST_LOSS = 1_000
    num_iter=20_000
    num_vis_interval=1_000
    # num_iter=2
    # num_vis_interval=1
    
    pbar = tqdm(range(num_iter))
    for i in pbar:

        #### data preparation ######################################
        with torch.no_grad():
            id_vert_, vn_list_, batch_R, batch_S, batch_S_inv, id_vert_R, id_vert_no_R_no_S = get_random_data(
                id_vert, vn_list, batch_size, device, USE_ROTATE, USE_SCALE, USE_NORMALIZE=False
            )
            
            if USE_ROTATE:
                ## GT rotation
                GT_R = rotation_matrix_from_vectors_batch_fast(id_vert_.float(), GT)
            
            if USE_SCALE:
                ## GT scale
                gt_scale = torch.linalg.norm(GT, axis=-1, keepdims=True) / torch.linalg.norm(id_vert_.float(), axis=-1, keepdims=True)
                GT_S = gt_scale.unsqueeze(-1) * torch.eye(3)[None,None].to(device)
                GT_S_inv = (1/gt_scale.unsqueeze(-1)) * torch.eye(3)[None,None].to(device)
            
        
        ## model prediction
        id_vert_in = torch.cat([id_vert_, vn_list_, PE], dim=-1).to(device)
        
        if USE_ROTATE:
            
            if MODEL_TYPE=='mlp':
                pred_R, raw_out = model_r_g(id_vert_in, return_raw=True)
                pred_R_l, raw_out_l = model_r_l(id_vert_in, return_raw=True)
            elif MODEL_TYPE=='mk2':
                pred_R, raw_out = model_r_g(id_vert_in, return_raw=True)
                pred_R_l, raw_out_l = model_r_l(id_vert_in, return_raw=True)
            elif MODEL_TYPE=='mk2_2':
                (pred_R, raw_out),(pred_R_l, raw_out_l) = model_r(id_vert_in, return_raw=True)
            elif MODEL_TYPE=='mk3':
                (pred_R, raw_out), trans_feat = model_r_g(id_vert_in, return_raw=True)
                (pred_R_l, raw_out_l), trans_feat_l = model_r_l(id_vert_in, return_raw=True)
            elif MODEL_TYPE=='mk4':
                pred_R, raw_out = model_r_g(
                    id_vert_in, 
                    batch_mass, batch_L, batch_evals, batch_evecs,
                    batch_grad_X, batch_grad_Y, batch_faces,
                    return_raw=True
                )
                pred_R_l, raw_out_l = model_r_l(
                    id_vert_in,
                    batch_mass, batch_L, batch_evals, batch_evecs,
                    batch_grad_X, batch_grad_Y, batch_faces,
                    return_raw=True
                )
        
        if USE_SCALE:
            if MODEL_TYPE=='mlp':
                pred_S, pred_S_inv = model_s(id_vert_in, return_inv=True)
            elif MODEL_TYPE=='mk2':
                pred_S, pred_S_inv = model_s(id_vert_in, return_inv=True)
            elif MODEL_TYPE=='mk2_2':
                pred_S, pred_S_inv = model_s(id_vert_in, return_inv=True)
            elif MODEL_TYPE=='mk3':
                (pred_S, pred_S_inv), trans_feat_s = model_s(id_vert_in, return_inv=True)
            elif MODEL_TYPE=='mk4':
                pred_S, pred_S_inv = model_s(
                    id_vert_in,
                    batch_mass, batch_L, batch_evals, batch_evecs,
                    batch_grad_X, batch_grad_Y, batch_faces,
                    return_inv=True
                )

            
        ## apply to mesh
        if USE_ROTATE:
            # pred_R_v = torch.einsum('bnck,bnk->bnc', pred_R, id_vert_.to(device))
            pred_GR_v = torch.einsum('bnck,bnk->bnc', pred_R, id_vert_.to(device)) # applying global rotation
            pred_R_v = torch.einsum('bnck,bnk->bnc', pred_R_l, pred_GR_v) # applying local rotation
            
        if USE_SCALE:
            pred_S_v = torch.einsum('bnck,bnk->bnc', pred_S, id_vert_.to(device))
    
        
        if USE_ROTATE and not USE_SCALE:
            pred_v = pred_R_v
        if not USE_ROTATE and USE_SCALE:
            pred_v = pred_S_v
        if USE_ROTATE and USE_SCALE:
            # pred_v = torch.einsum('bnck,bnk->bnc', pred_R@pred_S, id_vert_.to(device))
            pred_v = torch.einsum('bnck,bnk->bnc', pred_S, pred_R_v)
    
        
        loss = 0
        loss += criterion(GT, pred_v)
        
        if MODEL_TYPE=='mk3':
            loss += feature_transform_reguliarzer(trans_feat) * 0.001
            loss += feature_transform_reguliarzer(trans_feat_s) * 0.001
            loss += feature_transform_reguliarzer(trans_feat_l) * 0.001
            
        if USE_ROTATE:
            # gradient to global rotation only
            id_vert_no_R = torch.einsum('bck,bnk->bnc', batch_S, id_vert_no_R_no_S)
            loss += criterion(id_vert_no_R, pred_GR_v)
            
            # gradient to local rotation (non-rigid?) only
            with torch.no_grad():
                GT_R_v = torch.einsum('bnck,bnk->bnc', GT_R, id_vert_.to(device))
                GT_R_v = torch.einsum('bck,bnk->bnc', batch_S_inv, GT_R_v)
            pred_LR_v_only = torch.einsum('bnck,bnk->bnc', pred_R_l, id_vert_no_R_no_S.to(device))
            loss += criterion(GT_R_v, pred_LR_v_only) #* 1e+2#*0.2
            
            # gradient to global and local
            loss += criterion(GT_R_v, pred_R_v)
            
            
        if USE_SCALE:
            with torch.no_grad():
                GT_S_v = torch.einsum('bnck,bnk->bnc', GT_S, id_vert_.to(device))
            loss += criterion(GT_S_v, pred_S_v)

        
        # loss += criterion(T_, torch.eye(3)[None,None].repeat(batch_size, N, 1, 1).to(device))*0.5
        
        
        # if True: ## rotation to be identity
        #     loss += criterion(torch.einsum('bnki,bnkj->bnij', pred_R, pred_R), torch.eye(3)[None,None].repeat(batch_size, N, 1, 1).to(device))

        
        # if True: ## iff representation is quaternion
        #     sq_sum_out = (raw_out**2).sum(dim=-1)
        #     loss += ((sq_sum_out - 1.0) ** 2).mean()

        
        # if USE_ROTATE:
        #     # p3d_mesh_R_v = Meshes(verts=[v for v in pred_R_v], faces=[base_faces.to(device)]*batch_size)
        #     # loss_lap_R_v = mesh_laplacian_smoothing(p3d_mesh_R_v, method='cot')
        #     # loss += loss_lap_R_v
            
        #     p3d_mesh_LR_v = Meshes(verts=[v for v in pred_LR_v_only], faces=[base_faces.to(device)]*batch_size)
        #     loss_lap_LR_v = mesh_laplacian_smoothing(p3d_mesh_LR_v, method='cot')
        #     loss += loss_lap_LR_v * 0.1
            
        # if USE_SCALE:
        #     p3d_mesh_S_v = Meshes(verts=[v for v in pred_S_v], faces=[base_faces.to(device)]*batch_size)
        #     loss_lap_S_v = mesh_laplacian_smoothing(p3d_mesh_S_v, method='cot')
        #     loss += loss_lap_S_v * 0.1

        #### L(v) - L`(V`) ####################
        # p3d_mesh_2 = Meshes(verts=[v for v in pred_v], faces=[base_faces.to(device)]*batch_size)\
        # # cot_laplacian
        # L_pred, inv_areas = cot_laplacian(p3d_mesh_2.verts_packed(), p3d_mesh_2.faces_packed())
        # pred_norm_w = torch.sparse.sum(L_pred, dim=1).to_dense().view(-1, 1)
        # pred_idx = pred_norm_w > 0
        # pred_norm_w[pred_idx] = 1.0 / norm_w[pred_idx]
        # LV_pred = L_pred.mm(p3d_mesh_2.verts_packed()) * pred_norm_w - p3d_mesh_2.verts_packed()      
                
        p3d_mesh_2 = Meshes(verts=[v for v in pred_v], faces=[base_faces.to(device)]*batch_size)
        # loss_lap = mesh_laplacian_smoothing(p3d_mesh_2, method='cot')
        # loss += loss_lap * 0.1
        
        L_pred = p3d_mesh_2.laplacian_packed()
        LV_pred = L_pred.mm(p3d_mesh_2.verts_packed())
        loss += criterion(LV_pred, LV_gt)
        #######################################


        ## uses uniform laplacian #############
        # with torch.no_grad():
        #     p3d_mesh_LR = Meshes(verts=[v for v in GT_R_v], faces=[base_faces.to(device)]*batch_size)
        #     L_gt_LR = p3d_mesh_LR.laplacian_packed()
        #     LV_gt_LR = L_gt_LR.mm(p3d_mesh_LR.verts_packed())

        # p3d_mesh_LR_v = Meshes(verts=[v for v in pred_LR_v], faces=[base_faces.to(device)]*batch_size)
        # L_pred_LR = p3d_mesh_LR_v.laplacian_packed()
        # LV_pred_LR = L_pred_LR.mm(p3d_mesh_LR_v.verts_packed())
        # loss += criterion(LV_gt_LR, LV_pred_LR)
        #######################################
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        with torch.no_grad():
            val = ((GT - pred_v)**2).mean()
                        
        pbar.set_description(f"[{MODEL_TYPE}][train][{i:06d}] loss: {val.item():.3e}")

        
        if i % num_vis_interval == 0:
            with torch.no_grad():
                eval_ = ((GT - pred_v)**2).mean()
                
                if BEST_LOSS > eval_.item():
                    if MODEL_TYPE=='mk2_2':
                        torch.save(model_r.state_dict(), f'{logdir}/model_r_best.pth')
                        torch.save(model_s.state_dict(), f'{logdir}/model_s_best.pth')
                    else:
                        torch.save(model_r_g.state_dict(), f'{logdir}/model_r_g_best.pth')
                        torch.save(model_r_l.state_dict(), f'{logdir}/model_r_l_best.pth')
                        torch.save(model_s.state_dict(), f'{logdir}/model_s_best.pth')
                
                bR = batch_R[0].detach().cpu()
                # bS = batch_S[0].detach().cpu()
                bS = batch_S_inv[0].detach().cpu()
                
                v_list=[
                    id_vert_[0].detach().cpu(), 
                    id_vert_[0].detach().cpu() @ bR.T @ bS.T,
                    pred_v[0].detach().cpu().numpy(), 
                    base_verts[0],
                ]
                f_list=[ 
                    ict_f,
                    ict_f,
                    ict_f,
                    ict_f,
                ]
            
            if USE_ROTATE:
                pred_LR_v_detach = pred_LR_v_only
                GT_R_v_detach = GT_R_v
                    
                v_list.extend([
                    pred_LR_v_detach[0].detach().cpu(),
                    GT_R_v_detach[0].detach().cpu(),
                ])
                f_list.extend([
                    ict_f,
                    ict_f,
                ])
                        
            if USE_SCALE:
                pred_S_v_detach = torch.einsum('bck,bnk->bnc', batch_R, pred_S_v)
                GT_S_v_detach = torch.einsum('bck,bnk->bnc', batch_R, GT_S_v)
                
                v_list.extend([
                    pred_S_v_detach[0].detach().cpu(),
                    GT_S_v_detach[0].detach().cpu(),
                ])
                f_list.extend([
                    ict_f,
                    ict_f,
                ])
                
            if True:
                pred_GR_v_detach = torch.einsum('bck,bnk->bnc', batch_S_inv, pred_GR_v)
                GT_GR_v_detach = id_vert_no_R_no_S
                    
                v_list.extend([
                    pred_GR_v_detach[0].detach().cpu(),
                    GT_GR_v_detach[0].detach().cpu(),
                ])
                f_list.extend([
                    ict_f,
                    ict_f,
                ])
                        
            # xyz Euler angle to rotate the mesh
            rot_list=[ [0,0,0] ]*len(v_list)
            plot_mesh_gouraud(
                v_list, f_list,
                # Cs=v_list_n, is_color=True,
                rot_list=rot_list, size=SIZE, mode='normal', logdir=logdir, name=f"vis_{i:06d}", save=True)

    if MODEL_TYPE=='mk2_2':
        torch.save(model_r.state_dict(), f'{logdir}/model_r_last.pth')
        torch.save(model_s.state_dict(), f'{logdir}/model_s_last.pth')
    else:
        torch.save(model_r_g.state_dict(), f'{logdir}/model_r_g_last.pth')
        torch.save(model_r_l.state_dict(), f'{logdir}/model_r_l_last.pth')
        torch.save(model_s.state_dict(), f'{logdir}/model_s_last.pth')

    
    if MODEL_TYPE=='mk2_2':
        model_r.eval()
        model_s.eval()
    else:
        model_r_g.eval()
        model_r_l.eval()
        model_s.eval()
    
    # with torch.no_grad():
    #     val = ((GT - pred_v)**2).mean()
    # print(f"[{MODEL_TYPE}][{i:06d}] loss: {val.item():.3e}")
    

    ###########################################################
    ## Evaluation data
    with torch.no_grad():
        eval_id_code = ict_iden_vecs[100:100+batch_size]
        eval_id_vert = torch.tensor(ict.get_id_disp(eval_id_code)+ict.neutral_verts[None]).float()
        eval_vn_list=[]
        for id_v in eval_id_vert:
            _vn = compute_vertex_normals(id_v.numpy(), ict_f)
            eval_vn_list.append(_vn)
        eval_vn_list = np.array(eval_vn_list)
        eval_vn_list = torch.tensor(eval_vn_list).float()

        eval_id_vert = eval_id_vert.to(device)
        eval_vn_list = eval_vn_list.to(device)
        
        # PE = torch.tensor(get_colors(ict_v))[None].float()
        # PE = PE.repeat(batch_size, 1, 1)
        # PE = (PE / torch.linalg.norm(PE, axis=-1, keepdims=True))
        
        eval_PE_list = []
        eval_dfn_info_list = []
        for i in range(batch_size):
            i=i+100
            pth_file = f'/data/sihun/ICT-audio2face/precompute-synth-fullhead/{i:03d}_diff3f.pth'
            fm_feat = torch.load(pth_file)
            eval_PE_list.append(fm_feat[None])
            
            pkl_file = f'/data/sihun/ICT-audio2face/precompute-synth-fullhead/{i:03d}_dfn_info.pkl'
            with open(pkl_file, 'rb') as f:
                pkl_dfn_info = pickle.load(f)
            eval_dfn_info_list.append(pkl_dfn_info)
        
        eval_PE = torch.vstack(eval_PE_list).float().to(device)
        N_PE = eval_PE.shape[-1]
        if MODEL_TYPE=='mk4':
            batch_mass, batch_L, batch_evals, batch_evecs, batch_grad_X, batch_grad_Y, batch_faces = load_batch_dfn_ino(eval_dfn_info_list, device)
    ###########################################################
    
    # eval_id_vert_ 
    eval_id_vert_, eval_vn_list_, batch_R, batch_S, batch_S_inv, id_vert_R, id_vert_no_R_no_S = get_random_data(
        eval_id_vert, eval_vn_list, batch_size, device, USE_ROTATE, USE_SCALE, USE_NORMALIZE=False)

    id_vert_in = torch.cat([eval_id_vert_, eval_vn_list, eval_PE], dim=-1).to(device)
    
    ## model prediction
    if USE_ROTATE:        
        if MODEL_TYPE=='mlp':
            pred_R, raw_out = model_r_g(id_vert_in, return_raw=True)
            pred_R_l, raw_out_l = model_r_l(id_vert_in, return_raw=True)
        elif MODEL_TYPE=='mk2':
            pred_R, raw_out = model_r_g(id_vert_in, return_raw=True)
            pred_R_l, raw_out_l = model_r_l(id_vert_in, return_raw=True)
        elif MODEL_TYPE=='mk2_2':
            (pred_R, raw_out), (pred_R_l, raw_out_l) = model_r(id_vert_in, return_raw=True)
        elif MODEL_TYPE=='mk3':
            (pred_R, raw_out), trans_feat = model_r_g(id_vert_in, return_raw=True)
            (pred_R_l, raw_out_l), trans_feat_l = model_r_l(id_vert_in, return_raw=True)
        elif MODEL_TYPE=='mk4':
            pred_R, raw_out = model_r_g(
                id_vert_in, 
                batch_mass, batch_L, batch_evals, batch_evecs,
                batch_grad_X, batch_grad_Y, batch_faces,
                return_raw=True
            )
            pred_R_l, raw_out_l = model_r_l(
                id_vert_in,
                batch_mass, batch_L, batch_evals, batch_evecs,
                batch_grad_X, batch_grad_Y, batch_faces,
                return_raw=True
            )
    
    
    if USE_SCALE:
        if MODEL_TYPE=='mlp':
            pred_S, pred_S_inv = model_s(id_vert_in, return_inv=True)
        elif MODEL_TYPE=='mk2':
            pred_S, pred_S_inv = model_s(id_vert_in, return_inv=True)
        elif MODEL_TYPE=='mk2_2':
            pred_S, pred_S_inv = model_s(id_vert_in, return_inv=True)
        elif MODEL_TYPE=='mk3':
            (pred_S, pred_S_inv), trans_feat_s = model_s(id_vert_in, return_inv=True)
        elif MODEL_TYPE=='mk4':
            pred_S, pred_S_inv = model_s(
                id_vert_in,
                batch_mass, batch_L, batch_evals, batch_evecs,
                batch_grad_X, batch_grad_Y, batch_faces,
                return_inv=True
            )

        
    ## apply to mesh
    if USE_ROTATE and not USE_SCALE:
        eval_pred_v = torch.einsum('bnck,bnk->bnc', pred_R,   eval_id_vert_.to(device)) # applying global rotation
        eval_pred_v = torch.einsum('bnck,bnk->bnc', pred_R_l, eval_pred_v) # applying local rotation
    if not USE_ROTATE and USE_SCALE:
        eval_pred_v = torch.einsum('bnck,bnk->bnc', pred_S,   eval_id_vert_.to(device))
    if USE_ROTATE and USE_SCALE:
        eval_pred_v = torch.einsum('bnck,bnk->bnc', pred_R,   eval_id_vert_.to(device)) # applying global rotation
        eval_pred_v = torch.einsum('bnck,bnk->bnc', pred_R_l, eval_pred_v) # applying local rotation
        eval_pred_v = torch.einsum('bnck,bnk->bnc', pred_S,   eval_pred_v)

    with torch.no_grad():
        eval_result = ((GT - eval_pred_v)**2).mean()
    print(f"[{MODEL_TYPE}][test][{i:06d}] loss: {eval_result.item():.3e}")

    bR = batch_R[0].detach().cpu()
    bS = batch_S_inv[0].detach().cpu()
    
    if USE_SCALE:
        GT_R_or_S = GT_S_v_detach[0].detach().cpu()
    if USE_ROTATE:
        GT_R_or_S = GT_R_v_detach[0].detach().cpu()
    if USE_ROTATE and USE_SCALE:
        GT_R_or_S = GT[0].detach().cpu()
    
    v_list=[ eval_id_vert_[0].detach().cpu() @ bR.T @ bS.T, eval_pred_v[0].detach().cpu().numpy(), base_verts[0], GT_R_or_S ]
    f_list=[ ict_f, ict_f, ict_f, ict_f ]    
    # xyz Euler angle to rotate the mesh
    rot_list=[ [0,0,0] ]*len(v_list)
    plot_mesh_gouraud(
        v_list, f_list, rot_list=rot_list,
        size=SIZE, mode='shade', logdir=logdir, name=f"test_100-{eval_result:.3e}", save=True)
    return 

def set_seed(num):
    # set seed
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(num)
    random.seed(num)

if __name__=='__main__':
    # MODEL_TYPE = 'mlp' # MLP
    # MODEL_TYPE = 'mk2' # MLP + AdaIN
    # MODEL_TYPE = 'mk2_2' # MLP + AdaIN (global branch & local branch)
    # MODEL_TYPE = 'mk3' # PointNet
    # MODEL_TYPE = 'mk4' # DiffusionNet
    
    for MODEL_TYPE in ['mlp','mk2','mk2_2','mk3','mk4']:
    # for MODEL_TYPE in ['mk2_2','mk3','mk4','mlp','mk2']:
    # for MODEL_TYPE in ['mlp', 'mk3']:
        logdir = f'new_exp3/{MODEL_TYPE}'
        os.makedirs(logdir, exist_ok=True)
        
        set_seed(42)
        main(MODEL_TYPE, logdir)
    