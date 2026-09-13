import os
import torch
import torch.nn as nn
import numpy as np
import subprocess
import trimesh
import time
from tqdm import tqdm
# import mediapy as mp
import tempfile
import pickle

from subprocess import call
os.environ['PYOPENGL_PLATFORM'] = 'osmesa' #'osmesa' # 
# os.environ['PYOPENGL_EGL_DEVICE_ID'] = 'egl'
import pyrender
try:
    import cv2
except:
    import os
    # os.sys.cmd("pip install opencv-python==4.5.5.64")
    exit(f"install opencv-python==4.5.5.64")

import sys
from pathlib import Path
abs_path = str(Path.cwd().parents[0].absolute())
sys.path+=[abs_path, f'{abs_path}/utils']

from glob import glob
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.matplotlib_rnd import xrotate
from utils.remesh_utils import ICT_face_model, procrustes_LDM
import ffmpeg

def render_mesh_helper(\
                       mesh,\
                       t_center, \
                       camera_params, \
                       rot=np.zeros(3), \
                       tex_img=None, \
                       z_offset=0, \
                       vertex_color=None,
                       H=800,
                       W=800):

    frustum = {'near': 0.001, 'far': 10.0, 'height': H, 'width': W}
    
    # mesh_copy = trimesh.Trimesh(vertices=mesh.vertices - np.array([0, -0.15, 1.5]), faces=mesh.faces)
    mesh_copy = trimesh.Trimesh(vertices=mesh.vertices - np.array([0, 0, 3.6]), faces=mesh.faces)
    # mesh_copy = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    mesh_copy.vertices[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.vertices-t_center).T).T+t_center
    # intensity = 2.0
    intensity = 1.0

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                # baseColorFactor=[0.8, 0.8, 0.8, 1.0],
                baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                metallicFactor=0.8, 
                roughnessFactor=0.8, 
            )
    
    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.vertices, faces=mesh_copy.faces, vertex_colors=vertex_color)
    # tri_mesh = trimesh.Trimesh(vertices=mesh_copy.vertices, faces=mesh_copy.faces, vertex_colors=vertex_color)
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=None, smooth=True)
    # render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material,smooth=True)

    if True: # background black
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])
    else:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])
    
    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                      fy=camera_params['f'][1],
                                      cx=camera_params['c'][0],
                                      cy=camera_params['c'][1],
                                      znear=frustum['near'],
                                      zfar=frustum['far'])
    # pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
    # camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)

    scene.add(render_mesh, pose=np.eye(4))
        
    # #camera_pose = np.eye(4)
    # camera_pose = xrotate(-6)
    # camera_pose[:3,3] = np.array([0, 0.32, 1.0+z_offset])
    # # import pdb; pdb.set_trace()
    # scene.add(camera, pose=camera_pose)
    
    camera_pose = np.eye(4)
    camera_pose[:3,3] = np.array([0, 0, 1.0-z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    # angle = np.pi / 6.0
    angle = np.pi / 4.0
    
    pos = camera_pose[:3,3]
    light_color = np.array([1.0, 1.0, 1.0]) #* 0.8
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3,3] = pos
    scene.add(light, pose=light_pose.copy())
    
    light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    # flags = pyrender.RenderFlags.NONE
#     flags = pyrender.RenderFlags.ALL_WIREFRAME # | pyrender.RenderFlags.FLAT
    
    
    # try:
    r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
    color, _ = r.render(scene, flags=flags)
    # except:
    #     print('pyrender: Failed rendering frame')
    #     color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]

def get_mesh(selection, SELECT_MESH=0):
    if selection=='ict'or selection=='ict-cap':
        ict_face = ICT_face_model()
        id_vecs = torch.load(f'{abs_path}/ict_face_pt/ict_id_vecs_test.pt').numpy()
        id_disps = ict_face.get_id_disp(id_vecs[SELECT_MESH])
        mesh_v = ict_face.neutral_verts.squeeze() + id_disps.squeeze()
        
        return mesh_v, ict_face.faces
    else:
        # if selection=='voca' or selection=='coma':
        #     mesh = dataset.voca_mesh
        
        if selection=='voca' or selection=='coma':
            with open('/data/sihun/VOCA-COMA/voca_templates.pkl', 'rb') as f:
                mesh = pickle.load(f)
        if selection=='biwi':
            biwi_trimesh = trimesh.load(f'{abs_path}/test-mesh/BIWI.ply')
            with open(f'{__abs_path__}/test-mesh/biwi_templates.pkl', 'rb') as f:
                mesh = pickle.load(f)
            mesh['face']=biwi_trimesh.faces
        elif selection=='mf_SEN' or selection=='mf_ROM' or selection=='mf':
            with open(f'/data/sihun/pca/multiface_align/mf_templates.pkl', 'rb') as f:
                mesh = pickle.load(f)

        mesh_list = [idname for idname in mesh.keys() if idname!='face']
        mesh_v = mesh[mesh_list[SELECT_MESH]]

        if selection=='biwi':
            m_align = np.load(f'{abs_path}/utils/biwi/align.npy')
            mesh_v = np.concatenate((mesh_v,np.ones((mesh_v.shape[0],1))), axis=1) @ m_align.T
        return mesh_v, mesh['face']
        
def render_sequence(
        output_path = '../notebook/tmp/pyrender/',
        npy_file = "",
        filename = None,
        mesh_type='ict',
        use_seg_color=False,
        fps=30,
        debug=False,
        case_name=None,
        save_frames=True,
        align_to_mesh_idx=None,
    ):
    #filename=filename+'.mp4'

    os.makedirs(output_path, exist_ok=True)

    if case_name is not None:
        filename = case_name
        savepath_name = os.path.join(output_path, case_name)
    elif not debug:
        #NNN=1

        NNN=2 if '/verts' in npy_file else 1
        if filename is None:
            filename=npy_file.split('/')[-NNN]
            savepath_name = os.path.join(output_path, npy_file.split('/')[-(NNN+1)])
        else:
            filename=npy_file.split('/')[-NNN]
            savepath_name = os.path.join(output_path, npy_file.split('/')[-(NNN+1)])
    else:
        filename = "pyrender-test"
        savepath_name = os.path.join(output_path, filename)
    video_fname_pred = os.path.join(savepath_name, f'{filename}.mp4')
    os.makedirs(savepath_name, exist_ok=True)
    frames_dir = os.path.join(savepath_name, 'frames')
    if save_frames:
        os.makedirs(frames_dir, exist_ok=True)
    
    print("rendering sequence...")
    print(f"\t[save path]: {output_path}")
    print(f"\t[save filename]: {filename}")
    
    H, W = 800, 800
    
    
    camera_params = {
        'c': np.array([H//2, W//2]),
        # 'c': np.array([H//4, W//4]),
        'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
        # 'f': np.array([4754.97941935 / 12, 4754.97941935 / 12])
        # 'f': np.array([4754.97941935 / 9, 4754.97941935 / 9])
        # 'f': np.array([4754.97941935 / 6.2, 4754.97941935 / 6.2]) ####
        'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])
    }
        
    # import pdb; pdb.set_trace()
    # print(template_file)
    
    
    # template = trimesh.load(template_file, process=False, maintain_order=True)
    #predicted_vertices = verts    

    frame_vertices = sorted(glob(npy_file+"/*.npy"))
    num_frames = len(frame_vertices)
    if debug:
        frame_vertices = frame_vertices[:4]
        num_frames = len(frame_vertices)
    # import pdb;pdb.set_trace()
    
    tmp_vert = np.load(frame_vertices[0])
    center = np.mean(tmp_vert, axis=0)
    tmp_v, tmp_f = get_mesh(selection=mesh_type, SELECT_MESH=0)

    align_template_v = None
    if align_to_mesh_idx is not None:
        align_template_v, _ = get_mesh(selection=mesh_type, SELECT_MESH=align_to_mesh_idx)
    # with open(pkl_file, "rb") as f:
    #     tgt_pkl = pickle.load(f)
    # predicted_vertices = tgt_pkl["pred_outs"].detach().cpu().numpy()
    # vertex_seg = tgt_pkl["pred_seg"].detach().cpu().squeeze(0)
    
    # num_frames = predicted_vertices.shape[0]
    # num_frames = 5
    
    # ## segment to color
    # if use_seg_color:
    #     vertex_seg = F.softmax(vertex_seg, dim=-1)
    #     N = vertex_seg.shape[1]
    #     # #label_range = torch.arange(vertex_color.shape[1])
    #     label_range = torch.linspace(0, 18.2, 20)
    #     # vertex_color = (vertex_seg * label_range[None]).sum(-1)
    #     vertex_color = label_range[vertex_seg.argmax(-1)]
    #     vertex_color = 0.85 - (vertex_color / N)
    # #     import pdb;pdb.set_trace()
    #     #vertex_color = plt.get_cmap("nipy_spectral")(0.96-vertex_color)[..., :3]
    #     #vertex_color = plt.get_cmap("nipy_spectral")(1-vertex_color)[..., :3]
    #     #vertex_color = plt.get_cmap("turbo")(1-vertex_color)[..., :3]
    #     #vertex_color = plt.get_cmap("rainbow")(1-vertex_color)[..., :3] #
    #     #vertex_color = plt.get_cmap("hsv")(vertex_color)[..., :3]
    #     #vertex_color = plt.get_cmap("jet")(1-vertex_color)[..., :3]
    #     vertex_color = plt.get_cmap("gist_ncar")(vertex_color)[..., :3]
    # else:
    #     vertex_color = None
    vertex_color = None
    
    
    # center = np.mean(predicted_vertices[0], axis=0)
        
    tmp_video_file_pred = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=savepath_name)
    writer_pred = cv2.VideoWriter(tmp_video_file_pred.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H), True)
    
    # render video
    frames = []
    # for i_frame in tqdm(range(num_frames)):
    for i_frame, predicted_vertices_npy in tqdm(enumerate(frame_vertices)):
        predicted_vertices = np.load(predicted_vertices_npy)
        if align_template_v is not None:
            R, t, _ = procrustes_LDM(predicted_vertices, align_template_v)
            predicted_vertices = predicted_vertices @ R.T + t
        #render_mesh = Mesh(predicted_vertices[i_frame], template.f)
        # render_mesh = trimesh.Trimesh(vertices=predicted_vertices[i_frame], faces=template.faces)
        # render_mesh = trimesh.Trimesh(vertices=predicted_vertices*0.1, faces=tmp_f)
        render_mesh = trimesh.Trimesh(vertices=predicted_vertices*0.5, faces=tmp_f)
        pred_img = render_mesh_helper(render_mesh, center, camera_params, vertex_color=vertex_color, z_offset=1.3, H=H,W=W)
        pred_img = pred_img.astype(np.uint8)

        if save_frames:
            cv2.imwrite(
                os.path.join(frames_dir, f'{i_frame:06d}.png'),
                cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
            )
        writer_pred.write(pred_img)
        # frames.append(pred_img)
    # frames = np.stack(frames, axis=0)

    writer_pred.release()
    cmd = ('ffmpeg' + ' -i {0} -pix_fmt yuv420p -qscale 0 {1}'.format(
       tmp_video_file_pred.name, video_fname_pred
    )).split()
    call(cmd)


    
    # write
    #tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=output_path)
    #mp.write_video(f"{tmp_video_file.name}", frames, fps=30)
    
    # # ffmpeg video
    # #video_filename = os.path.join(output_path, 'tmp.mp4')
    # filename=filename+'.mp4'
    # video_filename = os.path.join(output_path, filename)
    # #cmd = f'ffmpeg -y -i {tmp_video_file.name} -pix_fmt yuv420p -qscale 0 {video_filename}'
    # #call(cmd, shell=True)

    # ########### mediapy #################################
    # mp.write_video(f"{video_filename}", frames, fps=30)
    # #####################################################

        
    # 비디오 출력 스트림을 설정합니다.
    # format='rgb24'은 각 픽셀을 3개의 바이트(R, G, B)로 표현합니다.
    # pix_fmt는 코덱에서 지원하는 픽셀 형식으로 설정합니다.
    # import pdb;pdb.set_trace()
    # process = (
    #     ffmpeg
    #     .input('pipe:', format='rawvideo', s='{}x{}'.format(W, H), pix_fmt='rgb24')
    #     .output(video_filename, framerate=30, vcodec='libx264')
    #     .run(input=frames.tobytes(), capture_stdout=True, capture_stderr=True)
    # )
    # print("비디오 생성이 완료되었습니다.")

def render_sequence_meshes(args,sequence_vertices, template, out_path,predicted_vertices_path,vt, ft ,tex_img):
    num_frames = sequence_vertices.shape[0]
    file_name_pred = predicted_vertices_path.split('/')[-1].split('.')[0]
    tmp_video_file_pred = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=out_path)
    writer_pred = cv2.VideoWriter(tmp_video_file_pred.name, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (800, 800), True)

    center = np.mean(sequence_vertices[0], axis=0)
    video_fname_pred = os.path.join(out_path, file_name_pred+'.mp4')
    for i_frame in range(num_frames):
        render_mesh = Mesh(sequence_vertices[i_frame], template.f)
        if vt is not None and ft is not None:
            render_mesh.vt, render_mesh.ft = vt, ft
        pred_img = render_mesh_helper(args,render_mesh, center, tex_img=tex_img)
        pred_img = pred_img.astype(np.uint8)
        img = pred_img
        writer_pred.write(img)

    writer_pred.release()
    cmd = ('ffmpeg' + ' -i {0} -pix_fmt yuv420p -qscale 0 {1}'.format(
       tmp_video_file_pred.name, video_fname_pred)).split()
    call(cmd)

if __name__ == "__main__":
    """
    Renders the 12 cases from report/render_retarget_request_2026-07-27.md
    using render_trimesh.py's own camera/lighting/resolution settings.

    Vertex data for all cases except exp1_1 (raw mf test source) must first
    be generated by ../vis_CBD_retarget_fig.py.

    Setup (see set_render.sh):
        apt-get install -y llvm-6.0 freeglut3 freeglut3-dev libosmesa6-dev
        pip install pyrender ffmpeg-python
        pip install pyopengl==3.1.4

        cd render
        python render_trimesh_fig.py
    """
    # Override via env vars for NC/NFS/NFR baselines, e.g.:
    #   CKPT_NAME=2025-10-20-00-15-40-CBD METHOD_NAME=NC python render_trimesh_fig.py
    CKPT_NAME = os.environ.get('CKPT_NAME', '2026-04-02-02-04-44-NGBCv5-dist')
    METHOD_NAME = os.environ.get('METHOD_NAME', 'ours')  # subfolder under vis_render_fig
    VIS_BASE = f'/source/sihun/NeuralFacialAnimation/vis_CBD/{CKPT_NAME}'
    OUTPUT_PATH = f'/source/sihun/NeuralFacialAnimation/vis_CBD/vis_render_fig/{METHOD_NAME}'

    RAW_MF_TEST_SOURCE = (
        '/data/sihun/multiface_align/ROM/test/vertices_npy/'
        'm--20190828--1318--002645310--GHS/EXP_free_face'
    )
    MF_TEST_IDX = 12  # mf_templates.pkl index for this mf test identity (matches vis_CBD_retarget_fig.py)

    # exp1_2/1_3/1_4/2_2/2_3 come from save_vis_data, which appends '-masked' to its
    # output folder (opts.use_t_mask is forced True in vis_CBD_retarget_fig.py's __main__).
    # The cyclic cases (produced by retarget_from_verts_folder) and exp2_1 (produced by
    # save_raw_source) do not carry that suffix.
    #
    # (case_name, npy_file, mesh_type, align_to_mesh_idx)
    # align_to_mesh_idx: only exp1_1 needs this -- it reads raw (unaligned) capture data
    # directly, bypassing dataloader_CBD.EvalDataset.get_mf_ROM (which now procrustes-aligns
    # each frame to its identity's neutral template). Aligning it here keeps it visually
    # consistent with exp1_2/1_3/1_4, whose model input is the aligned version of this
    # same source motion.
    cases = [
        ('exp1_1_mf_original',            RAW_MF_TEST_SOURCE,                                     'mf_ROM', MF_TEST_IDX),
        ('exp1_2_mf_to_ict_m00',          f'{VIS_BASE}/exp1_2_mf_to_ict_m00-masked/verts',         'ict',    None),
        ('exp1_3_mf_to_ict_m02',          f'{VIS_BASE}/exp1_3_mf_to_ict_m02-masked/verts',         'ict',    None),
        ('exp1_4_mf_to_mf_train',         f'{VIS_BASE}/exp1_4_mf_to_mf_train-masked/verts',        'mf_ROM', None),
        ('exp1_5_cyclic_ict_m00_to_mf',   f'{VIS_BASE}/exp1_5_cyclic_ict_m00_to_mf/verts',          'mf_ROM', None),
        ('exp1_6_cyclic_ict_m02_to_mf',   f'{VIS_BASE}/exp1_6_cyclic_ict_m02_to_mf/verts',          'mf_ROM', None),
        ('exp1_7_cyclic_mf_train_to_mf',  f'{VIS_BASE}/exp1_7_cyclic_mf_train_to_mf/verts',         'mf_ROM', None),
        ('exp2_1_ict_m02_original',       f'{VIS_BASE}/exp2_1_ict_m02_original/verts',              'ict',    None),
        ('exp2_2_ict_m02_to_ict_m05',     f'{VIS_BASE}/exp2_2_ict_m02_to_ict_m05-masked/verts',     'ict',    None),
        ('exp2_3_ict_m02_to_mf',          f'{VIS_BASE}/exp2_3_ict_m02_to_mf-masked/verts',          'mf_ROM', None),
        ('exp2_4_cyclic_ict_m05_to_ict_m02', f'{VIS_BASE}/exp2_4_cyclic_ict_m05_to_ict_m02/verts',  'ict',    None),
        ('exp2_5_cyclic_mf_to_ict_m02',   f'{VIS_BASE}/exp2_5_cyclic_mf_to_ict_m02/verts',          'ict',    None),
    ]

    for case_name, npy_file, mesh_type, align_to_mesh_idx in cases:
        if not os.path.isdir(npy_file):
            print(f'[skip] {case_name}: {npy_file} does not exist yet')
            continue
        render_sequence(
            output_path=OUTPUT_PATH,
            npy_file=npy_file,
            case_name=case_name,
            mesh_type=mesh_type,
            save_frames=True,
            align_to_mesh_idx=align_to_mesh_idx,
        )
