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
from utils.remesh_utils import ICT_face_model
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
            with open(f'{abs_path}/test-mesh/biwi_templates.pkl', 'rb') as f:
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
    ):
    #filename=filename+'.mp4'
    
    os.makedirs(output_path, exist_ok=True)
    
    if not debug:
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
        #render_mesh = Mesh(predicted_vertices[i_frame], template.f)
        # render_mesh = trimesh.Trimesh(vertices=predicted_vertices[i_frame], faces=template.faces)
        # render_mesh = trimesh.Trimesh(vertices=predicted_vertices*0.1, faces=tmp_f)
        render_mesh = trimesh.Trimesh(vertices=predicted_vertices*0.5, faces=tmp_f)
        pred_img = render_mesh_helper(render_mesh, center, camera_params, vertex_color=vertex_color, z_offset=1.3, H=H,W=W)
        pred_img = pred_img.astype(np.uint8)
        
        # cv2.imwrite(
        #     os.path.join(savepath_name, f'{i_frame:06d}.png'), 
        #     cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
        # )
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
    https://pyrender.readthedocs.io/en/latest/install/index.html#installmesa
    
    apt update
    apt upgrade ffmpeg

    wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb
    dpkg -i ./mesa_18.3.3-0.deb || true
    apt install -f

    apt-get install llvm-6.0 freeglut3 freeglut3-dev libosmesa6-dev
    pip install pyrender pyopengl==3.1.4 ffmpeg-python
    ~~pip install mediapy~~ # not used
    
    cd render
    python render_trimesh.py
    """
    #scp -P 31444 -r root@143.248.249.193:/source/sihun/NeuralFacialAnimation/notebook/tmp .
    
    # render_sequence()
    # output_path='../notebook/tmp/pyrender/'
    output_path='/source/sihun/NeuralFacialAnimation/video/'
#     output_path='../notebook/tmp/pyrender-seg/'
    use_seg_color=False
#     render_sequence(
#         output_path = output_path,
#         # template_file = "/data/sihun/multiface_align/obj/m--20190828--1318--002645310--GHS_mesh.obj",
#         npy_file = "/source/sihun/NeuralFacialAnimation/eval_CBD/2024-08-18-23-32-29-all/mf_ROM_test-to-mf_ROM_test",
#         filename = "pyrender--MF_test",
#         use_seg_color=use_seg_color,
#         mesh_type='mf_ROM',
#         # debug=True,
#     )
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/eval_CBD/2024-07-08-06-27-12-all/mf_ROM_test-to-mf_ROM_test",
#         filename = "pyrender--MF_test",
#         use_seg_color=use_seg_color,
#         mesh_type='mf_ROM',
#         # debug=True,
#     )
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/eval_CBD/2024-06-09-10-57-34-all/mf_ROM_test-to-mf_ROM_test",
#         filename = "pyrender--MF_test",
#         use_seg_color=use_seg_color,
#         mesh_type='mf_ROM',
#         # debug=True,
#     )

    ########################################################### design study
    # saved at video/NeuralFacialAnimation
    
    # data_name = 'coma-masked/verts'
    data_name = 'mf_ROM-masked-laplacian/verts'
    eval_path = '/source/sihun/NeuralFacialAnimation/eval_CBD/'
    #mesh_file_paths_w_relu = sorted(glob(eval_path+f'2025-10-23-17-32-49-NGBCv5-eval/{data_name}/*.npy'))
    
    
    
    render_sequence(
        output_path = os.path.join(output_path, 'design_study--ours-dist'),
        npy_file = os.path.join(eval_path,'2026-04-12-04-07-19-NGBCv5-eval', data_name),
        use_seg_color=use_seg_color, mesh_type='mf',
        filename=f'{data_name.split("/")[0]}--ours'
    )
    # render_sequence(
    #     output_path = os.path.join(output_path, 'design_study--ours-dist'),
    #     npy_file = os.path.join(eval_path,'2025-10-23-13-03-54-NGBCv5-eval', data_name),
    #     use_seg_color=use_seg_color, mesh_type='mf',
    #     filename=f'{data_name.split("/")[0]}--ours'
    # )
    
    # render_sequence(
    #     output_path = os.path.join(output_path, 'design_study--ours'),
    #     npy_file = os.path.join(eval_path,'2025-10-23-13-03-54-NGBCv5-eval', data_name),
    #     use_seg_color=use_seg_color, mesh_type='mf',
    #     filename=f'{data_name.split("/")[0]}--ours'
    # )

#     render_sequence(
#         output_path = os.path.join(output_path, 'design_study--w_ReLU'),
#         npy_file = os.path.join(eval_path,'2025-10-23-17-32-49-NGBCv5-eval', data_name),
#         use_seg_color=use_seg_color, mesh_type='mf',
#         filename=f'{data_name.split("/")[0]}--w_ReLU'
#     )
    
#     render_sequence(
#         output_path = os.path.join(output_path, 'design_study--no_act'),
#         npy_file = os.path.join(eval_path,'2025-09-30-16-28-51-NGBCv5-eval', data_name),
#         use_seg_color=use_seg_color, mesh_type='mf',
#         filename=f'{data_name.split("/")[0]}--no_act'
#     )
    
#     render_sequence(
#         output_path = os.path.join(output_path, 'design_study--w_softplus'),
#         npy_file = os.path.join(eval_path,'2025-09-28-14-50-34-NGBCv5-eval', data_name),
#         use_seg_color=use_seg_color, mesh_type='mf',
#         filename=f'{data_name.split("/")[0]}--w_softplus'
#     )
    
#     render_sequence(
#         output_path = os.path.join(output_path, 'design_study--w_ELU'),
#         npy_file = os.path.join(eval_path,'2025-10-08-02-55-02-NGBCv5-eval', data_name),
#         use_seg_color=use_seg_color, mesh_type='mf',
#         filename=f'{data_name.split("/")[0]}--w_ELU'
#     )
    
#     render_sequence(
#         output_path = os.path.join(output_path, 'design_study--delta'),
#         npy_file = os.path.join(eval_path,'2025-10-22-16-22-45-NGBCv5-eval', data_name),
#         use_seg_color=use_seg_color, mesh_type='mf',
#         filename=f'{data_name.split("/")[0]}--delta'
#     )
    
#     render_sequence(
#         output_path = os.path.join(output_path, 'design_study--matrix'),
#         npy_file = os.path.join(eval_path,'2025-10-22-16-26-26-NGBCv5-eval', data_name),
#         use_seg_color=use_seg_color, mesh_type='mf',
#         filename=f'{data_name.split("/")[0]}--matrix'
#     )
    
#     render_sequence(
#         output_path = os.path.join(output_path, 'design_study--softPOU'),
#         npy_file = os.path.join(eval_path,'2025-10-03-22-40-03-NGBCv5-eval', data_name),
#         use_seg_color=use_seg_color, mesh_type='mf',
#         filename=f'{data_name.split("/")[0]}--softPOU'
#     )
    
#     render_sequence(
#         output_path = os.path.join(output_path, 'design_study--wo_mask'),
#         npy_file = os.path.join(eval_path,'2025-09-26-10-16-32-NGBCv5-eval', data_name),
#         use_seg_color=use_seg_color, mesh_type='mf',
#         filename=f'{data_name.split("/")[0]}--wo_mask'
#     )
    ###########################################################
    
    ########################################################### GT
    # saved at video/NeuralFacialAnimation
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/GT-ict-cap_test-to-ict-cap_test_00",
    #     use_seg_color=use_seg_color, mesh_type='ict',
    # )
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/GT-ict-cap_test-to-ict-cap_test_01",
    #     use_seg_color=use_seg_color, mesh_type='ict',
    # )    
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/GT-mf_ROM_test-to-mf_ROM_test_00",
    #     use_seg_color=use_seg_color, mesh_type='mf_ROM',
    # )
    ###########################################################
    
    ########################################################### Ours
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-11-09-21-21-43-NGBCv5-50/ict-cap-ID_000_test-to-ict-cap-ID_000_test_01-masked/verts",
    #     use_seg_color=use_seg_color, mesh_type='ict',
    # )
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-11-09-21-21-43-NGBCv5-50/ict-cap-ID_000_test-to-ict-cap-ID_003_test_01-masked/verts",
    #     use_seg_color=use_seg_color, mesh_type='ict',
    # )
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-11-09-21-21-43-NGBCv5-50/ict-cap-ID_000_test-to-ict-cap-ID_009_test_01-masked/verts",
    #     use_seg_color=use_seg_color, mesh_type='ict',
    # )
    
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-11-09-21-21-43-NGBCv5-50/ict-cap-ID_002_test-to-ict-cap-ID_000_test_00-masked/verts",
    #     use_seg_color=use_seg_color, mesh_type='ict',
    # )
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-11-09-21-21-43-NGBCv5-50/ict-cap-ID_002_test-to-ict-cap-ID_002_test_00-masked/verts",
    #     use_seg_color=use_seg_color, mesh_type='ict',
    # )
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-11-09-21-21-43-NGBCv5-50/ict-cap-ID_002_test-to-ict-cap-ID_005_test_00-masked/verts",
    #     use_seg_color=use_seg_color, mesh_type='ict',
    # )
    
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-11-09-21-21-43-NGBCv5-50/mf_ROM_test-to-ict-cap_test-ID_000-masked/verts",
    #     use_seg_color=use_seg_color, mesh_type='ict',
    # )
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-11-09-21-21-43-NGBCv5-50/mf_ROM_test-to-mf_ROM_test-masked/verts",
    #     use_seg_color=use_seg_color, mesh_type='mf_ROM',
    # )
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-11-09-21-21-43-NGBCv5-50/mf_ROM_test-to-mf_ROM_train-masked/verts",
    #     use_seg_color=use_seg_color, mesh_type='mf_ROM',
    # )



    
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/nfr/ict-cap-ID_000_test-to-mf_ROM-ID_012_test_01",
    #     use_seg_color=use_seg_color, mesh_type='mf_ROM',
    #     filename='nfr'
    # )
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/nfr/ict-cap-ID_002_test-to-mf_ROM-ID_012_test_00",
    #     use_seg_color=use_seg_color, mesh_type='mf_ROM',
    #     filename='nfr-'
    # )
    
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/nfs/ict-cap-ID_000_test-to-mf_ROM-ID_012_test_01",
    #     use_seg_color=use_seg_color, mesh_type='mf_ROM',
    #     filename='nfs'
    # )
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/nfs/ict-cap-ID_002_test-to-mf_ROM-ID_012_test_00",
    #     use_seg_color=use_seg_color, mesh_type='mf_ROM',
    #     filename='nfs'
    # )

    ########################################################### NC
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-10-20-00-15-40-CBD/ict-cap-ID_000_test-to-ict-cap-ID_000_test_01-masked/verts",
    #     use_seg_color=use_seg_color, mesh_type='ict',
    # )
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-10-20-00-15-40-CBD/ict-cap-ID_000_test-to-ict-cap-ID_003_test_01-masked/verts",
    #     use_seg_color=use_seg_color, mesh_type='ict',
    # )
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-10-20-00-15-40-CBD/ict-cap-ID_000_test-to-ict-cap-ID_009_test_01-masked/verts",
    #     use_seg_color=use_seg_color, mesh_type='ict',
    # )
    
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-10-20-00-15-40-CBD/ict-cap-ID_002_test-to-ict-cap-ID_000_test_00-masked/verts",
    #     use_seg_color=use_seg_color, mesh_type='ict',
    # )
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-10-20-00-15-40-CBD/ict-cap-ID_002_test-to-ict-cap-ID_002_test_00-masked/verts",
    #     use_seg_color=use_seg_color, mesh_type='ict',
    # )
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-10-20-00-15-40-CBD/ict-cap-ID_002_test-to-ict-cap-ID_005_test_00-masked/verts",
    #     use_seg_color=use_seg_color, mesh_type='ict',
    # )
    
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-10-20-00-15-40-CBD/mf_ROM_test-to-ict-cap_test-ID_000-masked/verts",
    #     use_seg_color=use_seg_color, mesh_type='ict',
    # )
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-10-20-00-15-40-CBD/mf_ROM_test-to-mf_ROM_test-masked/verts",
    #     use_seg_color=use_seg_color, mesh_type='mf_ROM',
    # )
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-10-20-00-15-40-CBD/mf_ROM_test-to-mf_ROM_train-masked/verts",
    #     use_seg_color=use_seg_color, mesh_type='mf_ROM',
    # )


    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-10-20-00-15-40-CBD/ict-cap-ID_000_test-to-mf_ROM-ID_012_test_01-masked/verts",
    #     use_seg_color=use_seg_color, mesh_type='mf_ROM',
    # )
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-10-20-00-15-40-CBD/ict-cap-ID_002_test-to-mf_ROM-ID_012_test_00-masked/verts",
    #     use_seg_color=use_seg_color, mesh_type='mf_ROM',
    # )
    ##########################################################
    

    

    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-11-09-21-21-43-NGBCv5-50/ict-cap-ID_000_test-to-mf_ROM-ID_012_test_01-masked/verts",
    #     use_seg_color=use_seg_color, mesh_type='mf_ROM',
    # )
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-11-09-21-21-43-NGBCv5-50/ict-cap-ID_002_test-to-mf_ROM-ID_012_test_00-masked/verts",
    #     use_seg_color=use_seg_color, mesh_type='mf_ROM',
    # )
    ##########################################################
    
#     ########################################################### Ours (prev)
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-10-22-22-30-30-NGBCv5/ict-cap-ID_000_test-to-ict-cap-ID_000_test_01-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-10-22-22-30-30-NGBCv5/ict-cap-ID_000_test-to-ict-cap-ID_003_test_01-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-10-22-22-30-30-NGBCv5/ict-cap-ID_000_test-to-ict-cap-ID_009_test_01-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
    
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-10-22-22-30-30-NGBCv5/ict-cap-ID_002_test-to-ict-cap-ID_000_test_00-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-10-22-22-30-30-NGBCv5/ict-cap-ID_002_test-to-ict-cap-ID_002_test_00-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-10-22-22-30-30-NGBCv5/ict-cap-ID_002_test-to-ict-cap-ID_005_test_00-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
    
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-10-22-22-30-30-NGBCv5/mf_ROM_test-to-ict-cap_test-ID_000-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-10-22-22-30-30-NGBCv5/mf_ROM_test-to-mf_ROM_test-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='mf_ROM',
#     )
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2025-10-22-22-30-30-NGBCv5/mf_ROM_test-to-mf_ROM_train-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='mf_ROM',
#     )
    ###########################################################
    
    ########################################################### NFR (prev)
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/exp_019_ICT_MF-jacob_NFR/ict-cap-ID_000_test-to-ict-cap-ID_000_test_01-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/exp_019_ICT_MF-jacob_NFR/ict-cap-ID_000_test-to-ict-cap-ID_003_test_01-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/exp_019_ICT_MF-jacob_NFR/ict-cap-ID_000_test-to-ict-cap-ID_009_test_01-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
    
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/exp_019_ICT_MF-jacob_NFR/ict-cap-ID_002_test-to-ict-cap-ID_000_test_00-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/exp_019_ICT_MF-jacob_NFR/ict-cap-ID_002_test-to-ict-cap-ID_002_test_00-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/exp_019_ICT_MF-jacob_NFR/ict-cap-ID_002_test-to-ict-cap-ID_005_test_00-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
    
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/exp_019_ICT_MF-jacob_NFR/mf_ROM_test-to-ict-cap_test-ID_000-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/exp_019_ICT_MF-jacob_NFR/mf_ROM_test-to-mf_ROM_test-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='mf_ROM',
#     )
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/exp_019_ICT_MF-jacob_NFR/mf_ROM_test-to-mf_ROM_train-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='mf_ROM',
#     )
    ###########################################################
    
    ########################################################### NFS (prev)
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2024-08-18-23-32-29-all/ict-cap-ID_000_test-to-ict-cap-ID_000_test_01-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2024-08-18-23-32-29-all/ict-cap-ID_000_test-to-ict-cap-ID_003_test_01-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2024-08-18-23-32-29-all/ict-cap-ID_000_test-to-ict-cap-ID_009_test_01-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
    
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2024-08-18-23-32-29-all/ict-cap-ID_002_test-to-ict-cap-ID_000_test_00-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2024-08-18-23-32-29-all/ict-cap-ID_002_test-to-ict-cap-ID_002_test_00-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2024-08-18-23-32-29-all/ict-cap-ID_002_test-to-ict-cap-ID_005_test_00-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
    
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2024-08-18-23-32-29-all/mf_ROM_test-to-ict-cap_test-ID_000-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2024-08-18-23-32-29-all/mf_ROM_test-to-mf_ROM_test-masked/verts",
    #     use_seg_color=use_seg_color, mesh_type='mf_ROM',
    # )
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/2024-08-18-23-32-29-all/mf_ROM_test-to-mf_ROM_train-masked/verts",
#         use_seg_color=use_seg_color, mesh_type='mf_ROM',
#     )
    ###########################################################
    


    ########################################################### NFR 
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/nfr/ict-cap-ID_000_test-to-ict-cap-ID_000_test_01",
    #     use_seg_color=use_seg_color, mesh_type='ict',
    # )
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/nfr/ict-cap-ID_000_test-to-ict-cap-ID_003_test_01",
    #     use_seg_color=use_seg_color, mesh_type='ict',
    # )
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/nfr/ict-cap-ID_000_test-to-ict-cap-ID_009_test_01",
    #     use_seg_color=use_seg_color, mesh_type='ict',
    # )
    
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/nfr/ict-cap-ID_002_test-to-ict-cap-ID_000_test_00",
    #     use_seg_color=use_seg_color, mesh_type='ict',
    # )
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/nfr/ict-cap-ID_002_test-to-ict-cap-ID_002_test_00",
    #     use_seg_color=use_seg_color, mesh_type='ict',
    # )
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/nfr/ict-cap-ID_002_test-to-ict-cap-ID_005_test_00",
    #     use_seg_color=use_seg_color, mesh_type='ict',
    # )
    
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/nfr/mf_ROM_test-to-ict-cap_test-ID_000",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/nfr/mf_ROM_test-to-mf_ROM_test",
#         use_seg_color=use_seg_color, mesh_type='mf_ROM',
#     )
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/nfr/mf_ROM_test-to-mf_ROM_train",
#         use_seg_color=use_seg_color, mesh_type='mf_ROM',
#     )
    ##########################################################
    
    ########################################################### NFS
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/nfs/ict-cap-ID_000_test-to-ict-cap-ID_000_test_01",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/nfs/ict-cap-ID_000_test-to-ict-cap-ID_003_test_01",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/nfs/ict-cap-ID_000_test-to-ict-cap-ID_009_test_01",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
    
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/nfs/ict-cap-ID_002_test-to-ict-cap-ID_000_test_00",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/nfs/ict-cap-ID_002_test-to-ict-cap-ID_002_test_00",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
#     render_sequence(
#         output_path = output_path,
#         npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/nfs/ict-cap-ID_002_test-to-ict-cap-ID_005_test_00",
#         use_seg_color=use_seg_color, mesh_type='ict',
#     )
    
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/nfs/mf_ROM_test-to-ict-cap_test-ID_000",
    #     use_seg_color=use_seg_color, mesh_type='ict',
    # )
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/nfs/mf_ROM_test-to-mf_ROM_test",
    #     use_seg_color=use_seg_color, mesh_type='mf_ROM',
    # )
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "/source/sihun/NeuralFacialAnimation/vis_CBD/nfs/mf_ROM_test-to-mf_ROM_train",
    #     use_seg_color=use_seg_color, mesh_type='mf_ROM',
    # )
    ##########################################################

    
#     render_sequence(
#         output_path = output_path,
#         template_file = "/source/sihun/NeuralFacialAnimation/test-mesh/ict_neutral_rescale.obj",
#         pkl_file = "/source/sihun/NeuralFacialAnimation/notebook/tmp/arkit/arkit_CSH_shade-gt_frames.pkl",
#         filename = "pyrender-"+"arkit_CSH_shade-gt_frames",
#         use_seg_color=use_seg_color,
#     )

#     render_sequence(
#         output_path = output_path,
#         template_file = "/source/sihun/NeuralFacialAnimation/test-mesh/FLAME_sample_align.obj",
#         pkl_file = "/source/sihun/NeuralFacialAnimation/notebook/tmp/arkit/arkit_CSH_shade-FLAME_test_M.pkl",
#         filename = "pyrender-"+"arkit_CSH_shade-FLAME_test_M",
#         use_seg_color=use_seg_color,
#     )
#     render_sequence(
#         output_path = output_path,
#         template_file = "/source/sihun/NeuralFacialAnimation/test-mesh/FLAME_sample_align.obj",
#         pkl_file = "/source/sihun/NeuralFacialAnimation/notebook/tmp/arkit/arkit_CSH_shade-FLAME_test_F.pkl",
#         filename = "pyrender-"+"arkit_CSH_shade-FLAME_test_F",
#         use_seg_color=use_seg_color,
#     )
    # render_sequence(
    #     output_path = output_path,
    #     template_file = "/source/sihun/NeuralFacialAnimation/test-mesh/BIWI.ply",
    #     pkl_file = "/source/sihun/NeuralFacialAnimation/notebook/tmp/arkit/arkit_CSH_shade-BIWI_test1.pkl",
    #     filename = "pyrender-"+"arkit_CSH_shade-BIWI_test1",
    #     use_seg_color=use_seg_color,
    # )
    # render_sequence(
    #     output_path = output_path,
    #     template_file = "/source/sihun/NeuralFacialAnimation/test-mesh/face-reference.obj",
    #     pkl_file = "/source/sihun/NeuralFacialAnimation/notebook/tmp/arkit/arkit_CSH_shade-face-reference_test.pkl",
    #     filename = "pyrender-"+"arkit_CSH_shade-face-reference_test",
    #     use_seg_color=use_seg_color,
    # )
#     render_sequence(
#         output_path = output_path,
#         template_file = "/source/sihun/NeuralFacialAnimation/test-mesh/head-reference.obj",
#         pkl_file = "/source/sihun/NeuralFacialAnimation/notebook/tmp/arkit/arkit_CSH_shade-head-reference_test.pkl",
#         filename = "pyrender-"+"arkit_CSH_shade-head-reference_test",
#         use_seg_color=use_seg_color,
#     )
#     render_sequence(
#         output_path = output_path,
#         template_file = "/source/sihun/NeuralFacialAnimation/test-mesh/ict_live_100_000.obj",
#         pkl_file = "/source/sihun/NeuralFacialAnimation/notebook/tmp/arkit/arkit_CSH_shade-ICT_test.pkl",
#         filename = "pyrender-"+"arkit_CSH_shade-ICT_test",
#         use_seg_color=use_seg_color,
#     )
    # render_sequence(
    #     output_path = output_path,
    #     template_file = "/data/sihun/multiface_align/obj/m--20190828--1318--002645310--GHS_mesh.obj",
    #     pkl_file = "/source/sihun/NeuralFacialAnimation/notebook/tmp/arkit/arkit_CSH_shade-MF_test.pkl",
    #     filename = "pyrender-"+"arkit_CSH_shade-MF_test",
    #     use_seg_color=use_seg_color,
    # )
#     render_sequence(
#         output_path = output_path,
#         template_file = "/data/sihun/multiface_align/obj/m--20190529--1300--002421669--GHS_mesh.obj",
#         pkl_file = "/source/sihun/NeuralFacialAnimation/notebook/tmp/arkit/arkit_CSH_shade-MF_test2.pkl",
#         filename = "pyrender-"+"arkit_CSH_shade-MF_test2",
#         use_seg_color=use_seg_color,
#     )
#     render_sequence(
#         output_path = output_path,
#         template_file = "/source/sihun/NeuralFacialAnimation/test-mesh/morphy-align.obj",
#         pkl_file = "/source/sihun/NeuralFacialAnimation/notebook/tmp/arkit/arkit_CSH_shade-morphy_test.pkl",
#         filename = "pyrender-"+"arkit_CSH_shade-morphy_test",
#         use_seg_color=use_seg_color,
#     )
    
    
#     render_sequence(
#         output_path = output_path,
#         template_file = "/source/sihun/NeuralFacialAnimation/test-mesh/malcolm-align.obj",
#         pkl_file = "/source/sihun/NeuralFacialAnimation/notebook/tmp/arkit/arkit_CSH_shade-malcolm_test.pkl",
#         filename = "pyrender-"+"arkit_CSH_shade-malcolm_test",
#         use_seg_color=use_seg_color,
#     )
    # render_sequence(
    #     output_path = output_path,
    #     template_file = "/source/sihun/NeuralFacialAnimation/test-mesh/biwi_M5.obj",
    #     pkl_file = "/source/sihun/NeuralFacialAnimation/notebook/tmp/arkit/arkit_CSH_shade-biwi_test-M5.pkl",
    #     filename = "pyrender-"+"arkit_CSH_shade-biwi_test-M5",
    #     use_seg_color=use_seg_color,
    # )
#     render_sequence(
#         output_path = output_path,
#         template_file = "/source/sihun/NeuralFacialAnimation/test-mesh/mary-align.obj",
#         pkl_file = "/source/sihun/NeuralFacialAnimation/notebook/tmp/arkit/arkit_CSH_shade-mary_test.pkl",
#         filename = "pyrender-"+"arkit_CSH_shade-mary_test",
#         use_seg_color=use_seg_color,
#     )