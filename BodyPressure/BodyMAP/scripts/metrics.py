'''
This creates a tensorboard file writing the below metrics:
3D Pose:
    MPJPE
    PVE
3D Shape
    Height Error 
    Chest Error
    Waist Error 
    Hips Error 
3D Pressure Map 
    v2vP
    v2vP 1EA 
    v2vP 2EA
'''

import argparse
import json 
import numpy as np 
import os 
from scipy.spatial import ConvexHull
import trimesh
import shutil 
import sys 
sys.path.append('../PMM')
from tqdm import tqdm 

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


from SLPDataset import SLPDataset
import utils
import viz_utils
from constants import DEVICE, FACES, X_BUMP, Y_BUMP, BASE_PATH 


TRANS_PREV_GT = np.identity(4)
TRANS_PREV_GT[0, 3] = -2.0
TRANS_NEXT_GT = np.identity(4)
TRANS_NEXT_GT[0, 3] = -2.0
TRANS_NEXT_GT[1, 3] = 1.0

TRANS_PREV_PD = np.identity(4)
TRANS_NEXT_PD = np.identity(4)
TRANS_NEXT_PD[1, 3] = 1.0

data_path = os.path.join(BASE_PATH, 'BodyPressure', 'data_BP')
data_lines = utils.load_data_lines(os.path.join(BASE_PATH, 'BodyPressure', 'BodyMAP', 'data_files', 'real_val.txt'))

PI = np.load(os.path.join(BASE_PATH, 'BodyPressure', 'data_BP', 'parsed', 'segmented_mesh_idx_faces.npy'), allow_pickle=True).item()
EA1 = np.load(os.path.join(BASE_PATH, 'BodyPressure', 'data_BP', 'parsed', 'EA1.npy'), allow_pickle=True)
EA2 = np.load(os.path.join(BASE_PATH, 'BodyPressure', 'data_BP', 'parsed', 'EA2.npy'), allow_pickle=True)
FACES_NP = FACES.squeeze(0).to("cpu").numpy()
FACES = FACES.squeeze(0).long()

head_top_face_idx = 435
head_top_bc = torch.tensor([0.0, 1.0, 0.0]).float().to(DEVICE)
left_heel_face_idx = 5975
left_heel_bc = torch.tensor([0.0, 0.0, 1.0]).float().to(DEVICE)
chest_face_index = 11885
chest_bcs = torch.tensor([0.0, 0.0, 1.0]).float().to(DEVICE)
belly_face_index = 6833
belly_bcs = torch.tensor([0.0, 0.0, 1.0]).float().to(DEVICE)
hips_face_index = 1341
hips_bcs = torch.tensor([0.0, 1.0, 0.0]).float().to(DEVICE)


def create_metric_dict(PI):
    return {
            'MPJPE' : torch.tensor(0).float(),
            'PVE' : torch.tensor(0).float(),
            'height' : torch.tensor(0).float(),
            'chest' : torch.tensor(0).float(),
            'waist' : torch.tensor(0).float(),
            'hips' : torch.tensor(0).float(),
            'v2vP' : torch.tensor(0).float(),
            'v2vP_1EA' : torch.tensor(0).float(),
            'v2vP_2EA' : torch.tensor(0).float(),
            'count' : 0,
            'parts_v2vP' : {key: torch.tensor(0).float() for key in PI},
            }


def get_triangle_area_vert_weight(verts, faces, verts_idx_red = None):
    tri_verts = verts[faces, :]
    a = np.linalg.norm(tri_verts[:, 0] - tri_verts[:, 1], axis=1)
    b = np.linalg.norm(tri_verts[:, 1] - tri_verts[:, 2], axis=1)
    c = np.linalg.norm(tri_verts[:, 2] - tri_verts[:, 0], axis=1)
    s = (a + b + c) / 2
    A = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0.0))
    faces_flat = np.array(faces).flatten().astype(np.int64)
    weights = np.repeat(A, 3)
    area_sum = np.bincount(faces_flat, weights=weights, minlength=verts.shape[0])
    area_count = np.bincount(faces_flat, minlength=verts.shape[0])
    area_avg = np.zeros_like(area_sum, dtype=np.float64)
    np.divide(area_sum, area_count, out=area_avg, where=area_count > 0)
    area_avg_red = area_avg[area_avg > 0]
    if area_avg_red.size == 0:
        norm_area_avg = area_avg
    else:
        norm_area_avg = area_avg / np.sum(area_avg_red)
        norm_area_avg = norm_area_avg * area_avg_red.size
    if verts_idx_red is not None:
        try:
            norm_area_avg = norm_area_avg[verts_idx_red]
        except:
            norm_area_avg = norm_area_avg[verts_idx_red[:-1]]
    return norm_area_avg

def get_area_norm(verts, gt=False):
    if gt:
        trans_prev, trans_next = TRANS_PREV_GT.copy(), TRANS_NEXT_GT.copy()
    else:
        trans_prev, trans_next = TRANS_PREV_PD.copy(), TRANS_NEXT_PD.copy()
    
    verts_edit = verts.copy()
    
    smpl_verts_quad = np.concatenate((verts_edit, np.ones((verts.shape[0], 1))), axis = 1)
    smpl_verts_quad = np.swapaxes(smpl_verts_quad, 0, 1)
    smpl_verts = np.swapaxes(np.matmul(trans_prev, smpl_verts_quad), 0, 1)[:, 0:3] # gt over pressure mat

    vertices_pimg = np.array(smpl_verts)
    faces_pimg = np.array(FACES_NP.copy())

    vertices_pimg[:, 0] = vertices_pimg[:, 0] + trans_next[0, 3] - trans_prev[0, 3]
    vertices_pimg[:, 1] = vertices_pimg[:, 1] + trans_next[1, 3] - trans_prev[1, 3]

    area_norm = get_triangle_area_vert_weight(vertices_pimg, faces_pimg, None)
    return area_norm


def get_EA_pressure(pmap, ea):
    pmap_np = pmap.detach().cpu().numpy().astype(np.float64)
    lengths = np.array([len(x) for x in ea], dtype=np.int64)
    indices = np.concatenate([np.asarray(x, dtype=np.int64) for x in ea])
    bounds = np.concatenate(([0], np.cumsum(lengths[:-1])))
    sums = np.add.reduceat(pmap_np[indices], bounds)
    return torch.from_numpy(sums / lengths).float()

# Code borrowed from SHAPY (https://github.com/muelea/shapy/blob/master/mesh-mesh-intersection/body_measurements/body_measurements.py)
def compute_height(triangles):
    head_top_tri = triangles[:, head_top_face_idx]
    head_top = (head_top_tri[:, 0, :] * head_top_bc[0] +
                head_top_tri[:, 1, :] * head_top_bc[1] +
                head_top_tri[:, 2, :] * head_top_bc[2])
    head_top = (
        head_top_tri * head_top_bc.reshape(1, 3, 1)
    ).sum(dim=1)
    left_heel_tri = triangles[:, left_heel_face_idx]
    left_heel = (
        left_heel_tri * left_heel_bc.reshape(1, 3, 1)
    ).sum(dim=1)
    return torch.abs(head_top[:, 1] - left_heel[:, 1])


def get_plane_at_heights(height):
    device = height.device
    batch_size = height.shape[0]
    verts = torch.tensor(
        [[-1., 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]],
        device=device).unsqueeze(dim=0).expand(batch_size, -1, -1).clone()
    verts[:, :, 1] = height.reshape(batch_size, -1)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], device=device,
                            dtype=torch.long)
    return verts, faces, verts[:, faces]


def compute_peripheries(verts):
    batch_size = verts.shape[0]
    faces_np = FACES.squeeze(0).cpu().numpy()
    meas_data = {}
    meas_data['chest'] = (chest_face_index, chest_bcs)
    meas_data['waist'] = (belly_face_index, belly_bcs)
    meas_data['hips'] = (hips_face_index, hips_bcs)
    output = {}
    for name, (face_index, bcs) in meas_data.items():
        bcs_np = bcs.cpu().numpy()
        values = []
        for ii in range(batch_size):
            try:
                v = verts[ii].detach().cpu().numpy()
                y = float((v[faces_np[face_index]] * bcs_np.reshape(1, 3)).sum())
                mesh = trimesh.Trimesh(vertices=v, faces=faces_np, process=False)
                sec = mesh.section(plane_origin=[0, y, 0], plane_normal=[0, 1, 0])
                if sec is None or len(sec.vertices) < 3:
                    values.append(0.0)
                    continue
                pts = np.unique(np.round(sec.vertices[:, [0, 2]], 8), axis=0)
                if pts.shape[0] < 3:
                    values.append(0.0)
                    continue
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                closed = np.vstack([hull_pts, hull_pts[:1]])
                values.append(float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum()))
            except Exception:
                values.append(0.0)
        output[name] = {'tensor': torch.tensor(values)}
    return output

def compute_anatomy(verts):
    triangles = verts[:, FACES]
    height = (compute_height(triangles) * 100).to("cpu")
    peri = compute_peripheries(verts)
    chest = (peri['chest']['tensor'] * 100).to("cpu")
    waist = (peri['waist']['tensor'] * 100).to("cpu")
    hips =  (peri['hips']['tensor'] * 100).to("cpu")
    return height, chest, waist, hips


def parse_args():
    parser = argparse.ArgumentParser(description='save metric results')
    parser.add_argument('--files_dir', type=str, required=True, 
                        help='Full path of model inferences on the real dataset')
    parser.add_argument('--save_path', type=str, required=True, 
                        help='Full path of directory to save metric results')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model_data_path = args.files_dir 
    writer_path = args.save_path 
    if os.path.exists(writer_path):
        try:
            shutil.rmtree(os.path.join(writer_path, 'metric_overall'))
        except: 
            pass    
    os.makedirs(writer_path, exist_ok=True)
    writer_metric_overall = SummaryWriter(os.path.join(writer_path, 'metric_overall'))

    metric_overall = create_metric_dict(PI)
    metric_results = {}
    
    for cover_str in tqdm(['uncover', 'cover1', 'cover2'], desc='cover strs'):
        data_pressure_x, \
        data_depth_x, \
        data_label_y, \
        data_pmap_y, \
        data_verts_y, \
        data_names_y = SLPDataset(data_path).prepare_dataset(data_lines, 
                                                            cover_str, 
                                                            load_verts=True)

        metric_dict = create_metric_dict(PI)

        for i in tqdm(range(len(data_pressure_x)), desc=f'{cover_str} data'):
            org_pressure = data_pressure_x[i]
            org_depth = data_depth_x[i]
            label = torch.tensor(data_label_y[i]).float()
            gt_pmap = torch.tensor(data_pmap_y[i]).float()
            gt_verts = torch.tensor(data_verts_y[i]).float().reshape(-1, 3)
            name = data_names_y[i]
            name_split = name.split('_')

            gt_rest_verts = viz_utils.get_rest_verts(label[72:82].float().to(DEVICE), label[157]).unsqueeze(0)
            gt_height, gt_chest, gt_waist, gt_hips = compute_anatomy(gt_rest_verts)
            gt_path = os.path.join(data_path, 'GT_BP_data', 'slp2', f'{int(name_split[2]):05d}', f'{cover_str}')
            gt_jtr = np.load(os.path.join(gt_path, f'{int(name_split[-1]):06d}_gt_jtr.npy'))
            gt_jtr = torch.tensor(gt_jtr.reshape(24, 3)).float()

            pd_verts = np.load(os.path.join(model_data_path, f'{int(name_split[2]):05d}', f'{cover_str}', f'{int(name_split[-1]):06d}_pd_vertices.npy'))
            pd_verts = torch.tensor(pd_verts).float().reshape(-1, 3)
            pd_pmap = np.load(os.path.join(model_data_path, f'{int(name_split[2]):05d}', f'{cover_str}', f'{int(name_split[-1]):06d}_pd_pmap.npy'))
            pd_pmap = torch.tensor(pd_pmap).float()
            pd_jtr = np.load(os.path.join(model_data_path, f'{int(name_split[2]):05d}', f'{cover_str}', f'{int(name_split[-1]):06d}_pd_jtr.npy'))
            pd_jtr = torch.tensor(pd_jtr.reshape(24, 3)).float()
            pd_betas = torch.tensor(np.load(os.path.join(model_data_path, f'{int(name_split[2]):05d}', f'{cover_str}', f'{int(name_split[-1]):06d}_pd_betas.npy'))).float().to(DEVICE)
            pd_rest_verts = viz_utils.get_rest_verts(pd_betas, label[157]).unsqueeze(0)
            pd_height, pd_chest, pd_waist, pd_hips = compute_anatomy(pd_rest_verts)

            loss_verts = torch.norm(gt_verts - pd_verts, dim=1)
            loss_jtr = torch.norm(gt_jtr/1000. - pd_jtr/1000., dim=1)
            loss_height = torch.nn.functional.l1_loss(pd_height, gt_height, reduction='none')
            loss_chest = torch.nn.functional.l1_loss(pd_chest, gt_chest, reduction='none')
            loss_waist = torch.nn.functional.l1_loss(pd_waist, gt_waist, reduction='none')
            loss_hips = torch.nn.functional.l1_loss(pd_hips, gt_hips, reduction='none')

            area_norm_gt = torch.tensor(get_area_norm(gt_verts.numpy(), gt=True)).float()
            area_norm_pd = torch.tensor(get_area_norm(pd_verts.numpy(), gt=False)).float()

            gt_pmap_norm = gt_pmap * area_norm_gt
            pd_pmap_norm = pd_pmap * area_norm_pd
            loss_pmap = torch.nn.functional.mse_loss(pd_pmap_norm, gt_pmap_norm, reduction='none')

            gt_pmap_1ea = get_EA_pressure(gt_pmap, EA1)
            pd_pmap_1ea = get_EA_pressure(pd_pmap, EA1)
            loss_pmap_1ea = torch.nn.functional.mse_loss(pd_pmap_1ea * area_norm_pd, gt_pmap_1ea * area_norm_gt, reduction='none')

            gt_pmap_2ea = get_EA_pressure(gt_pmap, EA2)
            pd_pmap_2ea = get_EA_pressure(pd_pmap, EA2)
            loss_pmap_2ea = torch.nn.functional.mse_loss(pd_pmap_2ea * area_norm_pd, gt_pmap_2ea * area_norm_gt, reduction='none')

            for mdict in [metric_dict, metric_overall]:
                mdict['MPJPE'] += loss_jtr.sum()
                mdict['PVE'] += loss_verts.sum()
                mdict['height'] += loss_height.sum()
                mdict['chest'] += loss_chest.sum()
                mdict['waist'] += loss_waist.sum()
                mdict['hips'] += loss_hips.sum()
                mdict['v2vP'] += loss_pmap.sum()
                mdict['v2vP_1EA'] += loss_pmap_1ea.sum()
                mdict['v2vP_2EA'] += loss_pmap_2ea.sum()
                mdict['count'] += 1
                for key in PI:
                    mdict['parts_v2vP'][key] += loss_pmap[PI[key]].sum()

        metric_dict['MPJPE'] = round(((metric_dict['MPJPE']/(metric_dict['count']*24))*1000).item(), 2)
        metric_dict['PVE'] = round(((metric_dict['PVE']/(metric_dict['count']*6890))*1000).item(), 2)
        metric_dict['height'] = round((metric_dict['height']/metric_dict['count']).item(), 2)
        metric_dict['chest'] = round((metric_dict['chest']/metric_dict['count']).item(), 2)
        metric_dict['waist'] = round((metric_dict['waist']/metric_dict['count']).item(), 2)
        metric_dict['hips'] = round((metric_dict['hips']/metric_dict['count']).item(), 2)
        metric_dict['v2vP'] = (metric_dict['v2vP']/(metric_dict['count']*6890)).item()
        metric_dict['v2vP'] = round(133.32 * 133.32 * (1 / 1000000) * metric_dict['v2vP'], 3)
        metric_dict['v2vP_1EA'] = (metric_dict['v2vP_1EA']/(metric_dict['count']*6890)).item()
        metric_dict['v2vP_1EA'] = round(133.32 * 133.32 * (1 / 1000000) * metric_dict['v2vP_1EA'], 3)
        metric_dict['v2vP_2EA'] = (metric_dict['v2vP_2EA']/(metric_dict['count']*6890)).item()
        metric_dict['v2vP_2EA'] = round(133.32 * 133.32 * (1 / 1000000) * metric_dict['v2vP_2EA'], 3)
        for key in PI:
            metric_dict['parts_v2vP'][key] = (metric_dict['parts_v2vP'][key]/(metric_dict['count']*len(PI[key]))).item()
            metric_dict['parts_v2vP'][key] = round(133.32 * 133.32 * (1 / 1000000) * metric_dict['parts_v2vP'][key], 3)        

        writer_metric_overall.add_text(f'{cover_str}', json.dumps(metric_dict))
        metric_results[cover_str] = metric_dict

    metric_overall['MPJPE'] = round(((metric_overall['MPJPE']/(metric_overall['count']*24))*1000).item(), 2)
    metric_overall['PVE'] = round(((metric_overall['PVE']/(metric_overall['count']*6890))*1000).item(), 2)
    metric_overall['height'] = round((metric_overall['height']/metric_overall['count']).item(), 2)
    metric_overall['chest'] = round((metric_overall['chest']/metric_overall['count']).item(), 2)
    metric_overall['waist'] = round((metric_overall['waist']/metric_overall['count']).item(), 2)
    metric_overall['hips'] = round((metric_overall['hips']/metric_overall['count']).item(), 2)  
    metric_overall['v2vP'] = (metric_overall['v2vP']/(metric_overall['count']*6890)).item()
    metric_overall['v2vP'] = round(133.32 * 133.32 * (1 / 1000000) * metric_overall['v2vP'], 3)
    metric_overall['v2vP_1EA'] = (metric_overall['v2vP_1EA']/(metric_overall['count']*6890)).item()
    metric_overall['v2vP_1EA'] = round(133.32 * 133.32 * (1 / 1000000) * metric_overall['v2vP_1EA'], 3)
    metric_overall['v2vP_2EA'] = (metric_overall['v2vP_2EA']/(metric_overall['count']*6890)).item()
    metric_overall['v2vP_2EA'] = round(133.32 * 133.32 * (1 / 1000000) * metric_overall['v2vP_2EA'], 3)
    for key in PI:
        metric_overall['parts_v2vP'][key] = (metric_overall['parts_v2vP'][key]/(metric_overall['count']*len(PI[key]))).item()
        metric_overall['parts_v2vP'][key] = round(133.32 * 133.32 * (1 / 1000000) * metric_overall['parts_v2vP'][key], 3)        

    writer_metric_overall.add_text(f'overall', json.dumps(metric_overall))
    metric_results['overall'] = metric_overall
    writer_metric_overall.flush()
    writer_metric_overall.close()

    metrics_json_path = os.path.join(writer_path, 'metrics.json')
    with open(metrics_json_path, 'w') as metrics_fp:
        json.dump(metric_results, metrics_fp, indent=2)

    print(json.dumps(metric_results, indent=2))
    print(f'Metrics saved to {metrics_json_path}')

    print ('done')
            
