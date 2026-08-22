import json
import numpy as np
import os
from scipy.spatial import ConvexHull
import trimesh
from tqdm import tqdm 

import torch
from constants import *


FACES_NP = FACES.squeeze(0).to("cpu").numpy()

MEASUREMENTS = {
    'chest': (11885, np.array([0., 1., 0.])),
    'waist': (6833, np.array([0., 1., 0.])),
    'hips': (1341, np.array([0., 1., 0.])),
}


def compute_anatomy(verts):
    """BodyMAP/SHAPY-compatible height and convex cross-section perimeters in cm."""
    values = {'height': [], 'chest': [], 'waist': [], 'hips': []}
    for vertices in verts.detach().cpu().numpy():
        head = (vertices[FACES_NP[435]] * np.array([0., 1., 0.])[:, None]).sum(axis=0)
        heel = (vertices[FACES_NP[5975]] * np.array([0., 0., 1.])[:, None]).sum(axis=0)
        values['height'].append(abs(head[1] - heel[1]) * 100)
        mesh = trimesh.Trimesh(vertices=vertices, faces=FACES_NP, process=False)
        for name, (face_index, barycentric) in MEASUREMENTS.items():
            landmark = (vertices[FACES_NP[face_index]] * barycentric[:, None]).sum(axis=0)
            height = float(landmark[1])
            section = mesh.section(plane_origin=[0, height, 0], plane_normal=[0, 1, 0])
            if section is None or len(section.vertices) < 3:
                values[name].append(0.)
                continue
            points = np.unique(np.round(section.vertices[:, [0, 2]], 8), axis=0)
            if len(points) < 3:
                values[name].append(0.)
                continue
            hull = ConvexHull(points)
            polygon = points[hull.vertices]
            closed = np.vstack((polygon, polygon[:1]))
            values[name].append(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum() * 100)
    return {key: torch.tensor(value, device=DEVICE) for key, value in values.items()}

TRANS_PREV_GT = np.identity(4)
TRANS_PREV_GT[0, 3] = -2.0
TRANS_NEXT_GT = np.identity(4)
TRANS_NEXT_GT[0, 3] = -2.0
TRANS_NEXT_GT[1, 3] = 1.0

TRANS_PREV_PD = np.identity(4)
TRANS_NEXT_PD = np.identity(4)
TRANS_NEXT_PD[1, 3] = 1.0


def get_triangle_area_vert_weight(verts, faces, verts_idx_red = None):

    #first we need all the triangle areas
    tri_verts = verts[faces, :]
    a = np.linalg.norm(tri_verts[:,0]-tri_verts[:,1], axis = 1)
    b = np.linalg.norm(tri_verts[:,1]-tri_verts[:,2], axis = 1)
    c = np.linalg.norm(tri_verts[:,2]-tri_verts[:,0], axis = 1)
    s = (a+b+c)/2
    A = np.sqrt(s*(s-a)*(s-b)*(s-c))

    A = np.swapaxes(np.stack((A, A, A)), 0, 1) #repeat the area for each vert in the triangle
    A = A.flatten()
    faces = np.array(faces).flatten()
    i = np.argsort(faces) #sort the faces and the areas by the face idx
    faces_sorted = faces[i]
    A_sorted = A[i]
    last_face = 0
    area_minilist = []
    area_avg_list = []
    face_sort_list = [] #take the average area for all the trianges surrounding each vert
    for vtx_connect_idx in range(np.shape(faces_sorted)[0]):
        if faces_sorted[vtx_connect_idx] == last_face and vtx_connect_idx != np.shape(faces_sorted)[0]-1:
            area_minilist.append(A_sorted[vtx_connect_idx])
        elif faces_sorted[vtx_connect_idx] > last_face or vtx_connect_idx == np.shape(faces_sorted)[0]-1:
            if len(area_minilist) != 0:
                area_avg_list.append(np.mean(area_minilist))
            else:
                area_avg_list.append(0)
            face_sort_list.append(last_face)
            area_minilist = []
            last_face += 1
            if faces_sorted[vtx_connect_idx] == last_face:
                area_minilist.append(A_sorted[vtx_connect_idx])
            elif faces_sorted[vtx_connect_idx] > last_face:
                num_tack_on = np.copy(faces_sorted[vtx_connect_idx] - last_face)
                for i in range(num_tack_on):
                    area_avg_list.append(0)
                    face_sort_list.append(last_face)
                    last_face += 1
                    if faces_sorted[vtx_connect_idx] == last_face:
                        area_minilist.append(A_sorted[vtx_connect_idx])

    area_avg = np.array(area_avg_list)
    area_avg_red = area_avg[area_avg > 0] #find out how many of the areas correspond to verts facing the camera

    norm_area_avg = area_avg/np.sum(area_avg_red)
    norm_area_avg = norm_area_avg*np.shape(area_avg_red) #multiply by the REDUCED num of verts

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


def create_metric_dict(epoch=-1):
    return {
            'epoch' : epoch,
            '3D MPJPE' : torch.tensor(0).float().to(DEVICE),
            'PVE' : torch.tensor(0).float().to(DEVICE),
            'height' : torch.tensor(0).float().to(DEVICE),
            'chest' : torch.tensor(0).float().to(DEVICE),
            'waist' : torch.tensor(0).float().to(DEVICE),
            'hips' : torch.tensor(0).float().to(DEVICE),
            'v2vP' : torch.tensor(0).float().to(DEVICE), 
            'v2vP 1EA' : torch.tensor(0).float().to(DEVICE),
            'v2vP 2EA' : torch.tensor(0).float().to(DEVICE),
            'count' : 0,
            }


def PMMMetric(model, test_loader, writer=None, epoch=-1, pmap_norm=False, infer_pmap=False, infer_smpl=False, MOD1=None):
    ea1 = np.load(os.path.join(BASE_PATH, 'BodyPressure', 'data_BP', 'parsed', 'EA1.npy'), allow_pickle=True)
    ea2 = np.load(os.path.join(BASE_PATH, 'BodyPressure', 'data_BP', 'parsed', 'EA2.npy'), allow_pickle=True)

    def ea_matrix(neighborhoods):
        lengths = np.asarray([len(group) for group in neighborhoods])
        rows = np.repeat(np.arange(len(neighborhoods)), lengths)
        cols = np.concatenate([np.asarray(group) for group in neighborhoods])
        indexes = torch.tensor(np.stack((rows, cols)), dtype=torch.long, device=DEVICE)
        weights = torch.tensor(np.repeat(1.0 / lengths, lengths), dtype=torch.float32,
                               device=DEVICE)
        return torch.sparse_coo_tensor(indexes, weights, (len(neighborhoods), 6890)).coalesce()

    ea1_matrix, ea2_matrix = ea_matrix(ea1), ea_matrix(ea2)

    def ea_pressure(values, matrix):
        return torch.sparse.mm(matrix, values.unsqueeze(1)).squeeze(1)
    dict_map = {
            'uncover' : create_metric_dict(epoch), 
            'cover1' :  create_metric_dict(epoch), 
            'cover2' :  create_metric_dict(epoch), 
            'synth' :   create_metric_dict(epoch),
            'f' :       create_metric_dict(epoch), 
            'm' :       create_metric_dict(epoch),
            'overall' : create_metric_dict(epoch),
            }

    model.eval()
    with torch.no_grad():
        for _, batch_pressure_images, _, batch_depth_images, batch_labels, batch_pmap, batch_verts, batch_names in tqdm(iter(test_loader), desc='metric'):
            batch_depth_images = batch_depth_images.to(DEVICE)
            batch_pressure_images = batch_pressure_images.to(DEVICE)
            batch_labels_copy = batch_labels.clone().to(DEVICE)
            batch_pmap = batch_pmap.to(DEVICE)            

            if MOD1 is not None:
                batch_mesh_pred, _, img_feat, _ = MOD1.infer(batch_depth_images.clone(), batch_pressure_images.clone(), batch_labels[:, 157:159].clone())
                _, batch_pmap_pred, _, _ = model(batch_depth_images, batch_pressure_images, batch_labels[:, 157:159], batch_mesh_pred['out_verts'].clone(), img_feat)
            else:
                batch_mesh_pred, batch_pmap_pred, _, _ = model.infer(batch_depth_images, batch_pressure_images, batch_labels[:, 157:159])

            batch_labels = batch_labels.to(DEVICE)
            if infer_smpl:
                batch_mesh_pred['out_joint_pos'] = batch_mesh_pred['out_joint_pos'].reshape(-1, 24, 3)
            else:
                batch_mesh_pred = {
                    'out_joint_pos' : (batch_labels_copy[:, :72]/1000.).reshape(-1, 24, 3),
                    'out_verts' : batch_verts,
                    }
            if not infer_pmap:
                batch_pmap_pred = batch_pmap.clone()

            if infer_smpl:
                rest_gt = torch.zeros((batch_labels.shape[0], 88), device=DEVICE)
                rest_gt[:, :10] = batch_labels[:, 72:82]
                rest_gt[:, 13:16] = 1.0
                rest_pd = rest_gt.clone()
                rest_pd[:, :10] = batch_mesh_pred['out_betas']
                gt_anatomy = compute_anatomy(model.mesh_model.infer(
                    rest_gt, batch_labels[:, 157:159], is_gt=True)['out_verts'])
                pd_anatomy = compute_anatomy(model.mesh_model.infer(
                    rest_pd, batch_labels[:, 157:159], is_gt=True)['out_verts'])

            
            for i, file_name in enumerate(batch_names):
                file_name_contents = file_name.split('_')
                loss = torch.norm(batch_labels_copy[i, :72].reshape(24, 3)/1000. - batch_mesh_pred['out_joint_pos'][i], dim=1)
                loss_vertices = torch.norm(batch_verts[i].to(DEVICE) - batch_mesh_pred['out_verts'][i], dim=1)
                
                gt_pmap = batch_pmap[i]
                pd_pmap = batch_pmap_pred[i]
                if pmap_norm:
                    if batch_names[i][0] == 's':
                        pd_pmap *= MAX_PMAP_SYNTH
                        gt_pmap *= MAX_PMAP_SYNTH
                    elif batch_names[i][0] == 'r':
                        pd_pmap *= MAX_PMAP_REAL
                        gt_pmap *= MAX_PMAP_REAL
                    else:
                        print ('ERROR: Invaid data category in metric calculation')
                        exit(-1)
                gt_area_norm = torch.tensor(get_area_norm(batch_verts[i].numpy(), gt=True)).float().to(DEVICE)
                pd_area_norm = torch.tensor(get_area_norm(batch_mesh_pred['out_verts'][i].to("cpu").numpy(), gt=(not infer_smpl))).float().to(DEVICE)

                gt_pmap_norm = gt_pmap * gt_area_norm
                pd_pmap_norm = pd_pmap * pd_area_norm

                loss_pmap = torch.nn.functional.mse_loss(pd_pmap_norm, gt_pmap_norm, reduction='none')
                loss_pmap_1ea = torch.nn.functional.mse_loss(
                    ea_pressure(pd_pmap, ea1_matrix) * pd_area_norm,
                    ea_pressure(gt_pmap, ea1_matrix) * gt_area_norm, reduction='none')
                loss_pmap_2ea = torch.nn.functional.mse_loss(
                    ea_pressure(pd_pmap, ea2_matrix) * pd_area_norm,
                    ea_pressure(gt_pmap, ea2_matrix) * gt_area_norm, reduction='none')
                    
                metric_dict_data = dict_map[file_name_contents[1]]
                metric_dict_gender = dict_map[file_name_contents[3]]
                metric_dict_overall = dict_map['overall']
                for metric_dict in [metric_dict_data, metric_dict_gender, metric_dict_overall]:
                    metric_dict['3D MPJPE'] += loss.sum()
                    metric_dict['PVE'] += loss_vertices.sum()
                    if infer_smpl:
                        for measurement in ('height', 'chest', 'waist', 'hips'):
                            metric_dict[measurement] += torch.abs(
                                pd_anatomy[measurement][i] - gt_anatomy[measurement][i])
                    metric_dict['v2vP'] += loss_pmap.sum()
                    metric_dict['v2vP 1EA'] += loss_pmap_1ea.sum()
                    metric_dict['v2vP 2EA'] += loss_pmap_2ea.sum()
                    metric_dict['count'] += 1

        for mdict in dict_map.values():
            if mdict['count'] == 0:
                mdict['3D MPJPE'] = None
                mdict['PVE'] = None
                for measurement in ('height', 'chest', 'waist', 'hips'):
                    mdict[measurement] = None
                mdict['v2vP'] = None
                mdict['v2vP 1EA'] = None
                mdict['v2vP 2EA'] = None
                continue
            # divide by 24 for 24 joint positions, multiiply by 1000 for error in mm
            mdict['3D MPJPE'] = round(((mdict['3D MPJPE']/(mdict['count']*24))*1000).item(), 6)
            mdict['PVE'] = round(((mdict['PVE']/(mdict['count']*6890))*1000).item(), 6)
            for measurement in ('height', 'chest', 'waist', 'hips'):
                mdict[measurement] = round((mdict[measurement] / mdict['count']).item(), 6)
            mdict['v2vP'] = (mdict['v2vP']/(mdict['count']*6890)).item()
            mdict['v2vP'] = round(133.32 * 133.32 * (1 / 1000000) * mdict['v2vP'], 6)
            for metric in ('v2vP 1EA', 'v2vP 2EA'):
                value = (mdict[metric] / (mdict['count'] * 6890)).item()
                mdict[metric] = round(133.32 * 133.32 * (1 / 1000000) * value, 6)

    if writer is not None:
        for mkey, mdict in dict_map.items():
            mdict_str = json.dumps(mdict)
            writer.add_text(f'{mkey}', mdict_str)
    return dict_map
