import torch
import torch.nn as nn
import numpy as np
from ....ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from ....utils import common_utils


def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


class VoxelSetAbstraction(nn.Module):
    def __init__(self, backbone_channels, model_cfg, voxel_size, point_cloud_range, num_bev_features=None,
                 num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        SA_cfg = self.model_cfg.SA_LAYER

        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        c_in = 0
        c_in += 16
        for src_name in self.model_cfg.FEATURES_SOURCE:
            if src_name in ['bev']:
                continue
            mlps = SA_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [backbone_channels[src_name]] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=SA_cfg[src_name].QUERY_RANGES,
                nsamples=SA_cfg[src_name].NSAMPLE,
                radii=SA_cfg[src_name].POOL_RADIUS,
                mlps=mlps,
                pool_method=SA_cfg[src_name].POOL_METHOD,
            )

            self.SA_layers.append(pool_layer)
            self.SA_layer_names.append(src_name)

            c_in += sum([x[-1] for x in mlps])

        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            c_bev = num_bev_features
            c_in += c_bev



        # AFF Module
        channels = 128
        r = 4
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.sigmoid = nn.Sigmoid()

        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(c_in, self.model_cfg.NUM_OUTPUT_FEATURES, bias=False),
            nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_FEATURES),
            nn.ReLU(),
        )
        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES
        self.num_point_features_before_fusion = c_in

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        x_idxs = (keypoints[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            cur_point_mask = (keypoints[:, 0] == k)
            cur_x_idxs = x_idxs[cur_point_mask]
            cur_y_idxs = y_idxs[cur_point_mask]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features)

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        return point_bev_features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            point_features: (N, C)
            point_coords: (N, 4)

        """
        batch_size = batch_dict['batch_size']
        point_features = batch_dict['point_features']
        point_coords = batch_dict['point_coords']

        point_features_list = []
        point_features_list.append(point_features)
        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            point_bev_features = self.interpolate_from_bev_features(
                point_coords, batch_dict['spatial_features_2d'], batch_dict['batch_size'],
                bev_stride=batch_dict['spatial_features_stride']
            )
            point_features_list.append(point_bev_features)

        val = []
        for k in range(batch_size):
            point_mask = (point_coords[:, 0] == k)
            point = point_coords[point_mask][:, 1:4]
            val.append(point.cpu().numpy())
        max_point = max([len(x) for x in val])
        batch_point = np.zeros((batch_size, max_point, val[0].shape[-1]), dtype=np.float32)

        for j in range(batch_size):
            batch_point[j, :val[j].__len__(), :] = val[j]
        batch_point = torch.FloatTensor(batch_point).cuda()

        point_coords_x = (batch_point[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        point_coords_y = (batch_point[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        point_coords_z = (batch_point[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
        point_coords_xyz = torch.cat([point_coords_x, point_coords_y, point_coords_z], dim=-1)
        batch_idx = batch_point.new_zeros(batch_size, point_coords_xyz.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        point_batch_cnt = batch_point.new_zeros(batch_size).int().fill_(point_coords_xyz.shape[1])
        for k, src_name in enumerate(self.SA_layer_names):
            cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]
            cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
            cur_coords = batch_dict['multi_scale_3d_features'][src_name].indices
            cur_voxel_xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=cur_stride,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()

            v2p_ind_tensor = common_utils.generate_voxel2pinds(cur_sp_tensors)
            cur_point_coords = point_coords_xyz // cur_stride
            cur_point_coords = torch.cat([batch_idx, cur_point_coords], dim=-1)
            cur_point_coords = cur_point_coords.int()
            pooled_features = self.SA_layers[k](
                xyz=cur_voxel_xyz.contiguous(),
                xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                new_xyz=batch_point.contiguous().view(-1, 3),
                new_xyz_batch_cnt=point_batch_cnt,
                new_coords=cur_point_coords.contiguous().view(-1, 4),
                features=cur_sp_tensors.features.contiguous(),
                voxel2point_indices=v2p_ind_tensor
            )
            c_list =[]
            pooled_features_list = []
            for m in range(batch_size):
                c = val[m].shape[0]
                c_list.append(c)
            c = max(c_list)
            for n in range(batch_size):
                pooled_features_select = pooled_features[(n*c):(n*c+val[n].shape[0])]
                pooled_features_list.append(pooled_features_select)
            pooled_features = torch.cat(pooled_features_list, dim=0)
            point_features_list.append(pooled_features)

        point_features = torch.cat(point_features_list, dim=1)

        batch_dict['point_features_before_fusion'] = point_features
        point_features = self.vsa_point_feature_fusion(point_features)

        batch_dict['point_features'] = point_features  # (BxN, C)
        return batch_dict
