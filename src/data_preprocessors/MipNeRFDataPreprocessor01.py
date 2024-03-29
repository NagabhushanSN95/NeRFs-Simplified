# Shree KRISHNAYa Namaha
# Preprocesses data for Mip NeRF
# Author: Nagabhushan S N
# Last Modified: 23/09/2022

from typing import Optional, Union

import numpy
import skimage.io
import skimage.transform
import torch

from data_preprocessors.DataPreprocessorParent import DataPreprocessorParent
from utils import CommonUtils01 as CommonUtils


class MipNeRFDataPreprocessor(DataPreprocessorParent):
    def __init__(self, configs: dict, mode: str, raw_data_dict: Optional[dict] = None, model_configs: Optional[dict] = None):
        self.configs = configs
        self.mode = mode.lower()
        self.bd_factor = self.configs['data_loader']['bd_factor']
        self.use_batching = self.configs['data_loader']['batching']
        self.ndc = self.configs['data_loader']['ndc']
        self.downsampling_factor = self.configs['data_loader']['downsampling_factor']
        self.device = CommonUtils.get_device(self.configs['device'])

        self.raw_data_dict = raw_data_dict
        self.preprocessed_data_dict = None
        self.model_configs = model_configs
        self.i_batch = 0
        self.num_rays = self.configs['data_loader']['num_rays']

        if self.mode in ['train', 'validation']:
            self.preprocessed_data_dict = self.preprocess_data(self.raw_data_dict)
            if self.mode == 'train':
                self.model_configs = self.create_model_configs(self.preprocessed_data_dict)
        return

    def preprocess_data(self, raw_data_dict):
        preprocessed_data_dict = {}

        preprocessed_raw_data_dict = self.preprocess_raw_data(raw_data_dict)
        self.update_dict_recursively(preprocessed_data_dict, preprocessed_raw_data_dict)
        self.add_residual_dict_items(preprocessed_data_dict, preprocessed_raw_data_dict)

        if self.use_batching:
            cache_data_dict = self.create_cache(preprocessed_data_dict)
            self.update_dict_recursively(preprocessed_data_dict, cache_data_dict)
            self.add_residual_dict_items(preprocessed_data_dict, cache_data_dict)

        preprocessed_data_dict = CommonUtils.move_to_device(preprocessed_data_dict, self.device)
        return preprocessed_data_dict

    def create_model_configs(self, preprocessed_data_dict):
        # Create model configs to save
        model_configs = {
            'resolution': preprocessed_data_dict['nerf_data']['resolution'],
            'focal_length': list(preprocessed_data_dict['nerf_data']['focal_length']),
            'bounds': preprocessed_data_dict['nerf_data']['bounds'].tolist(),
            'translation_scale': preprocessed_data_dict['nerf_data'].get('sc', 1),
            f'{self.mode}_frame_nums': preprocessed_data_dict['frame_nums'].tolist(),
            'average_pose': preprocessed_data_dict['nerf_data']['average_pose'].tolist(),
            'near': preprocessed_data_dict['nerf_data']['near'],
            'far': preprocessed_data_dict['nerf_data']['far'],
        }
        if self.ndc:
            model_configs['near_ndc'] = preprocessed_data_dict['nerf_data']['near_ndc']
            model_configs['far_ndc'] = preprocessed_data_dict['nerf_data']['far_ndc']
        return model_configs

    def get_model_configs(self):
        return self.model_configs

    # --------------------- Methods that preprocess raw data ----------------------------------- #

    def preprocess_raw_data(self, raw_data_dict: dict):
        return_dict = {}

        return_dict['nerf_data'] = self.preprocess_raw_nerf_data(raw_data_dict, return_dict)

        self.add_residual_dict_items(return_dict, raw_data_dict)
        return return_dict

    def preprocess_raw_nerf_data(self, raw_data_dict: dict, processed_data_dict: dict):
        images = raw_data_dict['nerf_data']['images']
        poses = raw_data_dict['nerf_data']['poses']
        bds = raw_data_dict['nerf_data']['bounds']
        resolution = raw_data_dict['nerf_data']['resolution']
        focal_length = raw_data_dict['nerf_data']['focal_length']

        images = self.preprocess_images(images)
        resolution = [int(x) for x in resolution]

        if self.downsampling_factor > 1:
            images = numpy.stack([skimage.transform.rescale(image, 1/self.downsampling_factor, anti_aliasing=True, multichannel=True) for image in images])
            resolution = [x // self.downsampling_factor for x in resolution]
            focal_length = [x / self.downsampling_factor for x in focal_length]

        return_dict = {
            'images': images,
            'resolution': resolution,
            'focal_length': focal_length,
        }

        if self.mode == 'train':
            preprocessed_poses_dict = self.preprocess_poses({
                'poses': poses,
                'bounds': bds,
                'bd_factor': self.configs['data_loader']['bd_factor']
            },
                train_mode=True)
        else:
            preprocessed_poses_dict = self.preprocess_poses({
                'poses': poses,
                'bounds': bds,
                'translation_scale': self.model_configs['translation_scale'],
                'average_pose': numpy.array(self.model_configs['average_pose']),
            },
                train_mode=False)

        return_dict.update(preprocessed_poses_dict)

        bds = preprocessed_poses_dict['bounds']
        if not self.ndc:
            near = bds[0] * .9
            far = bds[1] * 1.
        else:
            bd_factor = self.bd_factor if self.bd_factor is not None else 1
            near = float(bds[0] * bd_factor)
            far = float(bds[1])
            near_ndc = 0.
            far_ndc = 1.

        return_dict['near'] = near
        return_dict['far'] = far
        if self.ndc:
            return_dict['near_ndc'] = near_ndc
            return_dict['far_ndc'] = far_ndc

        return return_dict

    # ----------------------- Methods that create cache, if batching is enabled ----------------------- #

    def create_cache(self, preprocessed_data_dict: dict):
        cache_data_dict = {}

        indices = self.generate_indices(preprocessed_data_dict, cache_data_dict)
        cache_data_dict['indices'] = indices

        cache_data_dict['nerf_data'] = self.preprocess_nerf_data(preprocessed_data_dict, cache_data_dict)
        return cache_data_dict

    def generate_indices(self, data_dict: dict, cache_data_dict: dict, iter_num: int = 0):
        n = len(data_dict['nerf_data']['images'])
        h, w = data_dict['nerf_data']['resolution']
        num_rays = n * h * w
        indices = numpy.arange(num_rays)

        if ('precrop_fraction' in self.configs['data_loader']) and \
                (self.configs['data_loader']['precrop_fraction'] < 1) and \
                (iter_num < self.configs['data_loader']['precrop_iterations']):
            h1 = int(round(h/2 * (1 - self.configs['data_loader']['precrop_fraction'])))
            h2 = int(round(h/2 * (1 + self.configs['data_loader']['precrop_fraction'])))
            w1 = int(round(w/2 * (1 - self.configs['data_loader']['precrop_fraction'])))
            w2 = int(round(w/2 * (1 + self.configs['data_loader']['precrop_fraction'])))
            indices_reshaped = numpy.reshape(indices, (n, h, w))
            cropped_indices = indices_reshaped[:, h1:h2, w1:w2]
            indices = cropped_indices.ravel()

        numpy.random.shuffle(indices)
        return indices
    
    def preprocess_nerf_data(self, data_dict: dict, cache_data_dict: dict):
        images = data_dict['nerf_data']['images']
        poses = data_dict['nerf_data']['poses']
        h, w = data_dict['nerf_data']['resolution']
        fx, fy = data_dict['nerf_data']['focal_length']
        near = data_dict['nerf_data']['near']
        
        # For random ray batching
        rays_o_list = []
        rays_d_list = []
        for i in range(poses.shape[0]):
            pose1 = poses[i, :3, :4]
            image_rays_o, image_rays_d = self.get_rays(h, w, fx, fy, pose1)
            rays_o_list.append(image_rays_o)
            rays_d_list.append(image_rays_d)
        rays_o = numpy.stack(rays_o_list, 0)  # [n, h, w, 3]
        rays_d = numpy.stack(rays_d_list, 0)  # [n, h, w, 3]
        radii = self.get_radii(rays_d)  # [n, h, w, 1]
        view_dirs = self.get_view_dirs(rays_d)  # [n, h, w, 3]
        if self.ndc:
            rays_o_ndc, rays_d_ndc = self.get_ndc_rays(rays_o, rays_d, h, w, fx, fy, near)
            radii_ndc = self.get_radii_ndc(rays_o_ndc)
        rays_o = numpy.reshape(rays_o, (-1, 3)).astype(numpy.float32)  # (n*h*w, 3)
        rays_d = numpy.reshape(rays_d, (-1, 3)).astype(numpy.float32)  # (n*h*w, 3)
        radii = numpy.reshape(radii, (-1, 1)).astype(numpy.float32)  # (n*h*w, 1)
        view_dirs = numpy.reshape(view_dirs, (-1, 3)).astype(numpy.float32)  # (n*h*w, 3)
        all_data_dict = {
            'rays_o': torch.from_numpy(rays_o), 
            'rays_d': torch.from_numpy(rays_d),
            'view_dirs': torch.from_numpy(view_dirs),
            'radii': torch.from_numpy(radii),
        }

        if self.ndc:
            rays_o_ndc = numpy.reshape(rays_o_ndc, (-1, 3)).astype(numpy.float32)  # (n*h*w, 3)
            rays_d_ndc = numpy.reshape(rays_d_ndc, (-1, 3)).astype(numpy.float32)  # (n*h*w, 3)
            radii_ndc = numpy.reshape(radii_ndc, (-1, 1)).astype(numpy.float32)  # (n*h*w, 1)
            all_data_dict['rays_o_ndc'] = torch.from_numpy(rays_o_ndc)
            all_data_dict['rays_d_ndc'] = torch.from_numpy(rays_d_ndc)
            all_data_dict['radii_ndc'] = torch.from_numpy(radii_ndc)

        target_rgb = numpy.reshape(images, (-1, 3)).astype(numpy.float32)  # (n*h*w, 3)
        all_data_dict['target_rgb'] = torch.from_numpy(target_rgb)
        return all_data_dict

    @staticmethod
    def get_rays(h, w, fx, fy, pose):
        x, y = numpy.meshgrid(numpy.arange(w, dtype=numpy.float32), numpy.arange(h, dtype=numpy.float32), indexing='xy')  # x: h x w, y: h x w
        dirs = numpy.stack([(x - w * 0.5 + 0.5) / fx, -(y - h * 0.5 + 0.5) / fy, -numpy.ones_like(x)], -1)  # dirs: h x w x 3
        # Rotate ray directions from camera frame to the world frame
        rays_d = numpy.sum(dirs[..., numpy.newaxis, :] * pose[:3, :3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = numpy.broadcast_to(pose[:3, -1], numpy.shape(rays_d))  # h x w x 3
        return rays_o, rays_d

    @staticmethod
    def get_ndc_rays(rays_o, rays_d, h, w, fx, fy, near):
        # Shift ray origins to near plane
        t = -(near + rays_o[..., 2]) / rays_d[..., 2]
        rays_o = rays_o + t[..., None] * rays_d

        # Projection
        o0 = -1. / (w / (2. * fx)) * rays_o[..., 0] / rays_o[..., 2]
        o1 = -1. / (h / (2. * fy)) * rays_o[..., 1] / rays_o[..., 2]
        o2 = 1. + 2. * near / rays_o[..., 2]

        d0 = -1. / (w / (2. * fx)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
        d1 = -1. / (h / (2. * fy)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
        d2 = -2. * near / rays_o[..., 2]

        rays_o_ndc = numpy.stack([o0, o1, o2], -1)
        rays_d_ndc = numpy.stack([d0, d1, d2], -1)
        return rays_o_ndc, rays_d_ndc

    @staticmethod
    def get_radii(rays_d):
        dx = numpy.sqrt(numpy.sum((rays_d[:, :-1, :, :] - rays_d[:, 1:, :, :]) ** 2, -1))
        dx = numpy.concatenate([dx, dx[:, -2:-1, :]], 1)
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = dx[..., None] * 2 / numpy.sqrt(12)
        return radii

    @staticmethod
    def get_radii_ndc(rays_o_ndc):
        # Distance from each unit-norm direction vector to its x-axis neighbor.
        dx = numpy.sqrt(numpy.sum((rays_o_ndc[:, :-1, :, :] - rays_o_ndc[:, 1:, :, :]) ** 2, -1))
        dx = numpy.concatenate([dx, dx[:, -2:-1, :]], 1)

        dy = numpy.sqrt(numpy.sum((rays_o_ndc[:, :, :-1, :] - rays_o_ndc[:, :, 1:, :]) ** 2, -1))
        dy = numpy.concatenate([dy, dy[:, :, -2:-1]], 2)
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii_ndc = (0.5 * (dx + dy))[..., None] * 2 / numpy.sqrt(12)
        return radii_ndc

    @staticmethod
    def get_view_dirs(rays_d):
        view_dirs = rays_d / numpy.linalg.norm(rays_d, ord=2, axis=-1, keepdims=True)
        return view_dirs

    # ---------------- Methods to return next batch during training/validation ------------------ #

    def get_next_batch(self, iter_num: int, image_num: int = None):
        if self.use_batching:
            return_dict = self.load_cached_next_batch(iter_num, image_num)
        else:
            return_dict = self.load_uncached_next_batch(iter_num, image_num)

        # Load data that is independent of cached data
        rays_o = return_dict['rays_o']
        near = self.preprocessed_data_dict['nerf_data']['near']
        far = self.preprocessed_data_dict['nerf_data']['far']
        near = near * torch.ones_like(rays_o[..., :1])
        far = far * torch.ones_like(rays_o[..., :1])
        return_dict['near'] = near
        return_dict['far'] = far
        if self.ndc:
            near_ndc = self.preprocessed_data_dict['nerf_data']['near_ndc']
            far_ndc = self.preprocessed_data_dict['nerf_data']['far_ndc']
            near_ndc = near_ndc * torch.ones_like(rays_o[..., :1])
            far_ndc = far_ndc * torch.ones_like(rays_o[..., :1])
            return_dict['near_ndc'] = near_ndc
            return_dict['far_ndc'] = far_ndc
        
        return return_dict
    
    def load_cached_next_batch(self, iter_num, image_num):
        return_dict = {}
        indices_dict = self.select_batch_indices(iter_num, image_num)
        return_dict.update(indices_dict)
        indices = indices_dict['indices']
        
        nerf_data_dict = self.load_nerf_cached_batch(iter_num, indices)
        return_dict.update(nerf_data_dict)
        return return_dict

    def load_uncached_next_batch(self, iter_num, image_num):
        """
        Random from one image.
        This function is not maintained well, since it is not in use.
        """
        images = self.preprocessed_data_dict['images']
        poses = self.preprocessed_data_dict['poses']
        h, w = self.preprocessed_data_dict['resolution']
        fx, fy = self.preprocessed_data_dict['focal_length']
        near = self.preprocessed_data_dict['near']

        if image_num is None:
            img_i = numpy.random.randint(0, images.shape[0])
        else:
            img_i = image_num
        target = images[img_i]
        pose = poses[img_i, :3, :4]

        rays_o, rays_d = self.get_rays(h, w, fx, fy, pose)  # (H, W, 3), (H, W, 3)
        # TODO: Compute radii also

        coords = torch.stack(torch.meshgrid(torch.linspace(0, h - 1, h), torch.linspace(0, w - 1, w)), -1)  # (H, W, 2)
        coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
        if image_num is None:
            select_inds = numpy.random.choice(rays_o.shape[0], size=[self.num_rays], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
        else:
            select_coords = coords

        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        target_rgb = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        if self.ndc:
            rays_o_ndc, rays_d_ndc = self.get_ndc_rays(rays_o, rays_d, h, w, fx, fy, near)

        return_dict = {
            'rays_o': torch.from_numpy(rays_o),
            'rays_d': torch.from_numpy(rays_d),
            'target_rgb': target_rgb,
            'iter_num': iter_num,
        }
        if self.ndc:
            return_dict['rays_o_ndc'] = torch.from_numpy(rays_o_ndc)
            return_dict['rays_d_ndc'] = torch.from_numpy(rays_d_ndc)
        return return_dict

    def select_batch_indices(self, iter_num, image_num):
        return_dict = {}

        if image_num is None:
            if iter_num == self.configs['data_loader']['precrop_iterations']:
                self.generate_indices(self.preprocessed_data_dict, None, iter_num)
            indices = self.preprocessed_data_dict['indices'][self.i_batch: self.i_batch + self.num_rays]
            self.i_batch += self.num_rays
            if self.i_batch >= self.preprocessed_data_dict['indices'].size:
                numpy.random.shuffle(self.preprocessed_data_dict['indices'])
                self.i_batch = 0
        else:
            h, w = self.preprocessed_data_dict['nerf_data']['resolution']
            indices = numpy.arange(h * w) + (image_num * h * w)

        return_dict['indices'] = indices
        return return_dict

    def load_nerf_cached_batch(self, iter_num, indices):
        rays_o = self.preprocessed_data_dict['nerf_data']['rays_o'][indices]
        rays_d = self.preprocessed_data_dict['nerf_data']['rays_d'][indices]
        view_dirs = self.preprocessed_data_dict['nerf_data']['view_dirs'][indices]
        radii = self.preprocessed_data_dict['nerf_data']['radii'][indices]
        target_rgb = self.preprocessed_data_dict['nerf_data']['target_rgb'][indices]

        return_dict = {
            'rays_o': rays_o,
            'rays_d': rays_d,
            'view_dirs': view_dirs,
            'radii': radii,
            'target_rgb': target_rgb,
            'iter_num': iter_num
        }

        if self.ndc:
            rays_o_ndc = self.preprocessed_data_dict['nerf_data']['rays_o_ndc'][indices]
            rays_d_ndc = self.preprocessed_data_dict['nerf_data']['rays_d_ndc'][indices]
            radii_ndc = self.preprocessed_data_dict['nerf_data']['radii_ndc'][indices]
            return_dict['rays_o_ndc'] = rays_o_ndc
            return_dict['rays_d_ndc'] = rays_d_ndc
            return_dict['radii_ndc'] = radii_ndc
        return return_dict

    # --------------------------------- Inference methods ----------------------------------- #

    def create_test_data(self, pose: numpy.ndarray, view_pose: Optional[numpy.ndarray] = None, preprocess_pose: bool = True):
        pose = pose.copy()
        if view_pose is not None:
            view_pose = view_pose.copy()
        if preprocess_pose:
            processed_pose = self.preprocess_poses({
                'poses': pose[None],
                'translation_scale': self.model_configs['translation_scale'],
                'average_pose': numpy.array(self.model_configs['average_pose']),
            },
                train_mode=False)['poses'][0]
        else:
            processed_pose = pose.astype('float32')
        h, w = self.model_configs['resolution']
        fx, fy = self.model_configs['focal_length']
        rays_o, rays_d = self.get_rays(h, w, fx, fy, processed_pose)
        radii = self.get_radii(rays_d[None])[0]
        if view_pose is not None:
            processed_view_pose = self.preprocess_poses({
                'poses': view_pose[None],
                'translation_scale': self.model_configs['translation_scale'],
                'average_pose': numpy.array(self.model_configs['average_pose']),
            },
                train_mode=False)['poses'][0]
            _, view_rays_d = self.get_rays(h, w, fx, fy, processed_view_pose)
            view_dirs = self.get_view_dirs(view_rays_d)
        else:
            view_dirs = self.get_view_dirs(rays_d)

        near, far = self.model_configs['near'], self.model_configs['far']
        near = near * numpy.ones_like(rays_d[..., :1])
        far = far * numpy.ones_like(rays_d[..., :1])

        input_batch = {
            'rays_o': torch.from_numpy(rays_o.copy()).reshape(-1, 3),
            'rays_d': torch.from_numpy(rays_d).reshape(-1, 3),
            'radii': torch.from_numpy(radii).reshape(-1, 1),
            'view_dirs': torch.from_numpy(view_dirs).reshape(-1, 3),
            'near': torch.from_numpy(near).reshape(-1, 1),
            'far': torch.from_numpy(far).reshape(-1, 1),
        }

        if self.ndc:
            near = self.model_configs['near']
            rays_o_ndc, rays_d_ndc = self.get_ndc_rays(rays_o, rays_d, h, w, fx, fy, near)
            radii_ndc = self.get_radii_ndc(rays_o_ndc[None])[0]
            input_batch['rays_o_ndc'] = torch.from_numpy(rays_o_ndc).reshape(-1, 3)
            input_batch['rays_d_ndc'] = torch.from_numpy(rays_d_ndc).reshape(-1, 3)
            input_batch['radii_ndc'] = torch.from_numpy(radii_ndc).reshape(-1, 1)

            near_ndc, far_ndc = self.model_configs['near_ndc'], self.model_configs['far_ndc']
            near_ndc = near_ndc * numpy.ones_like(rays_d[..., :1])
            far_ndc = far_ndc * numpy.ones_like(rays_d[..., :1])
            input_batch['near_ndc'] = torch.from_numpy(near_ndc).reshape(-1, 1)
            input_batch['far_ndc'] = torch.from_numpy(far_ndc).reshape(-1, 1)

        input_batch = CommonUtils.move_to_device(input_batch, self.device)
        return input_batch

    def retrieve_inference_outputs(self, network_outputs: dict):
        h, w = self.model_configs['resolution']
        processed_outputs = self.post_process_output(network_outputs)
        if ('num_samples_fine' in self.configs['model']) and (self.configs['model']['num_samples_fine'] > 0):
            suffix = '_fine'
        else:
            suffix = '_coarse'
        image = self.post_process_image(processed_outputs[f'rgb{suffix}'].reshape(h, w, 3))
        depth = self.post_process_depth(processed_outputs[f'depth{suffix}'].reshape(h, w))
        depth_var = self.post_process_depth(processed_outputs[f'depth_var{suffix}'].reshape(h, w))
        return_dict = {
            'image': image,
            'depth': depth,
            'depth_var': depth_var,
        }
        if self.ndc:
            depth_ndc = self.post_process_depth(processed_outputs[f'depth_ndc{suffix}'].reshape(h, w))
            depth_var_ndc = self.post_process_depth(processed_outputs[f'depth_var_ndc{suffix}'].reshape(h, w))
            return_dict['depth_ndc'] = depth_ndc
            return_dict['depth_var_ndc'] = depth_var_ndc
        return return_dict

    # ------------------------------- Data pre/post process methods -------------------------------- #

    def preprocess_images(self, images: numpy.ndarray):
        images = images.astype('float32') / 255
        if self.configs['model']['white_bkgd']:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]
        return images

    def preprocess_poses(self, input_dict: dict, train_mode: bool):
        return_dict = {}
        poses = input_dict['poses']

        if train_mode:
            if 'bounds' in input_dict:
                bds = input_dict['bounds']
                # Rescale if bd_factor is provided
                if ('bd_factor' in input_dict) and (input_dict['bd_factor'] is not None):
                    bd_factor = input_dict['bd_factor']
                    sc = 1./(float(bds[0]) * bd_factor)
                else:
                    sc = 1
                poses[:, :3, 3] *= sc
                bds *= sc
                return_dict['sc'] = sc
                return_dict['bounds'] = bds
            if self.configs['data_loader']['recenter_camera_poses']:
                avg_pose = self.compute_average_pose(poses)
            else:
                avg_pose = numpy.eye(4)
            return_dict['average_pose'] = avg_pose
        else:
            sc = input_dict['translation_scale']
            poses[:, :3, 3] *= sc
            if 'bounds' in input_dict:
                bds = input_dict['bounds']
                bds *= sc
                return_dict['bounds'] = bds
            avg_pose = input_dict['average_pose']

        poses = self.recenter_poses(poses, avg_pose)
        poses = self.convert_pose_to_standard_coordinates(poses)

        if self.configs['data_loader']['spherify']:
            poses, render_poses, bds = self.spherify_poses(poses, return_dict['bounds'])
            return_dict['bounds'] = bds

        poses = poses[:, :3].astype(numpy.float32)
        return_dict['poses'] = poses
        return return_dict

    @staticmethod
    def recenter_poses(poses, avg_pose):
        centered_poses = avg_pose[None] @ numpy.linalg.inv(poses)
        return centered_poses

    def convert_pose_to_standard_coordinates(self, poses):
        # Convert from Colmap/RE10K convention to NeRF convention: (x,-y,-z) to (x,y,z)
        perm_matrix = numpy.eye(3)
        perm_matrix[1, 1] = -1
        perm_matrix[2, 2] = -1
        std_poses = self.change_coordinate_system(poses, perm_matrix)
        return std_poses

    @staticmethod
    def compute_average_pose(poses: numpy.ndarray):
        def normalize(x):
            return x / numpy.linalg.norm(x)

        def viewmatrix(z, up, pos):
            vec2 = normalize(z)
            vec1_avg = up
            vec0 = normalize(numpy.cross(vec1_avg, vec2))
            vec1 = normalize(numpy.cross(vec2, vec0))
            m = numpy.stack([vec0, vec1, vec2, pos], 1)
            bottom = numpy.array([0, 0, 0, 1])[None]
            matrix = numpy.concatenate([m, bottom], axis=0)
            return matrix

        # compute average pose in camera2world system
        rot_mats = poses[:, :3, :3]
        rot_inverted = numpy.transpose(rot_mats, axes=[0, 2, 1])
        translations = poses[:, :3, 3:]
        rotated_translations = -rot_inverted @ translations
        avg_translation = numpy.mean(rotated_translations, axis=0)[:, 0]

        vec2 = normalize(rot_inverted[:, :3, 2].sum(0))
        up = rot_inverted[:, :3, 1].sum(0)
        avg_pose_c2w = viewmatrix(vec2, up, avg_translation)
        avg_pose = numpy.linalg.inv(avg_pose_c2w)  # convert avg_pose to world2camera system
        return avg_pose

    @staticmethod
    def change_coordinate_system(poses: numpy.ndarray, p: numpy.ndarray):
        changed_poses = []
        for pose in poses:
            r = pose[:3, :3]
            t = pose[:3, 3:]
            rc = p.T @ r @ p
            tc = p @ t
            changed_pose = numpy.concatenate([numpy.concatenate([rc, tc], axis=1), pose[3:]], axis=0)
            changed_poses.append(changed_pose)
        changed_poses = numpy.stack(changed_poses)
        return changed_poses

    def spherify_poses(self, poses, bds):
        # TODO SNB: Understand where spherify is required/used
        p34_to_44 = lambda p : numpy.concatenate([p, numpy.tile(numpy.reshape(numpy.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)

        rays_d = poses[:,:3,2:3]
        rays_o = poses[:,:3,3:4]

        def min_line_dist(rays_o, rays_d):
            A_i = numpy.eye(3) - rays_d * numpy.transpose(rays_d, [0,2,1])
            b_i = -A_i @ rays_o
            pt_mindist = numpy.squeeze(-numpy.linalg.inv((numpy.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
            return pt_mindist

        pt_mindist = min_line_dist(rays_o, rays_d)

        center = pt_mindist
        up = (poses[:,:3,3] - center).mean(0)

        vec0 = self.normalize(up)
        vec1 = self.normalize(numpy.cross([.1,.2,.3], vec0))
        vec2 = self.normalize(numpy.cross(vec0, vec1))
        pos = center
        c2w = numpy.stack([vec1, vec2, vec0, pos], 1)

        poses_reset = numpy.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

        rad = numpy.sqrt(numpy.mean(numpy.sum(numpy.square(poses_reset[:,:3,3]), -1)))

        sc = 1./rad
        poses_reset[:,:3,3] *= sc
        bds *= sc
        rad *= sc

        centroid = numpy.mean(poses_reset[:,:3,3], 0)
        zh = centroid[2]
        radcircle = numpy.sqrt(rad**2-zh**2)
        new_poses = []

        for th in numpy.linspace(0.,2.*numpy.pi, 120):

            camorigin = numpy.array([radcircle * numpy.cos(th), radcircle * numpy.sin(th), zh])
            up = numpy.array([0,0,-1.])

            vec2 = self.normalize(camorigin)
            vec0 = self.normalize(numpy.cross(vec2, up))
            vec1 = self.normalize(numpy.cross(vec2, vec0))
            pos = camorigin
            p = numpy.stack([vec0, vec1, vec2, pos], 1)

            new_poses.append(p)

        new_poses = numpy.stack(new_poses, 0)

        new_poses = numpy.concatenate([new_poses, numpy.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
        poses_reset = numpy.concatenate([poses_reset[:,:3,:4], numpy.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)

        return poses_reset, new_poses, bds

    def post_process_output(self, output_batch: Union[torch.Tensor, list, tuple, dict]) -> Union[numpy.ndarray, list, tuple, dict]:
        if isinstance(output_batch, torch.Tensor):
            processed_batch = output_batch.detach().cpu().numpy()
        elif isinstance(output_batch, list) or isinstance(output_batch, tuple):
            processed_batch = []
            for list_element in output_batch:
                processed_batch.append(self.post_process_output(list_element))
        elif isinstance(output_batch, dict):
            processed_batch = {}
            for key in output_batch.keys():
                processed_batch[key] = self.post_process_output(output_batch[key])
        else:
            raise RuntimeError(f'How do I post process an object of type: {type(output_batch)}?')
        return processed_batch

    @staticmethod
    def post_process_image(rgb: numpy.ndarray):
        clipped_image = numpy.clip(rgb, a_min=0, a_max=1)
        uint8_image = numpy.round(clipped_image * 255).astype('uint8')
        return uint8_image

    @staticmethod
    def post_process_depth(depth: numpy.ndarray):
        clipped_depth = numpy.clip(depth, a_min=0, a_max=numpy.inf).astype('float32')
        return clipped_depth

    # ------------------------------------ Utilities ------------------------------------------- #
    @staticmethod
    def update_dict_recursively(tgt_dict, src_dict):
        for key in src_dict:
            if not isinstance(src_dict[key], dict):
                continue
            if key not in tgt_dict:
                tgt_dict[key] = {}
            tgt_dict[key].update(src_dict[key])
        return

    @staticmethod
    def add_residual_dict_items(tgt_dict, src_dict):
        for key in src_dict:
            if key not in tgt_dict:
                tgt_dict[key] = src_dict[key]
        return
