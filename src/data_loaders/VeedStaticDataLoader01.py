# Shree KRISHNAYa Namaha
# Loads VEED-Static Data for NeRF
# Modified from VSR006/VeedStaticDataLoader03.py
# Author: Nagabhushan S N
# Last Modified: 01/08/2022

import re
from pathlib import Path
from typing import Optional

import numpy
import pandas
import skimage.io

from data_loaders.DataLoaderParent import DataLoaderParent


class VeedStaticDataLoader(DataLoaderParent):
    def __init__(self, configs: dict, data_dirpath: Path, mode: Optional[str]):
        super(VeedStaticDataLoader, self).__init__()
        self.configs = configs
        self.data_dirpath = data_dirpath
        self.mode = mode
        self.scene_id = configs['data_loader']['scene_id']
        self.scene_name, self.seq_num = self.parse_scene_id(self.scene_id)
        return
    
    @staticmethod
    def parse_scene_id(scene_id):
        pattern = '(\w+\d{2})_seq(\d{2})'
        matcher = re.match(pattern, scene_id)
        if matcher is None:
            raise RuntimeError(f'Unable to parse scene_id: {scene_id}')
        scene_name = matcher.group(1)
        seq_num = int(matcher.group(2))
        return scene_name, seq_num

    def load_data(self):
        frame_nums = self.get_frame_nums()
        data_dict = {
            'frame_nums': frame_nums,
        }

        data_dict['nerf_data'] = self.load_nerf_data(data_dict)

        return data_dict

    def get_frame_nums(self):
        set_num = self.configs['data_loader']['train_set_num']
        video_datapath = self.data_dirpath / f'TrainTestSets/Set{set_num:02}/{self.mode.capitalize()}VideosData.csv'
        video_data = pandas.read_csv(video_datapath)
        frame_nums = video_data.loc[(video_data['video_name'] == self.scene_name) & (video_data['seq_num'] == self.seq_num)]['pred_frame_num'].to_numpy()
        return frame_nums

    def load_nerf_data(self, data_dict):
        frame_nums = data_dict['frame_nums']
        images_dirpath = self.data_dirpath / f'all_short/v3/RenderedData/{self.scene_name}/seq{self.seq_num:02}/rgb'
        if not images_dirpath.exists():
            print(f'{images_dirpath.as_posix()} does not exist, returning.')
            return
        images_paths = [images_dirpath / f'{frame_num:04}.png' for frame_num in frame_nums]
        images = [self.read_image(image_path) for image_path in images_paths]
        images = numpy.stack(images)

        depths_dirpath = self.data_dirpath / f'all_short/v3/RenderedData/{self.scene_name}/seq{self.seq_num:02}/depth'
        if not depths_dirpath.exists():
            print(f'{depths_dirpath.as_posix()} does not exist, returning.')
            return
        depths_paths = [depths_dirpath / f'{frame_num:04}.npy' for frame_num in frame_nums]
        depths = [self.read_depth(depth_path) for depth_path in depths_paths]
        depths = numpy.stack(depths)
        bounds = numpy.array([depths.min(), depths.max()])

        extrinsics_path = self.data_dirpath / f'all_short/v3/RenderedData/{self.scene_name}/seq{self.seq_num:02}/TransformationMatrices.csv'
        extrinsic_matrices = numpy.loadtxt(extrinsics_path.as_posix(), delimiter=',').reshape((-1, 4, 4))
        poses = extrinsic_matrices[frame_nums]

        intrinsic_matrix = self.camera_intrinsic_matrix()
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        h, w = images.shape[1:3]

        return_dict = {
            'images': images,
            'poses': poses,
            'resolution': (h, w),
            'focal_length': (fx, fy),
            'bounds': bounds,
        }
        return return_dict

    @staticmethod
    def read_image(path: Path, mmap_mode: str = None):
        if path.suffix in ['.png']:
            image = skimage.io.imread(path.as_posix())
        elif path.suffix == '.npy':
            image = numpy.load(path.as_posix(), mmap_mode=mmap_mode)
        else:
            raise RuntimeError(f'Unknown image format: {path.as_posix()}')
        return image

    @staticmethod
    def read_depth(path: Path):
        if path.suffix == '.npy':
            depth = numpy.load(path.as_posix())
        elif path.suffix == '.npz':
            with numpy.load(path.as_posix()) as depth_data:
                depth = depth_data['depth']
        elif path.suffix == '.exr':
            import OpenEXR
            import Imath

            exr_file = OpenEXR.InputFile(path.as_posix())
            raw_bytes = exr_file.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
            depth_vector = numpy.frombuffer(raw_bytes, dtype=numpy.float32)
            height = exr_file.header()['displayWindow'].max.y + 1 - exr_file.header()['displayWindow'].min.y
            width = exr_file.header()['displayWindow'].max.x + 1 - exr_file.header()['displayWindow'].min.x
            depth = numpy.reshape(depth_vector, (height, width))
        else:
            raise RuntimeError(f'Unknown depth format: {path.as_posix()}')
        return depth

    @staticmethod
    def camera_intrinsic_matrix(capture_width=1920, capture_height=1080, patch_start_point: tuple = (0, 0)):
        start_y, start_x = patch_start_point
        camera_intrinsics = numpy.eye(3)
        camera_intrinsics[0, 0] = 2100
        camera_intrinsics[0, 2] = capture_width / 2.0 - start_x
        camera_intrinsics[1, 1] = 2100
        camera_intrinsics[1, 2] = capture_height / 2.0 - start_y
        return camera_intrinsics
