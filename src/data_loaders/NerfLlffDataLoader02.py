# Shree KRISHNAYa Namaha
# Loads NeRF_LLFF Data for NeRF
# Extended from NerfLLffDataLoader01.py. Complete intrinsic matrices are loaded instead of focal length.
# Author: Nagabhushan S N
# Last Modified: 06/01/2023

from pathlib import Path
from typing import Optional

import numpy
import pandas
import skimage.io

from data_loaders.DataLoaderParent import DataLoaderParent


class NerfLlffDataLoader(DataLoaderParent):
    def __init__(self, configs: dict, data_dirpath: Path, mode: Optional[str]):
        super(NerfLlffDataLoader, self).__init__()
        self.configs = configs
        self.data_dirpath = data_dirpath
        self.mode = mode
        self.scene_name = self.configs['data_loader']['scene_id']
        self.resolution_suffix = self.configs['data_loader']['resolution_suffix']
        return

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
        frame_nums = video_data.loc[video_data['scene_name'] == self.scene_name]['pred_frame_num'].to_numpy()
        return frame_nums

    def load_nerf_data(self, data_dict):
        frame_nums = data_dict['frame_nums']
        images_dirpath = self.data_dirpath / f'all/DatabaseData/{self.scene_name}/rgb{self.resolution_suffix}'
        if not images_dirpath.exists():
            print(f'{images_dirpath.as_posix()} does not exist, returning.')
            return
        images_paths = [images_dirpath / f'{frame_num:04}.png' for frame_num in frame_nums]
        images = [self.read_image(image_path) for image_path in images_paths]
        images = numpy.stack(images)

        bds_path = self.data_dirpath / f'all/DatabaseData/{self.scene_name}/DepthBounds.csv'
        bds = numpy.loadtxt(bds_path.as_posix(), delimiter=',')[frame_nums]
        bounds = numpy.array([bds.min(), bds.max()])

        extrinsics_path = self.data_dirpath / f'all/DatabaseData/{self.scene_name}/CameraExtrinsics.csv'
        extrinsic_matrices = numpy.loadtxt(extrinsics_path.as_posix(), delimiter=',').reshape((-1, 4, 4))
        extrinsics = extrinsic_matrices[frame_nums]

        intrinsics_path = self.data_dirpath / f'all/DatabaseData/{self.scene_name}/CameraIntrinsics{self.resolution_suffix}.csv'
        intrinsic_matrices = numpy.loadtxt(intrinsics_path.as_posix(), delimiter=',').reshape((-1, 3, 3))
        intrinsics = intrinsic_matrices[frame_nums]
        h, w = images.shape[1:3]

        return_dict = {
            'images': images,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
            'resolution': (h, w),
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
