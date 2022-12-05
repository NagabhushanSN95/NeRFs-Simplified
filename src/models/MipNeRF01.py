# Shree KRISHNAya Namaha
# Extended from NeRF01.py for MipNeRF
# Author: Nagabhushan S N
# Last Modified: 05/12/2022

import math

import torch
import torch.nn.functional as F

from utils import CommonUtils01 as CommonUtils


class MipNeRF(torch.nn.Module):
    def __init__(self, configs: dict, model_configs: dict):
        super().__init__()
        self.configs = configs
        self.model_configs = model_configs
        self.device = CommonUtils.get_device(self.configs['device'])
        self.ndc = self.configs['data_loader']['ndc']
        self.coarse_sampling_needed = ('num_samples_coarse' in self.configs['model']) and \
                                      (self.configs['model']['num_samples_coarse'] > 0)
        self.fine_sampling_needed = ('num_samples_fine' in self.configs['model']) and \
                                    (self.configs['model']['num_samples_fine'] > 0)

        nerf_dict = self.create_nerf()
        self.pts_pos_enc_fn = nerf_dict['pts_pos_enc_fn']
        self.views_pos_enc_fn = nerf_dict['views_pos_enc_fn']
        self.model = nerf_dict['network']
        return

    def create_nerf(self):
        pts_pos_enc = IntegratedPositionalEncoder(self.configs['model']['points_positional_encoding_degree'], input_dim=3, diag=True)
        pts_pos_enc_fn = pts_pos_enc.encode
        pts_input_dim = pts_pos_enc.output_dim

        views_input_dim = 0
        views_pos_enc_fn = None
        if self.configs['model']['use_view_dirs']:
            views_pos_enc = PositionalEncoder(self.configs['model']['views_positional_encoding_degree'], input_dim=3, append_identity=True)
            views_pos_enc_fn = views_pos_enc.encode
            views_input_dim = views_pos_enc.output_dim

        mlp = MLP(self.configs, pts_input_dim, views_input_dim).to(self.device)

        return_dict = {
            'network': mlp,
            'pts_pos_enc_fn': pts_pos_enc_fn,
            'views_pos_enc_fn': views_pos_enc_fn,
        }

        return return_dict

    def forward(self, input_batch: dict, retraw: bool = False):
        render_output_dict = self.render(input_batch, retraw=self.training or retraw)
        return render_output_dict

    def render(self, input_dict: dict, retraw: bool = False):
        all_ret = self.batchify_rays(input_dict, retraw)
        return all_ret

    def batchify_rays(self, input_dict: dict, retraw):
        """
        Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = {}
        num_rays = input_dict['rays_o'].shape[0]
        chunk = self.configs['model']['chunk']
        for i in range(0, num_rays, chunk):
            render_rays_dict = {}
            for key in input_dict:
                if isinstance(input_dict[key], torch.Tensor):
                    render_rays_dict[key] = input_dict[key][i:i+chunk]
                elif isinstance(input_dict[key], list):
                    render_rays_dict[key] = []
                    for i1 in range(len(input_dict[key])):
                        render_rays_dict[key].append(input_dict[key][i1][i:i+chunk])
                else:
                    render_rays_dict[key] = input_dict[key]

            # ret = self.render_rays(rays_flat[i:i+chunk], **kwargs)
            ret = self.render_rays(render_rays_dict, retraw)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = self.merge_mini_batch_data(all_ret)
        return all_ret

    def render_rays(self, input_dict: dict, retraw):
        rays_o = input_dict['rays_o']
        rays_d = input_dict['rays_d']
        radii = input_dict['radii']
        if self.ndc:
            rays_o_ndc = input_dict['rays_o_ndc']
            rays_d_ndc = input_dict['rays_d_ndc']
            radii_ndc = input_dict['radii_ndc']

        if self.configs['model']['use_view_dirs']:
            # provide ray directions as input
            view_dirs = input_dict['view_dirs']

        return_dict = {}
        if self.coarse_sampling_needed:
            z_vals_coarse = self.get_z_vals_coarse(input_dict)

            if not self.ndc:
                means_coarse, covs_coarse = self.get_gaussian_moments(rays_o, rays_d, radii, z_vals_coarse)
            else:
                means_coarse, covs_coarse = self.get_gaussian_moments(rays_o_ndc, rays_d_ndc, radii_ndc, z_vals_coarse)

            network_input_coarse = {
                'means': means_coarse,
                'covariances': covs_coarse,
            }
            if self.configs['model']['use_view_dirs']:
                network_input_coarse['view_dirs'] = view_dirs

            network_output_coarse = self.run_network(network_input_coarse, self.model)
            if not self.ndc:
                outputs_coarse = self.volume_rendering(network_output_coarse, z_vals=z_vals_coarse, rays_d=rays_d)
            else:
                outputs_coarse = self.volume_rendering(network_output_coarse, z_vals_ndc=z_vals_coarse,
                                                       rays_d_ndc=rays_d_ndc, rays_o=rays_o, rays_d=rays_d)
            weights_coarse = outputs_coarse['weights']

            return_dict['z_vals_coarse'] = z_vals_coarse
            for key in outputs_coarse:
                return_dict[f'{key}_coarse'] = outputs_coarse[key]
            if retraw:
                for key in network_output_coarse.keys():
                    return_dict[f'raw_{key}_coarse'] = network_output_coarse[key]

        if self.fine_sampling_needed:
            z_vals_fine = self.get_z_vals_fine(z_vals_coarse, weights_coarse)
            if not self.ndc:
                means_fine, covs_fine = self.get_gaussian_moments(rays_o, rays_d, radii, z_vals_fine)
            else:
                means_fine, covs_fine = self.get_gaussian_moments(rays_o_ndc, rays_d_ndc, radii_ndc, z_vals_fine)

            network_input_fine = {
                'means': means_fine,
                'covariances': covs_fine,
            }
            if self.configs['model']['use_view_dirs']:
                network_input_fine['view_dirs'] = view_dirs

            network_output_fine = self.run_network(network_input_fine, self.model)
            if not self.ndc:
                outputs_fine = self.volume_rendering(network_output_fine, z_vals=z_vals_fine, rays_d=rays_d)
            else:
                outputs_fine = self.volume_rendering(network_output_fine, z_vals_ndc=z_vals_fine, rays_d_ndc=rays_d_ndc,
                                                     rays_o=rays_o, rays_d=rays_d)

            return_dict['z_vals_fine'] = z_vals_fine
            for key in outputs_fine:
                return_dict[f'{key}_fine'] = outputs_fine[key]
            if retraw:
                for key in network_output_fine.keys():
                    return_dict[f'raw_{key}_fine'] = network_output_fine[key]

        if not self.training:
            del return_dict['z_vals_coarse'], return_dict['visibility_coarse'], return_dict['weights_coarse']
            del return_dict['z_vals_fine'], return_dict['visibility_fine'], return_dict['weights_fine']
        return return_dict

    def get_z_vals_coarse(self, input_dict: dict):
        num_rays = input_dict['rays_o'].shape[0]
        if not self.ndc:
            near, far = input_dict['near'], input_dict['far']
        else:
            near, far = input_dict['near_ndc'], input_dict['far_ndc']

        perturb = self.configs['model']['perturb']
        if not self.training:
            perturb = False
        lindisp = self.configs['model']['lindisp']

        num_samples_coarse = self.configs['model']['num_samples_coarse']
        t_vals = torch.linspace(0., 1., steps=num_samples_coarse + 1)
        if not lindisp:
            z_vals_coarse = near * (1.-t_vals) + far * t_vals
        else:
            z_vals_coarse = 1./(1. / near * (1.-t_vals) + 1. / far * t_vals)

        z_vals_coarse = z_vals_coarse.expand([num_rays, num_samples_coarse + 1])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
            upper = torch.cat([mids, z_vals_coarse[..., -1:]], -1)
            lower = torch.cat([z_vals_coarse[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals_coarse.shape)

            z_vals_coarse = lower + (upper - lower) * t_rand
        return z_vals_coarse

    def get_z_vals_fine(self, z_vals_coarse, weights_coarse):
        num_samples_fine = self.configs['model']['num_samples_fine']
        perturb = self.configs['model']['perturb']
        if not self.training:
            perturb = False
        stop_grad = self.configs['model']['stop_grad']
        resample_padding = self.configs['model']['resample_padding']

        weights_pad = torch.cat([weights_coarse[..., :1], weights_coarse, weights_coarse[..., -1:]], dim=-1)
        weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
        weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

        # Add in a constant (the sampling function will renormalize the PDF).
        weights_coarse = weights_blur + resample_padding

        z_vals_fine = self.sorted_piecewise_constant_pdf(z_vals_coarse, weights_coarse, num_samples_fine + 1, perturb)
        if stop_grad:
            z_vals_fine = z_vals_fine.detach()

        # z_vals_mid = .5 * (z_vals_coarse[...,1:] + z_vals_coarse[...,:-1])
        # z_samples = self.sample_pdf(z_vals_mid, weights_coarse[...,1:-1], num_samples_fine, det=(not perturb))
        # z_samples = z_samples.detach()
        #
        # z_vals_fine, _ = torch.sort(torch.cat([z_vals_coarse, z_samples], -1), -1)
        return z_vals_fine

    def get_gaussian_moments(self, rays_o, rays_d, radii, z_vals):
        z0 = z_vals[..., :-1]
        z1 = z_vals[..., 1:]
        if self.configs['model']['ray_shape'] == 'cone':
            means, covs = self.conical_frustum_to_gaussian(rays_d, z0, z1, radii, diag=True)
        elif self.configs['model']['ray_shape'] == 'cylinder':
            means, covs = self.cylinder_to_gaussian(rays_d, z0, z1, radii, diag=True)
        else:
            raise RuntimeError(f"Unknown ray shape: {self.configs['model']['ray_shape']}")
        means = means + rays_o[..., None, :]
        return means, covs

    def conical_frustum_to_gaussian(self, d, t0, t1, base_radius, diag, stable=True):
        """Approximate a conical frustum as a Gaussian distribution (mean+cov).

        Assumes the ray is originating from the origin, and base_radius is the
        radius at dist=1. Doesn't assume `d` is normalized.

        Args:
          d: jnp.float32 3-vector, the axis of the cone
          t0: float, the starting distance of the frustum.
          t1: float, the ending distance of the frustum.
          base_radius: float, the scale of the radius as a function of distance.
          diag: boolean, whether or the Gaussian will be diagonal or full-covariance.
          stable: boolean, whether or not to use the stable computation described in
            the paper (setting this to False will cause catastrophic failure).

        Returns:
          a Gaussian (mean and covariance).
        """
        if stable:
            mu = (t0 + t1) / 2
            hw = (t1 - t0) / 2
            t_mean = mu + (2 * mu * hw ** 2) / (3 * mu ** 2 + hw ** 2)
            t_var = (hw ** 2) / 3 - (4 / 15) * ((hw ** 4 * (12 * mu ** 2 - hw ** 2)) /
                                                (3 * mu ** 2 + hw ** 2) ** 2)
            r_var = base_radius ** 2 * ((mu ** 2) / 4 + (5 / 12) * hw ** 2 - 4 / 15 *
                                        (hw ** 4) / (3 * mu ** 2 + hw ** 2))
        else:
            t_mean = (3 * (t1 ** 4 - t0 ** 4)) / (4 * (t1 ** 3 - t0 ** 3))
            r_var = base_radius ** 2 * (3 / 20 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3))
            t_mosq = 3 / 5 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3)
            t_var = t_mosq - t_mean ** 2
        mean, cov = self.lift_gaussian(d, t_mean, t_var, r_var, diag)
        return mean, cov

    def cylinder_to_gaussian(self, d, t0, t1, radius, diag):
        """Approximate a cylinder as a Gaussian distribution (mean+cov).

        Assumes the ray is originating from the origin, and radius is the
        radius. Does not renormalize `d`.

        Args:
          d: jnp.float32 3-vector, the axis of the cylinder
          t0: float, the starting distance of the cylinder.
          t1: float, the ending distance of the cylinder.
          radius: float, the radius of the cylinder
          diag: boolean, whether or the Gaussian will be diagonal or full-covariance.

        Returns:
          a Gaussian (mean and covariance).
        """
        t_mean = (t0 + t1) / 2
        r_var = radius ** 2 / 4
        t_var = (t1 - t0) ** 2 / 12
        mean, cov = self.lift_gaussian(d, t_mean, t_var, r_var, diag)
        return mean, cov

    @staticmethod
    def lift_gaussian(d, t_mean, t_var, r_var, diag):
        """Lift a Gaussian defined along a ray to 3D coordinates."""
        mean = d[..., None, :] * t_mean[..., None]

        d_mag_sq = torch.maximum(torch.Tensor([1e-10]), torch.sum(d ** 2, dim=-1, keepdim=True))

        if diag:
            d_outer_diag = d ** 2
            null_outer_diag = 1 - d_outer_diag / d_mag_sq
            t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
            xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
            cov_diag = t_cov_diag + xy_cov_diag
            return mean, cov_diag
        else:
            d_outer = d[..., :, None] * d[..., None, :]
            eye = torch.eye(d.shape[-1])
            null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
            t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
            xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
            cov = t_cov + xy_cov
            return mean, cov

    @staticmethod
    def sorted_piecewise_constant_pdf(bins, weights, num_samples, perturb):
        """Piecewise-Constant PDF sampling from sorted bins.

        Args:
          bins: [batch_size, num_bins + 1].
          weights: [batch_size, num_bins].
          num_samples: int, the number of samples.
          perturb: bool, use randomized samples.

        Returns:
          t_samples: [batch_size, num_samples].
        """
        # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
        # avoids NaNs when the input is zeros or small, but has no effect otherwise.
        eps = 1e-5
        weight_sum = torch.sum(weights, dim=-1, keepdim=True)
        padding = torch.maximum(torch.Tensor([0]), eps - weight_sum)
        weights += padding / weights.shape[-1]
        weight_sum += padding

        # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
        # starts with exactly 0 and ends with exactly 1.
        pdf = weights / weight_sum
        cdf = torch.minimum(torch.Tensor([1]), torch.cumsum(pdf[..., :-1], dim=-1))
        cdf = torch.cat([torch.zeros(list(cdf.shape[:-1]) + [1]), cdf, torch.ones(list(cdf.shape[:-1]) + [1])], dim=-1)

        # Draw uniform samples.
        if perturb:
            s = 1 / num_samples
            u = torch.arange(num_samples) * s
            u = u[None, :] + torch.rand(list(cdf.shape[:-1]) + [num_samples]) * (s - torch.finfo(torch.float32).eps)
            # `u` is in [0, 1) --- it can be zero, but it can never be 1.
            u = torch.minimum(u, torch.Tensor([1. - torch.finfo(torch.float32).eps]))
        else:
            # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
            u = torch.linspace(0., 1. - torch.finfo(torch.float32).eps, num_samples)
            u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

        # TODO: check if torch.searchsorted() simplifies this
        # Identify the location in `cdf` that corresponds to a random sample.
        # The final `True` index in `mask` will be the start of the sampled interval.
        mask = u[..., None, :] >= cdf[..., :, None]

        def find_interval(x):
            # Grab the value where `mask` switches from True to False, and vice versa.
            # This approach takes advantage of the fact that `x` is sorted.
            x0 = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)[0]
            x1 = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2)[0]
            return x0, x1

        bins_g0, bins_g1 = find_interval(bins)
        cdf_g0, cdf_g1 = find_interval(cdf)

        t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
        samples = bins_g0 + t * (bins_g1 - bins_g0)
        return samples

    # Hierarchical sampling (section 5.2)
    @staticmethod
    def sample_pdf(bins, weights, N_samples, det=False):
        # Get pdf
        weights = weights + 1e-5  # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

        # Take uniform samples
        if det:
            u = torch.linspace(0., 1., steps=N_samples)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

        # Invert CDF
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds-1), inds-1)
        above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

        # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = (cdf_g[...,1]-cdf_g[...,0])
        denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
        t = (u-cdf_g[...,0])/denom
        samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

        return samples

    def run_network(self, input_dict, nerf_mlp):
        """
        Prepares inputs and applies network 'nerf_mlp'.
        """
        means, covs = input_dict['means'], input_dict['covariances']
        if self.configs['model']['disable_integration']:
            covs = torch.zeros_like(covs)
        encoded_pts = self.pts_pos_enc_fn((means, covs))
        encoded_pts_flat = torch.reshape(encoded_pts, [-1, encoded_pts.shape[-1]])
        network_input_dict = {
            'pts': encoded_pts_flat,
        }

        if self.configs['model']['use_view_dirs']:
            view_dirs = input_dict['view_dirs']
            if view_dirs.ndim == 2:
                view_dirs = view_dirs[:,None].expand(input_dict['means'].shape)
            encoded_view_dirs = self.views_pos_enc_fn(view_dirs)
            encoded_view_dirs_flat = torch.reshape(encoded_view_dirs, [-1, encoded_view_dirs.shape[-1]])
            network_input_dict['view_dirs'] = encoded_view_dirs_flat

        network_output_dict = self.batchify(nerf_mlp)(network_input_dict)

        for k, v in network_output_dict.items():
            if isinstance(v, torch.Tensor):
                network_output_dict[k] = torch.reshape(v, list(input_dict['means'].shape[:-1]) + [v.shape[-1]])
            elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
                for i in range(len(v)):
                    network_output_dict[k][i] = torch.reshape(v[i], list(input_dict['means'].shape[:-1]) + [v[i].shape[-1]])
            else:
                raise NotImplementedError
        return network_output_dict

    def batchify(self, nerf_mlp):
        """Constructs a version of 'nerf_mlp' that applies to smaller batches.
        """
        chunk = self.configs['model']['netchunk']
        if chunk is None:
            return nerf_mlp

        def ret(input_dict: dict):
            num_pts = input_dict['pts'].shape[0]
            network_output_chunks = {}
            for i in range(0, num_pts, chunk):
                network_input_chunk = {}
                for key in input_dict:
                    if isinstance(input_dict[key], torch.Tensor):
                        network_input_chunk[key] = input_dict[key][i:i+chunk]
                    elif isinstance(input_dict[key], list) and isinstance(input_dict[key][0], torch.Tensor):
                        network_input_chunk[key] = [input_dict[key][j][i:i+chunk] for j in range(len(input_dict[key]))]
                    else:
                        raise RuntimeError(key)

                network_output_chunk = nerf_mlp(network_input_chunk)

                for k in network_output_chunk.keys():
                    if k not in network_output_chunks:
                        network_output_chunks[k] = []
                    if isinstance(network_output_chunk[k], torch.Tensor):
                        network_output_chunks[k].append(network_output_chunk[k])
                    elif isinstance(network_output_chunk[k], list) and isinstance(network_output_chunk[k][0], torch.Tensor):
                        if len(network_output_chunks[k]) == 0:
                            for j in range(len(network_output_chunk[k])):
                                network_output_chunks[k].append([])
                        for j in range(len(network_output_chunk[k])):
                            network_output_chunks[k][j].append(network_output_chunk[k][j])
                    else:
                        raise RuntimeError

            for k in network_output_chunks:
                if isinstance(network_output_chunks[k][0], torch.Tensor):
                    network_output_chunks[k] = torch.cat(network_output_chunks[k], dim=0)
                elif isinstance(network_output_chunks[k][0], list) and isinstance(network_output_chunks[k][0][0], torch.Tensor):
                    for j in range(len(network_output_chunks[k])):
                        network_output_chunks[k][j] = torch.cat(network_output_chunks[k][j], dim=0)
                else:
                    raise NotImplementedError
            return network_output_chunks
        return ret

    def volume_rendering(self, network_output_dict,
                         z_vals=None, rays_o=None, rays_d=None,
                         z_vals_ndc=None, rays_d_ndc=None):
        if not self.ndc:
            z_mids = 0.5 * (z_vals[..., :-1] + z_vals[..., 1:])
            z_dists = z_vals[...,1:] - z_vals[...,:-1]  # [N_rays, N_samples]
            delta = z_dists * torch.norm(rays_d[...,None,:], dim=-1)
        else:
            z_mids_ndc = 0.5 * (z_vals_ndc[..., :-1] + z_vals_ndc[..., 1:])
            z_dists = z_vals_ndc[...,1:] - z_vals_ndc[...,:-1]  # [N_rays, N_samples]
            delta = z_dists * torch.norm(rays_d_ndc[...,None,:], dim=-1)

        rgb = network_output_dict['rgb']  # [N_rays, N_samples, 3]
        sigma = network_output_dict['sigma'][..., 0]  # [N_rays, N_samples]

        alpha = 1. - torch.exp(-sigma * delta)  # [N_rays, N_samples]
        visibility = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        weights = alpha * visibility
        rgb_map = torch.sum(weights[...,None] * rgb, dim=-2)  # [N_rays, 3]

        acc_map = torch.sum(weights, dim=-1)
        if not self.ndc:
            depth_map = torch.sum(weights * z_mids, dim=-1) / (acc_map + 1e-6)
            # TODO: what does jnp.nan_to_num() do?
            depth_map = torch.clip(depth_map, z_vals[..., 0], z_vals[..., -1])  # TODO: is ellipsis required?
            depth_var_map = torch.sum(weights * torch.square(z_mids - depth_map[..., None]), dim=-1)
        else:
            depth_map_ndc = torch.sum(weights * z_mids_ndc, dim=-1) / (acc_map + 1e-6)
            depth_map_ndc = torch.clip(depth_map_ndc, z_vals_ndc[..., 0], z_vals_ndc[..., -1])
            depth_var_map_ndc = torch.sum(weights * torch.square(z_mids_ndc - depth_map_ndc[..., None]), dim=-1)
            z_vals = self.convert_depth_from_ndc(z_vals_ndc, rays_o, rays_d)
            z_mids = 0.5 * (z_vals[..., :-1] + z_vals[..., 1:])
            depth_map = torch.sum(weights * z_mids, dim=-1) / (acc_map + 1e-6)
            depth_map = torch.clip(depth_map, z_vals[..., 0], z_vals[..., -1])
            depth_var_map = torch.sum(weights * torch.square(z_mids - depth_map[..., None]), dim=-1)

        if self.configs['model']['white_bkgd']:
            rgb_map = rgb_map + (1.-acc_map[...,None])

        return_dict = {
            'rgb': rgb_map,
            'acc': acc_map,
            'alpha': alpha,
            'visibility': visibility,
            'weights': weights,
            'depth': depth_map,
            'depth_var': depth_var_map,
        }

        if self.ndc:
            return_dict['depth_ndc'] = depth_map_ndc
            return_dict['depth_var_ndc'] = depth_var_map_ndc
        return return_dict

    @staticmethod
    def convert_depth_from_ndc(z_vals_ndc, rays_o, rays_d):
        """
        Converts depth in ndc to actual values
        From ndc write up, t' is z_vals_ndc and t is z_vals.
        t' = 1 - oz / (oz + t * dz)
        t = (oz / dz) * (1 / (1-t') - 1)
        But due to the final trick, oz is shifted. So, the actual oz = oz + tn * dz
        Overall t_act = t + tn = ((oz + tn * dz) / dz) * (1 / (1 - t') - 1) + tn
        """
        near = 1  # TODO: do not hard-code
        oz = rays_o[..., 2:3]
        dz = rays_d[..., 2:3]
        tn = -(near + oz) / dz
        constant = torch.where(z_vals_ndc == 1., 1e-3, 0.)
        # depth = (((oz + tn * dz) / (1 - z_vals_ndc + constant)) - oz) / dz
        depth = (oz + tn * dz) / dz * (1 / (1 - z_vals_ndc + constant) - 1) + tn
        return depth

    @staticmethod
    def append_to_dict_element(data_dict: dict, key: str, new_element):
        """
        Appends the `new_element` to the list identified by `key` in the dictionary `data_dict`
        """
        if key not in data_dict:
            data_dict[key] = []
        data_dict[key].append(new_element)
        return

    @staticmethod
    def merge_mini_batch_data(data_chunks: dict):
        merged_data = {}
        for key in data_chunks:
            if isinstance(data_chunks[key][0], torch.Tensor):
                merged_data[key] = torch.cat(data_chunks[key], dim=0)
            elif isinstance(data_chunks[key][0], list):
                merged_data[key] = []
                for i in range(len(data_chunks[key][0])):
                    merged_data[key].append(torch.cat([data_chunks[key][j][i] for j in range(len(data_chunks[key]))], dim=0))
            else:
                raise NotImplementedError
        return merged_data


class PositionalEncoder:
    def __init__(self, degree: int, input_dim: int, append_identity: bool = True):
        self.degree = degree
        self.input_dim = input_dim
        self.append_identity = append_identity

        self.output_dim = (self.input_dim * self.degree * 2) + (int(self.append_identity) * self.input_dim)
        self.scales = torch.Tensor([2 ** i for i in range(self.degree)])
        return

    def encode(self, x):
        xb = torch.reshape((x[..., None, :] * self.scales[:, None]), list(x.shape[:-1]) + [-1])
        encoded_x = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.append_identity:
            encoded_x = torch.cat([x] + [encoded_x], dim=-1)
        return encoded_x


class IntegratedPositionalEncoder:
    def __init__(self, degree: int, input_dim: int, diag: bool = True, *, num_dims: int = None):
        self.degree = degree
        self.input_dim = input_dim
        self.diag = diag

        self.output_dim = self.input_dim * self.degree * 2
        if self.diag:
            self.scales = torch.Tensor([2 ** i for i in range(self.degree)])
        else:
            self.basis = torch.cat([2 ** i * torch.eye(num_dims) for i in range(self.degree)], 1)
        return

    def encode(self, x_coord):
        if self.diag:
            x, x_cov_diag = x_coord
            shape = list(x.shape[:-1]) + [-1]
            y = torch.reshape(x[..., None, :] * self.scales[:, None], shape)
            y_var = torch.reshape(x_cov_diag[..., None, :] * self.scales[:, None] ** 2, shape)
        else:
            # TODO: This block has not been verified
            x, x_cov = x_coord
            y = torch.matmul(x, self.basis)
            # Get the diagonal of a covariance matrix (ie, variance). This is equivalent
            # to jax.vmap(jnp.diag)((basis.T @ covs) @ basis).
            y_var = torch.sum((torch.matmul(x_cov, self.basis)) * self.basis, dim=-2)

        encoded_x = self.expected_sin(
            torch.cat([y, y + 0.5 * math.pi], dim=-1),
            torch.cat([y_var] * 2, dim=-1))[0]
        return encoded_x

    def expected_sin(self, x, x_var):
        """Estimates mean and variance of sin(z), z ~ N(x, var)."""
        # When the variance is wide, shrink sin towards zero.
        y = torch.exp(-0.5 * x_var) * self.safe_sin(x)
        y_var = torch.maximum(torch.Tensor([0]), 0.5 * (1 - torch.exp(-2 * x_var) * self.safe_cos(2 * x)) - y ** 2)
        return y, y_var

    @staticmethod
    def safe_trig_helper(x, fn, t=100 * math.pi):
        return fn(torch.where(torch.abs(x) < t, x, x % t))

    def safe_cos(self, x):
        """jnp.cos() on a TPU may NaN out for large values."""
        return self.safe_trig_helper(x, torch.cos)

    def safe_sin(self, x):
        """jnp.sin() on a TPU may NaN out for large values."""
        return self.safe_trig_helper(x, torch.sin)


class MLP(torch.nn.Module):
    def __init__(self, configs, pts_input_dim=3, views_input_dim=3, coarse=True):
        """
        """
        super(MLP, self).__init__()
        self.configs = configs
        self.D = self.configs['model']['netdepth']
        self.W = self.configs['model']['netwidth']
        self.pts_input_dim = pts_input_dim
        self.views_input_dim = views_input_dim
        self.skips = [4]
        self.view_dep_rgb = self.configs['model']['view_dependent_rgb']
        self.raw_noise_std = self.configs['model']['raw_noise_std']
        self.sigma_bias = self.configs['model']['sigma_bias']
        self.rgb_padding = self.configs['model']['rgb_padding']

        self.pts_linears = torch.nn.ModuleList(
            [torch.nn.Linear(self.pts_input_dim, self.W)] +
            [torch.nn.Linear(self.W, self.W) if i not in self.skips else torch.nn.Linear(self.W + self.pts_input_dim, self.W) for i in range(self.D - 1)]
        )
        if self.view_dep_rgb:
            self.views_linears = torch.nn.ModuleList([torch.nn.Linear(self.views_input_dim + self.W, self.W // 2)])

        pts_output_dim = 1  # sigma
        views_output_dim = 0
        if self.view_dep_rgb:
            views_output_dim += 3
        else:
            pts_output_dim += 3

        self.pts_output_linear = torch.nn.Linear(self.W, pts_output_dim)
        if self.view_dep_rgb:
            self.feature_linear = torch.nn.Linear(self.W, self.W)
            self.views_output_linear = torch.nn.Linear(self.W // 2, views_output_dim)
        return

    def forward(self, input_batch):
        input_pts = input_batch['pts']
        input_views = input_batch['view_dirs']
        output_batch = {}

        pts_outputs = self.get_view_independent_outputs(input_pts)
        output_batch.update(pts_outputs)

        if self.view_dep_rgb:
            view_outputs = self.get_view_dependent_outputs(pts_outputs, input_views)
            output_batch.update(view_outputs)

        if 'feature' in output_batch: del output_batch['feature']
        return output_batch

    def get_view_independent_outputs(self, input_pts):
        output_dict = {}
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        pts_output = self.pts_output_linear(h)
        ch_i = 0  # Denotes number of channels which have already been taken output

        sigma = pts_output[..., ch_i:ch_i + 1]
        if self.training and (self.raw_noise_std > 0.):
            noise = torch.randn(sigma.shape) * self.raw_noise_std
            sigma = sigma + noise
        sigma = F.softplus(sigma + self.sigma_bias)
        output_dict['sigma'] = sigma
        ch_i += 1

        if self.view_dep_rgb:
            feature = self.feature_linear(h)
            output_dict['feature'] = feature
        else:
            rgb = pts_output[..., ch_i:ch_i+3]
            rgb = torch.sigmoid(rgb)
            rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
            output_dict['rgb'] = rgb
            ch_i += 3
        return output_dict

    def get_view_dependent_outputs(self, pts_outputs, input_views):
        output_dict = {}

        if self.view_dep_rgb:
            feature = pts_outputs['feature']
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            view_outputs = self.views_output_linear(h)
            ch_i = 0  # Denotes number of channels which have already been taken output

            rgb = view_outputs[..., ch_i:ch_i+3]
            rgb = torch.sigmoid(rgb)
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
            output_dict['rgb'] = rgb
            ch_i += 3
        return output_dict
