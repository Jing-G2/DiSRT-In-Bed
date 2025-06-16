import math
import numpy as np
import os
from scipy.ndimage.filters import gaussian_filter

from utils.constants import *
import utils.data_utils.data_utils as data_utils


class BPDataset:
    def __init__(self, data_path):
        super(BPDataset, self).__init__()
        self.pressure_data_dir = os.path.join(data_path, "synth")
        self.depth_data_dir = os.path.join(data_path, "synth_depth")
        self.pmap_data_dir = os.path.join(data_path, "GT_BP_data", "bp2")
        self.verts_data_dir = os.path.join(data_path, "GT_BP_data", "bp2")

        x_y_offset_synth = [12, -35]
        z_adj = -0.075
        self.z_adj_all = np.array(
            24 * [-x_y_offset_synth[0], x_y_offset_synth[1], z_adj * 1000]
        )
        self.z_adj_one = np.array(
            1
            * [-286.0 - x_y_offset_synth[0], -286.0 + x_y_offset_synth[1], z_adj * 1000]
        )
        self.gender_female = [1, 0]
        self.mass_normalizer_female = (62.5, 0.06878933937454557)
        self.gender_male = [0, 1]
        self.mass_normalizer_male = (78.4, 0.0828308574658067)

    def prepare_dataset(
        self,
        data_lines,
        cover_str_list=["uncover", "cover1"],
        syn_ratio=1,
        load_verts=False,
        load_pressure=False,
        for_infer=False,
    ):
        """Prepare BP dataset for training/inference
        Args:
            data_lines: list of file names
            cover_str_list: list of cover types
            syn_ratio: synthetic ratio
            load_verts: load vertices data
            for_infer: whether to prepare data for inference
            load_pressure: load pressure data
        """
        data_pressure_x = None
        data_depth_x = None
        data_label_y = None
        data_pmap_y = None
        data_verts_y = None
        data_names_y = None

        for file_name in data_lines:
            pressure_x = []
            depth_x = []
            label_y = []
            names_y = []

            pressure_file = os.path.join(self.pressure_data_dir, file_name)
            depth_file = os.path.join(
                self.depth_data_dir, file_name.split(".")[0] + "_depthims.p"
            )
            pmap_file = os.path.join(
                self.pmap_data_dir, file_name.split(".")[0] + "_gt_pmaps.npy"
            )
            verts_file = os.path.join(
                self.verts_data_dir, file_name.split(".")[0] + "_gt_vertices.npy"
            )

            # -------------------------
            # file_name, e.g.: train_slp_lay_f_1to40_8549.p
            file_name_split = file_name.split(".")[0].split("_")
            file_tag = f"{file_name_split[4]}_{file_name_split[2]}"

            if "f" in file_name:
                gender = self.gender_female
                mass_normalizer = self.mass_normalizer_female
                file_tag += "_f"
            else:
                gender = self.gender_male
                mass_normalizer = self.mass_normalizer_male
                file_tag += "_m"

            # -------------------------
            pressure_dict = data_utils.load_pickle(pressure_file)
            depth_dict = data_utils.load_pickle(depth_file)

            saved = 0
            limit = (
                int(syn_ratio * len(pressure_dict["images"]))
                if syn_ratio < 1
                else len(pressure_dict["images"])
            )
            if pressure_dict is not None and depth_dict is not None:
                for i in np.arange(len(pressure_dict["images"])):
                    image = pressure_dict["images"][i]
                    if not math.isnan(np.sum(image)):
                        # -------------------------
                        # Prepare Depth Image
                        depth = []
                        names = []
                        if "uncover" in cover_str_list:
                            uncover_depth = depth_dict["overhead_depthcam_noblanket"][
                                i
                            ].astype(np.float32)
                            # uncover_depth[127:, :] = BP_BED_HEIGHT
                            uncover_depth[127:, :] = (
                                0.0  # from unpack_depth_batch_lib, last row of depth image is set to zero
                            )
                            depth.append(uncover_depth)
                            names.append(f"synth_{file_tag}_{i}_uncover")
                        if "cover1" in cover_str_list:
                            cover_depth = depth_dict["overhead_depthcam"][i].astype(
                                np.float32
                            )
                            # cover_depth[127:, :] = BP_BED_HEIGHT
                            cover_depth[127:, :] = (
                                0.0  # from unpack_depth_batch_lib, last row of depth image is set to zero
                            )
                            depth.append(cover_depth)
                            names.append(f"synth_{file_tag}_{i}_cover1")

                        # getting 2:1, blanket:no blanket images in depth
                        if (i + 1) % 3 == 0:
                            depth_x.append([depth[0], depth[0]])
                            names_y.append(names[0])
                        else:
                            depth_x.append([depth[0], depth[1]])
                            names_y.append(names[1])

                        # use all depth images, 1:1 for blanket:no blanket
                        # for j in range(len(depth)):
                        #     depth_x.append([depth[0], depth[j]])
                        #     names_y.append(names[j])

                        # -------------------------
                        # Prepare Pressure Image
                        if load_pressure:
                            pressure_image = gaussian_filter(
                                image.reshape(64, 27).astype(np.float32), sigma=0.5
                            )
                            pimg_mass = (
                                pressure_dict["body_volume"][i]
                                * mass_normalizer[0]
                                / mass_normalizer[1]
                            )
                            pressure_image = (
                                pressure_image
                                * (
                                    (pimg_mass * 9.81)
                                    / (np.sum(pressure_image) * 0.0264 * 0.0286)
                                )
                                * (1 / 133.322)
                            )

                            pressure_image = np.clip(pressure_image, 0, 100)

                            for j in range(len(cover_str_list)):
                                pressure_x.append([pressure_image] * len(depth))

                        # -------------------------
                        # Prepare label
                        gt_markers = np.array(
                            pressure_dict["markers_xyz_m"][i][0:72]
                        ).reshape(24, 3)
                        gt_markers[:, 0:2] -= 0.286
                        gt_markers = gt_markers.reshape(72)
                        label = np.concatenate(
                            (
                                gt_markers * 1000.0 + self.z_adj_all,
                                pressure_dict["body_shape"][i][0:10],
                                pressure_dict["joint_angles"][i][0:72],
                                pressure_dict["root_xyz_shift"][i][0:3]
                                + self.z_adj_one / 1000.0,
                                gender,
                                [1],
                                [
                                    pressure_dict["body_volume"][i]
                                    * mass_normalizer[0]
                                    / mass_normalizer[1]
                                ],
                                [pressure_dict["body_height"][i]],
                            ),
                            axis=0,
                        )
                        label_y.append(label)

                        saved += 1

                        if saved == limit:
                            break

                        if for_infer and saved == USE_FOR_INFER:
                            break

            # -------------------------
            if load_pressure:
                pmap_y = np.load(pmap_file)
                if for_infer:
                    pmap_y = pmap_y[:USE_FOR_INFER, :]
                pmap_shape = pmap_y.shape
                pmap_y = pmap_y[:, np.newaxis, ...]
                pmap_y = np.repeat(pmap_y, len(cover_str_list), axis=1)
                pmap_y = pmap_y.reshape(-1, *pmap_shape[1:])
            else:
                pmap_y = np.array([])
            if load_verts:
                verts_y = np.load(verts_file)
                if for_infer:
                    verts_y = verts_y[:USE_FOR_INFER, :]
            else:
                verts_y = np.array([])

            assert pmap_y.shape[0] == len(
                pressure_x
            ), f"Error: pmap len={pmap_y.shape[0]}, pressure_x len={len(pressure_x)}, not equal for file = {file_name}"

            pressure_x = np.array(pressure_x)
            depth_x = np.array(depth_x)
            label_y = np.array(label_y)
            names_y = np.array(names_y)

            data_pressure_x = data_utils.concatenate(data_pressure_x, pressure_x)
            data_depth_x = data_utils.concatenate(data_depth_x, depth_x)
            data_label_y = data_utils.concatenate(data_label_y, label_y)
            data_pmap_y = data_utils.concatenate(data_pmap_y, pmap_y)
            data_verts_y = data_utils.concatenate(data_verts_y, verts_y)
            data_names_y = data_utils.concatenate(data_names_y, names_y)

        data_pressure_x = np.expand_dims(data_pressure_x, -1)  # (N, 2, 64, 27, 1)
        data_depth_x = np.expand_dims(data_depth_x, -1)  # (N, 2, 128, 54, 1)
        data_label_y = np.array(data_label_y)
        data_pmap_y = np.array(data_pmap_y)
        data_verts_y = np.array(data_verts_y)
        data_names_y = np.array(data_names_y)

        data = {
            "data_pressure_x": data_pressure_x if load_pressure else None,
            "data_ir_x": None,
            "data_depth_x": data_depth_x,
            "data_label_y": data_label_y,
            "data_pmap_y": data_pmap_y if load_pressure else None,
            "data_verts_y": data_verts_y,
            "data_names_y": data_names_y,
        }
        print("BP data:")
        for k, v in data.items():
            if v is not None:
                print(f"{k}: {v.shape}")
            else:
                print(f"{k}: None")

        return data
