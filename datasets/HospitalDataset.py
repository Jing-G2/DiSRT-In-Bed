import os
import numpy as np
import cv2
import scipy.io as sio
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import zoom

from utils.constants import *
import utils.data_utils.data_utils as data_utils
import utils.data_utils.angle_utils as angle_utils


class SLPHospitalDataset:
    def __init__(self):
        super(SLPHospitalDataset, self).__init__()
        """ 
        parameters for original SLP data
        reference: https://github.com/ostadabbas/SLP-Dataset-and-Code
        """
        self.slp_dir = os.path.join(SLP_PATH, "simLab")
        self.n_subj = 7
        self.n_frm = 45
        self.dct_li_PTr = data_utils.genPTr_dict(
            range(1, self.n_subj + 1), ["RGB", "IR", "depth", "PM"], self.slp_dir
        )
        self.d_bed = 2.264  # as meter
        # d_bed = 2.101

        self.sizes = {
            "RGB": [576, 1024],
            "PM": [84, 192],
            "IR": [120, 160],
            "depth": [424, 512],
        }
        self.gender_male = [0, 1]
        self.gender_female = [1, 0]
        self.pm_adjust_cm = [
            -1,
            4,
        ]  # pressure mattress adjustment (left/right shift, bottom)

    def prepare_dataset(
        self,
        data_lines,
        mod=["depth"],
        cover_str_list=["uncover", "cover1", "cover2"],
        load_verts=False,
    ):
        depth_x = []
        label_y = []
        verts_y = []
        names_y = []

        """ The original physique data of 7 participants
           gender(0 femal, 1 male),height, weight(kg), bust, waist, hip, right upper arm, right lower arm, righ upper leg, right lower leg 
        """
        phys_arr = np.load(os.path.join(self.slp_dir, "physiqueData.npy"))  # (7, 10)
        phys_arr[:, [2, 0]] = phys_arr[:, [0, 2]]  # 2-gender, 0-weight

        for person_num_str in data_lines:  # each line is a person number str
            # get the physique data of the person
            body_mass = phys_arr[int(person_num_str) - 1][0]
            body_height = phys_arr[int(person_num_str) - 1][1]
            gender = phys_arr[int(person_num_str) - 1][2]
            if gender == 0:
                gender_label = self.gender_female
                gender_str = "f"
            else:
                gender_label = self.gender_male
                gender_str = "m"

            PTr_depth2PM = data_utils.get_PTr_A2B(
                p_id=int(person_num_str), modA="depth", modB="PM", dsFd=self.slp_dir
            )
            # load 2D joints
            joints_gt_RGB_t = sio.loadmat(
                os.path.join(self.slp_dir, person_num_str, "joints_gt_RGB.mat")
            )[
                "joints_gt"
            ]  # [3 x n_jt x n_frm]
            joints_gt_RGB_t = joints_gt_RGB_t.transpose(
                [2, 1, 0]
            )  # [n_frm x n_jt x 3 (x,y,z)]

            joints_gt_RGB_t = joints_gt_RGB_t - 1  # transfer to 0 base

            # homography RGB to depth
            PTr_RGB2depth = data_utils.get_PTr_A2B(
                p_id=int(person_num_str), modA="RGB", modB="depth", dsFd=self.slp_dir
            )
            joints_gt_depth_t = np.array(
                list(
                    map(
                        lambda x: cv2.perspectiveTransform(
                            np.array([x]), PTr_RGB2depth
                        )[0],
                        joints_gt_RGB_t[:, :, :2],
                    )
                )
            )
            joints_gt_depth_t = np.concatenate(
                [joints_gt_depth_t, joints_gt_RGB_t[:, :, 2, None]], axis=2
            )  # [45, 14, 3(x,y,0)]

            # create the label
            gt_markers = np.zeros((72,))
            body_shape = np.zeros((10,))
            joint_angles = np.zeros((72,))
            root_xyz_shift = np.zeros((3,))
            label = np.concatenate(
                (
                    gt_markers,  # 72
                    body_shape,  # 10
                    joint_angles,  # 72
                    root_xyz_shift,  # 3
                    gender_label,  # 2
                    [1],
                    [body_mass],
                    [body_height / 100.0],
                ),
                axis=0,
            )  # (162,)

            # load depth images
            for i in range(self.n_frm):  # 45 poses
                depth_image_pairs = []
                for cover_str in cover_str_list:
                    try:
                        depth_image = np.load(
                            os.path.join(
                                self.slp_dir,
                                person_num_str,
                                "depthRaw",
                                cover_str,
                                f"{(i+1):06d}.npy",
                            )
                        ).astype(np.float32)
                    except:
                        # current type of cover_str is not available
                        continue

                    """Same as henry's cleaned depth data
                    refter to: https://github.com/Healthcare-Robotics/BodyPressure
                        BodyPressure/lib_py/slp_prep_lib_bp.py
                    original depth data: (424x512)
                        -> get PTr (depth to PM)
                        -> wrap depth to PM (84x192)
                        -> zoom depth with zoom factor=0.665, order=1
                        -> cut off the edges [:, 1:-1] (128x54)
                    """
                    depth_image = cv2.warpPerspective(
                        depth_image, PTr_depth2PM, self.sizes["PM"]
                    )
                    depth_image = depth_image[0 : 192 - self.pm_adjust_cm[1], 0:84]
                    if np.shape(depth_image)[0] < 192:
                        depth_image = np.concatenate(
                            (
                                depth_image,
                                np.zeros(
                                    (
                                        192 - np.shape(depth_image)[0],
                                        np.shape(depth_image)[1],
                                    )
                                ),
                            ),
                            axis=0,
                        )
                    depth_image = zoom(depth_image, (0.665, 0.665), order=1)
                    depth_image = depth_image[:, 1:-1]  # (128, 54)

                    depth_image_pairs.append(depth_image)
                    if len(depth_image_pairs) == 1:
                        depth_x.append([depth_image, depth_image])
                    else:
                        depth_x.append([depth_image_pairs[0], depth_image_pairs[-1]])

                    label_y.append(label)
                    names_y.append(
                        f"slpHos_{int(person_num_str):03d}_{gender_str}_{(i+1)}_{cover_str}"
                    )

        data_depth_x = np.expand_dims(np.array(depth_x), -1)  # (n, S, 128, 54, 1)
        data_label_y = np.array(label_y)  # (n, 162)
        data_verts_y = np.array(verts_y)  # (n, 6890, 3)
        data_names_y = np.array(names_y)  # (n, )

        data = {
            "data_pressure_x": None,
            "data_ir_x": None,
            "data_depth_x": data_depth_x,
            "data_label_y": data_label_y,
            "data_pmap_y": None,
            "data_verts_y": data_verts_y,
            "data_names_y": data_names_y,
        }

        print("Hospital data pairs: ")
        for k, v in data.items():
            print(k, v.shape if v is not None else None)
        return data
