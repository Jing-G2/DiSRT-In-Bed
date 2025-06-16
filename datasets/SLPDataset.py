import os
import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import zoom

from utils.constants import *
import utils.data_utils.data_utils as data_utils
import utils.data_utils.angle_utils as angle_utils


class SLPDataset:
    def __init__(self, data_path):
        super(SLPDataset, self).__init__()
        """ 
        parameters for original SLP data
        reference: https://github.com/ostadabbas/SLP-Dataset-and-Code
        """
        self.slp_dir = os.path.join(SLP_PATH, "danaLab")
        # get all mdoe
        self.n_subj = 102
        self.n_frm = 45
        self.dct_li_PTr = data_utils.genPTr_dict(
            range(1, self.n_subj + 1), ["RGB", "IR", "depth", "PM"], self.slp_dir
        )
        self.sizes = {
            "RGB": [576, 1024],
            "PM": [84, 192],
            "IR": [120, 160],
            "depth": [424, 512],
        }

        # --------------------------------------------------------
        """
        parameters for henry's data (pressure)
        """
        self.pressure_data_dir = os.path.join(data_path, "SLP", "danaLab")
        self.slp_real_cleaned_data_dir = os.path.join(data_path, "slp_real_cleaned")
        self.pmap_data_dir = os.path.join(data_path, "GT_BP_data", "slp2")
        self.verts_data_dir = os.path.join(data_path, "GT_BP_data", "slp2")
        self.slp_labels_dir = os.path.join(data_path, "SLP_SMPL_fits", "fits")

        assert os.path.exists(
            self.pressure_data_dir
        ), f"{self.pressure_data_dir} not exist"

        self.load_per_person = 45  # number of images/poses per person
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
        for_infer=False,
    ):
        pressure_x = []
        ir_x = []
        depth_x = []
        label_y = []
        pmap_y = []
        verts_y = []
        names_y = []

        """ The physique data of 102 participants
           wt(kg)
           height,
           gender(0 femal, 1 male),
           bust (cm),waist ,
           hip,
           right upper arm ,
           right lower arm,
           righ upper leg,
           right lower leg 
        """
        phys_arr = np.load(
            os.path.join(self.pressure_data_dir, "physiqueData.npy")
        )  # (102, 10)
        phys_arr[:, [2, 0]] = phys_arr[:, [0, 2]]  # 0-gender, 2-weight

        # slp world to camera transformation
        slp_T_cam_all = np.load(
            os.path.join(self.slp_real_cleaned_data_dir, "slp_T_cam_0to102.npy"),
            allow_pickle=True,
        )  # (102, 45, 3) every position of each person is the same

        if "depth" in mod:
            """load henry's cleaned depth data (uint16)
            refter to: https://github.com/Healthcare-Robotics/BodyPressure
                BodyPressure/lib_py/slp_prep_lib_bp.py
            original depth data: (424x512)
                -> get PTr (depth to PM)
                -> wrap depth to PM (84x192)
                -> zoom depth with zoom factor=0.665, order=1
                -> cut off the edges [:, 1:-1] (128x54)
                -> clean the noise from human hair
            """
            depth_all = {}
            for cover_str in cover_str_list:
                depth_all[cover_str] = np.load(
                    os.path.join(
                        self.slp_real_cleaned_data_dir,
                        f"depth_{cover_str}_cleaned_0to102.npy",
                    ),
                    allow_pickle=True,
                )  # (102, 45, 128, 54)

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

            # data paths
            pmap_path = os.path.join(self.pmap_data_dir, person_num_str, "uncover")
            verts_path = os.path.join(self.verts_data_dir, person_num_str, "uncover")

            if "IR" in mod:
                PTr_IR2PM = data_utils.get_PTr_A2B(
                    p_id=int(person_num_str), modA="IR", modB="PM", dsFd=self.slp_dir
                )

            # load the pressure and depth images
            for i in range(self.load_per_person):  # 45 poses
                if "PM" in mod:
                    pressure_image_pairs = []
                    for cover_str in cover_str_list:
                        pressure_image = np.load(
                            file=os.path.join(
                                self.slp_dir,
                                person_num_str,
                                "PMarray",
                                cover_str,
                                f"{(i+1):06d}.npy",
                            )
                        ).astype(
                            np.float32
                        )  # original PM (192, 84)

                        # PM_arr = PM_arr[1:191, 3:80] # cut off the edges because the original pressure mat is like 1.90 x 0.77 while this one is 1.92 x 0.84.
                        pressure_image = pressure_image[
                            0 : 191 - self.pm_adjust_cm[1], 0:84
                        ]  # adjust the bottom borders of the pressure images
                        if np.shape(pressure_image)[0] < 190:
                            pressure_image = np.concatenate(
                                (
                                    pressure_image,
                                    np.zeros(
                                        (
                                            190 - np.shape(pressure_image)[0],
                                            np.shape(pressure_image)[1],
                                        )
                                    ),
                                ),
                                axis=0,
                            )  # pad the top borders to keep shape (190, 84)
                        pressure_image = pressure_image[
                            :, 3 - self.pm_adjust_cm[0] : 80 - self.pm_adjust_cm[0]
                        ]  # (190, 77)

                        pressure_image = gaussian_filter(
                            pressure_image, sigma=0.5 / 0.345
                        )
                        pressure_image = zoom(
                            pressure_image, (0.335, 0.355), order=1
                        )  # (64, 27)

                        # normalizing by mass, converting to mmHg (64, 27)
                        pressure_image = (
                            pressure_image
                            * (
                                (body_mass * 9.81)
                                / (np.sum(pressure_image) * 0.0264 * 0.0286)
                            )
                        ) * (1 / 133.322)

                        pressure_image = np.clip(pressure_image, 0, 100)
                        pressure_image_pairs.append(pressure_image)  # (S, 64, 27)

                    for j in range(len(pressure_image_pairs)):
                        pressure_x.append(
                            [pressure_image_pairs[0], pressure_image_pairs[j]]
                        )

                if "IR" in mod:
                    ir_image_pairs = []
                    for cover_str in cover_str_list:
                        IR_image = np.load(
                            file=os.path.join(
                                self.slp_dir,
                                person_num_str,
                                "IRraw",
                                cover_str,
                                f"{(i+1):06d}.npy",
                            )
                        ).astype(
                            np.float32
                        )  # original IR (120, 160)

                        IR_image = cv2.warpPerspective(
                            IR_image, PTr_IR2PM, self.sizes["PM"]
                        )  # (192, 84)

                        if True:
                            """apply same processing as PM"""
                            # adjust the border correspondingly
                            IR_image = IR_image[
                                0 : 191 - self.pm_adjust_cm[1], 0:84
                            ]  # (190, 84)
                            if np.shape(IR_image)[0] < 190:
                                IR_image = np.concatenate(
                                    (
                                        IR_image,
                                        np.zeros(
                                            (
                                                190 - np.shape(IR_image)[0],
                                                np.shape(IR_image)[1],
                                            )
                                        ),
                                    ),
                                    axis=0,
                                )
                            IR_image = IR_image[
                                :, 3 - self.pm_adjust_cm[0] : 80 - self.pm_adjust_cm[0]
                            ]  # (190, 77)
                            IR_image = zoom(
                                IR_image, (0.335, 0.355), order=1
                            )  # (64, 27)
                        else:
                            """apply same processing as Depth (worse)"""
                            IR_image = zoom(IR_image, 0.665, order=1)  # (128, 56)
                            IR_image = IR_image[:, 1:-1]  # (128, 54)

                        ir_image_pairs.append(IR_image)  # (3, 64, 27)

                    for j in range(len(ir_image_pairs)):
                        ir_x.append([ir_image_pairs[0], ir_image_pairs[j]])

                if "depth" in mod:
                    depth_image_pairs = []
                    for cover_str in cover_str_list:
                        depth_image = depth_all[cover_str][
                            int(person_num_str) - 1, i
                        ]  # (128, 54)
                        # depth_image[127:, :] = 0.0 #* temporarily remove this line

                        depth_image_pairs.append(depth_image)  # (3, 128, 54)

                    for j in range(len(depth_image_pairs)):
                        depth_x.append([depth_image_pairs[0], depth_image_pairs[j]])

                """ pose data format (key, value.shape or value)
                  gender male
                  gt_joints (25, 2)
                  betas (10,)
                  body_pose (69,)
                  transl (3,)
                  global_orient (3,)
                  camera_rotation (3, 3)
                  camera_translation (3,)
                  camera_focal_length_x ()
                  camera_focal_length_y ()
                  camera_center (2,)
                  O_T_slp (3,)
                  slp_T_cam (3,)
                  cam_T_Bo (3,)
                  Bo_T_Br (3,)
                  markers_xyz_m (24, 3)
                """
                pose_data = data_utils.load_pickle(
                    os.path.join(
                        self.slp_labels_dir,
                        f"p{int(person_num_str):03d}",
                        f"proc_{(i+1):02d}.pkl",
                    )
                )
                pose_data["slp_T_cam"] = np.array(
                    slp_T_cam_all[int(person_num_str) - 1, i]
                )  # (3,)
                R_root = angle_utils.matrix_from_dir_cos_angles(
                    pose_data["global_orient"]
                )  # (3, 3)
                flip_root_euler = np.pi
                flip_root_R = angle_utils.eulerAnglesToRotationMatrix(
                    [flip_root_euler, 0.0, 0.0]
                )  # (3, 3)
                pose_data["global_orient"] = angle_utils.dir_cos_angles_from_matrix(
                    np.matmul(flip_root_R, R_root)
                )  # (3,)
                body_shape = pose_data["betas"]  # (10,)
                joint_angles = np.array(
                    list(pose_data["global_orient"]) + list(pose_data["body_pose"])
                )  # (72,)
                root_xyz_shift = (
                    pose_data["O_T_slp"]
                    + pose_data["slp_T_cam"]
                    + pose_data["cam_T_Bo"]
                    + pose_data["Bo_T_Br"]
                )  # (3,)
                root_xyz_shift[1:] *= -1
                # gt_markers = np.array(pose_data['markers_xyz_m']) + np.array([12/1000., -35/1000., 0.0])
                gt_markers = np.array(pose_data["markers_xyz_m"])  # (24, 3)
                gt_markers *= 1000.0
                gt_markers = gt_markers.reshape(-1)  # (72,)

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

                pmap = (
                    np.load(os.path.join(pmap_path, f"{(i+1):06d}_gt_pmap.npy"))
                    if "PM" in mod
                    else None
                )

                for j in range(len(cover_str_list)):
                    label_y.append(label)
                    pmap_y.append(pmap)
                    names_y.append(
                        f"slpHome_{int(person_num_str):03d}_{gender_str}_{(i+1)}_{cover_str_list[j]}",
                    )

                if load_verts:
                    verts = np.load(
                        os.path.join(verts_path, f"{(i+1):06d}_gt_vertices.npy")
                    )
                    for j in range(len(cover_str_list)):
                        verts_y.append(verts)

                if for_infer and i + 1 >= USE_FOR_INFER:
                    break

            if for_infer and len(names_y) >= USE_FOR_INFER * len(data_lines):
                break

        data_pressure_x = None
        data_ir_x = None
        data_depth_x = None

        if "PM" in mod:
            data_pressure_x = np.expand_dims(
                np.array(pressure_x), -1
            )  # (n, S, 64, 27, 1)
        if "IR" in mod:
            data_ir_x = np.expand_dims(
                np.array(ir_x), -1
            )  # (n, S, 64, 27, 1) align with PM
        if "depth" in mod:
            data_depth_x = np.expand_dims(np.array(depth_x), -1)  # (n, S, 128, 54, 1)

        data_label_y = np.array(label_y)  # (n, 162)
        data_pmap_y = np.array(pmap_y)  # (n, 6890)
        data_verts_y = np.array(verts_y)  # (n, 6890, 3)
        data_names_y = np.array(names_y)  # (n, )

        data = {
            "data_pressure_x": data_pressure_x if "PM" in mod else None,
            "data_ir_x": data_ir_x if "IR" in mod else None,
            "data_depth_x": data_depth_x,
            "data_label_y": data_label_y,
            "data_pmap_y": data_pmap_y if "PM" in mod else None,
            "data_verts_y": data_verts_y,
            "data_names_y": data_names_y,
        }

        print("SLP data pairs: ")
        for k, v in data.items():
            print(k, v.shape if v is not None else None)
        return data
