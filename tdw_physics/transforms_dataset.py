from typing import List, Tuple, Dict, Optional
from abc import ABC
import h5py
import numpy as np
import random
from tdw.tdw_utils import TDWUtils
from tdw.output_data import OutputData, Transforms, Images, CameraMatrices, Bounds
from tdw_physics.dataset import Dataset
from tdw_physics.controller import Controller
from tdw.librarian import ModelRecord
from tdw_physics.dataset import Dataset
from tdw_physics.util import xyz_to_arr, arr_to_xyz, MODEL_LIBRARIES
import matplotlib.pyplot as plt

from PIL import Image

class TransformsDataset(Dataset, ABC):
    """
    A dataset creator that receives and writes per frame: `Transforms`, `Images`, `CameraMatrices`.
    See README for more info.
    """

    def clear_static_data(self) -> None:
        super().clear_static_data()

        self.initial_positions = []
        self.initial_rotations = []

    def _write_static_data(self, static_group: h5py.Group) -> None:
        super()._write_static_data(static_group)

        # positions and rotations of objects
        static_group.create_dataset("initial_position",
                                    data=np.stack([xyz_to_arr(p) for p in self.initial_positions], 0))
        static_group.create_dataset("initial_rotation",
                                    data=np.stack([xyz_to_arr(r) for r in self.initial_rotations], 0))

    def random_model(self,
                     object_types: List[ModelRecord],
                     random_obj_id: bool = False,
                     add_data: bool = True) -> dict:
        obj_record = random.choice(object_types)
        obj_data = {
            "id": self.get_unique_id() if random_obj_id else self._get_next_object_id(),
            "name": obj_record.name
        }

        if add_data:
            self.model_names.append(obj_data["name"])

        return obj_record, obj_data

    def add_transforms_data(self, position, rotation):
        self.initial_positions = np.append(self.initial_positions, position)
        self.initial_rotations = np.append(self.initial_rotations, rotation)

    def add_transforms_object(self,
                              record: ModelRecord,
                              position: Dict[str, float],
                              rotation: Dict[str, float],
                              o_id: Optional[int] = None,
                              add_data: Optional[bool] = True,
                              library: str = ""
    ) -> dict:
        """
        This is a wrapper for `Controller.get_add_object()` and the `add_object` command.
        This caches the ID of the object so that it can be easily cleaned up later.

        :param record: The model record.
        :param position: The initial position of the object.
        :param rotation: The initial rotation of the object, in Euler angles.
        :param o_id: The unique ID of the object. If None, a random ID is generated.
        :param add_data: whether to add the chosen data to the hdf5

        :return: An `add_object` command.
        """

        if o_id is None:
            o_id: int = Controller.get_unique_id()

        # Log the static data.
        Dataset.OBJECT_IDS = np.append(Dataset.OBJECT_IDS, o_id)

        # print("appended id", o_id)

        if add_data:
            self.initial_positions = np.append(self.initial_positions, position)
            self.initial_rotations = np.append(self.initial_rotations, rotation)

        return {"$type": "add_object",
                "name": record.name,
                "url": record.get_url(),
                "scale_factor": record.scale_factor,
                "position": position,
                "rotation": rotation,
                "category": record.wcategory,
                "id": o_id}

        #
        # commands = Dataset.get_add_object(model_name=record.name, object_id=o_id, position=position, rotation=rotation, library=library)
        #
        # return commands


    @staticmethod
    def get_add_object(model_name: str, object_id: int, position: Dict[str, float] = None,
                       rotation: Dict[str, float] = None, library: str = "") -> dict:
        """
        Returns a valid add_object command.

        :param model_name: The name of the model.
        :param position: The position of the model. If None, defaults to `{"x": 0, "y": 0, "z": 0}`.
        :param rotation: The starting rotation of the model, in Euler angles. If None, defaults to `{"x": 0, "y": 0, "z": 0}`.
        :param library: The path to the records file. If left empty, the default library will be selected. See `ModelLibrarian.get_library_filenames()` and `ModelLibrarian.get_default_library()`.
        :param object_id: The ID of the new object.

        :return An add_object command that the controller can then send.
        """

        # Log the static data.
        Dataset.OBJECT_IDS = np.append(Dataset.OBJECT_IDS, object_id)


        return Dataset.get_add_object(model_name=model_name, object_id=object_id, position=position, rotation=rotation,
                                      library=library)

    def _get_send_data_commands(self) -> List[dict]:
        return [{"$type": "send_transforms",
                "frequency": "always"},
                {"$type": "send_camera_matrices",
                     "frequency": "always"},
                    {"$type": "send_bounds",
                     "frequency": "always"},
                    {"$type": "send_segmentation_colors",
                     "ids": [int(oid) for oid in Dataset.OBJECT_IDS],
                     "frequency": "once"}
                ]

    def _write_frame(self, frames_grp: h5py.Group, resp: List[bytes], frame_num: int, view_num: int) -> \
            Tuple[h5py.Group, h5py.Group, dict, bool]:
        num_objects = len(Dataset.OBJECT_IDS)

        # if view_num is None or view_num == 0:
        # Create a group for the frame
        frame = frames_grp.create_group(TDWUtils.zero_padding(frame_num, 4))

        # Create a group for images.
        # images = frame.create_group("images")

        camera_matrices = frame.create_group("camera_matrices")
        objs = frame.create_group("objects")
        # else:
        #     frame = frames_grp
        #     images = frame["images"]
        #     camera_matrices = frame["camera_matrices"]
        #     objs = frame["objects"]

        cam_suffix = '' if view_num is None else f'_cam{view_num}'

        # Transforms data.
        positions = np.empty(dtype=np.float32, shape=(num_objects, 3))
        forwards = np.empty(dtype=np.float32, shape=(num_objects, 3))
        rotations = np.empty(dtype=np.float32, shape=(num_objects, 4))

        # Bounds data.
        bounds = dict()
        for bound_type in ['front', 'back', 'left', 'right', 'top', 'bottom', 'center']:
            bounds[bound_type] = np.empty(dtype=np.float32, shape=(num_objects, 3))

        # Parse the data in an ordered manner so that it can be mapped back to the object IDs.
        tr_dict = dict()

        for r in resp[:-1]:
            r_id = OutputData.get_data_type_id(r)
            if r_id == "tran":
                tr = Transforms(r)
                for i in range(tr.get_num()):
                    pos = tr.get_position(i)
                    tr_dict.update({tr.get_id(i): {"pos": pos,
                                                   "for": tr.get_forward(i),
                                                   "rot": tr.get_rotation(i)}})
                # Add the Transforms data.
                for o_id, i in zip(Dataset.OBJECT_IDS, range(num_objects)):
                    if o_id not in tr_dict:
                        continue
                    positions[i] = tr_dict[o_id]["pos"]
                    forwards[i] = tr_dict[o_id]["for"]
                    rotations[i] = tr_dict[o_id]["rot"]
            elif r_id == "imag":
                im = Images(r)
                # Add each image.
                for i in range(im.get_num_passes()):
                    pass_mask = im.get_pass_mask(i) + cam_suffix
                    # Reshape the depth pass array.
                    # if pass_mask == "_depth" + cam_suffix:
                    #     img_d = TDWUtils.get_shaped_depth_pass(images=im, index=i)
                    #     image_data = TDWUtils.get_depth_values(img_d, width=img_d.shape[0], height=img_d.shape[1])[::-1, :]
                    #     # breakpoint()
                    # else:
                    #     image_data = im.get_image(i)
                    # images.create_dataset(pass_mask, data=image_data, compression="gzip")

                    # Save PNGs
                    # breakpoint()
                    # breakpoint()
                    sp_cam_suffix = [x+cam_suffix for x in self.save_passes]

                    if pass_mask in sp_cam_suffix:
                        # breakpoint()
                        filename = pass_mask[1:] + "_" + TDWUtils.zero_padding(frame_num, 4) + "." + im.get_extension(i)
                        path = self.png_dir.joinpath(filename)
                        if self.save_movies:
                            if pass_mask in ["_depth" + cam_suffix, "_depth_simple" + cam_suffix]:
                                #TODO: save as plt.imshow() fig
                                plt.imsave(path, image_data)
                            else:
                                with open(path, "wb") as f:
                                    f.write(im.get_image(i))
                        # breakpoint()
            # Add the camera matrices.
            elif r_id == "boun":
                bo = Bounds(r)
                bo_dict = dict()
                for i in range(bo.get_num()):
                    bo_dict.update({bo.get_id(i): {"front": bo.get_front(i),
                                                   "back": bo.get_back(i),
                                                   "left": bo.get_left(i),
                                                   "right": bo.get_right(i),
                                                   "top": bo.get_top(i),
                                                   "bottom": bo.get_bottom(i),
                                                   "center": bo.get_center(i)}})
                for o_id, i in zip(Dataset.OBJECT_IDS, range(num_objects)):
                    for bound_type in bounds.keys():
                        try:
                            bounds[bound_type][i] = bo_dict[o_id][bound_type]
                        except KeyError:
                            print("couldn't store bound data for object %d" % o_id)


            # Add the camera matrices.
            elif OutputData.get_data_type_id(r) == "cama":
                matrices = CameraMatrices(r)
                camera_matrices.create_dataset("projection_matrix" +  cam_suffix, data=matrices.get_projection_matrix())
                camera_matrices.create_dataset("camera_matrix" + cam_suffix, data=matrices.get_camera_matrix())

        # objs = frame.create_group("objects")
        objs.create_dataset("positions" + cam_suffix, data=positions.reshape(num_objects, 3), compression="gzip")
        objs.create_dataset("forwards" + cam_suffix, data=forwards.reshape(num_objects, 3), compression="gzip")
        objs.create_dataset("rotations" + cam_suffix, data=rotations.reshape(num_objects, 4), compression="gzip")
        for bound_type in bounds.keys():
            objs.create_dataset(bound_type + cam_suffix, data=bounds[bound_type], compression="gzip")

        return frame, objs, tr_dict, False

    def get_object_position(self, obj_id: int, resp: List[bytes]) -> None:
        position = None
        for r in resp:
            r_id = OutputData.get_data_type_id(r)
            if r_id == "tran":
                tr = Transforms(r)
                for i in range(tr.get_num()):
                    if tr.get_id(i) == obj_id:
                        position = tr.get_position(i)

        return position

