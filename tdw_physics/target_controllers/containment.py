from argparse import ArgumentParser
import h5py
import json
import copy
import importlib
import numpy as np
from enum import Enum
import random
from typing import List, Dict, Tuple
from weighted_collection import WeightedCollection
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelRecord, MaterialLibrarian
from tdw.output_data import OutputData, Transforms
from tdw_physics.rigidbodies_dataset import (RigidbodiesDataset,
                                             get_random_xyz_transform,
                                             get_range,
                                             handle_random_transform_args)
from tdw_physics.util import MODEL_LIBRARIES, get_parser, xyz_to_arr, arr_to_xyz, str_to_xyz

from tdw_physics.target_controllers.dominoes import Dominoes, MultiDominoes, get_args
from tdw_physics.postprocessing.labels import is_trial_valid

MODEL_NAMES = [r.name for r in MODEL_LIBRARIES['models_flex.json'].records]
M = MaterialLibrarian()
MATERIAL_TYPES = M.get_material_types()
MATERIAL_NAMES = {mtype: [m.name for m in M.get_all_materials_of_type(mtype)] \
                  for mtype in MATERIAL_TYPES}

'''
This controller pulls from the "towers.py" controller. 
You can think of it as a tower with only one block and a cap,
where the block and cap have a containment relationship. 

A probe object will collide with this "tower", possibly knocking the target out of its container.
'''

def none_or_str(value):
    if value == 'None':
        return None
    else:
        return value

def get_tower_args(dataset_dir: str, parse=True):
    """
    Combine Tower-specific args with general Dominoes args
    """
    common = get_parser(dataset_dir, get_help=False)
    domino, domino_postproc = get_args(dataset_dir, parse=False)
    parser = ArgumentParser(parents=[common, domino], conflict_handler='resolve', fromfile_prefix_chars='@')

    parser.add_argument("--remove_target",
                        type=int,
                        default=1,
                        help="Whether to remove the target object")
    parser.add_argument("--ramp",
                        type=int,
                        default=0,
                        help="Whether to place the probe object on the top of a ramp")    
    parser.add_argument("--collision_axis_length",
                        type=float,
                        default=3.0,
                        help="How far to put the probe and target")
    parser.add_argument("--num_blocks",
                        type=int,
                        default=1,
                        help="Number of rectangular blocks to build the tower base with")
    parser.add_argument("--mscale",
                        type=str,
                        default="[0.5,0.5]",
                        help="Scale or scale range for rectangular blocks to sample from")
    parser.add_argument("--mgrad",
                        type=float,
                        default=0.0,
                        help="Size of block scale gradient going from top to bottom of tower")
    parser.add_argument("--tower_cap",
                        type=str,
                        default="sphere",
                        help="Object types to use as a capper on the tower")
    parser.add_argument("--spacing_jitter",
                        type=float,
                        default=0.25,
                        help="jitter in how to space middle objects, as a fraction of uniform spacing")
    parser.add_argument("--mrot",
                        type=str,
                        default="[-45,45]",
                        help="comma separated list of initial middle object rotation values")
    parser.add_argument("--middle",
                        type=str,
                        default="bowl",
                        help="comma-separated list of possible middle objects")
    parser.add_argument("--probe",
                        type=str,
                        default="sphere",
                        help="comma-separated list of possible target objects")
    parser.add_argument("--pmass",
                        type=str,
                        default="[3.0,3.0]",
                        help="scale of probe objects")
    parser.add_argument("--pscale",
                        type=str,
                        default="[0.25,0.25]",
                        help="scale of probe objects")
    parser.add_argument("--tscale",
                        type=str,
                        default="[0.5,0.5]",
                        help="scale of target objects")
    parser.add_argument("--zscale",
                        type=str,
                        default="3.0,0.01,3.0",
                        help="scale of target zone")    
    parser.add_argument("--fscale",
                        type=str,
                        default="2.0",
                        help="range of scales to apply to push force")
    parser.add_argument("--frot",
                        type=str,
                        default="[-0,0]",
                        help="range of angles in xz plane to apply push force")
    parser.add_argument("--foffset",
                        type=str,
                        default="0.0,0.5,0.0",
                        help="offset from probe centroid from which to apply force, relative to probe scale")
    parser.add_argument("--fjitter",
                        type=float,
                        default=0.0,
                        help="jitter around object centroid to apply force")
    parser.add_argument("--fwait",
                        type=none_or_str,
                        default="[15,15]",
                        help="How many frames to wait before applying the force")        
    parser.add_argument("--camera_distance",
                        type=float,
                        default=2.5,
                        help="radial distance from camera to centerpoint")
    parser.add_argument("--camera_min_angle",
                        type=float,
                        default=0,
                        help="minimum angle of camera rotation around centerpoint")
    parser.add_argument("--camera_max_angle",
                        type=float,
                        default=60,
                        help="maximum angle of camera rotation around centerpoint")


    def postprocess(args):

        # whether to use a cap object on the tower
        if args.tower_cap is not None:
            cap_list = args.tower_cap.split(',')
            assert all([t in MODEL_NAMES for t in cap_list]), \
                "All target object names must be elements of %s" % MODEL_NAMES
            args.tower_cap = cap_list
        else:
            args.tower_cap = []

        return args

    args = parser.parse_args()
    args = domino_postproc(args)
    args = postprocess(args)

    return args

class Containment(MultiDominoes):

    def __init__(self,
                 port: int = 1071,
                 num_blocks=3,
                 middle_scale_range=[0.5,0.5],
                 middle_scale_gradient=0.0,
                 tower_cap=[],
                 use_ramp=False,
                 **kwargs):

        super().__init__(port=port, middle_scale_range=middle_scale_range, **kwargs)

        self.use_ramp = use_ramp

        # probe and target different colors
        self.match_probe_and_target_color = False

        # how many blocks in tower, sans cap
        self.num_blocks = num_blocks

        # how to scale the blocks
        self.middle_scale_gradient = middle_scale_gradient

        # whether to use a cap
        if len(tower_cap):
            self.use_cap = True
            self._cap_types = self.get_types(tower_cap)
        else:
            self.use_cap = False

        # scale the camera height
        self.camera_min_height *= 0.5 * (self.num_blocks + int(self.use_cap))
        self.camera_max_height *= 0.5 * (self.num_blocks + int(self.use_cap))
        self.camera_aim = 0.25 * (self.num_blocks + int(self.use_cap))
        self.camera_aim = {"x": 0., "y": self.camera_aim, "z": 0.}

    def clear_static_data(self) -> None:
        super().clear_static_data()

        self.cap_type = None
        self.did_fall = None
        

    def _write_static_data(self, static_group: h5py.Group) -> None:
        super()._write_static_data(static_group)

        static_group.create_dataset("cap_type", data=self.cap_type)
        static_group.create_dataset("use_cap", data=self.use_cap)
        static_group.create_dataset("num_blocks", data=self.num_blocks)

    @staticmethod
    def get_controller_label_funcs(classname = "Tower"):
        funcs = Dominoes.get_controller_label_funcs(classname)

        def num_middle_objects(f):
            try:
                return int(np.array(f['static']['num_blocks']))
            except KeyError:
                return int(len(f['static']['mass'] - 3))

        def did_tower_fall(f):
            return is_trial_valid(f, valid_key='did_fall')

        funcs += [num_middle_objects, did_tower_fall]
        
        return funcs

    def _write_frame_labels(self,
                            frame_grp: h5py.Group,
                            resp: List[bytes],
                            frame_num: int,
                            sleeping: bool) -> Tuple[h5py.Group, List[bytes], int, bool]:

        labels, resp, frame_num, done = super()._write_frame_labels(
            frame_grp, resp, frame_num, sleeping)

        if frame_num >= 30:
            labels.create_dataset("did_fall", data=bool(self.did_fall))
        else:
            labels.create_dataset("did_fall", data=False)

        return labels, resp, frame_num, done

    # def _get_zone_location(self, scale):
    #     bottom_block_width = get_range(self.middle_scale_range)[1]
    #     bottom_block_width += (self.num_blocks / 2.0) * np.abs(self.middle_scale_gradient)
    #     probe_width = get_range(self.probe_scale_range)[1]
    #     return {
    #         "x": 0.0,
    #         "y": 0.0,
    #         "z": -(0.5 + probe_width) * scale["z"] + bottom_block_width + 0.1,
    #     }

    def _get_zone_location(self, scale):
        return TDWUtils.VECTOR3_ZERO

    def _set_tower_height_now(self, resp: List[bytes]) -> None:
        top_obj_id = self.object_ids[-1]
        for r in resp[:-1]:
            r_id = OutputData.get_data_type_id(r)
            if r_id == "tran":
                tr = Transforms(r)
                for i in range(tr.get_num()):
                    if tr.get_id(i) == top_obj_id:
                        self.tower_height = tr.get_position(i)[1]

    def get_per_frame_commands(self, resp: List[bytes], frame: int) -> List[dict]:

        cmds = super().get_per_frame_commands(resp, frame)
        
        # check if tower fell
        if frame == 5:
            self._set_tower_height_now(resp)
            self.init_height = self.tower_height + 0.
        elif frame > 5:
            self._set_tower_height_now(resp)
            self.did_fall = (self.tower_height < 0.5 * self.init_height)

        return cmds

    def _build_intermediate_structure(self) -> List[dict]:
        print("middle color", self.middle_color)
        if self.randomize_colors_across_trials:
            self.middle_color = self.random_color(exclude=self.target_color) if self.monochrome else None
        self.cap_color = self.target_color
        commands = []

        commands.extend(self._build_stack())
        commands.extend(self._add_cap())

        return commands

    def _get_block_position(self, scale, y):
        jitter = lambda: random.uniform(-self.spacing_jitter, self.spacing_jitter)
        jx, jz = [scale["x"]*jitter(), scale["z"]*jitter()]
        return {"x": jx, "y": y, "z": jz}

    def _get_block_scale(self, offset) -> dict:
        print("scale range", self.middle_scale_range)
        scale = get_random_xyz_transform(self.middle_scale_range)
        scale = {k:v+offset for k,v in scale.items()}

        return scale

    def _build_stack(self) -> List[dict]:
        commands = []
        height = 0.

        # build the block scales
        mid = self.num_blocks / 2.0
        grad = self.middle_scale_gradient
        self.block_scales = [self._get_block_scale(offset=grad*(mid-i)) for i in range(self.num_blocks)]

        # place the blocks
        for m in range(self.num_blocks):
            record, data = self.random_primitive(
                self._middle_types,
                scale=self.block_scales[m],
                color=self.middle_color,
                exclude_color=self.target_color)
            self.middle_type = data["name"]
            o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]
            block_pos = self._get_block_position(scale, height)
            block_rot = self.get_y_rotation(self.middle_rotation_range)
            commands.extend(
                self.add_physics_object(
                    record=record,
                    position=block_pos,
                    rotation=block_rot,
                    mass=random.uniform(*get_range(self.middle_mass_range)),
                    dynamic_friction=random.uniform(0, 0.9),
                    static_friction=random.uniform(0, 0.9),
                    bounciness=random.uniform(0, 1),
                    o_id=o_id))

            # Set the block object material
            commands.extend(
                self.get_object_material_commands(
                    record, o_id, self.get_material_name(self.middle_material)))


            # Scale the object and set its color.
            commands.extend([
                {"$type": "set_color",
                 "color": {"r": rgb[0], "g": rgb[1], "b": rgb[2], "a": 1.},
                 "id": o_id},
                {"$type": "scale_object",
                 "scale_factor": scale,
                 "id": o_id}])

            print("placed middle object %s" % str(m+1))

            # update height
            _y = record.bounds['top']['y'] if self.middle_type != 'bowl' else (record.bounds['bottom']['y'] + 0.1)
            height += scale["y"] * _y

        self.tower_height = height

        return commands

    def _add_cap(self) -> List[dict]:
        commands = []

        # the cap becomes the target
        record, data = self.random_primitive(
            self._cap_types,
            scale=self.target_scale_range,
            color=self.target_color,
            add_data=self.use_cap
        )
        o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]
        self.cap  = record
        self.cap_type = data["name"]
        self._replace_target_with_object(record, data)

        commands.extend(
            self.add_physics_object(
                record=record,
                position={
                    "x": 0.,
                    "y": self.tower_height,
                    "z": 0.
                },
                rotation={"x":0.,"y":0.,"z":0.},
                mass=random.uniform(4.5,4.5),
                dynamic_friction=random.uniform(0, 0.9),
                static_friction=random.uniform(0, 0.9),
                bounciness=random.uniform(0, 1),
                o_id=o_id,
                add_data=self.use_cap
            ))

        # Set the cap object material
        commands.extend(
            self.get_object_material_commands(
                record, o_id, self.get_material_name(self.target_material)))

        # Scale the object and set its color.
        commands.extend([
            {"$type": "set_color",
             "color": {"r": rgb[0], "g": rgb[1], "b": rgb[2], "a": 1.},
             "id": o_id},
            {"$type": "scale_object",
             "scale_factor": scale,
             "id": o_id}])

        if not self.use_cap:
            commands.append(
                {"$type": self._get_destroy_object_command_name(o_id),
                 "id": int(o_id)})
            self.object_ids = self.object_ids[:-1]
        else:
            self.tower_height += scale["y"]

        return commands

    def is_done(self, resp: List[bytes], frame: int) -> bool:
        return frame > 300

if __name__ == "__main__":

    args = get_tower_args("towers")

    CC = Containment(
        # tower specific
        num_blocks=args.num_blocks,
        tower_cap=args.tower_cap,
        spacing_jitter=args.spacing_jitter,
        middle_rotation_range=args.mrot,
        middle_scale_range=args.mscale,
        middle_mass_range=args.mmass,
        middle_scale_gradient=args.mgrad,
        # domino specific
        target_zone=args.zone,
        zone_location=args.zlocation,
        zone_scale_range=args.zscale,
        zone_color=args.zcolor,
        target_objects=args.target,
        probe_objects=args.probe,
        middle_objects=args.middle,
        target_scale_range=args.tscale,
        target_rotation_range=args.trot,
        probe_scale_range=args.pscale,
        probe_mass_range=args.pmass,
        target_color=args.color,
        probe_color=args.pcolor,
        middle_color=args.mcolor,
        collision_axis_length=args.collision_axis_length,
        force_scale_range=args.fscale,
        force_angle_range=args.frot,
        force_offset=args.foffset,
        force_offset_jitter=args.fjitter,
        force_wait=args.fwait,
        remove_target=bool(args.remove_target),
        ## not scenario-specific
        room=args.room,
        randomize=args.random,
        seed=args.seed,
        camera_radius=args.camera_distance,
        camera_min_angle=args.camera_min_angle,
        camera_max_angle=args.camera_max_angle,
        camera_min_height=args.camera_min_height,
        camera_max_height=args.camera_max_height,
        monochrome=args.monochrome,
        material_types=args.material_types,
        target_material=args.tmaterial,
        probe_material=args.pmaterial,
        middle_material=args.mmaterial,
        zone_material=args.zmaterial,
        distractor_types=args.distractor,
        distractor_categories=args.distractor_categories,
        num_distractors=args.num_distractors,
        occluder_types=args.occluder,
        occluder_categories=args.occluder_categories,
        num_occluders=args.num_occluders,
        occlusion_scale=args.occlusion_scale,
        remove_middle=args.remove_middle,
        use_ramp=bool(args.ramp)
    )
    print(CC.num_blocks, [r.name for r in CC._cap_types])

    if bool(args.run):
        CC.run(num=args.num,
               output_dir=args.dir,
               temp_path=args.temp,
               width=args.width,
               height=args.height,
               save_passes=args.save_passes.split(','),
               save_movies=args.save_movies,
               save_labels=args.save_labels,               
               args_dict=vars(args)
        )
    else:
        CC.communicate({"$type": "terminate"})
