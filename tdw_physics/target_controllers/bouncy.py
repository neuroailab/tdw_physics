from argparse import ArgumentParser
import h5py
import json
import copy
import importlib
import numpy as np
from enum import Enum
import random
import ipdb
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

from tdw_physics.target_controllers.dominoes import Dominoes, MultiDominoes, get_args, none_or_str, none_or_int
from tdw_physics.postprocessing.labels import is_trial_valid

MODEL_NAMES = [r.name for r in MODEL_LIBRARIES['models_flex.json'].records]
M = MaterialLibrarian()
MATERIAL_TYPES = M.get_material_types()
MATERIAL_NAMES = {mtype: [m.name for m in M.get_all_materials_of_type(mtype)]
                  for mtype in MATERIAL_TYPES}

OCCLUDER_CATS = "coffee table,houseplant,vase,chair,dog,sofa,flowerpot,coffee maker,stool,laptop,laptop computer,globe,bookshelf,desktop computer,garden plant,garden plant,garden plant"
DISTRACTOR_CATS = "coffee table,houseplant,vase,chair,dog,sofa,flowerpot,coffee maker,stool,laptop,laptop computer,globe,bookshelf,desktop computer,garden plant,garden plant,garden plant"


def get_bouncy_args(dataset_dir: str, parse=True):

    common = get_parser(dataset_dir, get_help=False)
    domino, domino_postproc = get_args(dataset_dir, parse=False)
    parser = ArgumentParser(
        parents=[common, domino], conflict_handler='resolve', fromfile_prefix_chars='@')

    # Changed defaults
    # zone
    parser.add_argument("--zscale",
                        type=str,
                        default="1.2,0.01,2.0",
                        help="scale of target zone")

    parser.add_argument("--zone",
                        type=str,
                        default="cube",
                        help="comma-separated list of possible target zone shapes")

    parser.add_argument("--zjitter",
                        type=float,
                        default=0.,
                        help="amount of z jitter applied to the target zone")

    # force
    parser.add_argument("--fscale",
                        type=str,
                        default="[1.4,1.4]",
                        help="range of scales to apply to push force")

    parser.add_argument("--frot",
                        type=str,
                        default="[0,0]",
                        help="range of angles in xz plane to apply push force")

    parser.add_argument("--foffset",
                        type=str,
                        default="0.0,0.8,0.0",
                        help="offset from probe centroid from which to apply force, relative to probe scale")

    parser.add_argument("--fjitter",
                        type=float,
                        default=0.,
                        help="jitter around object centroid to apply force")

    parser.add_argument("--fupforce",
                        type=str,
                        default='[0,0]',
                        help="Upwards component of force applied, with 0 being purely horizontal force and 1 being the same force being applied horizontally applied vertically")

    # target
    parser.add_argument("--target",
                        type=str,
                        default="sphere",
                        help="comma-separated list of possible target objects")

    parser.add_argument("--tscale",
                        type=str,
                        default="0.2,0.2,0.2",
                        help="scale of target objects")

    parser.add_argument("--tlift",
                        type=float,
                        default=2.0,
                        help="Lift the target object off the floor/ramp. Useful for rotated objects")

    parser.add_argument("--tbounce",
                        type=str,
                        default="[0.5,0.9]",
                        help="range of bounciness setted for the target object")

    # layout
    parser.add_argument("--bouncy_axis_length",
                        type=float,
                        default=1.5,
                        help="Length of spacing between target object and zone.")

    # ramp
    parser.add_argument("--ramp_scale",
                        type=str,
                        default="[0.2,0.25,0.5]",
                        help="Scaling factor of the ramp in xyz.")

    # box_piles
    parser.add_argument("--use_box_piles",
                        type=int,
                        default=1,
                        help="Whether to place box_piles between the target and the zone")

    parser.add_argument("--use_blocker_with_hole",
                        type=int,
                        default=0,
                        help="Whether to use wall with a hole as a blocker")

    parser.add_argument("--box_piles",
                        type=str,
                        default="cube",
                        help="comma-separated list of possible box_piles objects")

    parser.add_argument("--box_piles_position",
                        type=float,
                        default=0.5,
                        help="Fraction between 0 and 1 where to place the box_piles on the axis")

    parser.add_argument("--box_piles_scale",
                        type=str,
                        default="[0.1,1.0,2.0]",
                        help="Scaling factor of the box_piles in xyz.")

    parser.add_argument("--box_piles_material",
                        type=none_or_str,
                        default=None,
                        help="Material name for boxes. If None, same as zone material")

    # occluder/distractors
    parser.add_argument("--occluder_categories",
                        type=none_or_str,
                        default=OCCLUDER_CATS,
                        help="the category ids to sample occluders from")
    parser.add_argument("--distractor_categories",
                        type=none_or_str,
                        default=DISTRACTOR_CATS,
                        help="the category ids to sample distractors from")

    def postprocess(args):
        args.fupforce = handle_random_transform_args(args.fupforce)
        args.tbounce = handle_random_transform_args(args.tbounce)
        args.ramp_scale = handle_random_transform_args(args.ramp_scale)

        # box_piles
        args.use_box_piles = bool(args.use_box_piles)
        args.use_blocker_with_hole = bool(args.use_blocker_with_hole)

        if args.box_piles is not None:
            targ_list = args.box_piles.split(',')
            assert all([t in MODEL_NAMES for t in targ_list]), \
                "All box_piles object names must be elements of %s" % MODEL_NAMES
            args.box_piles = targ_list
        else:
            args.box_piles = MODEL_NAMES

        args.box_piles_scale = handle_random_transform_args(
            args.box_piles_scale)

        return args

    args = parser.parse_args()
    args = domino_postproc(args)
    args = postprocess(args)

    return args


class Bouncy(MultiDominoes):

    def __init__(self,
                 port: int = None,
                 zjitter=0.,
                 fupforce=[0., 0.],
                 target_bounciness=[0., 0.],
                 ramp_scale=[0.2, 0.25, 0.5],
                 bouncy_axis_length=1.15,
                 use_ramp=True,
                 use_box_piles=True,
                 use_blocker_with_hole=False,
                 box_piles=['cube'],
                 box_piles_position=0.5,
                 box_piles_scale=[2, 1, 1],
                 box_piles_material="leather_fine",
                 #  box_piles_color = None,
                 target_lift=3.0,
                 **kwargs):
        # initialize everything in common w / Multidominoes
        super().__init__(port=port, **kwargs)
        self.zjitter = zjitter
        self.fupforce = fupforce
        self.use_ramp = use_ramp
        self.ramp_scale = ramp_scale
        self.bouncy_axis_length = self.collision_axis_length = bouncy_axis_length
        self.use_box_piles = use_box_piles
        self.use_blocker_with_hole = use_blocker_with_hole
        self.box_piles = box_piles
        self.box_piles_position = box_piles_position
        self.box_piles_scale = box_piles_scale
        self.box_piles_material = box_piles_material
        # self.box_piles_color = box_piles_color
        self.target_lift = target_lift
        self.target_bounciness = target_bounciness
        self.force_offset_jitter = 0.

        self.use_obi = False

    def get_trial_initialization_commands(self) -> List[dict]:
        """This is where we string together the important commands of the controller in order"""
        commands = []

        # randomization across trials
        if not(self.randomize):
            self.trial_seed = (self.MAX_TRIALS * self.seed) + self._trial_num
            random.seed(self.trial_seed)
        else:
            self.trial_seed = -1  # not used

        # Choose and place the target zone.
        commands.extend(self._place_target_zone())

        # Choose and place a target object.
        commands.extend(self._place_and_push_target_object())


        # Place box_piles between target and zone
        if self.use_box_piles:
            commands.extend(self._place_box_piles())


        # # Set the probe color
        # if self.probe_color is None:
        #     self.probe_color = self.target_color if (self.monochrome and self.match_probe_and_target_color) else None

        # # Choose, place, and push a probe object.
        # commands.extend(self._place_and_push_probe_object())

        # # Build the intermediate structure that captures some aspect of "intuitive physics."
        # commands.extend(self._build_intermediate_structure())

        # Teleport the avatar to a reasonable position
        a_pos = self.get_random_avatar_position(radius_min=self.camera_radius_range[0],
                                                radius_max=self.camera_radius_range[1],
                                                angle_min=self.camera_min_angle,
                                                angle_max=self.camera_max_angle,
                                                y_min=self.camera_min_height,
                                                y_max=self.camera_max_height,
                                                center=TDWUtils.VECTOR3_ZERO)

        commands.extend([
            {"$type": "teleport_avatar_to",
             "position": a_pos},
            {"$type": "look_at_position",
             "position": self.camera_aim},
            {"$type": "set_focus_distance",
             "focus_distance": TDWUtils.get_distance(a_pos, self.camera_aim)}
        ])

        # Set the camera parameters
        self._set_avatar_attributes(a_pos)

        self.camera_position = a_pos
        self.camera_rotation = np.degrees(np.arctan2(a_pos['z'], a_pos['x']))
        dist = TDWUtils.get_distance(a_pos, self.camera_aim)
        self.camera_altitude = np.degrees(
            np.arcsin((a_pos['y'] - self.camera_aim['y'])/dist))

        # Place distractor objects in the background
        commands.extend(self._place_background_distractors())

        # Place occluder objects in the background
        commands.extend(self._place_occluders())

        # test mode colors
        if self.use_test_mode_colors:
            self._set_test_mode_colors(commands)

        return commands



    def _place_and_push_target_object(self) -> List[dict]:
        """
        Place a probe object at the other end of the collision axis, then apply a force to push it.
        Using probe mass and location here to allow for ramps. There is no dedicated probe in this controller.
        """
        # create a target object
        record, data = self.random_primitive(self._target_types,
                                             scale=self.target_scale_range,
                                             color=self.target_color,
                                             add_data=False)
        # add_data=(not self.remove_target)
        o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]
        self.target = record
        self.target_type = data["name"]
        self.target_color = rgb
        #scale = {'x': 0.5, 'y': 0.5, 'z': 0.5}
        self.target_scale = self.middle_scale = scale
        self.target_id = o_id


        # Where to put the target
        if self.target_rotation is None:
            self.target_rotation = self.get_rotation(
                self.target_rotation_range)

        # Add the object with random physics values
        commands = []

        # TODO: better sampling of random physics values
        self.probe_mass = random.uniform(
            self.probe_mass_range[0], self.probe_mass_range[1])
        self.probe_initial_position = {
            "x": -1.8*self.collision_axis_length, "y": self.target_lift, "z": 0.}
        rot = self.get_rotation(self.target_rotation_range)
        print(rot)

        if self.use_ramp:
            commands.extend(self._place_ramp_under_probe())
            # HACK rotation might've led to the object falling of the back of the ramp, so we're moving it forward
            self.probe_initial_position['x'] += self.target_lift

        # commands.extend(
        #     self.add_physics_object(
        #         record=record,
        #         position=self.probe_initial_position,
        #         rotation=rot,
        #         mass=self.probe_mass,
        #         # dynamic_friction=0.5,
        #         # static_friction=0.5,
        #         # bounciness=0.1,
        #         dynamic_friction=0.4,
        #         static_friction=0.4,
        #         bounciness=0,
        #         o_id=o_id))
        self.star_bouncy = random.uniform(*self.target_bounciness)
        commands.extend(
            self.add_primitive(
                record=record,
                position=self.probe_initial_position,
                rotation=rot,
                mass=self.probe_mass,
                scale_mass=False,
                material=self.target_material,
                color=rgb,
                scale=scale,
                # dynamic_friction=0.5,
                # static_friction=0.5,
                # bounciness=0.1,
                dynamic_friction=0.0,
                static_friction=0.0,
                bounciness=self.star_bouncy,
                o_id=o_id,
                add_data=True
            ))

        # Set the target material
        # commands.extend(
        #     self.get_object_material_commands(
        #         record, o_id, self.get_material_name(self.target_material)))

        # the target is the probe
        self.target_position = self.probe_initial_position

        # Scale the object and set its color.
        # commands.extend([
        #     {"$type": "set_color",
        #      "color": {"r": rgb[0], "g": rgb[1], "b": rgb[2], "a": 1.},
        #      "id": o_id},
        # {"$type": "scale_object",
        #  "scale_factor": scale,
        #  "id": o_id}])

        # Set its collision mode
        commands.extend([
            # {"$type": "set_object_collision_detection_mode",
            #  "mode": "continuous_speculative",
            #  "id": o_id},
            {"$type": "set_object_drag",
             "id": o_id,
             "drag": 0., "angular_drag": 0.}])

        # Apply a force to the target object
        self.push_force = self.get_push_force(
            scale_range=self.probe_mass * np.array(self.force_scale_range),
            angle_range=self.force_angle_range)
        #self.push_force = self.rotate_vector_parallel_to_floor(
        #    self.push_force, -rot['y'], degrees=True)

        self.push_position = self.probe_initial_position
        if self.use_ramp:
            self.push_cmd = {
                "$type": "apply_force_to_object",
                "force": self.push_force,
                "id": int(o_id)
            }
        else:
            self.push_position = {
                k: v+self.force_offset[k]*self.rotate_vector_parallel_to_floor(
                    self.target_scale, rot['y'])[k]
                for k, v in self.push_position.items()}
            self.push_position = {
                k: v+random.uniform(-self.force_offset_jitter,
                                    self.force_offset_jitter)
                for k, v in self.push_position.items()}

            self.push_cmd = {
                "$type": "apply_force_at_position",
                "force": self.push_force,
                "position": self.push_position,
                "id": int(o_id)
            }

        # decide when to apply the force
        self.force_wait = int(random.uniform(
            *get_range(self.force_wait_range)))
        print("force wait", self.force_wait)

        if self.force_wait == 0:
            commands.append(self.push_cmd)

        return commands

    # def _place_target_object(self) -> List[dict]:
        """
        Place a primitive object at one end of the collision axis.
        """

        # create a target object
        record, data = self.random_primitive(self._target_types,
                                             scale=self.target_scale_range,
                                             color=self.target_color,
                                             add_data=(not self.remove_target)
                                             )
        o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]
        self.target = record
        self.target_type = data["name"]
        self.target_color = rgb
        self.target_scale = self.middle_scale = scale
        self.target_id = o_id

        if any((s <= 0 for s in scale.values())):
            self.remove_target = True

        # Where to put the target
        if self.target_rotation is None:
            self.target_rotation = self.get_rotation(
                self.target_rotation_range)

        if self.target_position is None:
            self.target_position = {
                "x": 0.5 * self.collision_axis_length,
                "y": 0. if not self.remove_target else 10.0,
                "z": 0. if not self.remove_target else 10.0
            }

        # Commands for adding hte object
        commands = []
        commands.extend(
            self.add_physics_object(
                record=record,
                position=self.target_position,
                rotation=self.target_rotation,
                mass=2.0,
                dynamic_friction=0.0,
                static_friction=0.0,
                bounciness=0.8,
                o_id=o_id,
                add_data=(not self.remove_target)
            ))

        # Set the object material
        commands.extend(
            self.get_object_material_commands(
                record, o_id, self.get_material_name(self.target_material)))

        # Scale the object and set its color.
        commands.extend([
            {"$type": "set_color",
             "color": {"r": rgb[0], "g": rgb[1], "b": rgb[2], "a": 1.},
             "id": o_id},
            {"$type": "scale_object",
             "scale_factor": scale if not self.remove_target else TDWUtils.VECTOR3_ZERO,
             "id": o_id}])

        # If this scene won't have a target
        if self.remove_target:
            commands.append(
                {"$type": self._get_destroy_object_command_name(o_id),
                 "id": int(o_id)})
            self.object_ids = self.object_ids[:-1]

        return commands

    def is_done(self, resp: List[bytes], frame: int) -> bool:
        return frame >= 300

    def _place_box_piles(self) -> List[dict]:
        """Places the box_piles on a location on the axis and fixes it in place"""
        assert self.use_box_piles, "need to use box_piles"
        commands = []
        stone_color = {"r": 0.5, "g": 0.5, "b": 0.5, "a": 1.}

        if self.use_blocker_with_hole:
            # add box with hole #1 (closest to yellow zone) bottom block #1
            record, data = self.random_primitive(
                object_types=self.get_types(self.box_piles),
                scale={"x": 0.1, "y":1.0, "z": 0.2},
                color=self.random_color(exclude=self.target_color),
            )
            o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]

            pos = {
                "x": 1.0 * self.box_piles_position * self.bouncy_axis_length + 0.250,
                "y": -0.5 * scale['y'],
                "z": -1.1
            }

            commands.extend(self.add_physics_object(
                record=record,
                position=pos,
                rotation=TDWUtils.VECTOR3_ZERO,
                mass=1000,
                dynamic_friction=1.0,
                static_friction=1.0,
                bounciness=0.1,
                o_id=o_id,
                add_data=True)
            )

            # Set the middle object material
            commands.extend(
                self.get_object_material_commands(
                    record, o_id, self.get_material_name(self.zone_material)))

            # Scale the object and set its color.
            commands.extend([
                {"$type": "set_color",
                    "color": stone_color,
                    "id": o_id},
                {"$type": "scale_object",
                    "scale_factor": scale,
                    "id": o_id}])

            # make it a "kinematic" object that won't move
            commands.extend([
                {"$type": "set_object_collision_detection_mode",
                "mode": "continuous_speculative",
                "id": o_id},
                {"$type": "set_kinematic_state",
                "id": o_id,
                "is_kinematic": True,
                "use_gravity": True}])

            # add box with hole #1 (closest to yellow zone) bottom block #2
            record, data = self.random_primitive(
                object_types=self.get_types(self.box_piles),
                scale={"x": 0.1, "y":1.0, "z": 0.2},
                color=self.random_color(exclude=self.target_color),
            )
            o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]

            pos = {
                "x": 1.0 * self.box_piles_position * self.bouncy_axis_length + 0.250,
                "y": -0.5 * scale['y'],
                "z": 1.1
            }

            commands.extend(self.add_physics_object(
                record=record,
                position=pos,
                rotation=TDWUtils.VECTOR3_ZERO,
                mass=1000,
                dynamic_friction=1.0,
                static_friction=1.0,
                bounciness=0.1,
                o_id=o_id,
                add_data=True)
            )

            # Set the middle object material
            commands.extend(
                self.get_object_material_commands(
                    record, o_id, self.get_material_name(self.zone_material)))

            # Scale the object and set its color.
            commands.extend([
                {"$type": "set_color",
                    "color": stone_color,
                    "id": o_id},
                {"$type": "scale_object",
                    "scale_factor": scale,
                    "id": o_id}])

            # make it a "kinematic" object that won't move
            commands.extend([
                {"$type": "set_object_collision_detection_mode",
                "mode": "continuous_speculative",
                "id": o_id},
                {"$type": "set_kinematic_state",
                "id": o_id,
                "is_kinematic": True,
                "use_gravity": True}])

            # add box with hole #1 (closest to yellow zone) upper block
            record, data = self.random_primitive(
                object_types=self.get_types(self.box_piles),
                scale={"x": 0.1, "y":0.5, "z": 2.4},
                color=self.random_color(exclude=self.target_color),
            )
            o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]

            pos = {
                "x": 1.0 * self.box_piles_position * self.bouncy_axis_length + 0.250,
                "y": 0.5,
                "z": 0.0
            }

            commands.extend(self.add_physics_object(
                record=record,
                position=pos,
                rotation=TDWUtils.VECTOR3_ZERO,
                mass=2000,
                dynamic_friction=1.0,
                static_friction=1.0,
                bounciness=0.1,
                o_id=o_id,
                add_data=True)
            )

            # Set the middle object material
            commands.extend(
                self.get_object_material_commands(
                    record, o_id, self.get_material_name(self.zone_material)))

            # Scale the object and set its color.
            commands.extend([
                {"$type": "set_color",
                    "color": stone_color,
                    "id": o_id},
                {"$type": "scale_object",
                    "scale_factor": scale,
                    "id": o_id}])

            # make it a "kinematic" object that won't move
            commands.extend([
                {"$type": "set_object_collision_detection_mode",
                "mode": "continuous_speculative",
                "id": o_id},
                {"$type": "set_kinematic_state",
                "id": o_id,
                "is_kinematic": True,
                "use_gravity": True}])

        else:
            # add box #1 (closest to yellow zone)
            record, data = self.random_primitive(
                object_types=self.get_types(self.box_piles),
                scale=self.box_piles_scale,
                color=self.random_color(exclude=self.target_color),
            )

            o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]

            pos = {
                "x": 1.0 * self.box_piles_position * self.bouncy_axis_length + 0.250,
                "y": -0.5 * scale['y'],
                "z": 0
            }

            commands.extend(self.add_physics_object(
                record=record,
                position=pos,
                rotation=TDWUtils.VECTOR3_ZERO,
                mass=1000,
                dynamic_friction=0.0,
                static_friction=0.0,
                bounciness=0.1,
                o_id=o_id,
                add_data=True)
            )

            # Set the middle object material
            commands.extend(
                self.get_object_material_commands(
                    record, o_id, self.get_material_name(self.zone_material)))

            # Scale the object and set its color.
            commands.extend([
                {"$type": "set_color",
                    "color": stone_color,
                    "id": o_id},
                {"$type": "scale_object",
                    "scale_factor": scale,
                    "id": o_id}])

            # make it a "kinematic" object that won't move
            commands.extend([
                {"$type": "set_object_collision_detection_mode",
                "mode": "continuous_speculative",
                "id": o_id},
                {"$type": "set_kinematic_state",
                "id": o_id,
                "is_kinematic": True,
                "use_gravity": True}])

        bouncy_zone_color = {"r": 0.0, "g": 0, "b": 1.0, "a": 1.}
        # add box #2 (second closest to yellow zone)
        record, data = self.random_primitive(
            object_types=self.get_types(self.box_piles),
            scale={"x": 0.8 + 0.45, "y": 0.2, "z": 2.0},
            color=self.random_color(exclude=self.target_color),
        )
        o_id, scale, rgb_var = [data[k] for k in ["id", "scale", "color"]]

        pos = {
            "x": 1.0 * self.box_piles_position * self.bouncy_axis_length - 0.5  + 0.185,
            "y": -0.5 * scale['y'],
            "z": 0
        }

        commands.extend(self.add_physics_object(
            record=record,
            position=pos,
            rotation=TDWUtils.VECTOR3_ZERO,
            mass=1000,
            dynamic_friction=0.0,
            static_friction=0.0,
            bounciness=0.8,
            o_id=o_id,
            add_data=True)
        )

        # Set the middle object material
        commands.extend(
            self.get_object_material_commands(
                record, o_id, self.get_material_name(self.zone_material)))

        # Scale the object and set its color.
        commands.extend([
            {"$type": "set_color",
                "color": bouncy_zone_color ,
                "id": o_id},
            {"$type": "scale_object",
                "scale_factor": scale,
                "id": o_id}])

        # make it a "kinematic" object that won't move
        commands.extend([
            {"$type": "set_object_collision_detection_mode",
             "mode": "continuous_speculative",
             "id": o_id},
            {"$type": "set_kinematic_state",
             "id": o_id,
             "is_kinematic": True,
             "use_gravity": True}])

        # add box #3 (third closest to yellow zone)
        # the stone
        record, data = self.random_primitive(
            object_types=self.get_types(self.box_piles),
            scale={"x": 1.0 - 0.25, "y": 0.98, "z": 2.0},
            color=self.random_color(exclude=self.target_color),
        )
        o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]

        pos = {
            "x": 1.0 * self.box_piles_position * self.bouncy_axis_length - 1.5 + 0.125,
            "y": 0,
            "z": 0
        }

        commands.extend(self.add_physics_object(
            record=record,
            position=pos,
            rotation=TDWUtils.VECTOR3_ZERO,
            mass=1000,
            dynamic_friction=0.0,
            static_friction=0.0,
            bounciness=0.0,
            o_id=o_id,
            add_data=True)
        )

        # Set the middle object material
        commands.extend(
            self.get_object_material_commands(
                record, o_id, self.get_material_name(self.zone_material)))

        # Scale the object and set its color.
        commands.extend([
            {"$type": "set_color",
                "color": {"r": 0.5, "g": 0.5, "b": 0.5, "a": 1.},
                "id": o_id},
            {"$type": "scale_object",
                "scale_factor": scale,
                "id": o_id}])

        # make it a "kinematic" object that won't move
        commands.extend([
            {"$type": "set_object_collision_detection_mode",
             "mode": "continuous_speculative",
             "id": o_id},
            {"$type": "set_kinematic_state",
             "id": o_id,
             "is_kinematic": True,
             "use_gravity": True}])

        # add box #4 (furtherest to yellow zone)
        record, data = self.random_primitive(
            object_types=self.get_types(self.box_piles),
            scale={"x": 1.25, "y": 1.0, "z": 2.0},
            color=self.random_color(exclude=self.target_color),
        )
        o_id, scale, rgb = [data[k] for k in ["id", "scale", "color"]]

        pos = {
            "x": 1.0 * self.box_piles_position * self.bouncy_axis_length - 2.51 + 0.125,
            "y": 0,
            "z": 0
        }

        commands.extend(self.add_physics_object(
            record=record,
            position=pos,
            rotation=TDWUtils.VECTOR3_ZERO,
            mass=1000,
            dynamic_friction=0.0,
            static_friction=0.0,
            bounciness=0.8,
            o_id=o_id,
            add_data=True)
        )

        # Set the middle object material
        commands.extend(
            self.get_object_material_commands(
                record, o_id, self.get_material_name(self.zone_material)))

        # Scale the object and set its color.
        commands.extend([
            {"$type": "set_color",
                "color": bouncy_zone_color ,
                "id": o_id},
            {"$type": "scale_object",
                "scale_factor": scale,
                "id": o_id}])

        # make it a "kinematic" object that won't move
        commands.extend([
            {"$type": "set_object_collision_detection_mode",
             "mode": "continuous_speculative",
             "id": o_id},
            {"$type": "set_kinematic_state",
             "id": o_id,
             "is_kinematic": True,
             "use_gravity": True}])

        # change time step for ball to go faster
        commands.extend([
            {"$type": "set_time_step", "time_step": 0.015}])

        return commands

    def _get_zone_location(self, scale):
        """Where to place the target zone? Right behind the target object."""
        BUFFER = 0
        return {
            # + 0.5 * self.zone_scale_range['x'] + BUFFER,
            "x": self.collision_axis_length + 0.125,
            "y": 0.0 if not self.remove_zone else 10.0,
            "z":  random.uniform(-self.zjitter, self.zjitter) if not self.remove_zone else 10.0
        }

    def clear_static_data(self) -> None:
        Dominoes.clear_static_data(self)
        # clear some other stuff

    def _write_static_data(self, static_group: h5py.Group) -> None:
        try:
            static_group.create_dataset("if_use_blocker_with_hole", data=self.use_blocker_with_hole)
        except (AttributeError,TypeError):
            pass

        try:
            static_group.create_dataset("if_use_box_piles", data=self.use_box_piles)
        except (AttributeError,TypeError):
            pass
        Dominoes._write_static_data(self, static_group)

        # static_group.create_dataset("bridge_height", data=self.bridge_height)

    @staticmethod
    def get_controller_label_funcs(classname="Bouncy"):

        funcs = Dominoes.get_controller_label_funcs(classname)

        return funcs


if __name__ == "__main__":
    import platform
    import os

    args = get_bouncy_args("bouncy")

    if platform.system() == 'Linux':
        if args.gpu is not None:
            os.environ["DISPLAY"] = ":" + str(args.gpu + 1)
        else:
            os.environ["DISPLAY"] = ":"

    ColC = Bouncy(
        room=args.room,
        randomize=args.random,
        seed=args.seed,
        target_zone=args.zone,
        zone_location=args.zlocation,
        zone_scale_range=args.zscale,
        zone_color=args.zcolor,
        zone_material=args.zmaterial,
        zone_friction=args.zfriction,
        target_objects=args.target,
        probe_objects=args.probe,
        target_scale_range=args.tscale,
        target_rotation_range=args.trot,
        target_bounciness=args.tbounce,
        probe_rotation_range=args.prot,
        probe_scale_range=args.pscale,
        probe_mass_range=args.pmass,
        target_color=args.color,
        probe_color=args.pcolor,
        collision_axis_length=args.collision_axis_length,
        force_scale_range=args.fscale,
        force_angle_range=args.frot,
        force_offset=args.foffset,
        force_offset_jitter=args.fjitter,
        force_wait=args.fwait,
        remove_target=bool(args.remove_target),
        remove_zone=bool(args.remove_zone),
        zjitter=args.zjitter,
        fupforce=args.fupforce,
        use_ramp=args.ramp,
        use_box_piles=args.use_box_piles,
        use_blocker_with_hole=args.use_blocker_with_hole,
        box_piles=args.box_piles,
        box_piles_position=args.box_piles_position,
        box_piles_scale=args.box_piles_scale,
        box_piles_material=args.box_piles_material,
        ## not scenario-specific
        camera_radius=args.camera_distance,
        camera_min_angle=args.camera_min_angle,
        camera_max_angle=args.camera_max_angle,
        camera_min_height=args.camera_min_height,
        camera_max_height=args.camera_max_height,
        monochrome=args.monochrome,
        material_types=args.material_types,
        target_material=args.tmaterial,
        probe_material=args.pmaterial,
        distractor_types=args.distractor,
        distractor_categories=args.distractor_categories,
        num_distractors=args.num_distractors,
        occluder_types=args.occluder,
        occluder_categories=args.occluder_categories,
        num_occluders=args.num_occluders,
        occlusion_scale=args.occlusion_scale,
        ramp_scale=args.ramp_scale,
        bouncy_axis_length=args.bouncy_axis_length,
        target_lift=args.tlift,
        flex_only=args.only_use_flex_objects,
        no_moving_distractors=args.no_moving_distractors,
        use_test_mode_colors=args.use_test_mode_colors
    )

    if bool(args.run):
        ColC.run(num=args.num,
                 output_dir=args.dir,
                 temp_path=args.temp,
                 width=args.width,
                 height=args.height,
                 save_passes=args.save_passes.split(','),
                 save_movies=args.save_movies,
                 save_labels=args.save_labels,
                 save_meshes=args.save_meshes,
                 write_passes=args.write_passes,
                 args_dict=vars(args)
                 )
    else:
        ColC.communicate({"$type": "terminate"})