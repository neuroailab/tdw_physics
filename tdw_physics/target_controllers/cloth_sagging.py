import sys, os, copy
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import random
import numpy as np
import h5py

from tdw.librarian import ModelRecord, MaterialLibrarian, ModelLibrarian
from tdw.tdw_utils import TDWUtils
from tdw_physics.target_controllers.dominoes import Dominoes, get_args, ArgumentParser
from tdw_physics.flex_dataset import FlexDataset, FlexParticles
from tdw_physics.rigidbodies_dataset import RigidbodiesDataset
from tdw_physics.util import MODEL_LIBRARIES, get_parser, none_or_str

# fluid
from tdw.flex.fluid_types import FluidTypes

MODEL_NAMES = [r.name for r in MODEL_LIBRARIES['models_flex.json'].records]
MODEL_CORE = [r.name for r in MODEL_LIBRARIES['models_core.json'].records]

def get_flex_args(dataset_dir: str, parse=True):

    common = get_parser(dataset_dir, get_help=False)
    domino, domino_postproc = get_args(dataset_dir, parse=False)
    parser = ArgumentParser(parents=[common, domino], conflict_handler='resolve', fromfile_prefix_chars='@')

    parser.add_argument("--all_flex_objects",
                        type=int,
                        default=1,
                        help="Whether all rigid objects should be FLEX")
    parser.add_argument("--step_physics",
                        type=int,
                        default=100,
                        help="How many physics steps to run forward after adding a solid FLEX object")
    parser.add_argument("--cloth",
                        action="store_true",
                        help="Demo: whether to drop a cloth")
    parser.add_argument("--squishy",
                        action="store_true",
                        help="Demo: whether to drop a squishy ball")
    parser.add_argument("--fluid",
                        action="store_true",
                        help="Demo: whether to drop fluid")
    parser.add_argument("--fwait",
                        type=none_or_str,
                        default="30",
                        help="How many frames to wait before applying the force")
    parser.add_argument("--collision_label_threshold",
                        type=float,
                        default=0.1,
                        help="Euclidean distance at which target and zone are said to be touching")
#    parser.add_argument("--drapeobject",
#                        type=str,
#                        default="alma_floor_lamp",
#                        help="object to use for revealing cloth properties (from models_core.json)")

    def postprocess(args):

        args = domino_postproc(args)
        args.all_flex_objects = bool(int(args.all_flex_objects))

        return args

    if not parse:
        return (parser, postproccess)

    args = parser.parse_args()
    args = postprocess(args)

    return args


class ClothSagging(Dominoes, FlexDataset):

    FLEX_RECORDS = ModelLibrarian(str(Path("flex.json").resolve())).records
    CLOTH_RECORD = MODEL_LIBRARIES["models_special.json"].get_record("cloth_square")
    SOFT_RECORD = MODEL_LIBRARIES["models_flex.json"].get_record("sphere")
    RECEPTACLE_RECORD = MODEL_LIBRARIES["models_special.json"].get_record("fluid_receptacle1x1")
    FLUID_TYPES = FluidTypes()

    def __init__(self, port: int = 1071,
                 all_flex_objects=True,
                 use_cloth=False,
                 use_squishy=False,
                 use_fluid=False,
                 step_physics=False,
                 drape_object="alma_floor_lamp", #alma_floor_lamp, metal_sculpture, white_lamp, vase_01, linbrazil_diz_armchair, desk_lamp
                 tether_stiffness_range = [0.1, 1.0],
                 bend_stiffness_range = [0.1, 1.0],#[0.0, 1.0],
                 stretch_stiffness_range = [0.1, 1.0],
                 distance_ratio_range = [0.25,0.45],#[0.2, 0.8],
                 anchor_locations = [-0.4, 0.4],
                 anchor_jitter = 0.2,#0.2,
                 height_jitter = 0.3,#0.3,
                 collision_label_threshold=0.1,
                 **kwargs):

        Dominoes.__init__(self, port=port, **kwargs)
        self._clear_flex_data()

        self.all_flex_objects = all_flex_objects
        self._set_add_physics_object()

        self.step_physics = step_physics
        self.use_cloth = use_cloth
        self.use_squishy = use_squishy
        self.use_fluid = use_fluid

        if self.use_fluid:
            self.ft_selection = random.choice(self.FLUID_TYPES.fluid_type_names)

        self.drape_object = drape_object
        self.tether_stiffness_range = tether_stiffness_range
        self.bend_stiffness_range = bend_stiffness_range
        self.stretch_stiffness_range = stretch_stiffness_range
        self.distance_ratio_range = distance_ratio_range
        self.anchor_locations = anchor_locations
        self.anchor_jitter = anchor_jitter
        self.height_jitter = height_jitter

        # for detecting collisions
        self.collision_label_thresh = collision_label_threshold

    def _set_add_physics_object(self):
        if self.all_flex_objects:
            self.add_physics_object = self.add_flex_solid_object
            self.add_primitive = self.add_flex_solid_object
        else:
            self.add_physics_object = self.add_rigid_physics_object


    def get_scene_initialization_commands(self) -> List[dict]:

        commands = Dominoes.get_scene_initialization_commands(self)
        commands[0].update({'convexify': True})
        create_container = {
            "$type": "create_flex_container",
            # "collision_distance": 0.001,
            "collision_distance": 0.025,
            "static_friction": 1.0,
            "dynamic_friction": 1.0,
            "radius": 0.1875,
            'max_particles': 50000}
            # 'max_particles': 200000}

        if self.use_fluid:
            create_container.update({
                'viscosity': self.FLUID_TYPES.fluid_types[self.ft_selection].viscosity,
                'adhesion': self.FLUID_TYPES.fluid_types[self.ft_selection].adhesion,
                'cohesion': self.FLUID_TYPES.fluid_types[self.ft_selection].cohesion,
                'fluid_rest': 0.05,
                'damping': 0.01,
                'subsetp_count': 5,
                'iteration_count': 8,
                'buoyancy': 1.0})

        commands.append(create_container)

        if self.use_fluid:
            commands.append({"$type": "set_time_step", "time_step": 0.005})

        return commands

    def get_trial_initialization_commands(self) -> List[dict]:

        # clear the flex data
        FlexDataset.get_trial_initialization_commands(self)
        return Dominoes.get_trial_initialization_commands(self)

    def _get_send_data_commands(self) -> List[dict]:
        commands = Dominoes._get_send_data_commands(self)
        commands.extend(FlexDataset._get_send_data_commands(self))
        return commands

    def add_rigid_physics_object(self, *args, **kwargs):
        """
        Make sure controller knows to treat probe, zone, target, etc. as non-flex objects
        """

        o_id = kwargs.get('o_id', None)
        if o_id is None:
            o_id: int = self.get_unique_id()
            kwargs['o_id'] = o_id

        commands = Dominoes.add_physics_object(self, *args, **kwargs)
        self.non_flex_objects.append(o_id)

        print("Add rigid physics object", o_id)

        return commands

    def add_flex_solid_object(self,
                              record: ModelRecord,
                              position: Dict[str, float],
                              rotation: Dict[str, float],
                              mesh_expansion: float = 0,
                              particle_spacing: float = 0.035,
                              mass: float = 1,
                              scale: Optional[Dict[str, float]] = {"x": 0.1, "y": 0.5, "z": 0.25},
                              material: Optional[str] = None,
                              color: Optional[list] = None,
                              exclude_color: Optional[list] = None,
                              o_id: Optional[int] = None,
                              add_data: Optional[bool] = True,
                              **kwargs) -> List[dict]:

        # so objects don't get stuck in each other -- an unfortunate feature of FLEX
        position = {'x': position['x'], 'y': position['y'] + 0.1, 'z': position['z']}

        commands = FlexDataset.add_solid_object(
            self,
            record = record,
            position = position,
            rotation = rotation,
            scale = scale,
            mesh_expansion = mesh_expansion,
            particle_spacing = particle_spacing,
            mass_scale = 1,
            o_id = o_id)

        # set mass
        commands.append({"$type": "set_flex_object_mass",
                         "mass": mass,
                         "id": o_id})

        # set material and color
        commands.extend(
            self.get_object_material_commands(
                record, o_id, self.get_material_name(material)))

        color = color if color is not None else self.random_color(exclude=exclude_color)
        commands.append(
            {"$type": "set_color",
             "color": {"r": color[0], "g": color[1], "b": color[2], "a": 1.},
             "id": o_id})

        # step physics
        if bool(self.step_physics):
            print("stepping physics forward", self.step_physics)
            commands.append({"$type": "step_physics",
                             "frames": self.step_physics})

        # add data
        print("Add FLEX physics object", o_id)
        if add_data:
            self._add_name_scale_color(record, {'color': color, 'scale': scale, 'id': o_id})
            self.masses = np.append(self.masses, mass)

        return commands

    def _get_push_cmd(self, o_id, position_or_particle=None):
        if not self.all_flex_objects:
            return Dominoes._get_push_cmd(self, o_id, position_or_particle)
        cmd = {"$type": "apply_force_to_flex_object",
               "force": self.push_force,
               "id": o_id,
               "particle": -1}
        print("PUSH CMD FLEX")
        print(cmd)
        return cmd

    def drop_cloth(self) -> List[dict]:

        self.cloth = self.CLOTH_RECORD
        self.cloth_id = self._get_next_object_id()
        self.cloth_position = {"x": 0.0, "y": 1.5, "z":-0.6}
        self.cloth_color = self.target_color if self.target_color is not None else self.random_color()
        self.cloth_scale = {'x': 1.0, 'y': 1.0, 'z': 1.0}
        self.cloth_mass = 0.5
        self.cloth_data = {"name": self.cloth.name, "color": self.cloth_color, "scale": self.cloth_scale, "id": self.cloth_id}

        commands = self.add_cloth_object(
            record = self.cloth,
            position = self.cloth_position,
            rotation = {k:0 for k in ['x','y','z']},
            scale=self.cloth_scale,
            mass_scale = 1,
            mesh_tesselation = 1,
            tether_stiffness = random.uniform(self.tether_stiffness_range[0], self.tether_stiffness_range[1]), # doesn't do much visually!
            bend_stiffness = random.uniform(self.bend_stiffness_range[0], self.bend_stiffness_range[1]), #changing this will lead to visible changes in cloth deformability
            stretch_stiffness = random.uniform(self.stretch_stiffness_range[0], self.stretch_stiffness_range[1]), # doesn't do much visually!
            o_id = self.cloth_id)

        # replace the target w the cloth
        self._replace_target_with_object(self.cloth, self.cloth_data)

        # set mass
        commands.append({"$type": "set_flex_object_mass",
                         "mass": self.cloth_mass,
                         "id": self.cloth_id})

        # color cloth
        commands.append(
            {"$type": "set_color",
             "color": {"r": self.cloth_color[0], "g": self.cloth_color[1], "b": self.cloth_color[2], "a": 1.},
             "id": self.cloth_id})

        self._add_name_scale_color(
            self.cloth, {'color': self.cloth_color, 'scale': self.cloth_scale, 'id': self.cloth_id})
        self.masses = np.append(self.masses, self.cloth_mass)

        self._replace_target_with_object(self.cloth, self.cloth_data)

        return commands

    def drop_squishy(self) -> List[dict]:

        self.squishy = self.SOFT_RECORD
        self.squishy_id = self._get_next_object_id()
        self.squishy_position = {'x': 0., 'y': 1.5, 'z': 0.}
        rotation = {k:0 for k in ['x','y','z']}

        self.squishy_color = [0.0,0.8,1.0]
        self.squishy_scale = {k:0.5 for k in ['x','y','z']}
        self.squishy_mass = 2.0

        commands = self.add_soft_object(
            record = self.squishy,
            position = self.squishy_position,
            rotation = rotation,
            scale=self.squishy_scale,
            o_id = self.squishy_id)

        # set mass
        commands.append({"$type": "set_flex_object_mass",
                         "mass": self.squishy_mass,
                         "id": self.squishy_id})

        commands.append(
            {"$type": "set_color",
             "color": {"r": self.squishy_color[0], "g": self.squishy_color[1], "b": self.squishy_color[2], "a": 1.},
             "id": self.squishy_id})

        self._add_name_scale_color(
            self.squishy, {'color': self.squishy_color, 'scale': self.squishy_scale, 'id': self.squishy_id})
        self.masses = np.append(self.masses, self.squishy_mass)

        return commands

    def drop_fluid(self) -> List[dict]:

        commands = []

        # create a pool for the fluid
        self.pool_id = self._get_next_object_id()
        print("POOL ID", self.pool_id)
        self.non_flex_objects.append(self.pool_id)
        commands.append(self.add_transforms_object(record=self.RECEPTACLE_RECORD,
                                                   position=TDWUtils.VECTOR3_ZERO,
                                                   rotation=TDWUtils.VECTOR3_ZERO,
                                                   o_id=self.pool_id,
                                                   add_data=True))
        commands.append({"$type": "set_kinematic_state",
                         "id": self.pool_id,
                         "is_kinematic": True,
                         "use_gravity": False})

        # add the fluid; this will also step physics forward 500 times
        self.fluid_id = self._get_next_object_id()
        print("FLUID ID", self.fluid_id)
        commands.extend(self.add_fluid_object(
            position={"x": 0.0, "y": 1.0, "z": 0.0},
            rotation=TDWUtils.VECTOR3_ZERO,
            o_id=self.fluid_id,
            fluid_type=self.ft_selection))
        self.fluid_object_ids.append(self.fluid_id)

        # restore usual time step
        commands.append({"$type": "set_time_step", "time_step": 0.01})

        return commands

    def _place_ramp_under_probe(self) -> List[dict]:

        cmds = Dominoes._place_ramp_under_probe(self)
        self.non_flex_objects.append(self.ramp_id)
        if self.ramp_base_height >= 0.01:
            self.non_flex_objects.append(self.ramp_base_id)
        return cmds

    def _place_and_push_probe_object(self):
        return []

    def _get_zone_location(self, scale):
        dratio = random.uniform(self.distance_ratio_range[0], self.distance_ratio_range[1])
        dist = max(self.anchor_locations) - min(self.anchor_locations)
        zonedist =  dratio * dist
        return {
            "x": max(self.anchor_locations)-zonedist,
            "y": 0.0 if not self.remove_zone else 10.0,
            "z": 0.0 if not self.remove_zone else 10.0
        }

    def is_done(self, resp: List[bytes], frame: int) -> bool:
        return frame >= 300

    def _build_intermediate_structure(self) -> List[dict]:

        commands = []

        # add two objects on each side of a target object
        self.objrec1 = MODEL_LIBRARIES["models_flex.json"].get_record("cube")
        self.objrec1_id = self._get_next_object_id()
        self.objrec1_position = {'x': min(self.anchor_locations)-random.uniform(0.0,self.anchor_jitter), 'y': 0., 'z': 0.}
        self.objrec1_rotation = {k:0 for k in ['x','y','z']}
        self.objrec1_scale = {'x': 0.2, 'y': 1.2+random.uniform(-self.height_jitter,self.height_jitter), 'z': 0.5}
        self.objrec1_mass = 25.0
        commands.extend(self.add_flex_solid_object(
                              record = self.objrec1,
                              position = self.objrec1_position,
                              rotation = self.objrec1_rotation,
                              mesh_expansion = 0.0,
                              particle_spacing = 0.035,
                              mass = self.objrec1_mass,
                              scale = self.objrec1_scale,
                              o_id = self.objrec1_id,
                              ))

        self.objrec2 = MODEL_LIBRARIES["models_flex.json"].get_record("cube")
        self.objrec2_id = self._get_next_object_id()
        self.objrec2_position = {'x': max(self.anchor_locations)+random.uniform(0.0,self.anchor_jitter), 'y': 0., 'z': 0.}
        self.objrec2_rotation = {k:0 for k in ['x','y','z']}
        self.objrec2_scale = {'x': 0.2, 'y': 0.7+random.uniform(-self.height_jitter,self.height_jitter), 'z': 0.5}
        self.objrec2_mass = 25.0
        commands.extend(self.add_flex_solid_object(
                               record = self.objrec2,
                               position = self.objrec2_position,
                               rotation = self.objrec2_rotation,
                               mesh_expansion = 0.0,
                               particle_spacing = 0.035,
                               mass = self.objrec2_mass,
                               scale = self.objrec2_scale,
                               o_id = self.objrec2_id,
                               ))

        self.objrec3 = MODEL_LIBRARIES["models_core.json"].get_record(self.drape_object)
        self.objrec3_id = self._get_next_object_id()
        self.objrec3_position = {'x': 0., 'y': 0., 'z': -1.5}
        self.objrec3_rotation = {k:0 for k in ['x','y','z']}
        self.objrec3_scale = {'x': 1.0, 'y': 0.8, 'z': 1.0}
        self.objrec3_mass = 100.0
        commands.extend(self.add_flex_solid_object(
                               record = self.objrec3,
                               position = self.objrec3_position,
                               rotation = self.objrec3_rotation,
                               mesh_expansion = 0.0,
                               particle_spacing = 0.035,
                               mass = self.objrec3_mass,
                               scale = self.objrec3_scale,
                               o_id = self.objrec3_id,
                               ))

        commands.extend(self.drop_cloth() if self.use_cloth else [])

        return commands

    @staticmethod
    def get_flex_object_collision(flex, obj1, obj2, collision_thresh=0.15):
        '''
        flex: FlexParticles Data
        '''
        collision = False
        p1 = p2 = None
        for n in range(flex.get_num_objects()):
            if flex.get_id(n) == obj1:
                p1 = flex.get_particles(n)
            elif flex.get_id(n) == obj2:
                p2 = flex.get_particles(n)

        if (p1 is not None) and (p2 is not None):

            p1 = np.array(p1)[:,0:3]
            p2 = np.array(p2)[:,0:3]

            dists = np.sqrt(np.square(p1[:,None] - p2[None,:]).sum(-1))
            collision = (dists < collision_thresh).max()
            print(obj1, p1.shape, obj2, p2.shape, "colliding?", collision)

        return collision

    def _write_frame_labels(self,
                            frame_grp: h5py.Group,
                            resp: List[bytes],
                            frame_num: int,
                            sleeping: bool) -> Tuple[h5py.Group, List[bytes], int, bool]:

        labels, resp, grame_num, done = RigidbodiesDataset._write_frame_labels(self, frame_grp, resp, frame_num, sleeping)

        has_target = (not self.remove_target) or self.replace_target
        has_zone = not self.remove_zone
        labels.create_dataset("has_target", data=has_target)
        labels.create_dataset("has_zone", data=has_zone)
        if not (has_target or has_zone):
            return labels, resp, frame_num, done

        print("frame num", frame_num)
        flex = None
        for r in resp[:-1]:
            if FlexParticles.get_data_type_id(r) == "flex":
                flex = FlexParticles(r)

        if has_target and has_zone and (flex is not None):
            are_touching = self.get_flex_object_collision(flex,
                                                          obj1=self.target_id,
                                                          obj2=self.zone_id,
                                                          collision_thresh=self.collision_label_thresh)
            labels.create_dataset("target_contacting_zone", data=are_touching)

        return labels, resp, frame_num, done


if __name__ == '__main__':
    import platform, os

    args = get_flex_args("flex_dominoes")

    print("core object types", MODEL_CORE)

    if platform.system() == 'Linux':
        if args.gpu is not None:
            os.environ["DISPLAY"] = ":0." + str(args.gpu)
        else:
            os.environ["DISPLAY"] = ":0"

        launch_build = False
    else:
        launch_build = True

    if platform.system() != 'Windows' and args.fluid:
        print("WARNING: Flex fluids are only supported in Windows")

    C = ClothSagging(
        port=args.port,
        launch_build=launch_build,
        all_flex_objects=args.all_flex_objects,
        use_cloth=args.cloth,
        use_squishy=args.squishy,
        use_fluid=args.fluid,
        step_physics=args.step_physics,
        room=args.room,
        num_middle_objects=args.num_middle_objects,
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
        middle_objects=args.middle,
        target_scale_range=args.tscale,
        target_rotation_range=args.trot,
        probe_rotation_range=args.prot,
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
        spacing_jitter=args.spacing_jitter,
        lateral_jitter=args.lateral_jitter,
        middle_scale_range=args.mscale,
        middle_rotation_range=args.mrot,
        middle_mass_range=args.mmass,
        horizontal=args.horizontal,
        remove_target=bool(args.remove_target),
        remove_zone=bool(args.remove_zone),
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
        middle_material=args.mmaterial,
        distractor_types=args.distractor,
        distractor_categories=args.distractor_categories,
        num_distractors=args.num_distractors,
        occluder_types=args.occluder,
        occluder_categories=args.occluder_categories,
        num_occluders=args.num_occluders,
        occlusion_scale=args.occlusion_scale,
        remove_middle=args.remove_middle,
        use_ramp=bool(args.ramp),
        ramp_color=args.rcolor,
        flex_only=args.only_use_flex_objects,
        no_moving_distractors=args.no_moving_distractors,
        collision_label_threshold=args.collision_label_threshold
    )

    if bool(args.run):
        C.run(num=args.num,
             output_dir=args.dir,
             temp_path=args.temp,
             width=args.width,
             height=args.height,
             write_passes=args.write_passes.split(','),
             save_passes=args.save_passes.split(','),
             save_movies=args.save_movies,
             save_labels=args.save_labels,
             args_dict=vars(args))
    else:
        end = C.communicate({"$type": "terminate"})
        print([OutputData.get_data_type_id(r) for r in end])
