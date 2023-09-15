import sys, os, subprocess, logging, time
from tqdm import tqdm
from tdw_physics.postprocessing.stimuli import pngs_to_mp4
import operator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from abc import ABC
import numpy as np
from tdw.librarian import ModelRecord
from .rigidbodies_dataset import RigidbodiesDataset
from tdw_physics.controller import Controller
from .dataset import Dataset, PASSES
from tdw.output_data import OutputData, Rigidbodies, Collision, EnvironmentCollision, Transforms, StaticRigidbodies
from dataclasses import dataclass
from typing import List, Dict, Optional
from tdw.tdw_utils import TDWUtils
from scipy.stats import vonmises, norm
from .noisy_utils import *
import json
from pathlib import Path
import h5py
import shutil
from tdw_physics.postprocessing.labels import get_labels_from
import random
from tdw.quaternion_utils import QuaternionUtils


XYZ = ['x', 'y', 'z']


@dataclass
class RigidNoiseParams:
    """
    TODO: confirm collision noise on force vs. momentum

    Holders for parameters defining the amount of noise in noisy simulations

    All noise parameters can take a `None` value to remove noise for that particular parameter; `None` is the default for all noise values
    
    position: Dict[str, float]: Gaussian noise around position of all dynamic objects (separate noise for each dimension)

    rotation: Dict[str, float]: vonMises precision for rotation noise along the x,y,z axes (separate noise for each dimension)

    velocity_dir: Dict[str, float]: vonMises precision for noise in the initial velocity direction along the x,y,z axes (separate noise for each dimension)

    velocity_mag: float: log-normal noise around the magnitude of initial speed

    mass: float: log-normal noise around the true object mass

    static_friction: float: Gaussian noise around the true object static friction

    dynamic_friction: float: Gaussian noise around the true object dynamic friction

    bounciness: float: Gaussian noise around the true object bounciness

    collision_dir: Dict[str, float]: vonMises precision for noise injected into the force normals for each collision along the x,y,z axes (separate noise for each dimension)

    collision_mag: float: log-normal noise around the true magnitude of the force normals for a collision

    coll_threshold: float: collisions below this threshold are ignored when adding noise
    """
    position: Dict[str, float] = None
    rotation: Dict[str, float] = None
    velocity_dir: Dict[str, float] = None
    velocity_mag: float = None
    mass: float = None
    static_friction: float = None
    dynamic_friction: float = None
    bounciness: float = None
    collision_dir: float = None
    collision_mag: float = None
    coll_threshold: float = None
    start_simulate: int = None

    def save(self, flpth):
        selfobj = {
            'position': self.position,
            'rotation': self.rotation,
            'velocity_dir': self.velocity_dir,
            'velocity_mag': self.velocity_mag,
            'mass': self.mass,
            'static_friction': self.static_friction,
            'dynamic_friction': self.dynamic_friction,
            'bounciness': self.bounciness,
            'collision_dir': self.collision_dir,
            'collision_mag': self.collision_mag,
            'coll_threshold': self.coll_threshold,
            'start_simulate': self.start_simulate
        }
        with open(flpth, 'w') as ofl:
            json.dump(selfobj, ofl)

    # def to_string(self):
    #     return'-'.join([x for x in .values()])

    @staticmethod
    def load(flpth):
        with open(flpth, 'r') as ifl:
            obj = json.load(ifl)
        return RigidNoiseParams(**obj)


# class IsotropicRigidNoiseParams(RigidNoiseParams):
#     """
#     As RigidNoiseParams except all [x,y,z] noises become floats
#     """

#     def __init__(self,
#                  position: float = None,
#                  rotation: float = None,
#                  velocity_dir: float = None,
#                  collision_dir: float = None,
#                  **kwargs):
#         if position is not None:
#             position = dict([[k, position] for k in XYZ])
#         if rotation is not None:
#             rotation = dict([[k, rotation] for k in XYZ])
#         if velocity_dir is not None:
#             velocity_dir = dict([[k, velocity_dir] for k in XYZ])
#         # if collision_dir is not None:
#         #     collision_dir = dict([[k, collision_dir] for k in XYZ])
#         RigidNoiseParams.__init__(self, position,
#                                   rotation, velocity_dir,
#                                   collision_dir, **kwargs)


NO_NOISE = RigidNoiseParams()



class NoisyRigidbodiesDataset(RigidbodiesDataset, ABC):
    """
    A dataset for running Rigidbody (PhysX) physics with injected noise
    """

    def __init__(self,
                 noise: RigidNoiseParams = NO_NOISE,
                 indexes: List[str] = [],
                 **kwargs):
        RigidbodiesDataset.__init__(self, **kwargs)
        self._noise_params = noise
        # self.indexes = [int(index) for index in indexes]
        try:
            self.indexes = [int(index) for index in indexes]
        except:
            self.indexes = [int(index) for index in range(self.MAX_TRIALS)]

        # how to generate collision noise
        # self._ongoing_collisions = []
        # self._lasttime_collisions = []
        self.set_collision_noise_generator(noise)
        # if self.collision_noise_generator is not None:
        #     print("example noise", self.collision_noise_generator())
        self._registered_objects = []


    def get_tr(self, resp: List[bytes]) -> dict:
        # Parse the data in an ordered manner so that it can be mapped back to the object IDs.
        num_objects = len(Dataset.OBJECT_IDS)
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
        return tr_dict

    def _write_frame(self, frames_grp: h5py.Group, resp: List[bytes], frame_num: int, view_num: int):
        if self._noise_params == NO_NOISE:
            return RigidbodiesDataset._write_frame(self, frames_grp=frames_grp, resp=resp, frame_num=frame_num, view_num=view_num)
        else:
            frame = frames_grp.create_group(TDWUtils.zero_padding(frame_num, 4))
            tr = self.get_tr(resp=resp)
            sleeping = True

            for r in resp[:-1]:
                r_id = OutputData.get_data_type_id(r)
                if r_id == "rigi":
                    ri = Rigidbodies(r)
                    for i in range(ri.get_num()):
                        # Check if any objects are sleeping that aren't in the abyss.
                        if not ri.get_sleeping(i) and tr[ri.get_id(i)]["pos"][1] >= -1:
                            sleeping = False
            return frame, None, None, sleeping
        
    # def combine_dicts(self, a, b, op=operator.add):
    #     assert a.keys() == b.keys()
    #     return dict([(k, op(a[k], b[k])) for k in a.keys()])
    
    # def get_scene_initialization_commands(self) -> List[dict]:
    #     print("we are in bussiness!")
    #     if self.room == 'mm_kitchen_1b_4x5':
    #         add_scene = [self.get_add_scene(scene_name='mm_kitchen_1b_4x5')]
    #         add_scene.extend([{"$type": "set_floorplan_roof",
    #             "show": False}])
    #         self.scene_record = Controller.SCENE_LIBRARIANS["scenes.json"].get_record(self.room)
    #         base_scene_record = Controller.SCENE_LIBRARIANS["scenes.json"].get_record(self.room.split('_4x5')[0])
    #         self.base_room_center =  base_scene_record.rooms[0].main_region.center

    #     if self.hdri_skybox is not None:
    #         add_scene.extend([{"$type": "add_hdri_skybox",
    #                            "name": self.hdri_skybox,  # ninomaru_teien_4k
    #                            "url": "file:///" + self.path_hdri + self.hdri_skybox,
    #                            "exposure": 1,
    #                            "initial_skybox_rotation": self.skybox_rotation,
    #                            "sun_elevation": self.sun_elevation,
    #                            "sun_initial_angle": self.sun_angle,
    #                            "sun_intensity": self.sun_intensity,
    #                            "location": 'interior'}])

    #         add_scene.extend([{"$type": "set_contrast",
    #              "contrast": 30},
    #             # {"$type": "set_saturation",
    #             #  "saturation": -5},
    #             {"$type": "set_screen_space_reflections",
    #              "enabled": True}])

    #     # else:
    #     add_scene += [{"$type": "set_aperture",
    #      "aperture": 8.0},
    #     {"$type": "set_post_process", "value": False},
    #     # {"$type": "set_focus_distance",
    #     #  "focus_distance": 2.25},
    #     {"$type": "set_post_exposure",
    #      "post_exposure": 0.4},
    #     {"$type": "set_ambient_occlusion_intensity",
    #      "intensity": 0.175},
    #     {"$type": "set_ambient_occlusion_thickness_modifier",
    #      "thickness": 3.5}]
    #     return add_scene

    # def get_trial_initialization_commands(self) -> List[dict]:
    #     commands = []

    #     # randomization across trials
    #     if not(self.randomize):

    #         self.trial_seed = (self.MAX_TRIALS * self.seed) + self._trial_num
    #         random.seed(self.trial_seed)
    #     else:
    #         # breakpoint()
    #         self.trial_seed = -1 # not used

    #     for room in self.scene_record.rooms:
    #         center = self.combine_dicts(room.main_region.center, self.base_room_center, operator.sub)
    #         # Choose and place the target zone.
    #         commands.extend(self._place_target_zone(center))
    #         # Choose and place the target object.
    #         commands.extend(self._place_target_object(center))
    #         # Choose, place, and push a probe object.
    #         commands.extend(self._place_and_push_probe_object(center))
    #         # Build the intermediate structure that captures some aspect of "intuitive physics."
    #         commands.extend(self._build_intermediate_structure(center))
    #         # Place distractor objects in the background
    #         commands.extend(self._place_background_distractors(center))
    #         # Place occluder objects in the background
    #         commands.extend(self._place_occluders(center))

    #     # Set the probe color
    #     if self.probe_color is None:
    #         self.probe_color = self.target_color if (self.monochrome and self.match_probe_and_target_color) else None

    #     # Teleport the avatar to a reasonable position based on the drop height.
    #     a_pos = self.get_random_avatar_position(radius_min=self.camera_radius_range[0],
    #                                             radius_max=self.camera_radius_range[1],
    #                                             angle_min=self.camera_min_angle,
    #                                             angle_max=self.camera_max_angle,
    #                                             y_min=self.camera_min_height,
    #                                             y_max=self.camera_max_height,
    #                                             center=TDWUtils.VECTOR3_ZERO,
    #                                             reflections=self.camera_left_right_reflections)

    #     # Set the camera parameters
    #     self._set_avatar_attributes(a_pos)

    #     commands.extend([
    #         {"$type": "teleport_avatar_to",
    #          "position": {"x": 0, "y": 50, "z": 0}},
    #         {"$type": "look_at_position",
    #          "position": self.camera_aim},
    #         {"$type": "set_focus_distance",
    #          "focus_distance": TDWUtils.get_distance(a_pos, self.camera_aim)}
    #     ])

    #     # make things stable
    #     # commands.extend(self.settle())

    #     # test mode colors
    #     if self.use_test_mode_colors:
    #         self._set_test_mode_colors(commands)

    #     return commands

    def trial(self, filepath: Path, temp_path: Path, trial_num: int, unload_assets_every: int) -> None:
        # return Dataset.trial(self, filepath, temp_path, trial_num, unload_assets_every)
        if self._noise_params == NO_NOISE:
            return Dataset.trial(self, filepath, temp_path, trial_num, unload_assets_every)
        else:
            """
            Run a trial. Write static and per-frame data to disk until the trial is done.

            :param filepath: The path to this trial's hdf5 file.
            :param temp_path: The path to the temporary file.
            :param trial_num: The number of the current trial.
            """
            # Clear the object IDs and other static data
            self.clear_static_data()
            self._trial_num = trial_num

            # Create the .hdf5 file.
            f = h5py.File(str(temp_path.resolve()), "a")

            commands = []
            # # Remove asset bundles (to prevent a memory leak).
            if trial_num%unload_assets_every == 0:
                commands.append({"$type": "unload_asset_bundles"})

            # Add commands to start the trial.
            commands.extend(self.get_trial_initialization_commands())
            commands.extend(self._get_send_data_commands())

            # azimuth_grp = f.create_group("azimuth")
            # multi_camera_positions = self.generate_multi_camera_positions(azimuth_grp, self.view_id_number)

            # commands.extend(self.move_camera_commands(multi_camera_positions, []))
            resp = self.communicate(commands)
            # use this stupid command to replace generate_multi_camera_positions
            random.uniform(6, 7.5)

            frame = 0
            # Add the first frame.
            done = False
            frames_grp = f.create_group("frames")
            frame_grp, _, _, _ = self._write_frame(frames_grp=frames_grp, resp=resp, frame_num=frame, view_num=None)
            self._write_frame_labels(frame_grp, resp, -1, False)
            t = time.time()
            while (not done) and (frame < self.max_frames):
                frame += 1
                # print('frame %d' % frame)
                # t1 = time.time()
                cmds = self.get_per_frame_commands(resp, frame)
                # t2 = time.time()
                # print(t2 - t1, " getting cmds", " commands at this frame", cmds)
                # for cmd in cmds:
                #     print(cmd)
                #     print(cmd['position']['x'], type(cmd['position']['x']))
                #     print(cmd['position']['y'], type(cmd['position']['y']))
                #     print(cmd['position']['z'], type(cmd['position']['z']))
                #     json.dumps(cmd)
                resp = self.communicate(cmds)
                # t3 = time.time()
                # print(t3-t2, " communicating")
                frame_grp, _, _, done = self._write_frame(frames_grp=frames_grp, resp=resp, frame_num=frame, view_num=None)
                _, _, _, _ = self._write_frame_labels(frame_grp, resp, frame, done)
                # t4 = time.time()
                # print(t4-t3, " writing frames")
                # print(t4-t1, " in total for this frame")

            # Cleanup.
            commands = []
            for o_id in Dataset.OBJECT_IDS:
                commands.append({"$type": self._get_destroy_object_command_name(o_id),
                                "id": int(o_id)})
            self.communicate(commands)
            
            #  # Compute the trial-level metadata. Save it per trial in case of failure mid-trial loop
            # # if self.save_labels:
            # meta = OrderedDict()
            # meta = get_labels_from(f, label_funcs=self.get_controller_label_funcs(type(self).__name__), res=meta)
            # self.trial_metadata.append(meta)

            # # Save the trial-level metadata
            # json_str = json.dumps(self.trial_metadata, indent=4)
            # self.meta_file.write_text(json_str, encoding='utf-8')
            # print("TRIAL %d LABELS" % self._trial_num)
            # print(json.dumps(self.trial_metadata[-1], indent=4))

            # Close the file.
            f.close()
            # # Move the file.
            shutil.move(temp_path, filepath)
            print("avg time to communicate", time.time() - t)

    def trial_loop(self,
                   num: int,
                   output_dir: str,
                   temp_path: str,
                   save_frame: int = None,
                   unload_assets_every: int = 1000000000,
                   update_kwargs: List[dict] = {},
                   do_log: bool = False) -> None:
        if self._noise_params == NO_NOISE:
            return Dataset.trial_loop(self, num, output_dir, temp_path, save_frame, unload_assets_every, update_kwargs, do_log)
        else:
            if not isinstance(update_kwargs, list):
                update_kwargs = [update_kwargs] * len(self.indexes)

            pbar = tqdm(total=len(self.indexes))
            # Skip trials that aren't on the disk, and presumably have been uploaded; jump to the highest number.
            exists_up_to = -1
            for f in output_dir.glob("*.hdf5"):
                if int(f.stem) > exists_up_to:
                    exists_up_to = int(f.stem)

            exists_up_to += 1

            if exists_up_to > 0:
                print('Trials up to %d already exist, skipping those' % exists_up_to)

            pbar.update(exists_up_to)
            t = time.time()
            for i in range(exists_up_to, num):
                if i not in self.indexes:
                    continue
                filepath = output_dir.joinpath(TDWUtils.zero_padding(i, 4) + ".hdf5")
                self.stimulus_name = '_'.join([filepath.parent.name, str(Path(filepath.name).with_suffix(''))])
                # if True: #not filepath.exists():
                if do_log:
                    start = time.time()
                    logging.info("Starting trial << %d >> with kwargs %s" % (i, update_kwargs[i]))
                # Save out images
                self.png_dir = None
                if any([pa in PASSES for pa in self.save_passes]):
                    self.png_dir = output_dir.joinpath("pngs_" + TDWUtils.zero_padding(i, 4))
                    if not self.png_dir.exists() and self.save_movies:
                        self.png_dir.mkdir(parents=True)

                # Do the trial.
                self.trial(filepath,
                        temp_path,
                        i,
                        unload_assets_every)

                _ = subprocess.run('rm -rf ' + str(self.png_dir), shell=True)
                if do_log:
                    end = time.time()
                    logging.info("Finished trial << %d >> with trial seed = %d (elapsed time: %d seconds)" % (
                    i, self.trial_seed, int(end - start)))
                pbar.update(1)
            # print("avg time to communicate", (time.time() - t)/num)
            pbar.close()

    def _add_perturbation(self, resp, sim_seed):
        ri_dict = dict()
        r_ids = [OutputData.get_data_type_id(r) for r in resp[:-1]]
        for i, r_id in enumerate(r_ids):
            # print("i: ", i, r_id)
            if r_id == 'rigi':
                # print('rigi!')
                rigi = Rigidbodies(resp[i])
                for j in range(rigi.get_num()):
                    object_id = rigi.get_id(j)
                    # print("j: ", j, 'object_id: ', object_id)
                    velocity = rigi.get_velocity(j)
                    angular_velocity = rigi.get_angular_velocity(j)
                    if object_id in ri_dict.keys():
                        ri_dict[object_id].update({'velocity': velocity, 'angular_velocity': angular_velocity})
                    else:
                        ri_dict[object_id] = {'velocity': velocity, 'angular_velocity': angular_velocity}
            if r_id == 'srig':
                # print('srig!')
                srigi = StaticRigidbodies(resp[i])
                for j in range(srigi.get_num()):
                    object_id = srigi.get_id(j)
                    # print("j: ", j, 'object_id: ', object_id)
                    mass = srigi.get_mass(j)
                    dynamic_friction = srigi.get_dynamic_friction(j)
                    static_friction = srigi.get_static_friction(j)
                    bounciness = srigi.get_bounciness(j)
                    if object_id in ri_dict.keys():
                        ri_dict[object_id].update({'mass': mass, 'dynamic_friction': dynamic_friction, 'static_friction': static_friction, 'bounciness': bounciness})
                    else:
                        ri_dict[object_id] = {'mass': mass, 'dynamic_friction': dynamic_friction, 'static_friction': static_friction, 'bounciness': bounciness}
            if r_id == 'tran':
                # print('tran!')
                tran = Transforms(resp[i])
                for j in range(tran.get_num()):
                    object_id = tran.get_id(j)
                    # print("j: ", j, 'object_id: ', object_id)
                    position = tran.get_position(j)
                    position = dict([[k, position[idx]]
                                            for idx, k in enumerate(XYZ)])
                    rotation = tran.get_rotation(j)
                    if object_id in ri_dict.keys():
                        ri_dict[object_id].update({'position': position, 'rotation': QuaternionUtils.quaternion_to_euler_angles(rotation)})
                    else:
                        ri_dict[object_id] = {'position': position, 'rotation': QuaternionUtils.quaternion_to_euler_angles(rotation)}

        cmds = []
        for o_id, vals in ri_dict.items():
            rotation = dict([[k, vals['rotation'][idx]]
                                            for idx, k in enumerate(XYZ)])
            # print(o_id, vals['position'], rotation, vals['mass'])
            position, rotation, mass, dynamic_friction, static_friction, bounciness = self._random_placement(vals['position'], rotation, vals['mass'], vals['dynamic_friction'], vals['static_friction'], vals['bounciness'], o_id, sim_seed)
            # print(o_id, position, rotation, mass)
            cmds.extend([{"$type": "teleport_object",
                                "id": o_id,
                                "position": dict([[k, np.float64(position[k])]
                                            for k in XYZ])}])
            cmds.extend([{"$type": "rotate_object_to_euler_angles",
                                "id": o_id,
                                "euler_angles": dict([[k, np.float64(rotation[k])]
                                            for k in XYZ])}])
            cmds.extend([{"$type": "set_mass", "mass": np.float64(mass), "id": o_id}])
            cmds.extend([{"$type": "set_physic_material",
                                "dynamic_friction": np.float64(dynamic_friction), 
                                "static_friction": np.float64(static_friction), 
                                "bounciness": np.float64(bounciness), 
                                "id": o_id}])
        # print(cmds)
        return cmds
    
    def _random_placement(self,
                          position: Dict[str, float],
                          rotation: Dict[str, float],
                          mass: float,
                          dynamic_friction: float,
                          static_friction: float,
                          bounciness: float,
                          o_id: int,
                          sim_seed: int):
        # print("----------------------------------------------------------------------------------------------------------------------------")
        # print("sim_seed: ", self.sim_seed)
        # print("original o_id: ", o_id)
        # print("noisy_params: ", self._noise_params)
        # print("original positions: ", position)
        # print("original rotations: ", rotation)
        # print("original masses: ", mass)
        # print("original dynamic_frictions: ", dynamic_friction)
        # print("original static_frictions: ", static_friction)
        # print("original bouncinesses: ", bounciness)
        n = self._noise_params
        rotrad = dict([[k, deg2rad(rotation[k])]
                       for k in rotation.keys()])
        for k in XYZ:
            if n.position is not None and k in n.position.keys()\
                    and n.position[k] is not None and position is not None:
                position[k] = norm.rvs(position[k],
                                       n.position[k], random_state=sim_seed)
                self.sim_seed += 1
            # this is adding vonmises noise to the Euler angles
            if n.rotation is not None and k in n.rotation.keys()\
                    and n.rotation[k] is not None and rotation is not None:
                rotrad[k] = vonmises.rvs(n.rotation[k], rotrad[k], random_state=sim_seed)
                self.sim_seed += 1
        rotation = dict([[k, rad2deg(rotrad[k])]
                         for k in rotrad.keys()])
        if (n.mass is not None) and (mass is not None):
            mass = max(0, norm.rvs(loc=mass, scale=n.mass, random_state=sim_seed))
            self.sim_seed += 1
        
        # Clamp frictions to be > 0
        if (n.dynamic_friction is not None) and (dynamic_friction is not None):
            dynamic_friction = max(0, norm.rvs(loc=dynamic_friction, scale=n.dynamic_friction, random_state=sim_seed))
            self.sim_seed += 1
        if (n.static_friction is not None) and (static_friction is not None):
            static_friction = max(0, norm.rvs(loc=static_friction, scale=n.static_friction, random_state=sim_seed))
            self.sim_seed += 1
        # Clamp bounciness between 0 and 1
        if (n.bounciness is not None) and (bounciness is not None):
            bounciness = max(0, norm.rvs(loc=bounciness, scale=n.bounciness, random_state=sim_seed))
            self.sim_seed += 1
        # print("perturbed positions: ", position)
        # print("perturbed rotations: ", rotation)
        # print("perturbed masses: ", mass)
        # print("perturbed dynamic_frictions: ", dynamic_friction)
        # print("perturbed static_frictions: ", static_friction)
        # print("perturbed bouncinesses: ", bounciness)
        # print("----------------------------------------------------------------------------------------------------------------------------")
        self._registered_objects.append(o_id)
        return position, rotation, mass, dynamic_friction, static_friction, bounciness

    def add_transforms_object(self,
                              record: ModelRecord,
                              position: Dict[str, float],
                              rotation: Dict[str, float],
                              o_id: Optional[int] = None,
                              add_data: Optional[bool] = True,
                              library: str = "",
                              sim_seed: int = None) -> dict:
        """
        Overwrites method from rigidbodies_dataset to add noise to objects when added to the scene
        """
        if self._noise_params != NO_NOISE and self._noise_params.start_simulate == 0:
            position, rotation, _, _, _, _ = self._random_placement(position, rotation, None, None, None, None, o_id, sim_seed)
        return RigidbodiesDataset.add_transforms_object(self,
            record, position, rotation, o_id, add_data, library)    
    
    def add_primitive(self,
                      record: ModelRecord,
                      position: Dict[str, float] = TDWUtils.VECTOR3_ZERO,
                      rotation: Dict[str, float] = TDWUtils.VECTOR3_ZERO,
                      scale: Dict[str, float] = {"x": 1., "y": 1., "z": 1},
                      o_id: Optional[int] = None,
                      material: Optional[str] = None,
                      color: Optional[list] = None,
                      exclude_color: Optional[list] = None,
                      mass: Optional[float] = 2.0,
                      dynamic_friction: Optional[float] = 0.1,
                      static_friction: Optional[float] = 0.1,
                      bounciness: Optional[float] = 0,
                      add_data: Optional[bool] = True,
                      scale_mass: Optional[bool] = True,
                      make_kinematic: Optional[bool] = False,
                      obj_list: Optional[list] = [],
                      apply_texture: Optional[bool] = True,
                      default_physics_values: Optional[bool] = True,
                      density:  Optional[float] = 5,
                      sim_seed: int = None
                      ) -> List[dict]:
        """
        Overwrites method from rigidbodies_dataset to add noise to objects when added to the scene
        """
        if self._noise_params != NO_NOISE and self._noise_params.start_simulate == 0:
            position, rotation, mass, dynamic_friction, static_friction, bounciness = self._random_placement(position, rotation, mass, dynamic_friction, static_friction, bounciness, o_id, sim_seed)
        return RigidbodiesDataset.add_primitive(self,
            record, position, rotation, scale, o_id, material, color, exclude_color, mass,
            dynamic_friction, static_friction,
            bounciness, add_data, scale_mass, make_kinematic, obj_list, apply_texture,
            default_physics_values, density)        

    def add_physics_object(self,
                           record: ModelRecord,
                           position: Dict[str, float],
                           rotation: Dict[str, float],
                           mass: float,
                           scale: Dict[str, float],
                           dynamic_friction: float,
                           static_friction: float,
                           bounciness: float,
                           o_id: Optional[int] = None,
                           add_data: Optional[bool] = True,
                           default_physics_values = True,
                           density = 5,
                           sim_seed = None
                           ) -> List[dict]:
        """
        Overwrites method from rigidbodies_dataset to add noise to objects when added to the scene
        """
        if self._noise_params != NO_NOISE and self._noise_params.start_simulate == 0:
            position, rotation, mass, dynamic_friction, static_friction, bounciness = self._random_placement(position, rotation, mass, dynamic_friction, static_friction, bounciness, o_id, sim_seed)
        return RigidbodiesDataset.add_physics_object(self,
            record, position, rotation, mass,
            scale, dynamic_friction, static_friction,
            bounciness, o_id, add_data,
            default_physics_values, density)        

    def get_per_frame_commands(self, resp: List[bytes], frame: int, sim_seed: int) -> List[dict]:
        """
        Overwrites abstract method to add collision noise commands

        Inhereting classes should *always* extend this output

        self._ongoing_collisions = []
        self._lasttime_collisions = []
        """
        # print('frame: ', frame)
        cmds = []
        if self.collision_noise_generator is not None and frame >= self._noise_params.start_simulate:
            if frame == self._noise_params.start_simulate:
                perturbation_cmds = self._add_perturbation(resp, sim_seed)
                cmds.extend(perturbation_cmds)
            coll_data = self._get_collision_data(resp)
            if len(coll_data) > 0:
                # """ some filtering and smoothing can be done below, e.g remove target zone collision
                for cd in coll_data:
                    # if '1' not in nm_ap:
                    # print("----------------------------------------------------------------------------------------------------------------------------")
                    coll_noise_cmds = self.apply_collision_noise(sim_seed, cd)
                    cmds.extend(coll_noise_cmds)
                    self.sim_seed += 1

                # new_collisions = []
                # for cd in coll_data:
                #     nm_ap = str(cd['agent_id']) + '_' + str(cd['patient_id'])
                #     if '3' in nm_ap:
                #         # Ensure consistency in naming / comparisons
                #         aid = cd['agent_id']
                #         pid = cd['patient_id']
                #         # nm_ap = str(aid) + '_' + str(pid)

                #         for r in resp[:-1]:
                #             r_id = OutputData.get_data_type_id(r)
                #             if r_id == "rigi":
                #                 ri = Rigidbodies(r)
                #                 for i in range(ri.get_num()):
                #                     if ri.get_id(i) == aid:
                #                         va = ri.get_velocity(i)
                #                     elif ri.get_id(i) == pid:
                #                         vp = ri.get_velocity(i)
                #         vel_diff = np.subtract(va, vp)
                #         if np.any(cd['impulse']):
                #             print('agent: ', aid)
                #             print('patient: ', pid)
                #             print('corr impulse delvel: ', np.dot(cd['impulse'], vel_diff)/(np.linalg.norm(cd['impulse'])*np.linalg.norm(vel_diff)))
                    # if cd['state'] == 'enter':
                    #     print('start rvel: ' + nm_ap + ' : '
                    #           + str(va) + '; ' + str(vp))
                    #     self._ongoing_collisions.append(nm_ap)
                    #     new_collisions.append(nm_ap)
                    # else:
                    #     if nm_ap in self._lasttime_collisions:
                    #         print('next rvel: ' + nm_ap + ' : '
                    #               + str(va) + '; ' + str(vp))

                    # if cd['state'] == 'exit':
                    #     self._ongoing_collisions = [
                    #         c for c in self._ongoing_collisions if
                    #         c != nm_ap
                    #     ]
                # self._lasttime_collisions = new_collisions
                # """
                # coll_noise_cmds = self.apply_collision_noise(coll_data)
                # cmds.extend(coll_noise_cmds)

        return cmds

    def apply_collision_noise(self, sim_seed, data=None):
        # nm_ap = str(data['agent_id']) + '_' + str(data['patient_id'])
        # print('collision objects: ', nm_ap)
        if data is None or np.linalg.norm(data['impulse']) < self._noise_params.coll_threshold:
            # print("NOT PERTURBED")
            return []
        # print("YES PERTURBED")
        # print("num_contacts: ", data['num_contacts'])
        p_id = data['patient_id']
        a_id = data['agent_id']
        # print('contact points: ', data['contact_points'])
        contact_points = [dict([[k, pt[idx]] for idx, k in enumerate(XYZ)]) for pt in data['contact_points']]
        # print('contact points formatted: ', contact_points)
        impulse = dict([[k, data['impulse'][idx]] for idx, k in enumerate(XYZ)])
        # print("original impulse: ", impulse)
        # print("norm original impulse: ", np.linalg.norm(list(impulse.values())))
        force = self.collision_noise_generator(sim_seed, data['impulse'])
        delta_force = self._calculate_collision_differential(impulse, force)
        # print("perturbed impulse: ", force)
        # print("norm perturbed impulse: ", np.linalg.norm(list(force.values())))
        # print("perturbed impulse delta: ", delta_force)
        force_avg_p = dict([[k, delta_force[k]/data['num_contacts']] for k in XYZ])
        force_avg_a = dict([[k, -delta_force[k]/data['num_contacts']] for k in XYZ])
        # if data['num_contacts'] != 0:
        #     force_avg_p = dict([[k, delta_force[k]/data['num_contacts']] for k in XYZ])
        #     force_avg_a = dict([[k, -delta_force[k]/data['num_contacts']] for k in XYZ])
        # else:
        #     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #     force_avg_p = delta_force
        #     force_avg_a = dict([[k, -delta_force[k]] for k in XYZ])
        # print("perturbed force_avg_p: ", force_avg_p)
        # print("perturbed force_avg_a: ", force_avg_a)        
        # see here: https://github.com/threedworld-mit/tdw/blob/ce177b9754e4fa7bc7094c59937bb12c01f978aa/Documentation/api/command_api.md#apply_force_at_position
        # why there are multiple contact normals: https://gamedev.stackexchange.com/questions/40048/why-doesnt-unitys-oncollisionenter-give-me-surface-normals-and-whats-the-mos
        cmds_p = [
            {
                "$type": "apply_force_at_position",
                "force": force_avg_p,
                "position": contact_point,
                "id": p_id
            } for contact_point in contact_points
        ]
        cmds_a = [
            {
                "$type": "apply_force_at_position",
                "force": force_avg_a,
                "position": contact_point,
                "id": a_id
            } for contact_point in contact_points
        ]
        cmds = cmds_p+cmds_a
        # print('cmd: ', cmds)
        # print("----------------------------------------------------------------------------------------------------------------------------")
        return cmds

    # """ INCOMPLETE - Adds force but not relative """

    """
    Helper function that takes in a collision momentum transfer vector, then calculates the momentum to apply
    in the next timestep to actualize the collision noise
    """
    def _calculate_collision_differential(self, original_momentum: Dict[str, float], perturbed_momentum: Dict[str, float]) -> Dict[str, float]:
        return dict([[k, perturbed_momentum[k]-original_momentum[k]] for k in XYZ])

    def set_collision_noise_generator(self,
                                      noise_obj: RigidNoiseParams):
        # Only make noise if there is noise to be added
        # if noise_obj.collision_dir is not None:
        #     ncd = copy.copy(noise_obj.collision_dir)
        # else:
        #     ncd = None
        ncd = noise_obj.collision_dir
        ncm = noise_obj.collision_mag
        if ncd is None and ncm is None:
            self.collision_noise_generator = None
        else:
            """ NOTE MAKE THIS ALL RELATIVE | ADD INPUT """
            if ncm is None:
                def cng(sim_seed, impulse):
                    impulse_rand_dir = rand_von_mises_fisher(sim_seed, impulse/np.linalg.norm(impulse),kappa=ncd)[0]
                    return rotmag2vec(dict([[k, impulse_rand_dir[idx]]
                                            for idx, k in enumerate(XYZ)]),
                                      np.linalg.norm(impulse))
            elif ncd is None:
                def cng(sim_seed, impulse):
                    impulse_mag = np.linalg.norm(impulse)
                    impulse_dir = impulse/impulse_mag
                    impulse_mag = max(0, norm.rvs(loc=impulse_mag, scale=ncm, random_state=sim_seed))
                    return rotmag2vec(dict([[k, impulse_dir[idx]] for idx, k in enumerate(XYZ)]),
                                      impulse_mag)
            else:
                def cng(sim_seed, impulse):
                    impulse_mag = np.linalg.norm(impulse)
                    impulse_dir = impulse/impulse_mag
                    impulse_mag = max(0, norm.rvs(loc=impulse_mag, scale=ncm, random_state=sim_seed))
                    impulse_rand_dir = rand_von_mises_fisher(sim_seed, impulse_dir, kappa=ncd)[0]
                    return rotmag2vec(dict([[k, impulse_rand_dir[idx]]
                                            for idx, k in enumerate(XYZ)]), impulse_mag)
            self.collision_noise_generator = cng

    # def settle(self):
    #     """
    #     After adding a set of objects in a noisy way,
    #     'settles' the world to avoid interpenetration
    #     check out this: https://github.com/threedworld-mit/tdw/blob/ce177b9754e4fa7bc7094c59937bb12c01f978aa/Documentation/lessons/semantic_states/overlap.md
    #     """
    #     # print("applying settle function!")
    #     cmds = []
    #     if (self._noise_params.position is not None) or (self._noise_params.rotation is not None):
    #         # disable gravity for all added objects
    #         for o_id in self._registered_objects:
    #             # print("disabling gravity for object: ", o_id)
    #             cmds.extend([{"$type": "set_kinematic_state",
    #                             "id": o_id,
    #                             "use_gravity": False}])
            
    #         # slowly resolve possible collisions
    #         cmds.extend([{"$type": "set_time_step",
    #                             "time_step": 0.0001},
    #                     {"$type": "step_physics",
    #                             "frames": 500}])

    #         # enable gravity again
    #         for o_id in self._registered_objects:
    #             # print("enabling object: ", o_id)
    #             cmds.extend([{"$type": "set_kinematic_state",
    #                             "id": o_id,
    #                             "use_gravity": True}])

    #         # set physics speed to normal
    #         cmds.extend([{"$type": "set_time_step",
    #                                 "time_step": 0.01}])
    #         self._registered_objects = []
    #         # print("finished applying settle function!")

    #     return cmds

    def _get_collision_data(self, resp: List[bytes]):
        coll_data = []
        r_ids = [OutputData.get_data_type_id(r) for r in resp[:-1]]
        for i, r_id in enumerate(r_ids):
            if r_id == 'coll':
                co = Collision(resp[i])
                state = co.get_state()
                agent_id = co.get_collider_id()
                patient_id = co.get_collidee_id()
                relative_velocity = co.get_relative_velocity()
                impulse = co.get_impulse()
                num_contacts = co.get_num_contacts()
                contact_points = [co.get_contact_point(i)
                                  for i in range(num_contacts)]
                contact_normals = [co.get_contact_normal(i)
                                   for i in range(num_contacts)]

                coll_data.append({
                    'agent_id': agent_id,
                    'patient_id': patient_id,
                    'relative_velocity': relative_velocity,
                    'impulse': impulse,
                    'num_contacts': num_contacts,
                    'contact_points': contact_points,
                    'contact_normals': contact_normals,
                    'state': state
                })
                if False:
                    print("agent: %d ---> patient %d" % (agent_id, patient_id))
                    print("relative velocity", relative_velocity)
                    print("impulse", impulse)
                    print("contact points", contact_points)
                    print("contact normals", contact_normals)
                    print("state", state)
            # collision between object and environment, not considered yet
            # elif  r_id == 'enco':
            #     co = EnvironmentCollision(resp[i])
            #     # Not implemented yet
                # pass
        return coll_data
