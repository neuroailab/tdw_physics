import sys, os, copy, subprocess, glob, logging, time
import platform
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
import stopit
from PIL import Image
import io
import h5py, json
from collections import OrderedDict
import numpy as np
import random
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.output_data import OutputData, SegmentationColors, Meshes, Images
from tdw.librarian import ModelRecord, MaterialLibrarian

from tdw_physics.postprocessing.stimuli import pngs_to_mp4
from tdw_physics.postprocessing.labels import (get_labels_from,
                                               get_all_label_funcs,
                                               get_across_trial_stats_from)
from tdw_physics.util_geom import save_obj
import shutil
import math

import signal
# from tdw.add_ons.interior_scene_lighting import InteriorSceneLighting


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

PASSES = ["_img", "_depth", "_normals", "_flow", "_id", "_category", "_albedo"]
M = MaterialLibrarian()
MATERIAL_TYPES = M.get_material_types()
MATERIAL_NAMES = {mtype: [m.name for m in M.get_all_materials_of_type(mtype)] \
                  for mtype in MATERIAL_TYPES}
MATERIAL_NAMES['selected'] = ['parquet_wood_mahogany', 'ceramic_tiles_floral_white', 'parquet_european_ash_grey',
                               'ceramic_tiles_pale_goldenrod', 'terracotta_simple']
MATERIAL_TYPES = MATERIAL_NAMES.keys()
# colors for the target/zone overlay
ZONE_COLOR = [255,255,0]
TARGET_COLOR = [255,0,0]

class Dataset(Controller, ABC):
    """
    Abstract class for a physics dataset.

    1. Create a dataset .hdf5 file.
    2. Send commands to initialize the scene.
    3. Run a series of trials. Per trial, do the following:
        1. Get commands to initialize the trial. Write "static" data (which doesn't change between trials).
        2. Run the trial until it is "done" (defined by output from the writer). Write per-frame data to disk,.
        3. Clean up the scene and start a new trial.
    """
    def __init__(self,
                 port: int = 1071,
                 check_version: bool=False,
                 launch_build: bool=True,
                 randomize: int=0,
                 seed: int=0,
                 save_args=True,
                 num_views=1,
                 start=0,
                 scale_factor_dict=None,
                 **kwargs
    ):
        # save the command-line args
        self.save_args = save_args
        self._trial_num = None

        if platform.system() == 'Linux':
            os.environ["DISPLAY"] = ":5"


            # if args.gpu is not None:
            #     os.environ["DISPLAY"] = ":" + str(args.gpu + 1)
            # else:
            #     os.environ["DISPLAY"] = ":0"

        super().__init__(port=port,
                         check_version=check_version,
                         launch_build=launch_build)

        # set random state
        self.randomize = randomize
        self.seed = seed
        self.num_views = num_views
        self.start = start

        if not bool(self.randomize):
            random.seed(self.seed)
            print("SET RANDOM SEED: %d" % self.seed)
            print("NUMBER OF VIEWS: %d" % self.num_views)
            print('STARTING TRIAL NUMBER: %d' % self.start)


        # fluid actors need to be handled separately
        self.fluid_object_ids = []
        self.scale_factor_dict = scale_factor_dict

        self.use_interior_scene_lighting = False

        if self.use_interior_scene_lighting:
            self.interior_scene_lighting = InteriorSceneLighting()
            self.add_ons.append(self.interior_scene_lighting)


    '''
    def communicate(self, commands) -> list:
        #Save a log of the commands so that they can be rerun
        with open(str(self.command_log), "at") as f:
            f.write(json.dumps(commands) + (" trial %s" % self._trial_num) + "\n")
        return super().communicate(commands)
    '''
    def clear_static_data(self) -> None:
        self.object_ids = np.empty(dtype=int, shape=0)
        self.object_scale_factors = []
        self.object_names = []
        self.model_names = []
        self._initialize_object_counter()

    @staticmethod
    def get_controller_label_funcs(classname = 'Dataset'):
        """
        A list of funcs with signature func(f: h5py.File) -> JSON-serializeable data
        """
        def stimulus_name(f):
            try:
                stim_name = str(np.array(f['static']['stimulus_name'], dtype=str))
            except TypeError:
                # happens if we have an empty stimulus name
                stim_name = "None"
            return stim_name
        def controller_name(f):
            return classname
        def git_commit(f):
            try:
                return str(np.array(f['static']['git_commit'], dtype=str))
            except TypeError:
                # happens when no git commit
                return "None"

        return [stimulus_name, controller_name, git_commit]

    def save_command_line_args(self, output_dir: str) -> None:
        if not self.save_args:
            return

        # save all the args, including defaults
        self._save_all_args(output_dir)

        # save just the commandline args
        output_dir = Path(output_dir)
        filepath = output_dir.joinpath("commandline_args.txt")
        if not filepath.exists():
            with open(filepath, 'w') as f:
                f.write('\n'.join(sys.argv[1:]))

        return

    def _save_all_args(self, output_dir: str) -> None:
        writelist = []
        for k,v in self.args_dict.items():
            writelist.extend(["--"+str(k),str(v)])

        self._script_args = writelist

        output_dir = Path(output_dir)
        filepath = output_dir.joinpath("args.txt")
        if not filepath.exists():
            with open(filepath, 'w') as f:
                f.write('\n'.join(writelist))
        return

    def get_initialization_commands(self,
                                    width: int,
                                    height: int) -> None:
        # Global commands for all physics datasets.
        commands = [{"$type": "set_screen_size",
                     "width": width,
                     "height": height},
                    {"$type": "set_render_quality",
                     "render_quality": 5},
                    {"$type": "set_physics_solver_iterations",
                     "iterations": 32},
                    {"$type": "set_vignette",
                     "enabled": False},
                    {"$type": "set_shadow_strength",
                     "strength": 1.0},
                    {"$type": "set_sleep_threshold",
                     "sleep_threshold": 0.01}]

        commands.extend(self.get_scene_initialization_commands())
        # Add the avatar.

        commands.extend([
             {"$type": "create_avatar", "type": "A_Img_Caps_Kinematic"},
             {"$type": "set_target_framerate", "framerate": self._framerate},
             {"$type": "set_pass_masks", "pass_masks": self.write_passes},
             {"$type": "set_field_of_view", "field_of_view": self.get_field_of_view()},
             {"$type": "send_images", "frequency": "always"},
             {"$type": "set_anti_aliasing", "mode": "subpixel"}
        ])
        print('FIELD OF VIEW: ', self.get_field_of_view())
        return commands

    def run(self,
            num: int,
            output_dir: str,
            temp_path: str,
            width: int,
            height: int,
            framerate: int = 30,
            write_passes: List[str] = PASSES,
            save_passes: List[str] = [],
            save_movies: bool = False,
            save_labels: bool = False,
            save_meshes: bool = False,
            terminate: bool = True,
            args_dict: dict={}) -> None:
        """
        Create the dataset.

        :param num: The number of trials in the dataset.
        :param output_dir: The root output directory.
        :param temp_path: Temporary path to a file being written.
        :param width: Screen width in pixels.
        :param height: Screen height in pixels.
        :param save_passes: a list of which passes to save out as PNGs (or convert to MP4)
        :param save_movies: whether to save out a movie of each trial
        :param save_labels: whether to save out JSON labels for the full trial set.
        """

        # If no temp_path given, place in local folder to prevent conflicts with other builds
        if temp_path == "NONE": temp_path = output_dir + "/temp.hdf5"

        self._height, self._width, self._framerate = height, width, framerate
        print("height: %d, width: %d, fps: %d" % (self._height, self._width, self._framerate))

        # the dir where files and metadata will go
        if not Path(output_dir).exists():
            Path(output_dir).mkdir(parents=True)

        # save a log of the commands send to TDW build
        self.command_log = Path(output_dir).joinpath('tdw_commands.json')

        # which passes to write to the HDF5
        self.write_passes = write_passes
        if isinstance(self.write_passes, str):
            self.write_passes = self.write_passes.split(',')
        self.write_passes = [p for p in self.write_passes if (p in PASSES)]

        # which passes to save as an MP4
        self.save_passes = save_passes
        if isinstance(self.save_passes, str):
            self.save_passes = self.save_passes.split(',')
        self.save_passes = [p for p in self.save_passes if (p in self.write_passes)]
        self.save_movies = save_movies

        # whether to send and save meshes
        self.save_meshes = save_meshes

        print("write passes", self.write_passes)
        print("save passes", self.save_passes)
        print("save movies", self.save_movies)
        print("save meshes", self.save_meshes)

        if self.save_movies:
            assert len(self.save_passes),\
                "You need to pass \'--save_passes [PASSES]\' to save out movies, where [PASSES] is a comma-separated list of items from %s" % PASSES

        # whether to save a JSON of trial-level labels
        self.save_labels = save_labels
        if self.save_labels:
            self.meta_file = Path(output_dir).joinpath('metadata.json')
            if self.meta_file.exists():
                self.trial_metadata = json.loads(self.meta_file.read_text())
            else:
                self.trial_metadata = []

        initialization_commands = self.get_initialization_commands(width=width, height=height)

        # Initialize the scene.
        self.communicate(initialization_commands)

        # Run trials
        self.trial_loop(num, output_dir, temp_path)

        # Terminate TDW
        # Windows doesn't know signal timeout
        if terminate:
            if platform.system() == 'Windows': end = self.communicate({"$type": "terminate"})
            else: #Unix systems can use signal to timeout
                with stopit.SignalTimeout(5) as to_ctx_mgr: #since TDW sometimes doesn't acknowledge being stopped we only *try* to close it
                    assert to_ctx_mgr.state == to_ctx_mgr.EXECUTING
                    end = self.communicate({"$type": "terminate"})
                if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
                    print("tdw closed successfully")
                elif to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
                    print("tdw failed to acknowledge being closed. tdw window might need to be manually closed")

        # Save the command line args
        if self.save_args:
            self.args_dict = copy.deepcopy(args_dict)
        self.save_command_line_args(output_dir)

        # Save the across-trial stats
        if self.save_labels:
            hdf5_paths = glob.glob(str(output_dir) + '/*.hdf5')
            stats = get_across_trial_stats_from(
                hdf5_paths, funcs=self.get_controller_label_funcs(classname=type(self).__name__))
            stats["num_trials"] = int(len(hdf5_paths))
            stats_str = json.dumps(stats, indent=4)
            stats_file = Path(output_dir).joinpath('trial_stats.json')
            stats_file.write_text(stats_str, encoding='utf-8')
            print("ACROSS TRIAL STATS")
            print(stats_str)


    def update_controller_state(self, **kwargs):
        """
        Change the state of the controller based on a set of kwargs.
        """
        return

    def trial_loop(self,
                   num: int,
                   output_dir: str,
                   temp_path: str,
                   save_frame: int=None,
                   unload_assets_every: int = 10,
                   update_kwargs: List[dict] = {},
                   do_log: bool = False) -> None:

        if not isinstance(update_kwargs, list):
            update_kwargs = [update_kwargs] * num

        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        temp_path = Path(temp_path)
        if not temp_path.parent.exists():
            temp_path.parent.mkdir(parents=True)
        # Remove an incomplete temp path.
        if temp_path.exists():
            temp_path.unlink()

        pbar = tqdm(total=num)
        # Skip trials that aren't on the disk, and presumably have been uploaded; jump to the highest number.
        exists_up_to = 0
        for f in output_dir.glob("*.hdf5"):
            if int(f.stem.replace('sc', '')) > exists_up_to:
                exists_up_to = int(f.stem.replace('sc', ''))

        if exists_up_to > 0:
            print('Trials up to %d already exist, skipping those' % exists_up_to)

        pbar.update(exists_up_to)

        if self.start > 0:
            exists_up_to = self.start
            num = self.start + num

        for i in range(exists_up_to, num):
            trial_num = i
            filepath = output_dir.joinpath('sc'+TDWUtils.zero_padding(trial_num, 4) + ".hdf5")
            self.stimulus_name = '_'.join([filepath.parent.name, str(Path(filepath.name).with_suffix(''))])

            ## update the controller state
            self.update_controller_state(**update_kwargs[i-self.start])

            if not filepath.exists():
                if do_log:
                    start = time.time()
                    logging.info("Starting trial << %d >> with kwargs %s" % (trial_num, update_kwargs[i-self.start]))

                # Save out images
                self.png_dir = None
                self.output_dir = output_dir

                # Do the trial.
                with timeout(seconds=320):


                    self.trial(filepath=filepath,
                               temp_path=temp_path,
                               trial_num=trial_num,
                               unload_assets_every=unload_assets_every)

                # # Save an MP4 of the stimulus
                # if self.save_movies:
                #     for pass_mask in self.save_passes:
                #         mp4_filename = str(filepath).split('.hdf5')[0] + pass_mask
                #         cmd, stdout, stderr = pngs_to_mp4(
                #             filename=mp4_filename,
                #             image_stem=pass_mask[1:]+'_',
                #             png_dir=self.png_dir,
                #             size=[self._height, self._width],
                #             overwrite=True,
                #             remove_pngs=(True if save_frame is None else False),
                #             use_parent_dir=False)
                #
                #     if save_frame is not None:
                #         frames = os.listdir(str(self.png_dir))
                #         sv = sorted(frames)[save_frame]
                #         png = output_dir.joinpath(TDWUtils.zero_padding(trial_num, 4) + ".png")
                #         _ = subprocess.run('mv ' + str(self.png_dir) + '/' + sv + ' ' + str(png), shell=True)
                #
                #     rm = subprocess.run('rm -rf ' + str(self.png_dir), shell=True)

                # if self.save_meshes:
                #     for o_id in self.object_ids:
                #         obj_filename = str(filepath).split('.hdf5')[0] + f"_obj{o_id}.obj"
                #         vertices, faces = self.object_meshes[o_id]
                #         save_obj(vertices, faces, obj_filename)
                #         save_obj(vertices, faces, os.path.join('./save_obj', f'{self.object_names[o_id-1]}.obj'))

                if do_log:
                    end = time.time()
                    logging.info("Finished trial << %d >> with trial seed = %d (elapsed time: %d seconds)" % (trial_num, self.trial_seed, int(end-start)))
            pbar.update(1)
        pbar.close()

    def trial(self,
              filepath: Path,
              temp_path: Path,
              trial_num: int,
              unload_assets_every: int=10) -> None:
        """
        Run a trial. Write static and per-frame data to disk until the trial is done.

        :param filepath: The path to this trial's hdf5 file.
        :param temp_path: The path to the temporary file.
        :param trial_num: The number of the current trial.
        """

        # Clear the object IDs and other static data
        self.clear_static_data()
        print('\tObject ids after clear static state: ', self.object_ids)
        self._trial_num = trial_num

        # Create the .hdf5 file.
        f = h5py.File(str(temp_path.resolve()), "a")

        commands = []
        # Remove asset bundles (to prevent a memory leak).
        if trial_num % unload_assets_every == 0:
            commands.append({"$type": "unload_asset_bundles"})

        # Add commands to start the trial.
        # if args.room == 'random_kitchen':
        #     commands.extend(self.get_scene_initialization_commands())
        commands.extend(self.get_trial_initialization_commands())
        # Add commands to request output data.
        commands.extend(self._get_send_data_commands())

        # Send the commands and start the trial.
        r_types = ['']
        count = 0
        resp = self.communicate(commands)

        print('\tObject ids after sending commands: ', self.object_ids)

        self._set_segmentation_colors(resp)

        self._get_object_meshes(resp)
        frame = 0
        # Write static data to disk.
        static_group = f.create_group("static")
        self._write_static_data(static_group)

        # Add the first frame.
        done = False
        frames_grp = f.create_group("frames")
        _, _, _, _, _ = self._write_frame(frames_grp=frames_grp, resp=resp, frame_num=frame)
        print('Warning not writing frame label')
        print('\n' * 5)
        # self._write_frame_labels(frames_grp, resp, -1, False)

        print('\tObject ids before looping: ', self.object_ids)

        # Continue the trial. Send commands, and parse output data.
        a_pos_dict = {}
        while not done:
            frame += 1
            print('frame %d' % frame)
            self.communicate([{"$type": "simulate_physics", "value": True}])

            resp = self.communicate(self.get_per_frame_commands(resp, frame))
            r_ids = [OutputData.get_data_type_id(r) for r in resp[:-1]]

            # Sometimes the build freezes and has to reopen the socket.
            # This prevents such errors from throwing off the frame numbering
            # if ('imag' not in r_ids) or ('tran' not in r_ids):
            #     print("retrying frame %d, response only had %s" % (frame, r_ids))
            #     frame -= 1
            #     continue
            start_frame = 5
            end_frame = 6

            if frame in range(start_frame, end_frame+1):
                camera_matrix_dict = {}
                azimuth_rotation = True

                if azimuth_rotation:
                    delta_az_list = [x / self.num_views * np.pi * 2 for x in range(self.num_views)]
                    az_range = 2 * np.pi / self.num_views

                    # origin_pos = {'x': 0.0, 'y': 2.8, 'z': 2.8}
                    # origin_pos = {'x': 0.0, 'y': 2.0, 'z': 3.0}
                    origin_pos =  {'x': -10.48 + 12.873, 'y': 1.81 - 1.85, 'z': - 6.583 + 5.75}

                    az, el, r = self.cart2sph(x=origin_pos['x'], y=origin_pos['z'], z=origin_pos['y']) # Note: Y-up to Z-up
                    # self.camera_aim =  {'x': -12.873 + 13.0, 'y': 1.85 - 1.0, 'z': - 5.75 + 5.0}
                    # self.camera_aim = {'x': -0.873, 'y': 0.85, 'z': -1.75}
                    self.camera_aim = {'x': 0, 'y': 0, 'z': 0}
                    self.camera_aim = self.add_room_center(self.camera_aim)
                else:
                    delta_angle = 2 * np.pi / self.num_views

                for view_id in range(self.num_views):

                    commands = []
                    if frame == start_frame:
                        if azimuth_rotation:

                            a_pos, az_ = self.get_rotating_camera_position_azimuth(az, el, r, delta_az_list[view_id], az_range)
                            a_pos = self.add_room_center(a_pos)
                            print('Camera position: ', a_pos)

                            # save azimuth
                            az_ori = az_ + math.pi  # since cam faces world origin, its orientation azimuth differs by pi
                            az_rot_mat_2d = np.array([[math.cos(az_ori), - math.sin(az_ori)],
                                                      [math.sin(az_ori), math.cos(az_ori)]])
                            az_rot_mat = np.eye(3)
                            az_rot_mat[:2, :2] = az_rot_mat_2d

                            transformation_save_name = os.path.join(self.output_dir, 'sc%s_frame%d_img%s_azi_rot.txt' % (format(trial_num, '04d'), frame, view_id))
                            # print('Save azi rot to ', transformation_save_name)
                            np.savetxt(transformation_save_name, az_rot_mat, fmt='%.5f')

                        else:

                            noise = (random.random() - 0.5) * delta_angle
                            a_pos = self.get_rotating_camera_position(center=TDWUtils.VECTOR3_ZERO,
                                                                      radius=self.camera_radius_range[1] * 1.5,
                                                                      angle= delta_angle * view_id + noise,
                                                                      height=self.camera_max_height * 1.5)
                        a_pos_dict[view_id] = a_pos
                    else:
                        a_pos = a_pos_dict[view_id]
                    # Set the camera parameters
                    self._set_avatar_attributes(a_pos)

                    commands.extend([
                        {"$type": "simulate_physics", "value": False},
                        {"$type": "teleport_avatar_to", "position": a_pos},
                        {"$type": "look_at_position", "position": self.camera_aim},
                        # {"$type": "set_focus_distance", "focus_distance": TDWUtils.get_distance(a_pos, self.camera_aim)},

                        # {"$type": "set_camera_clipping_planes", "near": 2., "far": 12}
                    ])

                    resp = self.communicate(commands)

                    _, objs_grp, tr_dict, done, camera_matrix = self._write_frame(
                        frames_grp=frames_grp, resp=resp, frame_num=frame, zone_id=self.zone_id, view_id=view_id, trial_num=trial_num)

                    camera_matrix_dict[f'view_{view_id}'] = camera_matrix.tolist()


                # with open('./tmp/camera_matrix.json', 'w') as fp:
                #     json.dump(camera_matrix_dict, fp, sort_keys=True, indent=4)

                # Write whether this frame completed the trial and any other trial-level data
                # labels_grp, _, _, done = self._write_frame_labels(frame, resp, frame, done)

                print('\tObject ids after end of frame %d: ' % frame, self.object_ids)

                if frame == end_frame:
                    break

        # Cleanup.
        commands = []
        print('Object ids before destroy: ', self.object_ids)
        for o_id in self.object_ids:
            commands.append({"$type": self._get_destroy_object_command_name(o_id),
                             "id": int(o_id)})
        for cmd in commands:
            print('Destroy: ', cmd)
            self.communicate(cmd)

        # Compute the trial-level metadata. Save it per trial in case of failure mid-trial loop
        if self.save_labels:
            meta = OrderedDict()
            meta = get_labels_from(f, label_funcs=self.get_controller_label_funcs(type(self).__name__), res=meta)
            self.trial_metadata.append(meta)

            # Save the trial-level metadata
            json_str =json.dumps(self.trial_metadata, indent=4)
            self.meta_file.write_text(json_str, encoding='utf-8')
            print("TRIAL %d LABELS" % self._trial_num)
            print(json.dumps(self.trial_metadata[-1], indent=4))


        '''
        
        # Save out the target/zone segmentation mask
        if (self.zone_id in self.object_ids) and (self.target_id in self.object_ids):

            _id = f['frames']['0000']['images']['_id']
            #get PIL image
            _id_map = np.array(Image.open(io.BytesIO(np.array(_id))))
            #get colors
            zone_idx = [i for i,o_id in enumerate(self.object_ids) if o_id == self.zone_id]
            zone_color = self.object_segmentation_colors[zone_idx[0] if len(zone_idx) else 0]
            target_idx = [i for i,o_id in enumerate(self.object_ids) if o_id == self.target_id]
            target_color = self.object_segmentation_colors[target_idx[0] if len(target_idx) else 1]
            #get individual maps
            zone_map = (_id_map == zone_color).min(axis=-1, keepdims=True)
            target_map = (_id_map == target_color).min(axis=-1, keepdims=True)
            #colorize
            zone_map = zone_map * ZONE_COLOR
            target_map = target_map * TARGET_COLOR
            joint_map = zone_map + target_map
            # add alpha
            alpha = ((target_map.sum(axis=2) | zone_map.sum(axis=2)) != 0) * 255
            joint_map = np.dstack((joint_map, alpha))
            #as image
            map_img = Image.fromarray(np.uint8(joint_map))
            #save image
            map_img.save(filepath.parent.joinpath(filepath.stem+"_map.png"))
        '''

        # Close the file.
        f.close()
        # Move the file.
        try:
            temp_path.replace(filepath)
        except OSError:
            shutil.move(temp_path, filepath)

    @staticmethod
    def rotate_vector_parallel_to_floor(
            vector: Dict[str, float],
            theta: float,
            degrees: bool = True) -> Dict[str, float]:

        v_x = vector['x']
        v_z = vector['z']
        if degrees:
            theta = np.radians(theta)

        v_x_new = np.cos(theta) * v_x - np.sin(theta) * v_z
        v_z_new = np.sin(theta) * v_x + np.cos(theta) * v_z

        return {'x': v_x_new, 'y': vector['y'], 'z': v_z_new}

    @staticmethod
    def cart2sph(x, y, z):
        hxy = np.hypot(x, y)
        r = np.hypot(hxy, z)
        el = np.arctan2(z, hxy)
        az = np.arctan2(y, x)
        return az, el, r

    @staticmethod
    def sph2cart(az, el, r):
        rcos_theta = r * np.cos(el)
        x = rcos_theta * np.cos(az)
        y = rcos_theta * np.sin(az)
        z = r * np.sin(el)
        return x, y, z


    @staticmethod
    def scale_vector(
            vector: Dict[str, float],
            scale: float) -> Dict[str, float]:
        return {k:vector[k] * scale for k in ['x','y','z']}

    @staticmethod
    def get_random_avatar_position(radius_min: float,
                                   radius_max: float,
                                   y_min: float,
                                   y_max: float,
                                   center: Dict[str, float],
                                   angle_min: float = 0,
                                   angle_max: float = 360,
                                   reflections: bool = False,
                                   ) -> Dict[str, float]:
        """
        :param radius_min: The minimum distance from the center.
        :param radius_max: The maximum distance from the center.
        :param y_min: The minimum y positional coordinate.
        :param y_max: The maximum y positional coordinate.
        :param center: The centerpoint.
        :param angle_min: The minimum angle of rotation around the centerpoint.
        :param angle_max: The maximum angle of rotation around the centerpoint.

        :return: A random position for the avatar around a centerpoint.
        """

        a_r = random.uniform(radius_min, radius_max)
        a_x = center["x"] + a_r
        a_z = center["z"] + a_r
        theta = np.radians(random.uniform(angle_min, angle_max))
        if reflections:
            theta2 = random.uniform(angle_min+180, angle_max+180)
            theta = random.choice([theta, theta2])
        a_y = random.uniform(y_min, y_max) + center["y"]
        a_x_new = np.cos(theta) * (a_x - center["x"]) - np.sin(theta) * (a_z - center["z"]) + center["x"]
        a_z_new = np.sin(theta) * (a_x - center["x"]) + np.cos(theta) * (a_z - center["z"]) + center["z"]
        a_x = a_x_new
        a_z = a_z_new

        return {"x": a_x, "y": a_y, "z": a_z}

    def get_rotating_camera_position(self, center, radius, angle, height):

        a_x = center["x"] + radius * np.sin(angle)
        a_z = center["z"] + radius * np.cos(angle)
        a_y = center["y"] + height

        return {"x": a_x, "y": a_y, "z": a_z}

    def get_rotating_camera_position_azimuth(self, az, el, r, delta_az, az_range, delta_dist=1.0):

        print('Warning no noise')
        print('\n' * 5)
        az_ = az + delta_az # + (random.random() - 0.5) * az_range
        el_ = el # + np.pi / 15
        r_ = r * delta_dist
        x_, z_, y_ = self.sph2cart(az_, el_, r_)  # Compute in Z-up

        return {"x": x_, "y": y_, "z": z_}, az_

    def is_done(self, resp: List[bytes], frame: int) -> bool:
        """
        Override this command for special logic to end the trial.

        :param resp: The output data response.
        :param frame: The frame number.

        :return: True if the trial is done.
        """

        return False

    @abstractmethod
    def get_scene_initialization_commands(self) -> List[dict]:
        """
        :return: Commands to initialize the scene ONLY for the first time (not per-trial).
        """

        raise Exception()

    @abstractmethod
    def get_trial_initialization_commands(self) -> List[dict]:
        """
        :return: Commands to initialize each trial.
        """

        raise Exception()

    @abstractmethod
    def _get_send_data_commands(self) -> List[dict]:
        """
        :return: A list of commands to request per-frame output data. Appended to the trial initialization commands.
        """

        raise Exception()

    def _write_static_data(self, static_group: h5py.Group) -> None:
        """
        Write static data to disk after assembling the trial initialization commands.

        :param static_group: The static data group.
        """
        # git commit and args
        res = subprocess.run('git rev-parse HEAD', shell=True, capture_output=True, text=True)
        self.commit = res.stdout.strip()
        static_group.create_dataset("git_commit", data=self.commit)

        # stimulus name
        static_group.create_dataset("stimulus_name", data=self.stimulus_name)
        static_group.create_dataset("object_ids", data=self.object_ids)
        static_group.create_dataset("model_names", data=[s.encode('utf8') for s in self.model_names])

        if self.object_segmentation_colors is not None:
            static_group.create_dataset("object_segmentation_colors", data=self.object_segmentation_colors)

    @abstractmethod
    def _write_frame(self, frames_grp: h5py.Group, resp: List[bytes], frame_num: int, zone_id: Optional[int] = None,
                     view_id: Optional[int] = None, trial_num:Optional[int] = None) -> \
            Tuple[h5py.Group, h5py.Group, dict, bool]:
        """
        Write a frame to the hdf5 file.

        :param frames_grp: The frames hdf5 group.
        :param resp: The response from the build.
        :param frame_num: The frame number.

        :return: Tuple: (The frame group, the objects group, a dictionary of Transforms, True if the trial is "done")
        """

        raise Exception()

    def _write_frame_labels(self,
                            frame_grp: h5py.Group,
                            resp: List[bytes],
                            frame_num: int,
                            sleeping: bool) -> Tuple[h5py.Group, bool]:
        """
        Writes the trial-level data for this frame.

        :param frame_grp: The hdf5 group for a single frame.
        :param resp: The response from the build.
        :param frame_num: The frame number.
        :param sleeping: Whether this trial timed out due to objects falling asleep.

        :return: Tuple(h5py.Group labels, bool done): the labels data and whether this is the last frame of the trial.
        """
        labels = frame_grp.create_group("labels")
        if frame_num > 0:
            complete = self.is_done(resp, frame_num)
        else:
            complete = False

        # If the trial is over, one way or another
        done = sleeping or complete

        # Write labels indicate whether and why the trial is over
        labels.create_dataset("trial_end", data=done)
        labels.create_dataset("trial_timeout", data=(sleeping and not complete))
        labels.create_dataset("trial_complete", data=(complete and not sleeping))

        # if done:
        #     print("Trial Ended: timeout? %s, completed? %s" % \
        #           ("YES" if sleeping and not complete else "NO",\
        #            "YES" if complete and not sleeping else "NO"))

        return labels, resp, frame_num, done

    def _get_destroy_object_command_name(self, o_id: int) -> str:
        """
        :param o_id: The object ID.

        :return: The name of the command used to destroy an object.
        """

        return "destroy_object"

    @abstractmethod
    def get_per_frame_commands(self, resp: List[bytes], frame: int) -> List[dict]:
        """
        :param resp: The output data response.
        :param frame: The frame number

        :return: Commands to send per frame.
        """
        raise Exception()

    @abstractmethod
    def get_field_of_view(self) -> float:
        """
        :return: The camera field of view.
        """

        raise Exception()

    def add_object(self, model_name: str, position={"x": 0, "y": 0, "z": 0}, rotation={"x": 0, "y": 0, "z": 0},
                   library: str = "") -> int:
        raise Exception("Don't use this function; see README for functions that supersede it.")

    def get_add_object(self, model_name: str, object_id: int, position={"x": 0, "y": 0, "z": 0},
                       rotation={"x": 0, "y": 0, "z": 0}, library: str = "") -> dict:
        raise Exception("Don't use this function; see README for functions that supersede it.")

    def _initialize_object_counter(self) -> None:
        self._object_id_counter = int(0)
        self._object_id_increment = int(1)

    def _increment_object_id(self) -> None:
        self._object_id_counter = int(self._object_id_counter + self._object_id_increment)

    def _get_next_object_id(self) -> int:
        self._increment_object_id()
        return int(self._object_id_counter)

    def add_room_center(self, vector):
        return {k: vector[k] + self.room_center[k] for k in vector.keys()}

    def get_material_name(self, material):

        if material is not None:
            if material in MATERIAL_TYPES:
                mat = random.choice(MATERIAL_NAMES[material])
            else:
                assert any((material in MATERIAL_NAMES[mtype] for mtype in self.material_types)), \
                    (material, self.material_types)
                mat = material
        else:
            mtype = random.choice(self.material_types)
            mat = random.choice(MATERIAL_NAMES[mtype])

        return mat

    def get_object_material_commands(self, record, object_id, material):
        commands = TDWUtils.set_visual_material(
            self, record.substructure, object_id, material, quality="high")
        return commands


    def _set_segmentation_colors(self, resp: List[bytes]) -> None:

        self.object_segmentation_colors = None

        if len(self.object_ids) == 0:
            self.object_segmentation_colors = []
            return

        for r in resp:
            if OutputData.get_data_type_id(r) == 'segm':
                seg = SegmentationColors(r)
                colors = {}
                for i in range(seg.get_num()):
                    try:
                        colors[seg.get_object_id(i)] = seg.get_object_color(i)
                    except:
                        print("No object id found for seg", i)

                self.object_segmentation_colors = []
                for o_id in self.object_ids:
                    if o_id in colors.keys():
                        self.object_segmentation_colors.append(
                            np.array(colors[o_id], dtype=np.uint8).reshape(1,3))
                    else:
                        self.object_segmentation_colors.append(
                            np.array([0,0,0], dtype=np.uint8).reshape(1,3))

                self.object_segmentation_colors = np.concatenate(self.object_segmentation_colors, 0)

    def _is_object_in_view(self, resp, o_id, pix_thresh=10) -> bool:

        id_map = None
        for r in resp[:-1]:
            r_id = OutputData.get_data_type_id(r)
            if r_id == "imag":
                im = Images(r)
                for i in range(im.get_num_passes()):
                    pass_mask = im.get_pass_mask(i)
                    if pass_mask == "_id":
                        id_map = np.array(Image.open(io.BytesIO(np.array(im.get_image(i))))).reshape(self._height, self._width, 3)

        if id_map is None:
            return True

        obj_index = [i for i,_o_id in enumerate(self.object_ids) if _o_id == o_id]
        if not len(obj_index):
            return True
        else:
            obj_index = obj_index[0]

        obj_seg_color = self.object_segmentation_colors[obj_index]
        obj_map = (id_map == obj_seg_color).min(axis=-1, keepdims=True)
        in_view = obj_map.sum() >= pix_thresh
        return in_view


    def _max_optical_flow(self, resp):

        flow_map = None
        for r in resp[:-1]:
            r_id = OutputData.get_data_type_id(r)
            if r_id == "imag":
                im = Images(r)
                for i in range(im.get_num_passes()):
                    pass_mask = im.get_pass_mask(i)
                    if pass_mask == "_flow":
                        flow_map = np.array(Image.open(io.BytesIO(np.array(im.get_image(i))))).reshape(self._height, self._width, 3)

        if flow_map is None:
            return float(0)

        else:
            return flow_map.sum(-1).max().astype(float)

    def _get_object_meshes(self, resp: List[bytes]) -> None:

        self.object_meshes = dict()
        # {object_id: (vertices, faces)}
        for r in resp:
            if OutputData.get_data_type_id(r) == 'mesh':
                meshes = Meshes(r)
                nmeshes = meshes.get_num()
                assert(len(self.object_ids) == nmeshes)
                for index in range(nmeshes):
                    o_id = meshes.get_object_id(index)
                    vertices = meshes.get_vertices(index)
                    faces = meshes.get_triangles(index)
                    self.object_meshes[o_id] = (vertices, faces)
