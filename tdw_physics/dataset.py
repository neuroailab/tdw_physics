import sys, os, copy, subprocess, glob, logging, time
import platform
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm
import stopit
from PIL import Image
import io
import h5py, json
from collections import OrderedDict
import numpy as np
import random
# from tdw.controller import Controller
from .controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.output_data import OutputData, SegmentationColors, Meshes, Images
from tdw.librarian import ModelRecord, MaterialLibrarian
from tdw.add_ons.interior_scene_lighting import InteriorSceneLighting
# from tdw_physics.data_utils import accept_stimuli

from tdw_physics.postprocessing.stimuli import pngs_to_mp4
from tdw_physics.postprocessing.labels import (get_labels_from,
                                               get_all_label_funcs,
                                               get_across_trial_stats_from)
from tdw_physics.util_geom import save_obj
import shutil
import tdw_physics.util as util
PASSES = ["_img", "_depth", "_normals", "_flow", "_id", "_category", "_albedo"]
M = MaterialLibrarian()
MATERIAL_TYPES = M.get_material_types()
MATERIAL_NAMES = {mtype: [m.name for m in M.get_all_materials_of_type(mtype)] \
                  for mtype in MATERIAL_TYPES}

# colors for the target/zone overlay
ZONE_COLOR = [255,255,0]
TARGET_COLOR = [255,0,0]

def pad_right(x, sz=7):
    ones = np.ones([x.shape[0], sz, x.shape[-1]])
    return np.concatenate([x, ones], 1)


def pad_below(x, sz=7):
    ones = np.ones([sz, x.shape[1], x.shape[-1]])
    return np.concatenate([x, ones], 0)


def concat_img_horz(x_gt):
    # x_gt is V x H x W x 3

    full_tensor = x_gt[0]

    for i in range(1, x_gt.shape[0]):
        img = x_gt[i]

        full_tensor = np.concatenate([pad_right(full_tensor), img], 1)

    return full_tensor

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

    # IDs of the objects in the current trial.
    OBJECT_IDS: np.array = np.empty(dtype=int, shape=0)

    def __init__(self, port: int = 1071, check_version: bool=False,
                 launch_build: bool=True,
                 randomize: int=0,
                 seed: int=0,
                 sim_seed: int=0,
                 save_args=True,
                 return_early=False,
                 custom_build=None,
                 ffmpeg_executable='ffmpeg',
                 path_obj='/mnt/fs3/rmvenkat/data/all_flex_meshes',
                 path_scene='/mnt/fs0/haw027/data/tdw_scenes/scenes',
                 view_id_number=0,
                 max_frames=250,
                 check_interpenet=True,
                 check_target_area=False,
                 **kwargs):

        # launch_build = False

        # breakpoint()

        # save the command-line args
        self.save_args = save_args
        self.max_frames = max_frames
        self.check_interpenet = check_interpenet
        self.ffmpeg_executable = ffmpeg_executable if ffmpeg_executable is not None else 'ffmpeg'
        self._trial_num = None
        self.command_log = None
        self.view_id_number = view_id_number
        self.check_target_area = check_target_area

        # ## get random port unless one is specified
        # if port is None:
        rng = np.random.default_rng(seed + sim_seed + (view_id_number*1251)%33 )
        port = rng.integers(1000,9999)
        # print("random port",port,"chosen. If communication with tdw build fails, set port to 1071 or update your tdw installation.")
        print("random port",port)

            # random.seed(self.seed)
            # print("SET RANDOM SEED: %d" % self.seed)

        if return_early:
            return

        super().__init__(port=port,
                        check_version=check_version,
                         launch_build=launch_build,
                         custom_build=custom_build,
                         mesh_folder=path_obj,
                         scene_folder=path_scene)

        # hdri_skybox = "table_mountain_1_4k"
        # interior_scene_lighting = InteriorSceneLighting(hdri_skybox=hdri_skybox, aperture=8, focus_distance=2.5, ambient_occlusion_intensity=0.125, ambient_occlusion_thickness_modifier=3.5, shadow_strength=0.1)
        # self.add_ons.append(interior_scene_lighting)

        from tdw.add_ons.logger import Logger

        # logger = Logger(path="log.txt")
        # self.add_ons.append(logger)
        #
        # from tdw.add_ons.obi import Obi
        # self.obi = Obi()
        # self.add_ons.extend([self.obi])

        # set random state
        self.randomize = randomize
        self.seed = seed
        self.sim_seed = sim_seed
        if not bool(self.randomize):
            random.seed(self.seed)
            # print("SET RANDOM SEED: %d" % self.seed)

        # fluid actors need to be handled separately
        self.fluid_object_ids = []
        self.num_views = kwargs.get('num_views', 1)


    def communicate(self, commands) -> list:
        '''
        Save a log of the commands so that they can be rerun
        '''
        if self.command_log is not None:
            with open(str(self.command_log), "at") as f:
                f.write(json.dumps(commands) + (" trial %s" % self._trial_num) + "\n")
        return super().communicate(commands)

    def clear_static_data(self) -> None:
        Dataset.OBJECT_IDS = np.empty(dtype=int, shape=0)
        self.model_names = []
        self._initialize_object_counter()

    @staticmethod
    def get_controller_label_funcs(classname='Dataset'):
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
                                    height: int) -> List:
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

        # commands = []

        commands.extend(self.get_scene_initialization_commands())
        # Add the avatar.


        commands.extend([{"$type": "create_avatar",
                          "type": "A_Img_Caps_Kinematic",
                          "id": "a"},
                         # {"$type": "set_target_framerate",
                         #  "framerate": 30},
                         {"$type": "set_time_step",
                          "time_step": 0.01},
                         {"$type": "set_field_of_view",
                          "field_of_view": self.get_field_of_view()},
                         {"$type": "set_anti_aliasing",
                          "mode": "subpixel"}
                         ])

        if len(self.write_passes) != 0:
            # breakpoint()
            commands.append({"$type": "set_pass_masks",
                          "pass_masks": self.write_passes})
            commands.append({"$type": "send_images",
                          "frequency": "always"})

        return commands

    def run(self, num: int, output_dir: str,
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
        # print("height: %d, width: %d, fps: %d" % (self._height, self._width, self._framerate))

        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        temp_path = Path(temp_path)
        if not temp_path.parent.exists():
            temp_path.parent.mkdir(parents=True)
        # Remove an incomplete temp path.
        if temp_path.exists():
            temp_path.unlink()

        # save a log of the commands send to TDW build
        self.command_log = Path(output_dir).joinpath('tdw_commands.json')

        # which passes to write to the HDF5
        self.write_passes = write_passes
        # print("self.write_passes", self.write_passes)
        if isinstance(self.write_passes, str):
            self.write_passes = self.write_passes.split(',')
        self.write_passes = [p for p in self.write_passes if (p in PASSES)]

        # which passes to save as an MP4
        self.save_passes = save_passes
        if isinstance(self.save_passes, str):
            self.save_passes = self.save_passes.split(',')
        # self.save_passes = [p for p in self.save_passes if (p in self.write_passes)]
        self.save_movies = save_movies

        # whether to send and save meshes
        self.save_meshes = save_meshes

        # print("write passes", self.write_passes)
        # print("save passes", self.save_passes)
        # print("save movies", self.save_movies)
        # print("save meshes", self.save_meshes)

        if self.save_movies and len(self.save_passes) == 0:
            self.save_movies = False
            print('Not saving movies since save_passes has len {}'.format(len(self.save_passes)))

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
        # print("initialization_commands: ", initialization_commands)
        self.communicate(initialization_commands)

        self.trial_loop(num, output_dir, temp_path)

        # Terminate TDW
        # Windows doesn't know signal timeout
        if terminate:
            if platform.system() == 'Windows':
                end = self.communicate({"$type": "terminate"})
            else:  # Unix systems can use signal to timeout
                with stopit.SignalTimeout(
                        5) as to_ctx_mgr:  # since TDW sometimes doesn't acknowledge being stopped we only *try* to close it
                    assert to_ctx_mgr.state == to_ctx_mgr.EXECUTING
                    end = self.communicate({"$type": "terminate"})
                if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
                    pass
                    # print("tdw closed successfully")
                elif to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
                    print("tdw failed to acknowledge being closed. tdw window might need to be manually closed")

        # Save the command line args
        if self.save_args:
            self.args_dict = copy.deepcopy(args_dict)
        self.save_command_line_args(output_dir)

    # def trialx(self, filepath: Path, temp_path: Path, trial_num: int, unload_assets_every: int) -> None:
    #
    #     return None

    def trial(self, filepath: Path, temp_path: Path, trial_num: int, unload_assets_every: int) -> None:

        # return None
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
        # Add commands to request output data.


        # breakpoint()


        commands.extend(self._get_send_data_commands())

        azimuth_grp = f.create_group("azimuth")

        multi_camera_positions = self.generate_multi_camera_positions(azimuth_grp, self.view_id_number)

        commands.extend(self.move_camera_commands(multi_camera_positions, []))
        # _resp = self.communicate(commands)


        # Send the commands and start the trial.
        r_types = ['']
        count = 0

        # print(commands)

        resp = self.communicate(commands)

        # breakpoint()

        self._set_segmentation_colors(resp)

        self._get_object_meshes(resp)
        frame = 0
        # Write static data to disk.
        static_group = f.create_group("static")
        self._write_static_data(static_group)

        # Add the first frame.
        done = False
        frames_grp = f.create_group("frames")


        # print("Hello***")
        # frame_grp, _, _, _ = self._write_frame(frames_grp=frames_grp, resp=resp, frame_num=frame, view_num=self.view_id_number)
        # self._write_frame_labels(frame_grp, resp, -1, False)

        # TODO: write the pngs here for img, id, depth, etc.

        # print("num views", self.num_views)

        # Continue the trial. Send commands, and parse output data.
        # if self.num_views > 1:

        # TODO: set as flag

        # t = time.time()
        while (not done) and (frame < self.max_frames):
            frame += 1
            # print('frame %d' % frame)
            # cmds, dict_masses = self.get_per_frame_commands(resp, frame)

            cmds = self.get_per_frame_commands(resp, frame)


            # mass_list = []
            # for idx, object_id in enumerate(Dataset.OBJECT_IDS):
            #     mass = dict_masses[object_id]
            #     mass_list.append(mass)

            # breakpoint()
            # if 'mass_scaled' not in static_group.keys():
            #     static_group.create_dataset(f"mass_scaled", data=np.array(mass_list))
            # mesh_group.create_dataset(f"vertices_{idx}", data=vertices)



            # Sometimes the build freezes and has to reopen the socket.
            # This prevents such errors from throwing off the frame numbering
            # if ('imag' not in r_ids) or ('tran' not in r_ids):
            #     print("retrying frame %d, response only had %s" % (frame, r_ids))
            #     frame -= 1
            #     continue

            # print()

            # if self.num_views > 1:
            resp = self.communicate(cmds)

            # breakpoint()

            # for view_num in range(self.num_views):
                # if view_num == 0 or frame == view_num:
            # camera_position = multi_camera_positions[0]

            # _resp = resp
            # print('\tview %d' % view_num)
            frame_grp, objs_grp, tr_dict, done = self._write_frame(
                frames_grp=frames_grp,
                resp=resp, frame_num=frame, view_num=self.view_id_number)

            # done = False

            # breakpoint()

            # else:
            #     resp = self.communicate(cmds)
            #     r_ids = [OutputData.get_data_type_id(r) for r in resp[:-1]]
            #     frame_grp, objs_grp, tr_dict, done = self._write_frame(frames_grp=frames_grp, resp=resp,
            #                                                            frame_num=frame, view_num=0)

            # TODO: write the pngs here for img, id, depth, etc. -- can make a function.

            # breakpoint()

            # Write whether this frame completed the trial and any other trial-level data
            # labels_grp, _, _, _ = self._write_frame_labels(frame_grp, resp, frame, done)
            #
            # if frame > 5:
            #     break
        # print("avg time to communicate", time.time() - t)

        #save_imgs for viz
        # if self.save_movies:
        #     for fr in range(1, frame+1):
        #         imgs = []
        #         for pass_mask in self.save_passes:

        #             all_imgs = []
        #             # for cam_no in range(self.num_views):
        #             cam_no = self.view_id_number
        #             filename = os.path.join(self.png_dir, pass_mask[1:] + '_' + 'cam' + str(cam_no) + '_' + str(fr).zfill(4) + '.png')
        #             img = plt.imread(filename)
        #             all_imgs.append(img)
        #             all_imgs = np.stack(all_imgs)
        #             all_imgs = concat_img_horz(all_imgs)
        #             # all_imgs = pad_below(all_imgs)
        #             imgs.append(all_imgs[:, :, :3])

        #         # breakpoint()

        #         imgs = (np.concatenate(imgs, 0)*255).astype('uint8')

        #         # breakpoint()

        #         # filename = os.path.join(self.png_dir, 'img_' + str(fr).zfill(4) + '.png')

        #         im_arr = Image.fromarray(imgs)
        #         shp = (im_arr.size[0] - im_arr.size[0]%2, im_arr.size[1] - im_arr.size[1]%2)
        #         im_arr = im_arr.resize(shp)
        #         # im_arr.save(filename)



        # breakpoint()

        # #write png file to png dir
        # for fr in range(frame+1):
        #
        #     # breakpoint()
        #
        #     img = frames_grp[str(fr).zfill(4)]['images']['_img_cam0']
        #     img = Image.open(io.BytesIO(np.array(img)))
        #     filename = os.path.join(self.png_dir, 'img_' + str(fr).zfill(4) + '.png')
        #     img.save(filename)
        #
        #     img_id = frames_grp[str(fr).zfill(4)]['images']['_id_cam0']
        #     img_id = Image.open(io.BytesIO(np.array(img_id)))
        #     filename = os.path.join(self.png_dir, 'id_' + str(fr).zfill(4) + '.png')
        #     img_id.save(filename)
        #
        #     img_depth = frames_grp[str(fr).zfill(4)]['images']['_depth_cam0']
        #     img_depth = Image.fromarray(np.array(img_depth))
        #     filename = os.path.join(self.png_dir, 'depth_' + str(fr).zfill(4) + '.png')
        #     img_depth.save(filename)

        # Cleanup.
        commands = []
        for o_id in Dataset.OBJECT_IDS:
            commands.append({"$type": self._get_destroy_object_command_name(o_id),
                             "id": int(o_id)})
        self.communicate(commands)

        # Compute the trial-level metadata. Save it per trial in case of failure mid-trial loop
        # if self.save_labels:
        meta = OrderedDict()
        meta = get_labels_from(f, label_funcs=self.get_controller_label_funcs(type(self).__name__), res=meta)
        self.trial_metadata.append(meta)

        # Save the trial-level metadata
        json_str = json.dumps(self.trial_metadata, indent=4)
        self.meta_file.write_text(json_str, encoding='utf-8')
        # print("TRIAL %d LABELS" % self._trial_num)
        # print(json.dumps(self.trial_metadata[-1], indent=4))

        # # Save out the target/zone segmentation mask
        # if (self.zone_id in Dataset.OBJECT_IDS) and (self.target_id in Dataset.OBJECT_IDS):
        # try:
        #     _id = f['frames']['0000']['images']['_id']
        # except:
        #     # print("inside cam0")
        #     _id = f['frames']['0000']['images']['_id_cam0']
        # # get PIL image
        # _id_map = np.array(Image.open(io.BytesIO(np.array(_id))))
        # # get colors
        # zone_idx = [i for i, o_id in enumerate(Dataset.OBJECT_IDS) if o_id == self.zone_id]
        # zone_color = self.object_segmentation_colors[zone_idx[0] if len(zone_idx) else 0]
        # target_idx = [i for i, o_id in enumerate(Dataset.OBJECT_IDS) if o_id == self.target_id]
        # target_color = self.object_segmentation_colors[target_idx[0] if len(target_idx) else 1]
        # # get individual maps
        # zone_map = (_id_map == zone_color).min(axis=-1, keepdims=True)
        # target_map = (_id_map == target_color).min(axis=-1, keepdims=True)
        # # colorize
        # zone_map = zone_map * ZONE_COLOR
        # target_map = target_map * TARGET_COLOR
        # joint_map = zone_map + target_map
        # # add alpha
        # alpha = ((target_map.sum(axis=2) | zone_map.sum(axis=2)) != 0) * 255
        # joint_map = np.dstack((joint_map, alpha))
        # # as image
        # map_img = Image.fromarray(np.uint8(joint_map))
        # # save image
        # map_img.save(filepath.parent.joinpath(filepath.stem + "_map.png"))

        # Close the file.
        f.close()
        # # Move the file.
        # try:
        #     temp_path.replace(filepath)
        # except OSError:
        shutil.move(temp_path, filepath)

        if self.save_movies:
            return [self._height, self._width]
        else:
            return [2, 2]

    def trial_loop(self,
                   num: int,
                   output_dir: str,
                   temp_path: str,
                   save_frame: int = None,
                   unload_assets_every: int = 10,
                   update_kwargs: List[dict] = {},
                   do_log: bool = False) -> None:

        # if not isinstance(update_kwargs, list):
        #     update_kwargs = [update_kwargs] * num

        # pbar = tqdm(total=num)
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

        # exists_up_to = 5

        pbar.update(exists_up_to)
        for i in range(exists_up_to, num):
            if i not in self.indexes:
                continue

            # if i==0:
            #     continue

            filepath = output_dir.joinpath(TDWUtils.zero_padding(i, 4) + ".hdf5")
            self.stimulus_name = '_'.join([filepath.parent.name, str(Path(filepath.name).with_suffix(''))])

            ## update the controller state
            # self.update_controller_state(**update_kwargs[i])

            if True: #not filepath.exists():
                if do_log:
                    start = time.time()
                    logging.info("Starting trial << %d >> with kwargs %s" % (i, update_kwargs[i]))
                # Save out images
                self.png_dir = None
                if any([pa in PASSES for pa in self.save_passes]):
                    self.png_dir = output_dir.joinpath("pngs_" + TDWUtils.zero_padding(i, 4))
                    if not self.png_dir.exists() and self.save_movies:
                        self.png_dir.mkdir(parents=True)

                # breakpoint()

                # Do the trial.
                shp = self.trial(filepath,
                           temp_path,
                           i,
                           unload_assets_every)

                #save only for cam0 "for now"
                cam_suffix = '_cam0'

                # Save an MP4 of the stimulus
                if self.save_movies:

                    for pass_mask in ['_img']:
                        pass_mask = pass_mask + cam_suffix
                        mp4_filename = str(filepath).split('.hdf5')[0].split('/')
                        name = mp4_filename[-1]
                        mp4_filename = '/'.join(mp4_filename[:-1]) + '_' + name + pass_mask

                        cmd, stdout, stderr = pngs_to_mp4(
                            filename=mp4_filename,
                            framerate=100,
                            executable= self.ffmpeg_executable, #'/ccn2/u/rmvenkat/ffmpeg',
                            image_stem=pass_mask[1:] + '_cam0_',
                            png_dir=self.png_dir,
                            size=[shp[0], shp[1]],#[self._height, self._width],#
                            overwrite=True,
                            remove_pngs=False,
                            use_parent_dir=False)

                        # print("saved:", mp4_filename)
                        # breakpoint()

                    if save_frame is not None:
                        frames = os.listdir(str(self.png_dir))
                        sv = sorted(frames)[save_frame]
                        png = output_dir.joinpath(TDWUtils.zero_padding(i, 4) + ".png")
                        _ = subprocess.run('mv ' + str(self.png_dir) + '/' + sv + ' ' + str(png), shell=True)

                    # rm = subprocess.run('rm -rf ' + str(self.png_dir), shell=True)

                # if self.save_meshes:
                #     for o_id in Dataset.OBJECT_IDS:
                #         obj_filename = str(filepath).split('.hdf5')[0] + f"_obj{o_id}.obj"
                #         vertices, faces = self.object_meshes[o_id]
                #         save_obj(vertices, faces, obj_filename)

                if do_log:
                    end = time.time()
                    logging.info("Finished trial << %d >> with trial seed = %d (elapsed time: %d seconds)" % (
                    i, self.trial_seed, int(end - start)))

            # if not accept_stimuli(str(filepath), check_interp=self.check_interpenet, check_area=self.check_target_area):
            #     print("stimiuli rejected due to interpenetration/target area too less")
            #     #create a folder for rejected stimuli
            #     rejected_stimuli_dir = output_dir.parent.joinpath('rejected_stimuli')
            #     if not rejected_stimuli_dir.exists():
            #         rejected_stimuli_dir.mkdir(parents=True)
            #     #move the rejected stimuli to the rejected_stimuli folder
            #     # breakpoint()
            #     xx_split = str(filepath).split('/')
            #     fp = xx_split[-2] + '_' + xx_split[-1]
            #     rejected_stimuli_path = rejected_stimuli_dir.joinpath(fp)
            #     if not rejected_stimuli_path.exists():
            #         shutil.move(filepath, rejected_stimuli_path)
            #         #move the corresponding png and mp4 files
            #         if self.save_movies:
            #             shutil.move(mp4_filename + '.mp4', rejected_stimuli_dir)
            #         # breakpoint()
            #         shutil.move(str(filepath).split('.')[0] + '_map.png', str(rejected_stimuli_path).split('.')[0] + '_map.png')
            #         # breakpoint()
            pbar.update(1)
        pbar.close()

    def generate_multi_camera_positions(self, azimuth_grp, i):
        '''
        Generate multiple camera positions based on azimuth rotation
        '''

        azimuth_delta = 2 * np.pi / self.num_views  # delta rotation angle
        init_pos_cart = self.camera_position # initial camera position (cartesian coordinate)
        init_pos_sphe = util.cart2sphe(init_pos_cart) # initial camera position (spherical coordinate)
        new_pos_sphe = copy.deepcopy(init_pos_sphe)

        camera_pos_list = []
        # for i in range(self.num_views):
        noise = (random.uniform(-0.2, 0.2)) * azimuth_delta # if add_noise else 0. # add noise to the azimuth rotation angles
        noise = 0 if i==0 else noise
        azimuth = init_pos_sphe['azimuth'] + i * azimuth_delta + noise # rotation for a new camera view
        new_pos_sphe['azimuth'] = azimuth # update the spherical coordinates
        new_pos_cart = util.sphe2cart(new_pos_sphe)  # convert to cartesian coordinates
        camera_pos_list.append(new_pos_cart)

        # save azimuth rotation matrix (for uORF training)
        az_ori = azimuth + np.pi  # since cam faces world origin, its orientation azimuth differs by pi
        az_rot_mat_2d = np.array([[np.cos(az_ori), - np.sin(az_ori)],
                                  [np.sin(az_ori), np.cos(az_ori)]])
        az_rot_mat = np.eye(3)
        az_rot_mat[:2, :2] = az_rot_mat_2d
        azimuth_grp.create_dataset(f"cam_{i}", data=az_rot_mat)

        return camera_pos_list[0]

    def move_camera_commands(self, camera_pos, commands):
        self._set_avatar_attributes(camera_pos)

        commands.extend([
            {"$type": "teleport_avatar_to", "position": camera_pos},
            {"$type": "look_at_position", "position": self.camera_aim},
            {"$type": "set_focus_distance", "focus_distance": TDWUtils.get_distance(camera_pos, self.camera_aim)},
            # {"$type": "set_camera_clipping_planes", "near": 2., "far": 12}
        ])
        return commands

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
    def scale_vector(
            vector: Dict[str, float],
            scale: float) -> Dict[str, float]:
        return {k:vector[k] * scale for k in ['x','y','z']}

    @staticmethod
    def get_random_avatar_position(radius_min: float, radius_max: float, y_min: float, y_max: float,
                                   center: Dict[str, float], angle_min: float = 0, reflections: bool = False,
                                   angle_max: float = 360) -> Dict[str, float]:
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


        # breakpoint()

        a_r = random.uniform(radius_min, radius_max)
        # a_x = center["x"] + a_r
        # a_z = center["z"] + a_r


        a_y = random.uniform(y_min, y_max)

        r_xy = np.sqrt(a_r**2 - a_y**2)



        #
        # # # if reflections:
        # # theta2 = 180
        theta = np.radians(random.uniform(angle_min, angle_max))
        theta2 = theta + np.pi
        theta = random.choice([theta, theta2])

        a_x = np.cos(theta)*r_xy + center["x"]

        a_z = np.sin(theta)*r_xy + center["z"]

        a_y = a_y + center["y"]

        return {"x": a_x, "y": a_y, "z": a_z}

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

        self.commit = ""  # res.stdout.strip()
        static_group.create_dataset("git_commit", data=self.commit)

        # stimulus name
        static_group.create_dataset("stimulus_name", data=self.stimulus_name)
        static_group.create_dataset("object_ids", data=Dataset.OBJECT_IDS)

        static_group.create_dataset("model_names", data=[s.encode('utf8') for s in self.model_names])

        if self.object_segmentation_colors is not None:
            static_group.create_dataset("object_segmentation_colors", data=self.object_segmentation_colors)

    @abstractmethod
    def _write_frame(self, frames_grp: h5py.Group, resp: List[bytes], frame_num: int, view_num: int) -> \
            Tuple[h5py.Group, h5py.Group, dict, bool]:
        """
        Write a frame to the hdf5 file.

        :param frames_grp: The frames hdf5 group.
        :param resp: The response from the build.
        :param frame_num: The frame number.

        :return: Tuple: (The frame group, the objects group, a dictionary of Transforms, True if the trial is "done")
        """

        raise Exception()

    def _initialize_object_counter(self) -> None:
        self._object_id_counter = int(0)
        self._object_id_increment = int(1)

    def _write_frame_labels(self,
                            frame_grp: h5py.Group,
                            resp: List[bytes],
                            frame_num: int,

                            sleeping: bool) -> Tuple[h5py.Group, List[bytes], int, bool]:
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

        if len(Dataset.OBJECT_IDS) == 0:
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
                for o_id in Dataset.OBJECT_IDS:
                    if o_id in colors.keys():
                        self.object_segmentation_colors.append(
                            np.array(colors[o_id], dtype=np.uint8).reshape(1, 3))
                    else:
                        self.object_segmentation_colors.append(
                            np.array([0, 0, 0], dtype=np.uint8).reshape(1, 3))

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
                        id_map = np.array(Image.open(io.BytesIO(np.array(im.get_image(i))))).reshape(self._height,
                                                                                                     self._width, 3)

        if id_map is None:
            return True

        obj_index = [i for i, _o_id in enumerate(Dataset.OBJECT_IDS) if _o_id == o_id]
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
                # breakpoint()
                # print("len(Dataset.OBJECT_IDS)", len(Dataset.OBJECT_IDS), nmeshes)
                # assert (len(Dataset.OBJECT_IDS) == nmeshes)
                for index in range(nmeshes):

                    o_id = meshes.get_object_id(index)

                    if o_id not in Dataset.OBJECT_IDS:
                        continue

                    vertices = meshes.get_vertices(index)
                    faces = meshes.get_triangles(index)
                    self.object_meshes[o_id] = (vertices, faces)