from typing import Dict, List
import random
from tdw.tdw_utils import TDWUtils
import numpy as np
from tdw.librarian import ModelLibrarian, MaterialLibrarian, SceneLibrarian
from tdw.tdw_utils import TDWUtils
import argparse

# Every model library, sorted by name.
MODEL_LIBRARIES: Dict[str, ModelLibrarian] = {}
for filename in ModelLibrarian.get_library_filenames():
    MODEL_LIBRARIES.update({filename: ModelLibrarian(filename)})

# All the models with flex enabled
FLEX_MODELS: Dict[str, ModelLibrarian] = {}
for filename in MODEL_LIBRARIES.keys():
    FLEX_MODELS[filename] = {
        record for record in MODEL_LIBRARIES[filename].records if record.flex == True}

# All the model categories
MODEL_CATEGORIES = []
for record in MODEL_LIBRARIES["models_full.json"].records:
    MODEL_CATEGORIES.append(record.wcategory)
MODEL_CATEGORIES = list(set(MODEL_CATEGORIES))

# All the material types and materials
M = MaterialLibrarian()
MATERIAL_TYPES = M.get_material_types()
MATERIAL_NAMES = {mtype: [m.name for m in M.get_all_materials_of_type(mtype)] \
                  for mtype in MATERIAL_TYPES}

S = SceneLibrarian()
ROOMS = [r.name for r in S.records]


# The names of the image passes
PASSES = ["_img", "_depth", "_normals", "_flow", "_id", "_category", "_albedo"]

def str_to_xyz(s: str, to_json=False):
    xyz = s.split(',')
    if len(xyz) == 3:
        s ={"x":float(xyz[0]), "y":float(xyz[1]), "z":float(xyz[2])}
    return s

def xyz_to_arr(xyz : dict):
    if xyz is not None:
        arr = np.array(
            [xyz[k] for k in ["x","y","z"]], dtype=np.float32)
        return arr
    return xyz

def arr_to_xyz(arr : np.ndarray):
    if arr is not None:
        xyz = {k:arr[i] for i,k in enumerate(["x","y","z"])}
        return xyz
    return arr


def get_move_along_direction(pos: Dict[str, float], target: Dict[str, float], d: float, noise: float = 0) -> \
        Dict[str, float]:
    """
    :param pos: The object's position.
    :param target: The target position.
    :param d: The distance to teleport.
    :param noise: Add a little noise to the teleport.

    :return: A position from pos by distance d along a directional vector defined by pos, target.
    """
    direction = TDWUtils.array_to_vector3((TDWUtils.vector3_to_array(target) - TDWUtils.vector3_to_array(pos)) /
                                          TDWUtils.get_distance(pos, target))

    return {"x": pos["x"] + direction["x"] * d + random.uniform(-noise, noise),
            "y": pos["y"],
            "z": pos["z"] + direction["z"] * d + random.uniform(-noise, noise)}


def get_object_look_at(o_id: int, pos: Dict[str, float], noise: float = 0) -> List[dict]:
    """
    :param o_id: The ID of the object to be rotated.
    :param pos: The position to look at.
    :param noise: Rotate the object randomly by this much after applying the look_at command.

    :return: A list of commands to rotate an object to look at the target position.
    """

    commands = [{"$type": "object_look_at_position",
                 "id": o_id,
                 "position": pos}]
    if noise > 0:
        commands.append({"$type": "rotate_object_by",
                         "angle": random.uniform(-noise, noise),
                         "axis": "yaw",
                         "id": o_id,
                         "is_world": True})
    return commands

def none_or_int(value):
    if value is None:
        return None
    elif value == 'None':
        return None
    else:
        return int(value)

def none_or_str(value):
    if value == 'None':
        return None
    else:
        return value

def int_or_bool(value):
    if isinstance(value, int):
        return False if value == 0 else True
    elif isinstance(value, bool):
        return value
    elif isinstance(value, str):
        if value in ['True', 'true', 't', '1']:
            return True
        elif value in ['False', 'false', 'f', '0']:
            return False
        else:
            raise ValueError("%s is not a valid int_or_bool" % value)
    else:
        return False

def get_parser(dataset_dir: str, get_help: bool=False):
    """
    :param dataset_dir: The default name of the dataset.

    :return: Parsed command-line arguments common to all controllers.
    """
    parser = argparse.ArgumentParser(add_help=get_help)
    parser.add_argument("--port", type=int, default=None, help="Which port to communicate with build")
    parser.add_argument("--dir", type=str, default=f"D:/{dataset_dir}", help="Root output directory.")
    parser.add_argument("--num", type=int, default=3, help="The number of trials in the dataset.")
    parser.add_argument("--sim_id", type=int, default=0, help="The index of the simulation.")
    parser.add_argument("--noise", type=str, default="null", help="Path to the noise parameters file.")
    parser.add_argument("--indexes", type=str, default='', help="Comma-separated list of Which trials to run")
    parser.add_argument("--num_multiplier", type=float, default=1.0, help="Multiply the number of trials by this factor")
    parser.add_argument("--temp", type=str, default="NONE", help="Temp path for incomplete files. 'NONE' to place the temp file in the output folder.")
    parser.add_argument("--width", type=int, default=256, help="Screen width in pixels.")
    parser.add_argument("--height", type=int, default=256, help="Screen width in pixels.")
    parser.add_argument("--framerate", type=int, default=30, help="Framerate for rendered movies")
    parser.add_argument("--gpu", type=none_or_int, default=0, help="ID of the gpu to run on")
    parser.add_argument("--display", type=str, default='0', help="ID of the display to run on")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed with which to initialize scenario")
    parser.add_argument("--random", type=int, default=1, help="Whether to set trials randomly")
    parser.add_argument("--num_views", type=int, default=1, help="How many possible viewpoints to render trial from")
    parser.add_argument("--viewpoint", type=int, default=0, help="which viewpoint to render from")
    parser.add_argument("--run", type=int, default=1, help="run the simulation or not")
    parser.add_argument("--monochrome", type=int_or_bool, default=0, help="whether to set all colorable objects to the same color")
    parser.add_argument("--room", type=str, default="box", help="Which room to use. Either 'box' or 'tdw'.")
    parser.add_argument("--write_passes", type=str, default=','.join(PASSES), help="Comma-separated list of which passes to write to the HDF5: _img, _depth, _normals, _id, _flow")
    parser.add_argument("--save_passes", type=str, default='', help="Comma-separated list of Which passes to save to PNGs/MP4s: _img, _depth, _normals, _id, _flow")
    parser.add_argument("--save_movies", action='store_true', help="Whether to write out MP4s of each trial")
    parser.add_argument("--save_labels", action='store_true', help="Whether to save out JSON labels for the full trial set.")
    parser.add_argument("--save_meshes", action='store_true', help="Whether to save meshes sent from the build")
    parser.add_argument("--unload_assets_every", type=int, default=10, help="Unload assets after how many trials")
    parser.add_argument("--local_models_path",
                        type=none_or_str,
                        default=None,
                        help="path to local models if they exist")
    parser.add_argument("--local_materials_path",
                        type=none_or_str,
                        default=None,
                        help="path to local materials if they exist")
    parser.add_argument("--check_compat", action='store_true', help="check compatibility of phsics models")
    parser.add_argument("--hdri_skybox", type=none_or_str, default=None, help="Which skybox to use")
    parser.add_argument("--custom_build", type=none_or_str, default=None, help="use custom build")
    parser.add_argument("--ffmpeg_executable", type=none_or_str, default=None, help="ffmpeg exec")
    parser.add_argument("--seed_global", type=int, default=1024, help="Global Seed")
    parser.add_argument("--path_obj", type=none_or_str, default=None, help="path to flex obj files")
    parser.add_argument("--local_scenes_path", type=none_or_str, default=None, help="path to local materials if they exist")
    parser.add_argument("--path_hdri", type=none_or_str, default=None, help="path to hdri skyboxes")
    parser.add_argument("--sun_angle",
                        type=str,
                        default="[60,60]",
                        help="comma separated list of sun_angle")
    parser.add_argument("--sun_intensity",
                        type=str,
                        default="[1.4,1.4]",
                        help="comma separated list of sun_intensity")
    parser.add_argument("--skybox_rotation",
                        type=str,
                        default="[30,30]",
                        help="comma separated list of skybox_rotation")
    parser.add_argument("--sun_elevation",
                        type=str,
                        default="[60,120]",
                        help="comma separated list of sun elevation")
    parser.add_argument("--view_id_number", type=int, default=0, help="Which view to render")
    # parser.add_argument("--return_early", action='store_true', help="check compatibility of phsics models")

    args = parser.parse_args('')

    # breakpoint()

    return parser

def get_args(dataset_dir: str):

    parser = get_parser(dataset_dir, get_help=True)
    return parser.parse_args()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--flex", action="store_true",
                        help="Print all the FLEX-enabled models")
    parser.add_argument("--material_type",
                        type=none_or_str,
                        default=None,
                        help="Print all the materials with this material_type")
    parser.add_argument("--model_category",
                        type=none_or_str,
                        default=None,
                        help="Print all the models with this category id")
    parser.add_argument("--model_name",
                        type=none_or_str,
                        default=None,
                        help="Print all the models with this substring in their name")

    args = parser.parse_args()

    print("=====TDW LIBRARIES====")

    if args.flex and not (args.model_category or args.model_name):
        print("=====FLEX-ENABLED MODELS=====")
        for lib, models in FLEX_MODELS.items():
            print("Library: %s" % lib)
            print("%d out of %d models are FLEX-enabled" % \
                  (len(models), len(MODEL_LIBRARIES[lib].records)))
            print([r.name for r in FLEX_MODELS[lib]])

    if args.material_type is not None:
        print("=====MATERIAL TYPES=====")
        print(MATERIAL_TYPES, len(MATERIAL_TYPES), "total materials = %d" % sum([len(MATERIAL_NAMES[mtype]) for mtype in MATERIAL_TYPES]))
        assert args.material_type in MATERIAL_TYPES, "Must pass a valid material type"


        print("=====MATERIAL NAMES OF TYPE %s======" % args.material_type)
        print(MATERIAL_NAMES[args.material_type], len(MATERIAL_NAMES[args.material_type]))

    if args.model_category is not None:
        print("=====MODEL CATEGORIES=====")
        print(MODEL_CATEGORIES, len(MODEL_CATEGORIES))
        assert args.model_category in MODEL_CATEGORIES, "Must pass a valid model category"

        print("=====%s MODELS OF CATEGORY %s=====" % \
              ("FLEX" if args.flex else "ALL", args.model_category))
        if args.flex:
            for lib, models in FLEX_MODELS.items():
                print("Library: %s" % lib)
                print([r.name for r in models \
                       if r.wcategory == args.model_category])
        else:
            for lib, _models in MODEL_LIBRARIES.items():
                print("Library: %s" % lib)
                models = _models.records
                print([r.name for r in models \
                       if r.wcategory == args.model_category])

    if args.model_name is not None:
        print("=====%s MODEL NAMES CONTAINING %s=====" %
              ("FLEX" if args.flex else "ALL", args.model_name))
        if args.flex:
            for lib, models in FLEX_MODELS.items():
                print("Library: %s" % lib)
                print([r.name for r in models \
                       if args.model_name in r.name])
        else:
            for lib, _models in MODEL_LIBRARIES.items():
                print("Library: %s" % lib)
                models = _models.records
                print([r.name for r in models \
                       if args.model_name in r.name])

    print("DO NOT USE")
    for lib, _models in MODEL_LIBRARIES.items():
        models = _models.records
        print([r.name for r in models \
               if r.do_not_use])


def cart2sphe(cart):
    """ Convert cartesian coordinates to spherical coordinates (Y-up) """
    hypot_xz = np.hypot(cart['x'], cart['z'])
    sphe = {
        'azimuth': np.arctan2(cart['z'], cart['x']),
        'elevation': np.arctan2(cart['y'], hypot_xz),
        'radius': np.hypot(hypot_xz, cart['y'])
    }
    return sphe

def sphe2cart(sphe):
    """ Convert spherical coordinates to cartesian coordinates (Y-up)"""
    hypot_xz = sphe['radius'] * np.cos(sphe['elevation'])
    cart = {
        'x': hypot_xz * np.cos(sphe['azimuth']),
        'z': hypot_xz * np.sin(sphe['azimuth']),
        'y': sphe['radius'] * np.sin(sphe['elevation']),
    }
    return cart