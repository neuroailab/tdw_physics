import random
from pathlib import Path
import pkg_resources
import io
import json
from tdw.tdw_utils import TDWUtils
from tdw.output_data import OutputData, Rigidbodies, Collision, EnvironmentCollision, StaticRigidbodies
from typing import List, Tuple, Dict, Optional
import tdw_physics
from tdw.librarian import ModelRecord
from abc import ABC
import h5py
import numpy as np
from tdw_physics.controller import Controller
from tdw.output_data import OutputData, Rigidbodies, Collision, EnvironmentCollision
from tdw.tdw_utils import TDWUtils
from tdw_physics.transforms_dataset import TransformsDataset
from tdw_physics.dataset import Dataset
from tdw_physics.physics_info import PhysicsInfo, PHYSICS_INFO
from tdw_physics.util import MODEL_LIBRARIES, str_to_xyz, xyz_to_arr, arr_to_xyz

def handle_random_transform_args(args):

    if args is not None:
        if type(args) in [float,int,bool,list,dict]: # don't apply any parsing to simple datatypes
            return args
        try:
            args = json.loads(args)
        except:
            try: args = eval(args) #this allows us to read dictionaries etc.
            except: args = str_to_xyz(args)

        if hasattr(args, 'keys'):
            if 'class' in args:
                data = args['data']
                modname, classname = args['class']
                mod = importlib.import_module(modname)
                klass = get_attr(mod, classname)
                args = klass(data)
                assert callable(args)
            else:
                assert "x" in args, args
                assert "y" in args, args
                assert "z" in args, args
        elif hasattr(args, '__len__'):
            if len(args) == 3:
                args = {k:args[i] for i, k in enumerate(["x","y","z"])}
            else:
                assert len(args) == 2, (args, len(args))
        else:
            args + 0.0
    return args



def get_range(scl):
    if scl is None:
        return None, None
    elif hasattr(scl, '__len__'):
        return scl[0], scl[1]
    else:
        scl + 0.0
        return scl, scl


def get_random_xyz_transform(generator):
    if callable(generator):
        s = generator()
    elif hasattr(generator, 'keys'):
        sx0, sx1 = get_range(generator["x"])
        sx = random.uniform(sx0, sx1)
        sy0, sy1 = get_range(generator["y"])
        sy = random.uniform(sy0, sy1)
        sz0, sz1 = get_range(generator["z"])
        sz = random.uniform(sz0, sz1)
        s = {"x": sx, "y": sy, "z": sz}
    elif hasattr(generator, '__len__'):
        s0 = random.uniform(generator[0], generator[1])
        s = {"x": s0, "y": s0, "z": s0}
    else:
        generator + 0.0
        s = {"x": generator, "y": generator, "z": generator}
    assert hasattr(s, 'keys'), s
    assert 'x' in s, s
    assert 'y' in s, s
    assert 'z' in s, s
    return s


class RigidbodiesDataset(TransformsDataset, ABC):
    """
    A dataset for Rigidbody (PhysX) physics.
    """


    # Static physics data.
    MASSES: np.array = np.empty(dtype=np.float32, shape=0)
    STATIC_FRICTIONS: np.array = np.empty(dtype=np.float32, shape=0)
    DYNAMIC_FRICTIONS: np.array = np.empty(dtype=np.float32, shape=0)
    BOUNCINESSES: np.array = np.empty(dtype=np.float32, shape=0)
    # The physics info of each object instance. Useful for referencing in a controller, but not written to disk.
    PHYSICS_INFO: Dict[int, PhysicsInfo] = dict()

    def __init__(self, port: int = 1071, monochrome: bool = False, **kwargs):

        TransformsDataset.__init__(self, port=port, **kwargs)

        # Whether the objects will be set to the same color
        self.monochrome = monochrome

    def clear_static_data(self) -> None:
        super().clear_static_data()

        RigidbodiesDataset.MASSES = np.empty(dtype=np.float32, shape=0)
        RigidbodiesDataset.STATIC_FRICTIONS = np.empty(dtype=np.float32, shape=0)
        RigidbodiesDataset.DYNAMIC_FRICTIONS = np.empty(dtype=np.float32, shape=0)
        RigidbodiesDataset.BOUNCINESSES = np.empty(dtype=np.float32, shape=0)
        self.colors = np.empty(dtype=np.float32, shape=(0,3))
        self.scales = []

    def _xyz_to_arr(self, xyz: dict):
        arr = np.array(
            [xyz[k] for k in ["x", "y", "z"]], dtype=np.float32)
        return arr

    def _arr_to_xyz(self, arr: np.ndarray):
        xyz = {k: arr[i] for i, k in enumerate(["x", "y", "z"])}
        return xyz

    def random_color(self, exclude=None, exclude_range=0.33):
        rgb = [random.random(), random.random(), random.random()]
        if exclude is None:
            return rgb

        assert len(exclude) == 3, exclude
        while any([np.abs(exclude[i] - rgb[i]) < exclude_range for i in range(3)]):
            rgb = [random.random(), random.random(), random.random()]

        return rgb

    def random_color_from_rng(self, exclude=None, exclude_range=0.33, seed=0):

        rng = np.random.RandomState(seed)
        rgb = [rng.random(), rng.random(), rng.random()]

        if exclude is None:
            return rgb
        assert len(exclude) == 3, exclude
        while any([np.abs(exclude[i] - rgb[i]) < exclude_range for i in range(3)]):
            rgb = [rng.random(), rng.random(), rng.random()]

        return rgb

    def get_random_scale_transform(self, scale):
        return get_random_xyz_transform(scale)

    def _add_name_scale(self, record, data) -> None:
        self.model_names.append(record.name)
        self.scales.append(data['scale'])
        # self.colors = np.concatenate([self.colors, np.array(data['color']).reshape((1, 3))], axis=0)

    def _add_name_scale_color(self, record, data) -> None:
        self.model_names.append(record.name)
        self.scales.append(data['scale'])
        self.colors = np.concatenate([self.colors, np.array(data['color']).reshape((1, 3))], axis=0)

    def random_primitive(self,
                         object_types: List[ModelRecord],
                         scale: List[float] = [0.2, 0.3],
                         color: List[float] = None,
                         exclude_color: List[float] = None,
                         exclude_range: float = 0.25,
                         add_data: bool = True,
                         random_obj_id: bool = False,
                         xz_same_scale=False
    ) -> dict:
        obj_record = random.choice(object_types)
        s = self.get_random_scale_transform(scale)
        if xz_same_scale:
            s['x'] = s['z']

        obj_data = {
            "id": self.get_unique_id() if random_obj_id else self._get_next_object_id(),
            "scale": s,
            "color": np.array(color if color is not None else self.random_color(exclude_color, exclude_range)),
            "name": obj_record.name
        }

        if add_data:
            self._add_name_scale_color(obj_record, obj_data)
            # self.model_names.append(obj_data["name"])
            # self.scales.append(obj_data["scale"])
            # self.colors = np.concatenate([self.colors, obj_data["color"].reshape((1,3))], axis=0)
        return obj_record, obj_data

    #For backward compatibility with old version.
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
                           ) -> List[dict]:

        if o_id is None:
            o_id: int = self.get_unique_id()

        # breakpoint()

        lib = PHYSICS_INFO[record.name].library if record.name in PHYSICS_INFO.keys() else "models_flex.json"


        if add_data:
            self.add_transforms_data(position, rotation)

            # # if add_data:
            # data = {'name': record.name, 'id': o_id,
            #         'scale': scale,
            #         'mass': mass,
            #         'dynamic_friction': dynamic_friction,
            #         'static_friction': static_friction,
            #         'bounciness': bounciness}
            # self._add_name_scale(record, data)
            # obj_list.append((record, data))

        return self.get_add_physics_object(model_name=record.name,
                                           object_id=o_id,
                                           position=position,
                                           rotation=rotation,
                                           mass=mass,
                                           scale_factor=scale,
                                           library=lib,
                                           dynamic_friction=dynamic_friction,
                                           static_friction=static_friction,
                                           bounciness=bounciness,
                                           default_physics_values=default_physics_values, density=density)

    #TODO: in the controllers, replace add_physics_object with get_add_physics_object
    @staticmethod
    def get_add_physics_object(model_name: str,
                               object_id: int,
                               position: Dict[str, float] = None,
                               rotation: Dict[str, float] = None,
                               library: str = "",
                               scale_factor: Dict[str, float] = None,
                               kinematic: bool = False,
                               gravity: bool = True,
                               default_physics_values: bool = True,
                               mass: float = 1, dynamic_friction: float = 0.3,
                               static_friction: float = 0.3,
                               bounciness: float = 0.7,
                               scale_mass: bool = True,
                               density: float = 5) -> List[dict]:
        """
        Add an object to the scene with physics values (mass, friction coefficients, etc.).

        :param model_name: The name of the model.
        :param position: The position of the model. If None, defaults to `{"x": 0, "y": 0, "z": 0}`.
        :param rotation: The starting rotation of the model, in Euler angles. If None, defaults to `{"x": 0, "y": 0, "z": 0}`.
        :param library: The path to the records file. If left empty, the default library will be selected. See `ModelLibrarian.get_library_filenames()` and `ModelLibrarian.get_default_library()`.
        :param object_id: The ID of the new object.
        :param scale_factor: The [scale factor](../api/command_api.md#scale_object).
        :param kinematic: If True, the object will be [kinematic](../api/command_api.md#set_kinematic_state).
        :param gravity: If True, the object won't respond to [gravity](../api/command_api.md#set_kinematic_state).
        :param default_physics_values: If True, use default physics values. Not all objects have default physics values. To determine if object does: `has_default_physics_values = model_name in DEFAULT_OBJECT_AUDIO_STATIC_DATA`.
        :param mass: The mass of the object. Ignored if `default_physics_values == True`.
        :param dynamic_friction: The [dynamic friction](../api/command_api.md#set_physic_material) of the object. Ignored if `default_physics_values == True`.
        :param static_friction: The [static friction](../api/command_api.md#set_physic_material) of the object. Ignored if `default_physics_values == True`.
        :param bounciness: The [bounciness](../api/command_api.md#set_physic_material) of the object. Ignored if `default_physics_values == True`.
        :param scale_mass: If True, the mass of the object will be scaled proportionally to the spatial scale.

        :return: A list of commands to add the object and apply physics values.
        """

        # Use override physics values.
        #TODO: did you mean this the other way around?? if default_physics_values and model_name in PHYSICS_INFO: default_physics_values = False
        # if default_physics_values and model_name in PHYSICS_INFO:
        #
        #     default_physics_values = False
        #     mass = PHYSICS_INFO[model_name].mass
        #     dynamic_friction = PHYSICS_INFO[model_name].dynamic_friction
        #     static_friction = PHYSICS_INFO[model_name].static_friction
        #     bounciness = PHYSICS_INFO[model_name].bounciness



        # record = Controller.MODEL_LIBRARIANS[library].get_record(model_name)

        # breakpoint()

        commands = TransformsDataset.get_add_physics_object(model_name=model_name,
                                                            object_id=object_id,
                                                            position=position,
                                                            rotation=rotation,
                                                            library=library,
                                                            scale_factor=scale_factor,
                                                            kinematic=kinematic,
                                                            gravity=True,
                                                            default_physics_values=default_physics_values,
                                                            mass=mass,
                                                            dynamic_friction=dynamic_friction,
                                                            static_friction=static_friction,
                                                            bounciness=bounciness,
                                                            scale_mass=scale_mass,
                                                            density=density)
        # Log the object ID.
        Dataset.OBJECT_IDS = np.append(Dataset.OBJECT_IDS, object_id)
        # Get the static data from the commands (these values might be automatically set).

        #TODO: get updated mass (in case it is updated and set it here.)
        for command in commands:
            if command["$type"] == "set_mass":
                mass = command["mass"]
            elif command["$type"] == "set_physic_material":
                dynamic_friction = command["dynamic_friction"]
                static_friction = command["static_friction"]
                bounciness = command["bounciness"]
        # Cache the static data.
        RigidbodiesDataset.MASSES = np.append(RigidbodiesDataset.MASSES, mass)
        RigidbodiesDataset.DYNAMIC_FRICTIONS = np.append(RigidbodiesDataset.DYNAMIC_FRICTIONS, dynamic_friction)
        RigidbodiesDataset.STATIC_FRICTIONS = np.append(RigidbodiesDataset.STATIC_FRICTIONS, static_friction)
        RigidbodiesDataset.BOUNCINESSES = np.append(RigidbodiesDataset.BOUNCINESSES, bounciness)
        # Cache the physics info.
        record = Controller.MODEL_LIBRARIANS[library].get_record(model_name)
        lib = PHYSICS_INFO[record.name].library if record.name in PHYSICS_INFO.keys() else "models_flex.json"
        RigidbodiesDataset.PHYSICS_INFO[object_id] = PhysicsInfo(record=record,
                                                                 mass=mass,
                                                                 dynamic_friction=dynamic_friction,
                                                                 static_friction=static_friction,
                                                                 bounciness=bounciness,
                                                                 library=lib)


        return commands, mass

    #TODO: this need not be here. can put into some utlils.
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
                      ) -> List[dict]:

        cmds = []

        if o_id is None:
            o_id = self._get_next_object_id()


        # breakpoint()


        lib = PHYSICS_INFO[record.name].library if record.name in PHYSICS_INFO.keys() else "models_flex.json"

        # print("default_physics_values", default_physics_values)

        # default_physics_values = True

        # breakpoint()

        if add_data:
            self.add_transforms_data(position, rotation)

        commands, mass = self.get_add_physics_object(model_name=record.name,
                                                 object_id=o_id,
                                                 position=position,
                                                 rotation=rotation,
                                                 library=lib,
                                                 scale_factor=scale,
                                                 kinematic=make_kinematic,
                                                 gravity=True,
                                                 default_physics_values=default_physics_values,
                                                 mass=mass,
                                                 dynamic_friction=dynamic_friction,
                                                 static_friction=static_friction,
                                                 bounciness=bounciness,
                                                 scale_mass=True,
                                                 density=density)
        # add the physics stuff
        cmds.extend(commands)

        if scale_mass:
            # print("mass", mass)
            mass = mass * np.prod(xyz_to_arr(scale))
            # print("mass", mass)

        # set the material and color
        if apply_texture:
            cmds.extend(
                self.get_object_material_commands(
                    record, o_id, self.get_material_name(material)))

            color = color if color is not None else self.random_color(exclude=exclude_color)
            cmds.append(
                {"$type": "set_color",
                 "color": {"r": color[0], "g": color[1], "b": color[2], "a": 1.},
                 "id": o_id})

        if make_kinematic:
            cmds.extend([
                {"$type": "set_object_collision_detection_mode",
                 "mode": "continuous_speculative",
                 "id": o_id},
                {"$type": "set_kinematic_state",
                 "id": o_id,
                 "is_kinematic": True,
                 "use_gravity": True}])

        if add_data:
            data = {'name': record.name, 'id': o_id,
                    'scale': scale, 'color': color, 'material': material,
                    'mass': mass,
                    'dynamic_friction': dynamic_friction,
                    'static_friction': static_friction,
                    'bounciness': bounciness}
            self._add_name_scale_color(record, data)
            obj_list.append((record, data))

        return cmds, mass

    def add_ramp(self,
                 record: ModelRecord,
                 position: Dict[str, float] = TDWUtils.VECTOR3_ZERO,
                 rotation: Dict[str, float] = TDWUtils.VECTOR3_ZERO,
                 scale: Dict[str, float] = {"x": 1., "y": 1., "z": 1},
                 o_id: Optional[int] = None,
                 material: Optional[str] = None,
                 color: Optional[list] = None,
                 mass: Optional[float] = None,
                 dynamic_friction: Optional[float] = None,
                 static_friction: Optional[float] = None,
                 bounciness: Optional[float] = None,
                 add_data: Optional[bool] = True
                 ) -> List[dict]:

        # get a named ramp or choose a random one
        ramp_records = {r.name: r for r in MODEL_LIBRARIES['models_full.json'].records \
                        if 'ramp' in r.name}
        if record.name not in ramp_records.keys():
            record = ramp_records[random.choice(sorted(ramp_records.keys()))]

        cmds = []

        # add the ramp
        # info = PHYSICS_INFO[record.name]
        info = {
            "bounciness": 0,
            "dynamic_friction": 0.01,
            "static_friction": 0.01,
            "mass": 500,
        }
        cds, _ = self.add_physics_object(
                record = record,
                position = position,
                rotation = rotation,
                mass = mass or info["mass"],
                dynamic_friction = dynamic_friction or info["dynamic_friction"],
                static_friction = static_friction or info["static_friction"],
                bounciness = bounciness or info["bounciness"],
                o_id = o_id, default_physics_values=False,
                add_data = add_data, scale=scale)
        cmds.extend(cds)

        if o_id is None:
            o_id = cmds[-self.num_sim]["id"]

        # # scale the ramp
        # cmds.append(
        #     {"$type": "scale_object",
        #      "scale_factor": scale,
        #      "id": o_id})

        for i in range(self.num_sim):
            cmds.extend(
                self.get_object_material_commands(
                    record, o_id+i*self.interval, self.get_material_name(material)))
            cmds.extend([
                {"$type": "set_color",
                "color": {"r": color[0], "g": color[1], "b": color[2], "a": 1.},
                "id": o_id+i*self.interval},
                {"$type": "set_object_collision_detection_mode",
                "mode": "continuous_speculative",
                "id": o_id+i*self.interval},
                {"$type": "set_kinematic_state",
                "id": o_id+i*self.interval,
                "is_kinematic": True,
                "use_gravity": True}])
            
        # # texture and color it
        # cmds.extend(
        #     self.get_object_material_commands(
        #         record, o_id, self.get_material_name(material)))

        # cmds.append(
        #     {"$type": "set_color",
        #      "color": {"r": color[0], "g": color[1], "b": color[2], "a": 1.},
        #      "id": o_id})

        # # need to make ramp a kinetimatic object
        # cmds.extend([
        #     {"$type": "set_object_collision_detection_mode",
        #      "mode": "continuous_speculative",
        #      "id": o_id},
        #     {"$type": "set_kinematic_state",
        #      "id": o_id,
        #      "is_kinematic": True,
            #  "use_gravity": True}])

        if add_data:
            self._add_name_scale_color(record, {'color': color, 'scale': scale})
            # self.model_names.append(record.name)
            # self.colors = np.concatenate([self.colors, np.array(color).reshape((1,3))], axis=0)
            # self.scales.append(scale)

        return cmds

    # def trial(self, filepath: Path, temp_path: Path, trial_num: int) -> None:
    #     # Clear data.
    #     RigidbodiesDataset.MASSES = np.empty(dtype=int, shape=0)
    #     RigidbodiesDataset.DYNAMIC_FRICTIONS = np.empty(dtype=int, shape=0)
    #     RigidbodiesDataset.STATIC_FRICTIONS = np.empty(dtype=int, shape=0)
    #     RigidbodiesDataset.BOUNCINESSES = np.empty(dtype=int, shape=0)
    #     super().trial(filepath=filepath, temp_path=temp_path, trial_num=trial_num)

    @staticmethod
    def get_objects_by_mass(mass: float) -> List[int]:
        """
        :param mass: The mass threshold.

        :return: A list of object IDs for objects with mass <= the mass threshold.
        """

        return [o for o in RigidbodiesDataset.PHYSICS_INFO.keys() if RigidbodiesDataset.PHYSICS_INFO[o].mass < mass]

    def get_falling_commands(self, mass: float = 3) -> List[List[dict]]:
        """
        :param mass: Objects with <= this mass might receive a force.

        :return: A list of lists; per-frame commands to make small objects fly up.
        """

        per_frame_commands: List[List[dict]] = []

        # Get a list of all small objects.
        small_ids = self.get_objects_by_mass(mass)
        random.shuffle(small_ids)
        max_num_objects = len(small_ids) if len(small_ids) < 8 else 8
        min_num_objects = max_num_objects - 3
        if min_num_objects <= 0:
            min_num_objects = 1
        # Add some objects.
        for i in range(random.randint(min_num_objects, max_num_objects)):
            o_id = small_ids.pop(0)
            force_dir = np.array([random.uniform(-0.125, 0.125), random.uniform(0.7, 1), random.uniform(-0.125, 0.125)])
            force_dir = force_dir / np.linalg.norm(force_dir)
            min_force = RigidbodiesDataset.PHYSICS_INFO[o_id].mass * 2
            max_force = RigidbodiesDataset.PHYSICS_INFO[o_id].mass * 4
            force = TDWUtils.array_to_vector3(force_dir * random.uniform(min_force, max_force))
            per_frame_commands.append([{"$type": "apply_force_to_object",
                                        "force": force,
                                        "id": o_id}])
            # Wait some frames.
            for j in range(10, 30):
                per_frame_commands.append([])
        return per_frame_commands

    def _get_send_data_commands(self) -> List[dict]:
        commands = super()._get_send_data_commands()
        commands.extend([{"$type": "send_collisions",
                          "enter": True,
                          "exit": True,
                          "stay": True,
                          "collision_types": ["obj", "env"]},
                         {"$type": "send_static_rigidbodies",
                          "frequency": "always"},
                         {"$type": "send_rigidbodies",
                          "frequency": "always"}])

        if self.save_meshes:
            commands.append({"$type": "send_meshes", "frequency": "once"})

        return commands

    def _write_static_data(self, static_group: h5py.Group) -> None:
        super()._write_static_data(static_group)

        static_group.create_dataset("mass", data=RigidbodiesDataset.MASSES)
        static_group.create_dataset("static_friction", data=RigidbodiesDataset.STATIC_FRICTIONS)
        static_group.create_dataset("dynamic_friction", data=RigidbodiesDataset.DYNAMIC_FRICTIONS)
        static_group.create_dataset("bounciness", data=RigidbodiesDataset.BOUNCINESSES)

        ## size and colors
        static_group.create_dataset("color", data=self.colors)
        static_group.create_dataset("scale", data=np.stack([xyz_to_arr(_s) for _s in self.scales], 0))
        static_group.create_dataset("scale_x", data=[_s["x"] for _s in self.scales])
        static_group.create_dataset("scale_y", data=[_s["y"] for _s in self.scales])
        static_group.create_dataset("scale_z", data=[_s["z"] for _s in self.scales])

        if self.save_meshes:
            mesh_group = static_group.create_group("mesh")

            obj_points = []
            # breakpoint()
            for idx, object_id in enumerate(Dataset.OBJECT_IDS):
                vertices, faces = self.object_meshes[object_id]
                mesh_group.create_dataset(f"faces_{idx}", data=faces)
                mesh_group.create_dataset(f"vertices_{idx}", data=vertices)

    def _write_frame(self, frames_grp: h5py.Group, resp: List[bytes], frame_num: int, view_num: int) -> \
            Tuple[h5py.Group, h5py.Group, dict, bool]:
        frame, objs, tr, done = super()._write_frame(frames_grp=frames_grp, resp=resp, frame_num=frame_num, view_num=view_num)
        num_objects = len(Dataset.OBJECT_IDS)
        # Physics data.
        velocities = np.empty(dtype=np.float32, shape=(num_objects, 3))

        masses = np.empty(dtype=np.float32, shape=(num_objects, ))
        sfrict = np.empty(dtype=np.float32, shape=(num_objects,))
        dfrict = np.empty(dtype=np.float32, shape=(num_objects,))
        bness = np.empty(dtype=np.float32, shape=(num_objects,))

        angular_velocities = np.empty(dtype=np.float32, shape=(num_objects, 3))
        # Collision data.
        collision_ids = np.empty(dtype=np.int32, shape=(0, 2))
        collision_relative_velocities = np.empty(dtype=np.float32, shape=(0, 3))
        collision_contacts = np.empty(dtype=np.float32, shape=(0, 2, 3))
        collision_states = np.empty(dtype=str, shape=(0, 1))
        # Environment Collision data.
        env_collision_ids = np.empty(dtype=np.int32, shape=(0, 1))
        env_collision_contacts = np.empty(dtype=np.float32, shape=(0, 2, 3))

        sleeping = True

        for r in resp[:-1]:
            r_id = OutputData.get_data_type_id(r)
            if r_id == "rigi":
                ri = Rigidbodies(r)
                ri_dict = dict()
                for i in range(ri.get_num()):
                    ri_dict.update({ri.get_id(i): {"vel": ri.get_velocity(i),
                                                   "ang": ri.get_angular_velocity(i)}})
                    # Check if any objects are sleeping that aren't in the abyss.
                    if not ri.get_sleeping(i) and tr[ri.get_id(i)]["pos"][1] >= -1:
                        sleeping = False
                # Add the Rigibodies data.
                for o_id, i in zip(Dataset.OBJECT_IDS, range(num_objects)):
                    try:
                        velocities[i] = ri_dict[o_id]["vel"]
                        angular_velocities[i] = ri_dict[o_id]["ang"]
                    except KeyError:
                        print("Couldn't store velocity data for object %d" % o_id)
                        print("frame num", frame_num)
                        print("ri_dict", ri_dict)
                        print([OutputData.get_data_type_id(r) for r in resp])

            elif r_id == "srig":
                srig = StaticRigidbodies(r)
                ri_dict = dict()
                for j in range(srig.get_num()):

                    mass = srig.get_mass(j)
                    dynamic_friction = srig.get_dynamic_friction(j)
                    static_friction = srig.get_static_friction(j)
                    bounciness = srig.get_bounciness(j)

                    ri_dict.update({srig.get_id(j): {"mass": mass,
                                                   "dfrict":dynamic_friction,
                                                   "sfrict":static_friction,
                                                   "bness":bounciness}})
                for o_id, i in zip(Dataset.OBJECT_IDS, range(num_objects)):
                    masses[i] = ri_dict[o_id]["mass"]
                    dfrict[i] = ri_dict[o_id]["dfrict"]
                    sfrict[i] = ri_dict[o_id]["sfrict"]
                    bness[i] = ri_dict[o_id]["bness"]

            elif r_id == "coll":
                co = Collision(r)
                collision_states = np.append(collision_states, co.get_state())
                collision_ids = np.append(collision_ids, [co.get_collider_id(), co.get_collidee_id()])
                collision_relative_velocities = np.append(collision_relative_velocities, co.get_relative_velocity())
                for i in range(co.get_num_contacts()):
                    collision_contacts = np.append(collision_contacts, (co.get_contact_normal(i),
                                                                        co.get_contact_point(i)))
            elif r_id == "enco":
                en = EnvironmentCollision(r)
                env_collision_ids = np.append(env_collision_ids, en.get_object_id())
                for i in range(en.get_num_contacts()):
                    env_collision_contacts = np.append(env_collision_contacts, (en.get_contact_normal(i),
                                                                                en.get_contact_point(i)))

        if not 'velocities' in objs.keys():

            objs.create_dataset("masses", data=masses, compression="gzip")
            objs.create_dataset("dfrict", data=dfrict, compression="gzip")
            objs.create_dataset("sfrict", data=sfrict, compression="gzip")
            objs.create_dataset("bness", data=bness, compression="gzip")

            objs.create_dataset("velocities", data=velocities.reshape(num_objects, 3), compression="gzip")
            objs.create_dataset("angular_velocities", data=angular_velocities.reshape(num_objects, 3), compression="gzip")
            collisions = frame.create_group("collisions")
            collisions.create_dataset("object_ids", data=collision_ids.reshape((-1, 2)), compression="gzip")
            collisions.create_dataset("relative_velocities", data=collision_relative_velocities.reshape((-1, 3)),
                                      compression="gzip")
            collisions.create_dataset("contacts", data=collision_contacts.reshape((-1, 2, 3)), compression="gzip")
            collisions.create_dataset("states", data=collision_states.astype('S'), compression="gzip")
            env_collisions = frame.create_group("env_collisions")
            env_collisions.create_dataset("object_ids", data=env_collision_ids, compression="gzip")
            env_collisions.create_dataset("contacts", data=env_collision_contacts.reshape((-1, 2, 3)),
                                          compression="gzip")
        return frame, objs, tr, sleeping

    def get_object_target_collision(self, obj_id: int, target_id: int, resp: List[bytes]):

        contact_points = []
        contact_normals = []

        for r in resp[:-1]:
            r_id = OutputData.get_data_type_id(r)
            if r_id == "coll":
                co = Collision(r)
                coll_ids = [co.get_collider_id(), co.get_collidee_id()]
                if [obj_id, target_id] == coll_ids or [target_id, obj_id] == coll_ids:
                    contact_points = [co.get_contact_point(i) for i in range(co.get_num_contacts())]
                    contact_normals = [co.get_contact_normal(i) for i in range(co.get_num_contacts())]

        return (contact_points, contact_normals)

    def get_object_environment_collision(self, obj_id: int, resp: List[bytes]):

        contact_points = []
        contact_normals = []

        for r in resp[:-1]:
            r_id = OutputData.get_data_type_id(r)
            if r_id == 'enco':
                en = EnvironmentCollision(r)
                if en.get_object_id() == obj_id:
                    contact_points = [np.array(en.get_contact_point(i)) for i in range(en.get_num_contacts())]
                    contact_normals = [np.array(en.get_contact_normal(i)) for i in range(en.get_num_contacts())]

        return (contact_points, contact_normals)
