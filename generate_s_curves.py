import os
import socket
import glob
import h5py
import math
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

import collections

#folder = "/media/htung/Extreme SSD/fish/tdw_physics/dump/mass_dominoes_pp/mass_dominoes_num_middle_objects_3"
#folder="/home/hsiaoyut/2021/tdw_physics/dump/mass_dominoes_pp/mass_dominoes_num_middle_objects_0"

def split_info(filename):
    infos = filename.split("-")[1:]
    info_dict = dict()
    for info in infos:
        arg_name, arg_value = info.split("=")
        info_dict[arg_name] = arg_value
    return info_dict

if "ccncluster" in socket.gethostname():
    #data_root = "/mnt/fs4/hsiaoyut/physion++/analysis"
    data_root = "/mnt/fs4/hsiaoyut/physion++/data_v1"
else:
    data_root = "/media/htung/Extreme SSD/fish/tdw_physics/dump_mini4"

sname = "deform_clothhang_pp"
folder = os.path.join(data_root, sname)
#import ipdb; ipdb.set_trace()
filenames = os.listdir(folder)
restrict = "sphere" #deform_clothhang-zdloc=1" #mass_dominoes-num_middle_objects=0-star_putfirst=0-remove_middle=0" #pilot_dominoes_2distinct_1middle_tdwroom_fixedcam_curtain" #"pilot_it2_drop_simple_box" #"bouncy_platform-use_blocker_with_hole=1" #"target_cone-tscale_0.35,0.5,0.35"
remove = ""#"-is_single_ramp=1-zdloc=1" #"simple_box1"
filenames = [filename for filename in filenames if restrict in filename]

target_varname = "star_deform" #"star_mass" #"star_mass", "star_deform"
merge_by = "target" # "all" #"zld" #"all" #"num_middle_objects" #"all""tscale"
#merge_by = "tscale"

set_dict = collections.defaultdict(list)

for filename in filenames:
    info_dict = split_info(filename)
    if merge_by == "all":
        set_dict["all"].append(filename)
    elif merge_by:
        set_dict[info_dict[merge_by]].append(filename)
    else:
        set_dict[filename].append(filename)
import ipdb; ipdb.set_trace()

for set_id, merge_var_name in enumerate(set_dict):
    target_params = []
    labels = []

    for filename in tqdm(set_dict[merge_var_name]):
        if remove and remove in filename:
            continue
        #print(filename)
        pkl_filenames = [os.path.join(folder, filename, x) for x in os.listdir(os.path.join(folder, filename)) if x.endswith(".pkl")]

        for pkl_file in pkl_filenames: # glob.glob(os.path.join(folder, filename) + "/*.pkl"):
            print(pkl_file)
            with open(pkl_file, "rb") as f:
                f = pickle.load(f)

            #print(f['static']['cloth_material'])
            target_params.append(f["static"][target_varname])
            labels.append(float(f["static"]["does_target_contact_zone"]))
            #print(f["static"]["seed"], f["static"]["randomize"], f["static"]["trial_seed"], f["static"]["trial_num"])
            #import ipdb; ipdb.set_trace()

    print(labels)
    print(target_params)
    #import ipdb; ipdb.set_trace()
    """

    for hdf5_file in glob.glob(folder + "/*_001.hdf5"):
    	print(hdf5_file)

    	f = h5py.File(hdf5_file, "r")

    	target_params.append(f["static"][target_varname][()])
    	labels.append(float(f["static"]["does_target_contact_zone"][()]))
    """

    if target_varname in ["star_mass"]:
        target_params = [math.log10(param) for param in target_params]
    else:
        target_params = [param for param in target_params]

    #scatter plot
    #for oid, param in enumerate(target_params):
    #    if param > 0.5 and labels[oid] < 0.2:
    #        print("heavy object with negative outcome", oid)

    plt.scatter(target_params, labels)
    if target_varname in ["star_mass"]:
        plt.xlabel(f"log({target_varname})")
    else:
        plt.xlabel(f"{target_varname}")
    plt.ylabel("red hits yellow")
    ax = plt.gca()
    ax.set_ylim([-0.1, 1.6])


    nbins = 8
    n, _ = np.histogram(target_params, bins=nbins)
    sy, _ = np.histogram(target_params, bins=nbins, weights=labels)
    sy2, _ = np.histogram(target_params, bins=nbins, weights=labels)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, yerr=std, fmt='-', label=merge_var_name)

    #hist plot
#plt.legend(loc="lower left")
plt.legend(loc="upper left")
plt.show()

rstr, mstr = "", ""
if restrict != "":
    rstr = f"_r{restrict}"
if merge_by != "":
    mstr = f"_m{merge_by}"
plt.savefig(f"s_curve_{sname}{rstr}{mstr}.png")

print(len(labels))

#print(target_params)
