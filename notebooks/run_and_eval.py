import sys
import os
sys.path.insert(0, os.path.abspath("../lib/"))
sys.path.insert(1, '../conf/')

import json

from util import JSONExtendedEncoder

#%load_ext autoreload
#%autoreload 2

#%matplotlib inline
from matplotlib.pyplot import *

import pandas as pd
import numpy as np

import multiprocessing as mp

from itertools import product

import itertools
import shutil

import os

import paramexplo_blueprints



# datasets to run
datasetnames = [
    "ecoli_colombos",
    "ecoli_dream5",
    "yeast_gpl2529",
    "yeast_dream5",
    "synth_ecoli_regulondb",
    "synth_yeast_macisaac",
    "human_tcga",
    "human_gtex",
    "human_seek_gpl5175",
    "ecoli_precise2"
]

# choose the method to evaluate
# method_name = "dummy" # use the dummy method to check if everything works correctly
method_name = "agglom" # this method runs very fast, and has the best performance among clustering methods
# method_name = "ica_zscore" # this method runs very slow, but has approx. the highest performance in the benchmark
# method_name = "spectral_biclust" # top biclustering method
# method_name = "MAK"
# method_name = "meanshift"


#blueprints = os.system("python conf/paramexplo_blueprints.py")
#exec(open('conf/paramexplo_blueprints.py').read())
#execfile("conf/paramexplo_blueprints.py")
blueprints = paramexplo_blueprints.run_blueprints()
#print(blueprints)
methodblueprint = blueprints[method_name]
print(methodblueprint)

#%%

params_folder = "conf/paramexplo/" + method_name + "/"
if os.path.exists("../" + params_folder):
    shutil.rmtree("../" +  params_folder)
os.makedirs("../" + params_folder)

methodsettings = []
method_locations = []
i = 0
for dynparam_combination in list(itertools.product(*[methodblueprint["dynparams"][param] for param in sorted(methodblueprint["dynparams"].keys())])):
    method = {"params":{}}
    method["params"] = methodblueprint["staticparams"].copy()
    method["params"].update(dict(zip(sorted(methodblueprint["dynparams"].keys()), dynparam_combination)))
    method["location"] = "../" + params_folder + str(i) + ".json"
    method["seed"] = 0

    methodsettings.append(method)

    json.dump(method, open(  method["location"], "w"), cls=JSONExtendedEncoder)

    method_locations.append( method["location"])

    i+=1

settings_name = "paramexplo/{method_name}".format(method_name=method_name)
settings = []
for datasetname in datasetnames:
    for setting_ix, methodsetting in enumerate(methodsettings):
        settingid = datasetname + "_" + str(setting_ix)
        settings.append({
            "dataset_location": "conf/datasets/" + datasetname + ".json",
            "dataset_name": datasetname,
            "method_location": methodsetting["location"],
            "output_folder": "../results/" + methodblueprint["type"] + "/{settings_name}/{settingid}/".format(
                settings_name=settings_name, settingid=settingid),
            "settingid": settingid
        })
json.dump(settings, open("../conf/settings/{settings_name}.json".format(settings_name=settings_name), "w"))

# %%

settings_dataset = pd.DataFrame(
    [dict(settingid=setting["settingid"], **json.load(open("../" +setting["dataset_location"]))["params"]) for setting
     in settings])
settings_method = pd.DataFrame(
    [dict(settingid=setting["settingid"], **json.load(open(setting["method_location"]))["params"]) for setting
     in settings])

# %%

commands = ""
for i, setting in enumerate(settings):
    # commands += "python ../scripts/moduledetection.py {method_location} {dataset_location} {output_folder} 0 test\n".format(**setting)
    commands += "python3 ../scripts/" + methodblueprint[
        "type"] + ".py {method_location} ../{dataset_location} {output_folder}\n".format(**setting)

commands_location = "../tmp/{settings_name}.txt".format(**locals())
os.makedirs(  os.path.dirname(commands_location), exist_ok=True)
with open(  commands_location, "w") as outfile:
    outfile.write(commands)
commands_location = "../tmp/{settings_name}.txt".format(**locals())
os.makedirs(os.path.dirname(commands_location), exist_ok=True)
with open( commands_location, "w") as outfile:
    outfile.write(commands)

script_location = generate_batchcode(commands_location, settings_name, len(settings), {"memory":"10G", "numcores":1}, "biclust_comp2")

# this command can be used on most linux computers to run the different parameter settings in parallel
#print("parallel -j 4 -a " + commands_location)


from modulescomparison import ModevalKnownmodules, ModevalCoverage

if "pool" in locals().keys():
    pool.close()
pool = mp.Pool(mp.cpu_count()-1)

if "pool" in locals().keys():
    pool.close()
pool = mp.Pool(mp.cpu_count()-1)
# pool = mp.Pool(1)

#%%

settings_filtered = [setting for setting in settings if not setting["dataset_name"].startswith("human")] # only evaluate non-human datasets
modeval = ModevalKnownmodules(settings_filtered, baseline = True)

#%%

modeval.run(pool)
modeval.save(settings_name)

#%%

modeval.load(settings_name)

#%%

modeval.scores

modeval.scores.to_csv('modeval.scores.tsv', index=False, sep="\t")
