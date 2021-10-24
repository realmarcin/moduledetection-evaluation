import sys
import os
sys.path.insert(0,os.path.abspath("../lib/"))
sys.path.insert(1, os.path.abspath("../conf/"))

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.pyplot import subplots
import numpy as np
import pandas as pd
from itertools import product
import json


#from paramexplo_blueprints import methodparamsoi
from paramexplo_blueprints import *
#from importlib import reload # reload
#reload(paramexplo_blueprints)

from modulescomparison import ModevalKnownmodules, ModevalCoverage



# choose the method to evaluate
# method_name = "dummy" # use the dummy method to check if everything works correctly
method_name = "agglom" # this method runs very fast, and has the best performance among clustering methods
# method_name = "ica_zscore" # this method runs very slow, but has approx. the highest performance in the benchmark
# method_name = "spectral_biclust" # top biclustering method
# method_name = "MAK"
# method_name = "meanshift"

def score_method(scores):
    methodscores = []
    for ((datasetoi, goldstandardoi), scoresoi), ((datasetor, goldstandardor), scoresor) in product \
            (scores.groupby(["datasetname", "goldstandard"]), scores.groupby(["datasetname", "goldstandard"])):
        if (datasetor.split("_")[0 ] == "synth" and datasetoi.split("_")[0 ] != "synth") or \
                (datasetor.split("_")[0] != "synth" and datasetoi.split("_")[0] == "synth"):
            continue

        if (goldstandardoi.split("#")[-1] != goldstandardor.split("#")[-1]):
            if (datasetoi.startswith("human") != datasetor.startswith("human")):
                ""
            else:
                continue

        # find the most optimal method parameters in the reference dataset (test dataset)
        bestparams = scoresor[paramsoi].loc[scoresor["score"].idxmax()]

        try:
            rowids = scoresoi.index[
                np.where(np.all([scoresoi[param] == paramvalue for param, paramvalue in bestparams.items()], 0))[0]]
        except:
            print(scoresoi)

        # now find these parameters in the dataset of interest (training dataset)
        rowids = scoresoi.index[
            np.where(np.all([scoresoi[param] == paramvalue for param, paramvalue in bestparams.items()], 0))[0]]

        if len(rowids) == 0:
            print("parameters could not be matched!!", datasetoi, datasetor)
            print(bestparams)
            print([scoresoi[param] == paramvalue for param, paramvalue in bestparams.items()])
        if len(rowids) > 1:
            print(datasetoi)
            print("multiple matched parameters")
            print(scoresoi.loc[rowids][paramsoi])

        methodscores.append({
            "datasetoi": datasetoi,
            "datasetor": datasetor,
            "score": scoresoi.loc[rowids, "score"].max(),
            "method": methodname,
            "goldstandardoi": goldstandardoi,
            "goldstandardor": goldstandardor,
            "ofinterest": datasetoi + "#" + goldstandardoi,
            "ofreference": datasetor + "#" + goldstandardor,
            "runningtime": scoresoi.loc[rowids, "runningtime"].mean() if "runningtime" in scoresoi.columns else 0,
            "moduledef": scoresoi.loc[rowids, "moduledef"].tolist()[0],
            "organismoi": scoresoi.loc[rowids, "organism"].tolist()[0],
        })

    return pd.DataFrame(methodscores)


# %%

methodnames = ["dummy", "agglom", "ica_zscore", "spectral_biclust", "meanshift"]

# %%

finalscores = []
for methodname in methodnames:
    settings_name = "paramexplo/" + methodname
    path ="../conf/settings/{}.json".format(settings_name)
    print(path)
    settings = json.load(open(path))
    settings_dataset = pd.DataFrame(
        [dict(settingid=setting["settingid"], **json.load(open("../" + setting["dataset_location"]))["params"]) for
         setting in settings])
    settings_method = pd.DataFrame(
        [dict(settingid=setting["settingid"], **json.load(open( "../" + setting["method_location"]))["params"]) for
         setting in settings])

    print(methodname)
    paramsoi = methodparamsoi[methodname]

    scores = pd.DataFrame()

    modeval = ModevalKnownmodules(settings_name)
    modeval.load(settings_name)
    modeval.scores["score"] = modeval.scores["F1rprr_permuted"]
    modeval.scores["moduledef"] = [modulesname if modulesname in ["minimal", "strict"] else "interconnected" for
                                   modulesname in modeval.scores["knownmodules_name"]]
    modeval.scores = modeval.scores.merge(settings_dataset, on="settingid").merge(settings_method, on="settingid")
    scores = scores.append(modeval.scores, ignore_index=True)

    modeval = ModevalCoverage(settings_name)
    modeval.load(settings_name)
    modeval.scores["score"] = modeval.scores["aucodds_permuted"]
    modeval.scores = modeval.scores.merge(settings_dataset, on="settingid").merge(settings_method, on="settingid")
    scores = scores.append(modeval.scores, ignore_index=True)

    methodscores = score_method(scores)

    methodscores["organismnetoi"] = [dataset.split("_")[0] for dataset in methodscores["goldstandardoi"]]
    methodscores["organismnetor"] = [dataset.split("_")[0] for dataset in methodscores["goldstandardor"]]

    finalscores.append(methodscores)
finalscores = pd.concat(finalscores, ignore_index=True)

# %% md
#The final scores contains all the comparisons we made, together with a final score int he score column:
# %%

finalscores

# %%

finalscores.query("method == 'ica_zscore'")

# %% md
#We add weights to the test scores, because e.g. E. coli datasets will have many more test scores as there are more "reference" datasets available.
# %%

def add_weights(scores):
    weights = []
    scores["moduledef"] = scores["moduledef"].fillna("")
    for organismoi, subscores in scores.groupby("organismoi"):
        moduledef_weights = 1 / subscores.groupby("moduledef")["score"].count()
        for moduledef, weight in moduledef_weights.items():
            weights.append({
                "organism": organismoi,
                "moduledef": moduledef,
                "weight": weight / len(moduledef_weights)
            })
    weights = pd.DataFrame(weights).set_index(["organism", "moduledef"])["weight"]

    scores["weight"] = weights.loc[pd.Index(scores[["organismoi", "moduledef"]])].tolist()

    return scores


# %%

trainingscores_ = add_weights(finalscores.loc[(finalscores["ofinterest"] == finalscores["ofreference"])])
testscores_ = add_weights(finalscores.loc[(finalscores["ofinterest"] != finalscores["ofreference"]) & (
            finalscores["organismnetoi"] != finalscores["organismnetor"])])

# %% md

#Do a weighted mean:

# %%

trainingscores = trainingscores_.groupby("method").apply(lambda x: np.average(x.score, weights=x.weight))
testscores = testscores_.groupby("method").apply(lambda x: np.average(x.score, weights=x.weight))

# %%

testscores_.to_csv("../results/testscores_.tsv", sep="\t")
trainingscores_.to_csv("../results/trainingscores_.tsv", sep="\t")

# %%

trainingscores

# %%

testscores


#Visualization of overall training and test scores:
#%%

# A bar chart is actually not the ideal representation here, given that we're working with ratios.
# This way of plotting is kept here because it most closely resembles that of the paper.

fig, ax = subplots(figsize=(len(trainingscores)/2, 5))

methodorder = testscores.sort_values(ascending=False).index

ax.axhline(1, color = "black")
ax.bar(range(len(methodorder)), trainingscores[methodorder], color="grey")
ax.bar(range(len(methodorder)), testscores[methodorder], color="#333333")
ax.set_xticks(np.arange(len(methodorder)))
ax.set_xticklabels(methodorder, rotation=45, ha="right", va="top")
ax.tick_params(labelsize=14)

fig.savefig('modeval_barchart.png')

#%%

# A better way to visualize the data would be dotplot

fig, ax = subplots(figsize=(5, len(trainingscores)/2))

methodorder = testscores.sort_values(ascending=True).index

ax.axvline(1, color = "black")
for y, method in enumerate(methodorder):
    ax.plot([trainingscores[method], testscores[method]], [y, y], zorder = 0, color = "grey")
ax.scatter(trainingscores[methodorder], range(len(methodorder)), color="grey", s = 20)
ax.scatter(testscores[methodorder], range(len(methodorder)), color="#333333", s = 100)
ax.set_yticks(np.arange(len(methodorder)))
ax.set_yticklabels(methodorder)
ax.tick_params(labelsize=14)
ax.set_xlim([0, ax.get_xlim()[1]])

fig.savefig('modeval_dotplot.png')