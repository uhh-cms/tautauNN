# coding: utf-8

from __future__ import annotations

import os
from typing import Any

import tensorflow as tf
import cmsml

from util import get_device

# model save dir
this_dir: str = os.path.dirname(os.path.abspath(__file__))
model_dir: str = os.getenv("TN_MODEL_DIR", os.path.join(this_dir, "models"))

# model name to load
model_name: str = "hbtres_ED5_LU5x125_CTfcn_ACTelu_BNy_LT50_DO0_BS4096_LR3.0e-03_SPINy_MASSy_FI{fold}_SD{seed}"

# where to save exported models
export_dir: str = os.getenv("TN_MODEL_DIR", os.path.join(this_dir, "models"))

# number of expected input features (including mass and spin as last features if those are parametrized)
n_cont_inputs: int = 55
n_cat_inputs: int = 6

# number of classes for classification
n_classes: int = 3

# folds and seeds trained per fold
fold_seeds: dict[int, list[int]] = {
    0: [1],
}
# all folds must have the same number of seeds
n_seeds: set[int] = set(map(len, fold_seeds.values()))
assert len(n_seeds) == 1
n_seeds: int = n_seeds.pop()


# ensemble layer for simpler export
class Ensemble(tf.keras.layers.Layer):

    def __init__(
        self,
        model_dir: str,
        model_name: str,
        fold: int,
        seeds: list[int],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.model_dir: str = model_dir
        self.model_name: str = model_name
        self.fold: int = fold
        self.seeds: list[int] = seeds

        self.models: list[tf.kers.Model] = []

    def load_model(self, seed: int) -> tf.keras.Model:
        path = os.path.join(self.model_dir, self.model_name.format(fold=self.fold, seed=seed))
        return tf.saved_model.load(path)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()

        config["model_dir"] = self.model_dir
        config["model_name"] = self.model_name
        config["fold"] = self.fold
        config["seeds"] = self.seeds
        config["ensembling_mode"] = self.ensembling_mode

        return config

    def build(self, input_shapes):
        # load models
        self.models.clear()
        for seed in self.seeds:
            model = self.load_model(seed)
            self.models.append(model)

        super().build(input_shapes)

    def call(self, x: list[tf.Tensor], training: bool = False) -> tf.Tensor:
        # unpack inputs
        cont_inputs, cat_inputs = x

        # call models
        output = tf.concat(
            [
                model([cont_inputs, cat_inputs], training=False)[:, None, :]
                for model in self.models
            ],
            axis=1,
        )

        # ensembling (via mixture of experts)
        output = tf.reduce_mean(output, axis=1)

        return output


# device
cpu = get_device("cpu")

# build and save ensembles for all folds
for fold, seeds in fold_seeds.items():
    with cpu:
        # build the model
        x_cont = tf.keras.Input(n_cont_inputs, dtype=tf.float32, name="cont_input")
        x_cat = tf.keras.Input(n_cat_inputs, dtype=tf.int32, name="cat_input")
        y = Ensemble(model_dir, model_name, fold, seeds, name="hbt_ensemble")([x_cont, x_cat])
        ensemble_model = tf.keras.Model(inputs=[x_cont, x_cat], outputs=y)

        # test it
        cont_inputs = tf.constant([1.0] * n_cont_inputs, dtype=tf.float32)[None, ...]
        cat_inputs = tf.constant([0, -1, 0, -1, -1, 2], dtype=tf.int32)[None, ...]
        print(f"fold {fold} -> {ensemble_model([cont_inputs, cat_inputs], training=False)}")

    # save it
    save_name = model_name.format(fold=fold, seed=0).replace("_SD0", "") + "_moe"
    save_path = os.path.join(model_dir, save_name)
    cmsml.tensorflow.save_frozen_graph(save_path, ensemble_model, variables_to_constants=True)
    print(f"saved ensemble model at {save_path}")
