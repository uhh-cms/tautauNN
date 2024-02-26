# coding: utf-8

from __future__ import annotations

import os
from typing import Any

import tensorflow as tf
import cmsml

from tautaunn.tf_util import get_device


class Ensemble(tf.keras.layers.Layer):

    def __init__(self, model_dirs: list[str], **kwargs) -> None:
        super().__init__(**kwargs)

        self.model_dirs: str = model_dirs
        self.models: list[tf.kers.Model] = []

    def load_model(self, model_dir: str) -> tf.keras.Model:
        return tf.saved_model.load(model_dir)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()

        config["model_dirs"] = self.model_dirs

        return config

    def build(self, input_shapes):
        # load models
        self.models.clear()
        self.models += list(map(self.load_model, self.model_dirs))

        super().build(input_shapes)

    def call(self, x: list[tf.Tensor], training: bool = False) -> tf.Tensor:
        # unpack inputs
        cont_inputs, cat_inputs = x

        # call models
        output = tf.concat(
            [model([cont_inputs, cat_inputs], training=False)[:, None, :] for model in self.models],
            axis=1,
        )

        # ensembling (via mixture of experts)
        output = tf.reduce_mean(output, axis=1)

        return output


def export_ensemble(
    model_dirs: list[str],
    ensemble_dir: str,
    n_cont_inputs: int,
    n_cat_inputs: int,
):
    model_dirs = list(model_dirs)

    with get_device("cpu"):
        # build the model
        x_cont = tf.keras.Input(n_cont_inputs, dtype=tf.float32, name="cont_input")
        x_cat = tf.keras.Input(n_cat_inputs, dtype=tf.int32, name="cat_input")
        y = Ensemble(model_dirs, name="hbt_ensemble")([x_cont, x_cat])
        ensemble_model = tf.keras.Model(inputs=[x_cont, x_cat], outputs=y)

        # test it
        # cont_inputs = tf.constant([1.0] * n_cont_inputs, dtype=tf.float32)[None, ...]
        # cat_inputs = tf.constant([0, -1, 0, -1, -1, 2], dtype=tf.int32)[None, ...]
        # print(f"evaluate -> {ensemble_model([cont_inputs, cat_inputs], training=False)}")

        # save it
        tf.keras.saving.save_model(
            ensemble_model,
            ensemble_dir,
            overwrite=True,
            save_format="tf",
            include_optimizer=False,
        )

        # also save a frozen version for use in c++
        cmsml.tensorflow.save_frozen_graph(
            os.path.join(ensemble_dir, "frozen.pb"),
            ensemble_model,
            output_names=["Identity"],
            variables_to_constants=True,
        )
        cmsml.tensorflow.save_frozen_graph(
            os.path.join(ensemble_dir, "frozen.pb.txt"),
            ensemble_model,
            output_names=["Identity"],
            variables_to_constants=True,
        )

        print(f"saved ensemble model at {ensemble_dir}")


if __name__ == "__main__":
    # model save dir
    this_dir: str = os.path.dirname(os.path.abspath(__file__))
    model_dir: str = os.getenv("TN_MODEL_DIR", os.path.join(this_dir, "models"))

    # model name to load
    model_name: str = "hbtres_ED5_LU5x125_CTfcn_ACTelu_BNy_LT50_DO0_BS4096_LR3.0e-03_SPINy_MASSy_FI{fold}_SD{seed}_binary"

    # where to save exported models
    export_dir: str = os.getenv("TN_MODEL_DIR", os.path.join(this_dir, "models"))

    # number of expected input features (including mass and spin as last features if those are parametrized)
    n_cont_inputs: int = 55
    n_cat_inputs: int = 6

    # test fold 0 with a single seed
    models_dirs = [
        os.path.join(model_dir, model_name.format(fold=0, seed=seed))
        for seed in [1]
    ]
    ensemble_name = model_name.format(fold=0, seed=0).replace("_SD0", "") + "_moe"
    ensemble_dir = os.path.join(export_dir, ensemble_name)
    export_ensemble(models_dirs, n_cont_inputs, n_cat_inputs, ensemble_dir)
