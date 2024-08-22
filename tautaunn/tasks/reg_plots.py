# coding: utf-8

from __future__ import annotations

import time

import luigi
import law
import numpy as np
import tensorflow as tf
import awkward as ak
import vector


from tautaunn.tasks.base import SkimWorkflow, MultiSkimTask
from tautaunn.tasks.reg_training import RegTraining, RegMultiFoldParameters
from tautaunn.util import calc_new_columns
from tautaunn.tf_util import get_device
import tautaunn.config as cfg


class RegEvaluationParameters(RegMultiFoldParameters):

    spins = law.CSVParameter(
        cls=luigi.IntParameter,
        default=tuple(cfg.spins),
        description=f"spins to evaluate; default: {','.join(map(str, cfg.spins))}",
        brace_expand=True,
    )
    masses = law.CSVParameter(
        cls=luigi.FloatParameter,
        default=tuple(cfg.masses),
        description=f"masses to evaluate; default: {','.join(map(str, cfg.masses))}",
        brace_expand=True,
    )


class RegEvaluateSkims(SkimWorkflow, RegEvaluationParameters):

    default_store = "$TN_STORE_DIR_TOBI"

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["models"] = {i: RegTraining.req(self, fold=i) for i in self.flat_folds}
        return reqs

    def requires(self):
        return {i: RegTraining.req(self, fold=i) for i in self.flat_folds}

    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "model", self.get_model_name())
        return parts

    def output(self):
        return self.local_target(f"output_{self.branch_data}.root")

    @law.decorator.localize(input=False, output=True)
    def run(self):
        t_start = time.perf_counter()

        # set inter and intra op parallelism threads of tensorflow
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

        # prepare input models
        models = dict(self.input().items())
        n_models = len(models)
        assert n_models == 1 or set(models.keys()) == set(range(self.n_folds))

        # helpers
        flatten = lambda r, t: r.astype([(n, t) for n in r.dtype.names], copy=False).view(t).reshape((-1, len(r.dtype)))
        def col_name(mass, spin, out_name):
            return f"regdnn_m{int(mass)}_s{int(spin)}_{out_name.lower()}"

        def calc_inputs(arr, dyn_names, cfg):
            arr = calc_new_columns(arr, {name: cfg.dynamic_columns[name] for name in dyn_names})

            # prepare model inputs
            cont_inputs = flatten(np.asarray(arr[cont_input_names]), np.float32)
            cat_inputs = flatten(np.asarray(arr[cat_input_names]), np.int32)

            # add year
            y = self.sample.year_flag
            cat_inputs = np.append(cat_inputs, y * np.ones(len(cat_inputs), dtype=np.int32)[..., None], axis=1)

            # reserve column for mass
            cont_inputs = np.append(cont_inputs, -1 * np.ones(len(cont_inputs), dtype=np.float32)[..., None], axis=1)

            # reserve column for spin (must be behind year!)
            cat_inputs = np.append(cat_inputs, -1 * np.ones(len(cat_inputs), dtype=np.int32)[..., None], axis=1)

            # create a mask to only select events whose categorical features were seen during training
            cat_mask = np.ones(len(arr), dtype=bool)
            for i, name in enumerate(cat_input_names):
                cat_mask &= np.isin(cat_inputs[:, i], np.unique(cfg.embedding_expected_inputs[name]))
            self.publish_message(f"events passing cat_mask: {cat_mask.mean() * 100:.2f}%")

            return arr, cont_inputs, cat_inputs

        def predict(model, cont_inputs, cat_inputs, eval_mask, class_names, out_tree):
            # selection and categorization

            spins = self.spins if self.sample.spin < 0 else [self.sample.spin]
            masses = self.masses if self.sample.mass < 0 else [self.sample.mass]
            for spin in spins:
                # insert spin
                cat_inputs[:, -1] = int(spin)
                for mass in masses:
                    # insert mass
                    cont_inputs[:, -1] = float(mass)

                    # evaluate
                    predictions = model(
                        {
                            "cont_input": cont_inputs[eval_mask],
                            "cat_input": cat_inputs[eval_mask],
                        },
                        training=False,
                    )

                    # insert class predictions
                    for i, class_name in class_names.items():
                        field = col_name(mass, spin, class_name)
                        if field not in out_tree:
                            out_tree[field] = -1 * np.ones(len(eval_mask), dtype=np.float32)
                        out_tree[field][eval_mask] = predictions["classification_output_softmax"][:, i]

                    # insert regression predictions
                    for i, reg_name in enumerate(reg_names):
                        field = col_name(mass, spin, reg_name)
                        if field not in out_tree:
                            out_tree[field] = -1 * np.ones(len(eval_mask), dtype=np.float32)
                        out_tree[field][eval_mask] = predictions["regression_output_hep"][:, i]

                    dau1 = vector.array(
                        {
                            "px": out_tree["dau1_px"],
                            "py": out_tree["dau1_py"],
                            "pz": out_tree["dau1_pz"],
                            "e": out_tree["dau1_e"],
                        },
                    )
                    dau2 = vector.array(
                        {
                            "px": out_tree["dau2_px"],
                            "py": out_tree["dau2_py"],
                            "pz": out_tree["dau2_pz"],
                            "e": out_tree["dau2_e"],
                        },
                    )
                    nu1 = vector.array(
                        {
                            "px": out_tree[col_name(mass, spin, "nu1_px")],
                            "py": out_tree[col_name(mass, spin, "nu1_py")],
                            "pz": out_tree[col_name(mass, spin, "nu1_pz")],
                            "e": np.sqrt(
                                out_tree[col_name(mass, spin, "nu1_px")]**2 +
                                out_tree[col_name(mass, spin, "nu1_py")]**2 +
                                out_tree[col_name(mass, spin, "nu1_pz")]**2,
                            ),
                        },
                    )

                    nu2 = vector.array(
                        {
                            "px": out_tree[col_name(mass, spin, "nu2_px")],
                            "py": out_tree[col_name(mass, spin, "nu2_py")],
                            "pz": out_tree[col_name(mass, spin, "nu2_pz")],
                            "e": np.sqrt(
                                out_tree[col_name(mass, spin, "nu2_px")]**2 +
                                out_tree[col_name(mass, spin, "nu2_py")]**2 +
                                out_tree[col_name(mass, spin, "nu2_pz")]**2,
                            ),
                        },
                    )

                    h_bb = vector.array(
                        {
                            "px": out_tree["bH_px"],
                            "py": out_tree["bH_py"],
                            "pz": out_tree["bH_pz"],
                            "e": out_tree["bH_e"],
                        },
                    )

                    h_tt = dau1 + dau2 + nu1 + nu2
                    hh = h_bb + h_tt

                    out_tree[f"reg_H_mass_m{int(mass)}_{int(spin)}"] = h_tt.m
                    out_tree[f"reg_H_pt_m{int(mass)}_{int(spin)}"] = h_tt.pt
                    out_tree[f"reg_HH_mass_m{int(mass)}_{int(spin)}"] = hh.m
                    out_tree[f"reg_HH_pt_m{int(mass)}_{int(spin)}"] = hh.pt

        def sel_mass_window_res(array: ak.Array) -> ak.Array:
            return (
                (array.tauH_mass >= 20.0) &
                (array.bH_mass >= 40.0)
            )

        def sel_mass_window_boosted(array: ak.Array) -> ak.Array:
            return (
                (array.tauH_mass >= 20.0) &
                (array.tauH_mass <= 130.0)
            )

        def sel_trigger(array: ak.Array) -> ak.Array:
            return (
                (array.isLeptrigger == 1) | (array.isMETtrigger == 1) | (array.isSingleTautrigger == 1)
            )

        def sel_common(array: ak.Array) -> ak.Array:
            return (
                (array.isOS == 1) &
                (array.nleps == 0) &
                (array.dau2_deepTauVsJet >= 5) &
                sel_trigger(array)
            )

        def sel_mutau(array: ak.Array) -> ak.Array:
            return (
                (array.pairType == 0) &
                (array.dau1_iso < 0.15)
            )

        def sel_etau(array: ak.Array) -> ak.Array:
            return (
                (array.pairType == 1) &
                (array.dau1_eleMVAiso == 1)
            )

        def sel_tautau(array: ak.Array) -> ak.Array:
            return (
                (array.pairType == 2) &
                (array.dau1_deepTauVsJet >= 5)
            )

        def sel_btag_m(array: ak.Array) -> ak.Array:
            return (
                (array.bjet1_bID_deepFlavor > cfg.btag_wps[self.sample.year]["medium"]) &
                (array.bjet2_bID_deepFlavor <= cfg.btag_wps[self.sample.year]["medium"])
            ) | (
                (array.bjet1_bID_deepFlavor <= cfg.btag_wps[self.sample.year]["medium"]) &
                (array.bjet2_bID_deepFlavor > cfg.btag_wps[self.sample.year]["medium"])
            )

        def sel_btag_mm(array: ak.Array) -> ak.Array:
            return (
                (array.bjet1_bID_deepFlavor > cfg.btag_wps[self.sample.year]["medium"]) &
                (array.bjet2_bID_deepFlavor > cfg.btag_wps[self.sample.year]["medium"])
            )

        def sel_pnet_l(array: ak.Array) -> ak.Array:
            return (
                (array.fatjet_particleNetMDJetTags_score > cfg.pnet_wps[self.sample.year])
            )

        def sel_mutau_res1b(array: ak.Array) -> ak.Array:
            return (
                sel_common(array) &
                sel_mutau(array) &
                sel_btag_m(array) &
                sel_mass_window_res(array) &
                (array.nbjetscand > 1) &
                (array.isBoosted == 0)
            )

        def sel_mutau_res2b(array: ak.Array) -> ak.Array:
            return (
                sel_common(array) &
                sel_mutau(array) &
                sel_btag_mm(array) &
                sel_mass_window_res(array) &
                (array.nbjetscand > 1) &
                (array.isBoosted == 0)
            )

        def sel_mutau_boosted(array: ak.Array) -> ak.Array:
            return (
                sel_common(array) &
                sel_mutau(array) &
                sel_pnet_l(array) &
                sel_mass_window_boosted(array) &
                (array.isBoosted == 1)
            )

        def sel_etau_res1b(array: ak.Array) -> ak.Array:
            return (
                sel_common(array) &
                sel_etau(array) &
                sel_btag_m(array) &
                sel_mass_window_res(array) &
                (array.nbjetscand > 1) &
                (array.isBoosted == 0)
            )

        def sel_etau_res2b(array: ak.Array) -> ak.Array:
            return (
                sel_common(array) &
                sel_etau(array) &
                sel_btag_mm(array) &
                sel_mass_window_res(array) &
                (array.nbjetscand > 1) &
                (array.isBoosted == 0)
            )

        def sel_etau_boosted(array: ak.Array) -> ak.Array:
            return (
                sel_common(array) &
                sel_etau(array) &
                sel_pnet_l(array) &
                sel_mass_window_boosted(array) &
                (array.isBoosted == 1)
            )

        def sel_tautau_res1b(array: ak.Array) -> ak.Array:
            return (
                sel_common(array) &
                sel_tautau(array) &
                sel_btag_m(array) &
                sel_mass_window_res(array) &
                (array.nbjetscand > 1) &
                (array.isBoosted == 0)
            )

        def sel_tautau_res2b(array: ak.Array) -> ak.Array:
            return (
                sel_common(array) &
                sel_tautau(array) &
                sel_btag_mm(array) &
                sel_mass_window_res(array) &
                (array.nbjetscand > 1) &
                (array.isBoosted == 0)
            )

        def sel_tautau_boosted(array: ak.Array) -> ak.Array:
            return (
                sel_common(array) &
                sel_tautau(array) &
                sel_pnet_l(array) &
                sel_mass_window_boosted(array) &
                (array.isBoosted == 1)
            )

        def select_category(array: ak.Array) -> ak.Array:
            cat_ids = np.zeros(len(array), dtype=np.int32)
            cat_ids[sel_mutau_res1b(array)] = cfg.category_indices["mutau_res1b"]
            cat_ids[sel_mutau_res2b(array)] = cfg.category_indices["mutau_res2b"]
            cat_ids[sel_mutau_boosted(array)] = cfg.category_indices["mutau_boosted"]
            cat_ids[sel_etau_res1b(array)] = cfg.category_indices["etau_res1b"]
            cat_ids[sel_etau_res2b(array)] = cfg.category_indices["etau_res2b"]
            cat_ids[sel_etau_boosted(array)] = cfg.category_indices["etau_boosted"]
            cat_ids[sel_tautau_res1b(array)] = cfg.category_indices["tautau_res1b"]
            cat_ids[sel_tautau_res2b(array)] = cfg.category_indices["tautau_res2b"]
            cat_ids[sel_tautau_boosted(array)] = cfg.category_indices["tautau_boosted"]
            return cat_ids

        # determine columns to read
        columns_to_read = set()
        columns_to_read |= set(cfg.cont_feature_sets[self.cont_feature_set])
        columns_to_read |= set(cfg.cat_feature_sets[self.cat_feature_set])
        columns_to_read |= set(cfg.klub_index_columns)
        columns_to_read |= set(cfg.klub_category_columns)
        columns_to_read |= set(cfg.klub_weight_columns)
        columns_to_read |= set(cfg.reg_plot_columns)
        # expand dynamic columns, keeping track of those that are needed
        all_dyn_names = set(cfg.dynamic_columns)
        dyn_names = set()
        while (to_expand := columns_to_read & all_dyn_names):
            for name in to_expand:
                columns_to_read |= set(cfg.dynamic_columns[name][0])
            columns_to_read -= to_expand
            dyn_names |= to_expand
        dyn_names = sorted(dyn_names, key=list(cfg.dynamic_columns.keys()).index)

        # determine names of inputs
        cont_input_names = list(cfg.cont_feature_sets[self.cont_feature_set])
        cat_input_names = list(cfg.cat_feature_sets[self.cat_feature_set])

        # get class names
        class_names = {
            label: data["name"].lower()
            for label, data in cfg.label_sets[self.label_set].items()
        }

        # get regression label names
        reg_names = ["nu1_px", "nu1_py", "nu1_pz", "nu2_px", "nu2_py", "nu2_pz"]

        # load the input tree
        skim_file = self.get_skim_file(self.branch_data)
        in_file = skim_file.load(formatter="uproot")
        in_tree = in_file["HTauTauTree"]

        # read columns and insert dynamic ones
        arr = in_tree.arrays(
            list(columns_to_read - {"year_flag"}),
            aliases=cfg.klub_aliases,
            library="ak",
        )
        arr = ak.with_field(arr, self.sample.year_flag, "year_flag")

        # selection / categorization
        cat_ids = select_category(arr)
        sel_mask = cat_ids > 0
        arr = arr[sel_mask]
        cat_ids = cat_ids[sel_mask]

        arr, cont_inputs, cat_inputs = calc_inputs(arr, dyn_names, cfg)
        # prepare tree-like structure for outputs
        out_tree = {
            c: np.asarray(arr[c])
            for c in list(set(cfg.klub_index_columns) | set(cfg.klub_weight_columns) | set(cfg.reg_plot_columns))
        }
        out_tree["category_id"] = cat_ids

        # loop over models to keep only one in memory at a time
        for fold_index, inps in models.items():
            with self.publish_step(f"\nloading model for fold {fold_index} ..."), get_device("cpu"):
                model = inps["saved_model"].load(formatter="tf_saved_model")

                eval_mask = np.ones(len(arr), dtype=np.int32)
                if n_models > 1:
                    eval_mask = np.asarray((arr.EventNumber % self.n_folds) == fold_index)

                # evaluate the data
                with self.publish_step(f"evaluating model on {eval_mask.sum()} events ..."):
                    predict(model, cont_inputs, cat_inputs, eval_mask, class_names, out_tree)

        # free memory
        del models

        # save outputs
        with self.output().dump(formatter="uproot", mode="recreate") as f:
            f["hbtres"] = out_tree
            f["h_eff"] = in_file["h_eff"]

        print(f"full evaluation took {law.util.human_duration(seconds=time.perf_counter() - t_start)}")


class RegEvaluateSkimsWrapper(MultiSkimTask, RegMultiFoldParameters, law.WrapperTask):

    def requires(self):
        return {
            skim_name: RegEvaluateSkims.req(self, skim_name=skim_name)
            for skim_name in self.skim_names
        }
