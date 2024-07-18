# coding: utf-8

from __future__ import annotations

import time

import luigi
import law
import numpy as np
import tensorflow as tf
import awkward as ak

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

    # @property
    # def priority(self):
    #     # higher priority value = picked earlier by scheduler
    #     # priotize (tt, ttH, ttZ > data > rest) across all years
    #     if re.match(r"^(TT_SemiLep|TT_FullyLep|ttHToTauTau|TTZToLLNuNu)$", self.sample.name):
    #         return 10
    #     if re.match(r"^(EGamma|MET|Muon|Tau)(A|B|C|D|E|F|G|H)$", self.sample.name):
    #         return 5
    #     return 1

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

        def calc_inputs(arr, dyn_names, cfg, fold_index):
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

            # merge with fold mask in case there are multiple models
            eval_mask = cat_mask
            if n_models > 1:
                eval_mask &= np.asarray((arr.EventNumber % self.n_folds) == fold_index)

            return cont_inputs, cat_inputs, eval_mask

        def predict(model, cont_inputs, cat_inputs, eval_mask, class_names, shape_name, out_tree):
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

        def sel_os(array: ak.Array) -> ak.Array:
            return array.isOS == 1

        def select_category(array: ak.Array) -> ak.Array:
            cat_ids = np.zeros(len(array), dtype=np.int32)
            cat_ids[sel_os(array)] = 1
            cat_ids[~sel_os(array)] = 2
            return cat_ids

        # def sel_btag_m(array: ak.Array, year: str) -> ak.Array:
        #     return (
        #         (array.bjet1_bID_deepFlavor > cfg.btag_wps[year]["medium"]) &
        #         (array.bjet2_bID_deepFlavor <= cfg.btag_wps[year]["medium"])
        #     ) | (
        #         (array.bjet1_bID_deepFlavor <= cfg.btag_wps[year]["medium"]) &
        #         (array.bjet2_bID_deepFlavor > cfg.btag_wps[year]["medium"])
        #     )

        # def sel_btag_mm(array: ak.Array, year: str) -> ak.Array:
        #     return (
        #         (array.bjet1_bID_deepFlavor > cfg.btag_wps[year]["medium"]) &
        #         (array.bjet2_bID_deepFlavor > cfg.btag_wps[year]["medium"])
        #     )

        # def sel_first_lep(array: ak.Array) -> ak.Array:
        #     return (
        #         ((array.pairType == 0) & (array.dau1_iso < 0.15)) |
        #         ((array.pairType == 1) & (array.dau1_eleMVAiso == 1)) |
        #         ((array.pairType == 2) & (array.dau1_deepTauVsJet >= 5))
        #     )

        # def sel_pnet_l(array: ak.Array, year: str) -> ak.Array:
        #     return (
        #         (array.fatjet_particleNetMDJetTags_score > cfg.pnet_wps[year])
        #     )

        # def sel_cats(array: ak.Array, year: str) -> ak.Array:
        #     return (
        #         (array.nleps == 0) &
        #         sel_first_lep(array) &
        #         sel_trigger(array) &
        #         (
        #             (  # boosted (pnet cut left out to be looser)
        #                 (array.isBoosted == 1)
        #             ) |
        #             (  # res1b (no ~isBoosted cut to be looser)
        #                 (array.nbjetscand > 1) &
        #                 sel_btag_m(array, year)
        #             ) |
        #             (  # res2b (no ~isBoosted cut to be looser)
        #                 (array.nbjetscand > 1) &
        #                 sel_btag_mm(array, year)
        #             )
        #         )
        #     )

        # determine columns to read
        columns_to_read = set()
        columns_to_read |= set(cfg.cont_feature_sets[self.cont_feature_set])
        columns_to_read |= set(cfg.cat_feature_sets[self.cat_feature_set])
        columns_to_read |= set(cfg.klub_index_columns)
        columns_to_read |= set(cfg.klub_category_columns)
        columns_to_read |= set(cfg.klub_weight_columns)
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

        # prepare tree-like structure for outputs
        out_tree = {
            c: np.asarray(arr[c])
            for c in list(set(cfg.klub_index_columns) | set(cfg.klub_weight_columns) | set(cfg.klub_category_columns))
        }
        out_tree["category_id"] = cat_ids

        # loop over models to keep only one in memory at a time
        for fold_index, inps in models.items():
            with self.publish_step(f"\nloading model for fold {fold_index} ..."), get_device("cpu"):
                model = inps["saved_model"].load(formatter="tf_saved_model")

                cont_inputs, cat_inputs, eval_mask = calc_inputs(arr, dyn_names, cfg, fold_index)
                # evaluate the data
                with self.publish_step(f"evaluating model for nominal on {eval_mask.sum()} events ..."):
                    predict(model, cont_inputs, cat_inputs, eval_mask, class_names, "nominal", out_tree)

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
