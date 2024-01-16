# coding: utf-8

from __future__ import annotations

import os
import itertools

import luigi
import law
import numpy as np

from tautaunn.tasks.base import SkimWorkflow, MultiSkimTask
from tautaunn.tasks.training import MultiFoldParameters, ExportEnsemble
from tautaunn.util import calc_new_columns
import tautaunn.config as cfg


class EvaluationParameters(MultiFoldParameters):

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


class EvaluateSkims(SkimWorkflow, EvaluationParameters):

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["ensembles"] = {i: ExportEnsemble.req(self, fold=i) for i in self.flat_folds}
        return reqs

    def requires(self):
        return {i: ExportEnsemble.req(self, fold=i) for i in self.flat_folds}

    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "ensemble", self.get_model_name())
        return parts

    def output(self):
        return self.local_target(f"output_{self.branch_data}.root")

    @law.decorator.safe_output
    def run(self):
        # helpers
        flatten = lambda r, t: r.astype([(n, t) for n in r.dtype.names], copy=False).view(t).reshape((-1, len(r.dtype)))
        col_name = lambda mass, spin, class_name: f"hbtresdnn_mass{int(mass)}_spin{int(spin)}_{class_name.lower()}"

        # load the input tree
        in_tree = self.get_skim_file(self.branch_data).load(formatter="uproot")["HTauTauTree"]

        # determine columns to read
        columns_to_read = set()
        columns_to_read |= set(cfg.cont_feature_sets[self.cont_feature_set])
        columns_to_read |= set(cfg.cat_feature_sets[self.cat_feature_set])
        if self.regression_cfg is not None:
            columns_to_read |= set(cfg.cont_feature_sets[self.regression_cfg.cont_feature_set])
            columns_to_read |= set(cfg.cat_feature_sets[self.regression_cfg.cat_feature_set])
        if self.lbn_cfg is not None:
            columns_to_read |= set(self.lbn_cfg.input_features) - {None}
        columns_to_read |= set(cfg.klub_index_columns)
        # expand dynamic columns, keeping track of those that are needed
        all_dyn_names = set(cfg.dynamic_columns)
        dyn_names = set()
        while (to_expand := columns_to_read & all_dyn_names):
            for name in to_expand:
                columns_to_read |= set(cfg.dynamic_columns[name][0])
            columns_to_read -= to_expand
            dyn_names |= to_expand
        dyn_names = sorted(dyn_names, key=list(cfg.dynamic_columns.keys()).index)

        # preprocess input data
        with self.publish_step("preparing data ..."):
            # read columns and insert dynamic ones
            rec = in_tree.arrays(list(columns_to_read), aliases=cfg.klub_aliases, library="ak").to_numpy()
            self.publish_message(f"read {len(rec)} events")
            rec = calc_new_columns(rec, {name: cfg.dynamic_columns[name] for name in dyn_names})

            # determine names of inputs
            cont_input_names = list(cfg.cont_feature_sets[self.cont_feature_set])
            cat_input_names = list(cfg.cat_feature_sets[self.cat_feature_set])
            if self.regression_cfg is not None:
                for name in cfg.cont_feature_sets[self.regression_cfg.cont_feature_set]:
                    if name not in cont_input_names:
                        cont_input_names.append(name)
                for name in cfg.cat_feature_sets[self.regression_cfg.cat_feature_set]:
                    if name not in cat_input_names:
                        cat_input_names.append(name)
            if self.lbn_cfg is not None:
                for name in self.lbn_cfg.input_features:
                    if name and name not in cont_input_names:
                        cont_input_names.append(name)

            # prepare model inputs
            cont_inputs = flatten(rec[cont_input_names], np.float32)
            cat_inputs = flatten(rec[cat_input_names], np.int32)

            # add year
            y = self.sample.year_flag
            cat_inputs = np.append(cat_inputs, y * np.ones(len(cat_inputs), dtype=np.int32)[..., None], axis=1)

            # reserve column for mass
            cont_inputs = np.append(cont_inputs, -1 * np.ones(len(cont_inputs), dtype=np.float32)[..., None], axis=1)

            # reserve column for spin (must be behind year!)
            cat_inputs = np.append(cat_inputs, -1 * np.ones(len(cat_inputs), dtype=np.int32)[..., None], axis=1)

            # determine the fold index per event
            fold = rec["EventNumber"] % self.n_folds

        # create a mask to only select events whose categorical features were seen during training
        cat_mask = np.ones(len(rec), dtype=bool)
        for i, name in enumerate(cat_input_names):
            cat_mask &= np.isin(cat_inputs[:, i], np.unique(cfg.embedding_expected_inputs[name]))
        self.publish_message(f"events passing cat_mask: {cat_mask.mean() * 100:.2f}%")

        # get class names
        class_names = {label: data["name"].lower() for label, data in cfg.label_sets[self.label_set].items()}

        # prepare the output tree structure
        out_tree = {c: rec[c] for c in cfg.klub_index_columns}
        for spin in self.spins:
            for mass in self.masses:
                for class_name in class_names.values():
                    out_tree[col_name(mass, spin, class_name)] = -999.0 * np.ones(len(rec), dtype=np.float32)

        # load models
        with self.publish_step("loading models ..."):
            models = {
                i: inps["saved_model"].load(formatter="tf_saved_model")
                for i, inps in self.input().items()
            }

        # testing: when there is just a single fold 0, copy it to all other folds
        if len(models) == 1 and list(models.keys())[0] == 0:
            for i in range(1, self.n_folds):
                models[i] = models[0]
        assert set(models.keys()) == set(range(self.n_folds))

        # evaluate the data
        with self.publish_step("evaluating model ..."):
            for spin in self.spins:
                # insert spin
                cat_inputs[:, -1] = int(spin)
                for mass in self.masses:
                    # insert mass
                    cont_inputs[:, -1] = float(mass)

                    # evaluate per fold
                    for _fold, model in models.items():
                        fold_mask = fold == _fold
                        eval_mask = cat_mask & fold_mask
                        predictions = model([cont_inputs[eval_mask], cat_inputs[eval_mask]], training=False)

                        # insert into output tree
                        for i, class_name in enumerate(class_names.values()):
                            out_tree[col_name(mass, spin, class_name)][eval_mask] = predictions[:, i]

        # save the output tree
        with self.output().dump(formatter="uproot", mode="recreate") as f:
            f["evaluation"] = out_tree


class EvaluateSkimsWrapper(MultiSkimTask, EvaluationParameters, law.WrapperTask):

    def requires(self):
        return {
            skim_name: EvaluateSkims.req(self, skim_name=skim_name)
            for skim_name in self.skim_names
        }


_default_categories = ("2017_*tau_resolved?b_os_iso", "2017_*tau_boosted_os_iso", "2017_*tau_vbf_os_iso")


class WriteDatacards(MultiSkimTask, EvaluationParameters):

    categories = law.CSVParameter(
        default=_default_categories,
        description=f"comma-separated patterns of categories to produce; default: {','.join(_default_categories)}",
        brace_expand=True,
    )
    qcd_estimation = luigi.BoolParameter(
        default=True,
        description="whether to estimate QCD contributions from data; default: True",
    )
    binning = luigi.ChoiceParameter(
        default="flats",
        choices=("flats", "equal"),
        description="binning to use; choices: flats,equal; default: flats",
    )
    n_bins = luigi.IntParameter(
        default=10,
        description="number of bins to use; default: 10",
    )
    variable = luigi.Parameter(
        default="hbtresdnn_mass{mass}_spin{spin}_hh",
        description="variable to use; template values 'mass' and 'spin' are replaced automatically; "
        "default: 'hbtresdnn_mass{mass}_spin{spin}_hh'",
    )
    parallel_read = luigi.IntParameter(
        default=4,
        description="number of parallel processes to use for reading; default: 4",
    )
    parallel_write = luigi.IntParameter(
        default=4,
        description="number of parallel processes to use for writing; default: 4",
    )
    output_suffix = luigi.Parameter(
        default=law.NO_STR,
        description="suffix to append to the output directory; default: ''",
    )
    rewrite_existing = luigi.BoolParameter(
        default=False,
        significant=False,
        description="whether to rewrite existing datacards; default: False",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # only one year at a time is supported for now
        years = set(sample.year for sample in self.samples)
        if len(years) != 1:
            raise ValueError(f"only one year at a time is supported for now, got {','.join(years)}")
        self.year = years.pop()
        self.skim_dir = cfg.skim_dirs[self.year]

        # TODO: complain when the year does not match the category patterns (but how to do that for 2016/APV?)

        self.card_pattern = "cat_{category}_spin_{spin}_mass_{mass}"
        self._card_names = None

    @property
    def card_names(self):
        if self._card_names is None:
            from tautaunn.write_datacards import expand_categories
            categories = expand_categories(self.categories)
            self._card_names = [
                self.card_pattern.format(category=category, spin=spin, mass=mass)
                for spin, mass, category in itertools.product(self.spins, self.masses, categories)
            ]

        return self._card_names

    def requires(self):
        return {
            skim_name: EvaluateSkims.req(self, skim_name=skim_name)
            for skim_name in self.skim_names
        }

    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "ensemble", self.get_model_name())
        return parts

    def output(self):
        # prepare the output directory
        dirname = f"{self.binning}{self.n_bins}"
        if self.output_suffix not in ("", law.NO_STR):
            dirname += f"_{self.output_suffix.lstrip('_')}"
        d = self.local_target(dirname, dir=True)

        return law.SiblingFileCollection({
            name: {
                "datacard": d.child(f"datacard_{name}.txt", type="f"),
                "shapes": d.child(f"shapes_{name}.root", type="f"),
            }
            for name in self.card_names
        })

    @law.decorator.safe_output
    def run(self):
        # load the datacard creating function
        from tautaunn.write_datacards import write_datacards

        # prepare inputs
        inp = self.input()
        sample_names = list(inp)

        # define arguments
        datacard_kwargs = dict(
            spin=list(self.spins),
            mass=list(self.masses),
            category=list(self.categories),
            skim_directory=self.skim_dir,
            eval_directory=inp[sample_names[0]].collection.dir.parent.path,
            output_directory=self.output().dir.path,
            output_pattern=self.card_pattern,
            variable_pattern=self.variable,
            # force using all samples, disabling the feature to select a subset
            # sample_names=[sample_name.replace("SKIM_", "") for sample_name in sample_names],
            binning=(self.n_bins, 0.0, 1.0, "equal_distance" if self.binning == "equal" else "flat_s"),
            qcd_estimation=self.qcd_estimation,
            n_parallel_read=self.parallel_read,
            n_parallel_write=self.parallel_write,
            cache_directory=os.path.join(os.environ["TN_DATA_DIR"], "datacard_cache"),
            skip_existing=not self.rewrite_existing,
        )

        # create the cards
        write_datacards(**datacard_kwargs)
