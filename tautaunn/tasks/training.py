# coding: utf-8

import os
import fnmatch
from typing import Any

import luigi
import law

from tautaunn.tasks.base import Task
from tautaunn.util import create_model_name
import tautaunn.config as cfg


class TrainingParameters(Task):

    model_name = luigi.Parameter(
        default=law.NO_STR,
        description="custom model name",
    )
    model_prefix = luigi.Parameter(
        default="hbtres",
        description="custom model prefix; default: hbtres",
    )
    model_suffix = luigi.Parameter(
        default=law.NO_STR,
        description="custom model suffix",
    )
    embedding_output_dim = luigi.IntParameter(
        default=5,
        description="dimension of the categorical embedding; default: 5",
    )
    units = law.CSVParameter(
        cls=luigi.IntParameter,
        default=5 * (128,),
        description="number of units per layer; default: 128,128,128,128,128",
        brace_expand=True,
    )
    connection_type = luigi.ChoiceParameter(
        default="fcn",
        choices=["fcn", "res", "dense"],
        description="connection type between layers; choices: fcn,res,dense; default: fcn",
    )
    activation = luigi.Parameter(
        default="elu",
        description="activation function; default: elu",
    )
    l2_norm = luigi.FloatParameter(
        default=50.0,
        description="weight-normalized l2 regularization; default: 50.0",
    )
    dropout_rate = luigi.FloatParameter(
        default=0.0,
        description="dropout percentage; default: 0.0",
    )
    batch_norm = luigi.BoolParameter(
        default=True,
        description="enable batch normalization between layers; default: True",
    )
    batch_size = luigi.IntParameter(
        default=4096,
        description="batch size; default: 4096",
    )
    optimizer = luigi.ChoiceParameter(
        default="adam",
        choices=["adam", "adamw"],
        description="the optimizer the use; choices: adam,adamw; default: adam",
    )
    learning_rate = luigi.FloatParameter(
        default=3e-3,
        description="learning rate; default: 3e-3",
    )
    learning_rate_patience = luigi.IntParameter(
        default=8,
        description="non-improving steps before reducing learning rate; default: 10",
    )
    learning_rate_reductions = luigi.IntParameter(
        default=6,
        description="number of possible learning rate reductions; default: 6",
    )
    early_stopping_patience = luigi.IntParameter(
        default=12,
        description="non-improving steps before stopping training; default: 10",
    )
    background_weight = luigi.FloatParameter(
        default=1.0,
        description="relative weight of background classes; default: 1.0",
    )
    fold = luigi.IntParameter(
        default=0,
        description="number of the fold to train for; default: 0",
    )
    seed = luigi.IntParameter(
        default=1,
        description="random seed; default: 1",
    )
    max_epochs = luigi.IntParameter(
        default=10000,
        significant=False,
        description="maximum number of epochs; default: 10000",
    )
    validate_every = luigi.IntParameter(
        default=500,
        significant=False,
        description="validate every n batches; default: 500",
    )
    label_set = luigi.ChoiceParameter(
        default="binary",
        choices=list(cfg.label_sets.keys()),
        description="name of label set; default: binary",
    )
    sample_set = luigi.ChoiceParameter(
        default="default",
        choices=list(cfg.sample_sets.keys()),
        description="name of sample set; default: default",
    )
    cont_feature_set = luigi.ChoiceParameter(
        default="reg",
        choices=list(cfg.cont_feature_sets.keys()),
        description="name of continuous feature set; default: reg",
    )
    cat_feature_set = luigi.ChoiceParameter(
        default="reg",
        choices=list(cfg.cat_feature_sets.keys()),
        description="name of categorical feature set; default: reg",
    )
    regression_set = luigi.ChoiceParameter(
        default=law.NO_STR,
        choices=[law.NO_STR] + list(cfg.regression_sets.keys()),
        description="name of a regression set to use; default: empty",
    )
    skip_tensorboard = luigi.BoolParameter(
        default=False,
        significant=False,
        description="skip tensorboard logging; default: False",
    )

    def get_model_name_kwargs(self) -> dict[str, Any]:
        kwargs = dict(
            model_name=None if self.model_name in (None, "", law.NO_STR) else self.model_name,
            model_prefix=None if self.model_prefix in (None, "", law.NO_STR) else self.model_prefix,
            model_suffix=None if self.model_suffix in (None, "", law.NO_STR) else self.model_suffix,
            label_set=self.label_set,
            sample_set=self.sample_set,
            feature_set=f"{self.cont_feature_set}-{self.cat_feature_set}",
            embedding_output_dim=self.embedding_output_dim,
            units=list(self.units),
            connection_type=self.connection_type,
            activation=self.activation,
            l2_norm=self.l2_norm,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            batch_size=self.batch_size,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            parameterize_spin=True,
            parameterize_mass=True,
            regression_set="none" if self.regression_set in (None, "", law.NO_STR) else self.regression_set,
            fold_index=self.fold,
            seed=self.seed,
        )

        # optionals
        if self.background_weight != 1.0:
            kwargs["background_weight"] = self.background_weight

        return kwargs

    def get_model_name(self) -> str:
        return create_model_name(**self.get_model_name_kwargs())


class Training(TrainingParameters):

    def output(self):
        return {
            "saved_model": (model_dir := self.local_target(self.get_model_name(), dir=True)),
            "meta": model_dir.child("meta.json", type="f"),
        }

    @law.decorator.safe_output
    def run(self):
        # load the training function
        from tautaunn.train_combined import train

        # prepare samples to use and corresponding class names
        class_names: dict[int, str] = {}
        samples: list[cfg.Sample] = []
        for label, data in cfg.label_sets[self.label_set].items():
            class_names[label] = data["name"]
            loss_weight = 1.0 if label == 0 else self.background_weight
            for sample in cfg.sample_sets[self.sample_set]:
                if any(fnmatch.fnmatch(sample.name, pattern) for pattern in data["sample_patterns"]):
                    samples.append(sample.with_label_and_loss_weight(label, loss_weight))
                    continue

        # define arguments
        train_kwargs = dict(
            model_name=self.get_model_name(),
            model_prefix="",
            model_suffix="",
            data_dirs={
                "2016": os.environ["TN_SKIMS_2016"],
                "2016APV": os.environ["TN_SKIMS_2016APV"],
                "2017": os.environ["TN_SKIMS_2017"],
                "2018": os.environ["TN_SKIMS_2018"],
            },
            cache_dir=os.path.join(os.environ["TN_DATA_DIR"], "training_cache"),
            tensorboard_dir=(
                None
                if self.skip_tensorboard
                else os.getenv("TN_TENSORBOARD_DIR", os.path.join(os.environ["TN_DATA_DIR"], "tensorboard"))
            ),
            clear_existing_tensorboard=True,
            model_dir=self.output()["saved_model"].parent.path,
            samples=samples,
            class_names=class_names,
            cont_input_names=cfg.cont_feature_sets[self.cont_feature_set],
            cat_input_names=cfg.cat_feature_sets[self.cat_feature_set],
            units=self.units,
            connection_type=self.connection_type,
            embedding_output_dim=self.embedding_output_dim,
            activation=self.activation,
            l2_norm=self.l2_norm,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            batch_size=self.batch_size,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            learning_rate_patience=self.learning_rate_patience,
            learning_rate_reductions=self.learning_rate_reductions,
            early_stopping_patience=self.early_stopping_patience,
            max_epochs=self.max_epochs,
            validate_every=self.validate_every,
            parameterize_spin=True,
            parameterize_mass=True,
            regression_set=None if self.regression_set in (None, "", law.NO_STR) else self.regression_set,
            fold_index=self.fold,
            validation_folds=3,
            seed=self.seed,
        )

        # run the training
        ret = train(**train_kwargs)
        if ret is None:
            raise Exception("training did not provide a model")


class MultiSeedParameters(TrainingParameters):

    seed = None
    seeds = law.MultiRangeParameter(
        default=((1, 6),),
        single_value=True,
        require_end=True,
        description="random seeds to use for the ensemble; default: 1:6",
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.flat_seeds = sorted(set.union(*map(set, map(law.util.range_expand, self.seeds))))

    def get_model_name_kwargs(self) -> dict[str, Any]:
        kwargs = super().get_model_name_kwargs()
        kwargs.pop("seed")
        kwargs["seeds"] = f"x{len(self.flat_seeds)}"
        return kwargs


class ExportEnsemble(MultiSeedParameters):

    def requires(self):
        return {seed: Training.req(self, seed=seed) for seed in self.flat_seeds}

    def output(self):
        return {
            "saved_model": (model_dir := self.local_target(self.get_model_name(), dir=True)),
            "frozen": model_dir.child("frozen.pb", type="f"),
        }

    @law.decorator.safe_output
    def run(self):
        # load the export function
        from tautaunn.export_ensembles import export_ensemble

        # define arguments
        export_kwargs = dict(
            model_dirs=[self.input()[seed]["saved_model"].path for seed in self.flat_seeds],
            ensemble_dir=self.output()["saved_model"].path,
            n_cont_inputs=len(cfg.cont_feature_sets[self.cont_feature_set]) + 1,  # +1 for mass
            n_cat_inputs=len(cfg.cat_feature_sets[self.cat_feature_set]) + 1,  # +1 for spin
        )

        # export it
        export_ensemble(**export_kwargs)


class MultiFoldParameters(MultiSeedParameters):

    fold = None
    folds = law.MultiRangeParameter(
        default=((0,),),
        single_value=True,
        require_end=True,
        description="folds to use for training; default: 0",
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.flat_folds = sorted(set.union(*map(set, map(law.util.range_expand, self.folds))))

    def get_model_name_kwargs(self) -> dict[str, Any]:
        kwargs = super().get_model_name_kwargs()
        kwargs.pop("fold_index")
        kwargs["fold_indices"] = f"x{len(self.flat_folds)}"
        return kwargs


class ExportEnsembleWrapper(MultiFoldParameters, law.WrapperTask):

    def requires(self):
        return {
            fold: ExportEnsemble.req(self, fold=fold)
            for fold in self.flat_folds
        }
