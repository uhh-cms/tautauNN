# coding: utf-8

import os
from typing import Any

import luigi
import law

from tautaunn.tasks.base import Task
from tautaunn.util import create_model_name, match
import tautaunn.config as cfg


class RegTrainingParameters(Task):

    model_name = luigi.Parameter(
        default=law.NO_STR,
        description="custom model name",
    )
    model_prefix = luigi.Parameter(
        default="tautaureg",
        description="custom model prefix; default: tautaureg",
    )
    model_suffix = luigi.Parameter(
        default=law.NO_STR,
        description="custom model suffix",
    )
    embedding_output_dim = luigi.IntParameter(
        default=10,
        description="dimension of the categorical embedding; default: 10",
    )
    units = law.MultiCSVParameter(
        cls=luigi.IntParameter,
        default=((128,) * 5, (128,) * 4),
        description="number of units per layer; default: 128,128,128,128,128:128,128,128,128",
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
        default="adamw",
        choices=["adam", "adamw"],
        description="the optimizer the use; choices: adam,adamw; default: adamw",
    )
    learning_rate = luigi.FloatParameter(
        default=3e-3,
        description="learning rate; default: 3e-3",
    )
    learning_rate_patience = luigi.IntParameter(
        default=10,
        description="non-improving steps before reducing learning rate; default: 10",
    )
    early_stopping_patience = luigi.IntParameter(
        default=15,
        description="non-improving steps before stopping training; default: 15",
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
    selection_set = luigi.ChoiceParameter(
        default="new_baseline",
        choices=list(cfg.selection_sets.keys()),
        description="name of selection set; default: new_baseline",
    )
    label_set = luigi.ChoiceParameter(
        default="multi4",
        choices=list(cfg.label_sets.keys()),
        description="name of label set; default: multi4",
    )
    sample_set = luigi.ChoiceParameter(
        default="default",
        choices=list(cfg.sample_sets.keys()),
        description="name of sample set; default: default",
    )
    cont_feature_set = luigi.ChoiceParameter(
        default="default_daurot_fatjet",
        choices=list(cfg.cont_feature_sets.keys()),
        description="name of continuous feature set; default: default_daurot_fatjet",
    )
    cat_feature_set = luigi.ChoiceParameter(
        default="default_extended",
        choices=list(cfg.cat_feature_sets.keys()),
        description="name of categorical feature set; default: default_extended",
    )
    skip_tensorboard = luigi.BoolParameter(
        default=False,
        significant=False,
        description="skip tensorboard logging; default: False",
    )
    n_folds = 5

    def get_model_name_kwargs(self) -> dict[str, Any]:
        kwargs = dict(
            model_name=None if self.model_name in (None, "", law.NO_STR) else self.model_name,
            model_prefix=None if self.model_prefix in (None, "", law.NO_STR) else self.model_prefix,
            model_suffix=None if self.model_suffix in (None, "", law.NO_STR) else self.model_suffix,
            selection_set=self.selection_set,
            label_set=self.label_set,
            sample_set=self.sample_set,
            feature_set=f"{self.cont_feature_set}-{self.cat_feature_set}",
            embedding_output_dim=self.embedding_output_dim,
            units=tuple(map(list, self.units)),
            connection_type=self.connection_type,
            activation=self.activation,
            l2_norm=self.l2_norm,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            batch_size=self.batch_size,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            parameterize_year=True,
            parameterize_spin=True,
            parameterize_mass=True,
            fold_index=self.fold,
            seed=self.seed,
        )

        # optionals
        if self.background_weight != 1.0:
            kwargs["background_weight"] = self.background_weight

        return kwargs

    def get_model_name(self) -> str:
        return create_model_name(**self.get_model_name_kwargs())


class RegTraining(RegTrainingParameters):

    def output(self):
        return {
            "saved_model": (model_dir := self.local_target(self.get_model_name(), dir=True)),
            "meta": model_dir.child("meta.json", type="f"),
        }

    @law.decorator.safe_output
    def run(self):
        # load the training function
        from tautaunn.train_regression import train

        # prepare samples to use and corresponding class names
        class_names: dict[int, str] = {}
        samples: list[cfg.Sample] = []
        for label, data in cfg.label_sets[self.label_set].items():
            class_names[label] = data["name"]
            loss_weight = 1.0 if label == 0 else self.background_weight
            for sample in cfg.sample_sets[self.sample_set]:
                if any(match(sample.skim_name, pattern) for pattern in data["sample_patterns"]):
                    samples.append(sample.with_label_and_loss_weight(label, loss_weight))
                    continue

        # define arguments
        train_kwargs = dict(
            model_name=self.get_model_name(),
            model_prefix="",
            model_suffix="",
            data_dirs=dict(cfg.skim_dirs),
            cache_dir=os.path.join(os.environ["TN_DATA_DIR"], "training_cache"),
            tensorboard_dir=(
                None
                if self.skip_tensorboard
                else os.getenv("TN_TENSORBOARD_DIR", os.path.join(os.environ["TN_DATA_DIR"], "tensorboard"))
            ),
            tensorboard_version=self.version,
            clear_existing_tensorboard=True,
            model_dir=self.output()["saved_model"].parent.path,
            samples=samples,
            selections=cfg.selection_sets[self.selection_set],
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
            validation_batch_size=self.batch_size * 16,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            learning_rate_patience=self.learning_rate_patience,
            early_stopping_patience=self.early_stopping_patience,
            max_epochs=self.max_epochs,
            validate_every=self.validate_every,
            parameterize_year=True,
            parameterize_spin=True,
            parameterize_mass=True,
            n_folds=self.n_folds,
            fold_index=self.fold,
            validation_fraction=0.25,
            seed=self.seed,
        )

        # run the training
        ret = train(**train_kwargs)
        if ret is None:
            raise Exception("training did not provide a model, probably due to manual stop (ctrl+c)")
