# coding: utf-8

"""
Interface to the X -> HH -> bbtautau classifier.

Details:

- Folds:
    The fold_index is defined as "event_number % 5". The model corresponding to a fold index f was trained on events
    that did not see events with that fold index (neither in training nor in validation data). When the event was seen
    during the training of that fold, nan's are returned.
- Ensembling:
    Each model file can contain a single model or an ensemble of models whose outputs are aggregated by averaring
    (mixture-of-experts approach). The interace is identical in both cases since ensembles are created on graph level.
- Inputs:
    See NNInterface.predict() for the list of input features. Please stick to the type hints. When the event has a fat
    jet that passes the pnet cut, the fatjet features are expected and bjet features are automatically set to default
    values (and vice-versa). Input normalization is done internally by the model.
- Outputs:
    The model predicts three (softmax'ed) class probabilities for HH, TT and DY.
"""

from __future__ import annotations

import os
import enum

import numpy as np
import tensorflow as tf  # type: ignore[import-untyped]


class Era(enum.Enum):

    e2016APV = 0
    e2016 = 1
    e2017 = 2
    e2018 = 3


def rotate_to_phi(
    ref_phi: float,
    px: float,
    py: float,
) -> tuple[float, float]:
    """
    Rotates a momentum vector given by *px* and *py* in the transverse plane to a reference phi angle *ref_phi*.
    The rotated px and py components are returned in a 2-tuple.
    """
    new_phi = np.arctan2(py, px) - ref_phi
    pt = (px**2 + py**2)**0.5
    return pt * np.cos(new_phi), pt * np.sin(new_phi)


class NNInterface(object):

    n_folds = 5
    n_out = 3

    def __init__(self, fold_index: int, model_path: str) -> None:
        super().__init__()

        # attributes
        self.model_path = os.path.expandvars(model_path)
        self.fold_index = fold_index

        # load the model
        self.model = tf.saved_model.load(model_path)

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.predict(*args, **kwargs)

    def predict(
        self,
        # event number for fold check
        event_number: int,
        # parameterized input features
        spin: int,
        mass: float,
        # categorical input features
        era: Era,
        pair_type: int,  # 0: mutau, 1: etau, 2: tautau
        dau1_dm: int,
        dau2_dm: int,
        dau1_charge: int,
        dau2_charge: int,
        has_bjet_pair: int,  # whether the event has a bjet pair (same as nbjetscand > 1)
        is_boosted: int,  # whether the event has a fatjet
        # continuous input features
        met_px: float,
        met_py: float,
        met_cov00: float,
        met_cov01: float,
        met_cov11: float,
        dau1_e: float,
        dau1_px: float,
        dau1_py: float,
        dau1_pz: float,
        dau2_e: float,
        dau2_px: float,
        dau2_py: float,
        dau2_pz: float,
        bjet1_e: float,
        bjet1_px: float,
        bjet1_py: float,
        bjet1_pz: float,
        bjet1_btag_df: float,  # prob_b + prob_bb + prob_blep
        bjet1_cvsb: float,  # prob_c / (prob_c + prob_b + prob_bb + prob_blep)
        bjet1_cvsl: float,  # prob_c / (prob_c + prob_uds + prob_g)
        bjet1_hhbtag: float,
        bjet2_e: float,
        bjet2_px: float,
        bjet2_py: float,
        bjet2_pz: float,
        bjet2_btag_df: float,
        bjet2_cvsb: float,
        bjet2_cvsl: float,
        bjet2_hhbtag: float,
        fatjet_e: float,
        fatjet_px: float,
        fatjet_py: float,
        fatjet_pz: float,
    ) -> np.ndarray:
        # when the event was seen during the training of that fold, return nan's
        event_fold_index = event_number % self.n_folds
        if event_fold_index != self.fold_index:
            return np.full(self.n_out, np.nan, dtype=np.float32)

        # compute phi of visible leptons
        phi_lep = np.arctan2(dau1_py + dau2_py, dau1_px + dau2_px)

        # rotate all four-vectors to phi_lep
        met_px, met_py = rotate_to_phi(phi_lep, met_px, met_py)
        dau1_px, dau1_py = rotate_to_phi(phi_lep, dau1_px, dau1_py)
        dau2_px, dau2_py = rotate_to_phi(phi_lep, dau2_px, dau2_py)

        # rotate bjets if existing or set all features to defaults
        if has_bjet_pair:
            bjet1_px, bjet1_py = rotate_to_phi(phi_lep, bjet1_px, bjet1_py)
            bjet2_px, bjet2_py = rotate_to_phi(phi_lep, bjet2_px, bjet2_py)
        else:
            bjet1_e, bjet1_px, bjet1_py, bjet1_pz = 4 * (0.0,)
            bjet2_e, bjet2_px, bjet2_py, bjet2_pz = 4 * (0.0,)
            bjet1_btag_df, bjet1_cvsb, bjet1_cvsl, bjet1_hhbtag = 4 * (-1.0,)
            bjet2_btag_df, bjet2_cvsb, bjet2_cvsl, bjet2_hhbtag = 4 * (-1.0,)

        # rotate fatjet if existing or set all features to defaults
        if is_boosted:
            fatjet_px, fatjet_py = rotate_to_phi(phi_lep, fatjet_px, fatjet_py)
        else:
            fatjet_e, fatjet_px, fatjet_py, fatjet_pz = 4 * (0.0,)

        # build input tensors
        cont_inputs = [
            met_px, met_py, met_cov00, met_cov01, met_cov11,
            dau1_px, dau1_py, dau1_pz, dau1_e,
            dau2_px, dau2_py, dau2_pz, dau2_e,
            bjet1_px, bjet1_py, bjet1_pz, bjet1_e, bjet1_btag_df, bjet1_cvsb, bjet1_cvsl, bjet1_hhbtag,
            bjet2_px, bjet2_py, bjet2_pz, bjet2_e, bjet2_btag_df, bjet2_cvsb, bjet2_cvsl, bjet2_hhbtag,
            fatjet_px, fatjet_py, fatjet_pz, fatjet_e,
            mass,
        ]
        cat_inputs = list(map(int, [
            pair_type, dau1_dm, dau2_dm, dau1_charge, dau2_charge, is_boosted, has_bjet_pair, era.value, spin,
        ]))
        cont_inputs = tf.constant([cont_inputs], dtype=tf.float32)
        cat_inputs = tf.constant([cat_inputs], dtype=tf.int32)

        # evaluate the model
        predictions = self.model([cont_inputs, cat_inputs], training=False)

        return predictions.numpy()[0]


if __name__ == "__main__":
    nn0 = NNInterface(
        fold_index=0,
        model_path="/gpfs/dust/cms/user/riegerma/taunn_data/store/Training/dev_new_skims/hbtres_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_fatjet-default_pnet_ED10_LU8x128_CTdense_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR3.0e-03_YEARy_SPINy_MASSy_RSv4pre_lbn_ft_lt20_lr1_LBdefault_daurot_fatjet_FI0_SD1",  # noqa
    )

    predictions = nn0(
        event_number=0,
        spin=0,
        mass=400.0,
        era=Era.e2016,
        pair_type=0,
        dau1_dm=0,
        dau2_dm=0,
        dau1_charge=1,
        dau2_charge=-1,
        has_bjet_pair=True,
        is_boosted=False,
        met_px=230.0,
        met_py=-50.0,
        met_cov00=40000.0,
        met_cov01=35000.0,
        met_cov11=38800.0,
        dau1_e=50.0,
        dau1_px=10.0,
        dau1_py=90.0,
        dau1_pz=-30.0,
        dau2_e=75.0,
        dau2_px=20.0,
        dau2_py=-80.0,
        dau2_pz=5.0,
        bjet1_e=95.0,
        bjet1_px=50.0,
        bjet1_py=60.0,
        bjet1_pz=-10.0,
        bjet1_btag_df=0.9,
        bjet1_cvsb=0.1,
        bjet1_cvsl=0.5,
        bjet1_hhbtag=1.11,
        bjet2_e=150.0,
        bjet2_px=-50.0,
        bjet2_py=-60.0,
        bjet2_pz=40.0,
        bjet2_btag_df=0.8,
        bjet2_cvsb=0.05,
        bjet2_cvsl=0.4,
        bjet2_hhbtag=0.98,
        fatjet_e=0.0,
        fatjet_px=0.0,
        fatjet_py=0.0,
        fatjet_pz=0.0,
    )
    print(predictions)
