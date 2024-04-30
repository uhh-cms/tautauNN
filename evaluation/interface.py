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
    See NNInterface.array_inputs() for the list of input features. Please stick to the type hints. Input normalization
    is done internally by the model.
- Outputs:
    The model predicts three (softmax'ed) class probabilities for HH, TT and DY.
"""

from __future__ import annotations

import os
import enum
from typing import Any

import numpy as np
import numpy.typing as npt
import tensorflow as tf  # type: ignore[import-untyped]


class Era(enum.Enum):

    e2016APV = 0
    e2016 = 1
    e2017 = 2
    e2018 = 3


def rotate_to_phi(
    ref_phi: npt.NDArray[np.float32],
    px: npt.NDArray[np.float32],
    py: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
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

    array_inputs = [
        # int64
        "event_number",
        # int32
        "pair_type",  # 0: mutau, 1: etau, 2: tautau
        "dau1_dm", "dau2_dm", "dau1_charge", "dau2_charge", "is_boosted", "has_bjet_pair",
        # float32
        "met_px", "met_py", "met_cov00", "met_cov01", "met_cov11",
        "dau1_e", "dau1_px", "dau1_py", "dau1_pz",
        "dau2_e", "dau2_px", "dau2_py", "dau2_pz",
        "bjet1_e", "bjet1_px", "bjet1_py", "bjet1_pz", "bjet1_btag_df", "bjet1_cvsb", "bjet1_cvsl", "bjet1_hhbtag",
        "bjet2_e", "bjet2_px", "bjet2_py", "bjet2_pz", "bjet2_btag_df", "bjet2_cvsb", "bjet2_cvsl", "bjet2_hhbtag",
        "fatjet_e", "fatjet_px", "fatjet_py", "fatjet_pz",
    ]

    def __init__(self, fold_index: int, model_path: str) -> None:
        super().__init__()

        # attributes
        self.model_path = os.path.expandvars(model_path)
        self.fold_index = fold_index

        # load the model
        self.model = tf.saved_model.load(model_path)

    def __call__(self, *args, **kwargs) -> npt.NDArray[np.float32]:
        return self.predict(*args, **kwargs)

    def predict(
        self,
        # parameterized input features
        spin: int,
        mass: float,
        era: Era,  # a single value, assuming that the predict() is not called in parallel over multuple eras
        # features are defined in the array_inputs list
        **features: dict[str, npt.NDArray[np.int64] | npt.NDArray[np.int32] | npt.NDArray[np.float32]],
    ) -> npt.NDArray[np.float32]:
        # shorthand for features
        f = DotDict(features)

        # prepare and output array that is updated in place
        batch_size = f.event_number.shape[0]
        pred = np.full((batch_size, self.n_out), np.nan, dtype=np.float32)

        # evaluate which events have the right fold index and evaluate only those
        fold_mask = (f.event_number % self.n_folds) == self.fold_index

        # when no events pass the mask, return the nan-filled array right away
        if not np.any(fold_mask):
            return pred

        # apply the mask to all arrays to ensure that preprocessing only affects events that will be evaluated
        for key, arr in f.items():
            f[key] = arr[fold_mask]

        # compute phi of visible leptons
        phi_lep = np.arctan2(f.dau1_py + f.dau2_py, f.dau1_px + f.dau2_px)

        # rotate all four-vectors to phi_lep
        f.met_px, f.met_py = rotate_to_phi(phi_lep, f.met_px, f.met_py)
        f.dau1_px, f.dau1_py = rotate_to_phi(phi_lep, f.dau1_px, f.dau1_py)
        f.dau2_px, f.dau2_py = rotate_to_phi(phi_lep, f.dau2_px, f.dau2_py)
        f.bjet1_px, f.bjet1_py = rotate_to_phi(phi_lep, f.bjet1_px, f.bjet1_py)
        f.bjet2_px, f.bjet2_py = rotate_to_phi(phi_lep, f.bjet2_px, f.bjet2_py)
        f.fatjet_px, f.fatjet_py = rotate_to_phi(phi_lep, f.fatjet_px, f.fatjet_py)

        # composite particles
        f.htt_e = f.dau1_e + f.dau2_e
        f.htt_px = f.dau1_px + f.dau2_px
        f.htt_py = f.dau1_py + f.dau2_py
        f.htt_pz = f.dau1_pz + f.dau2_pz
        f.hbb_e = f.bjet1_e + f.bjet2_e
        f.hbb_px = f.bjet1_px + f.bjet2_px
        f.hbb_py = f.bjet1_py + f.bjet2_py
        f.hbb_pz = f.bjet1_pz + f.bjet2_pz
        f.htthbb_e = f.htt_e + f.hbb_e
        f.htthbb_px = f.htt_px + f.hbb_px
        f.htthbb_py = f.htt_py + f.hbb_py
        f.htthbb_pz = f.htt_pz + f.hbb_pz
        f.httfatjet_e = f.htt_e + f.fatjet_e
        f.httfatjet_px = f.htt_px + f.fatjet_px
        f.httfatjet_py = f.htt_py + f.fatjet_py
        f.httfatjet_pz = f.htt_pz + f.fatjet_pz

        # mask bjet-related features when there was actually no bjet pair
        bj_mask = f.has_bjet_pair == 1
        f.bjet1_e[bj_mask] = f.bjet1_px[bj_mask] = f.bjet1_py[bj_mask] = f.bjet1_pz[bj_mask] = 0.0  # noqa
        f.bjet2_e[bj_mask] = f.bjet2_px[bj_mask] = f.bjet2_py[bj_mask] = f.bjet2_pz[bj_mask] = 0.0  # noqa
        f.bjet1_btag_df[bj_mask] = f.bjet1_cvsb[bj_mask] = f.bjet1_cvsl[bj_mask] = -1.0
        f.bjet2_btag_df[bj_mask] = f.bjet2_cvsb[bj_mask] = f.bjet2_cvsl[bj_mask] = -1.0
        f.hbb_e[bj_mask] = f.hbb_px[bj_mask] = f.hbb_py[bj_mask] = f.hbb_pz[bj_mask] = 0.0  # noqa
        f.htthbb_e[bj_mask] = f.htthbb_px[bj_mask] = f.htthbb_py[bj_mask] = f.htthbb_pz[bj_mask] = 0.0  # noqa

        # mask fatjet features when there was actually no fatjet
        fj_mask = f.is_boosted == 1
        f.fatjet_e[fj_mask] = f.fatjet_px[fj_mask] = f.fatjet_py[fj_mask] = f.fatjet_pz[fj_mask] = 0.0  # noqa

        # build input tensors
        cont_ones = np.ones_like(f.met_px)
        cont_inputs = tf.concat(
            [t[..., None] for t in [
                f.met_px, f.met_py, f.met_cov00, f.met_cov01, f.met_cov11,
                f.dau1_px, f.dau1_py, f.dau1_pz, f.dau1_e,
                f.dau2_px, f.dau2_py, f.dau2_pz, f.dau2_e,
                f.bjet1_px, f.bjet1_py, f.bjet1_pz, f.bjet1_e, f.bjet1_btag_df, f.bjet1_cvsb, f.bjet1_cvsl, f.bjet1_hhbtag,  # noqa
                f.bjet2_px, f.bjet2_py, f.bjet2_pz, f.bjet2_e, f.bjet2_btag_df, f.bjet2_cvsb, f.bjet2_cvsl, f.bjet2_hhbtag,  # noqa
                f.fatjet_px, f.fatjet_py, f.fatjet_pz, f.fatjet_e,
                f.htt_e, f.htt_px, f.htt_py, f.htt_pz,
                f.hbb_e, f.hbb_px, f.hbb_py, f.hbb_pz,
                f.htthbb_e, f.htthbb_px, f.htthbb_py, f.htthbb_pz,
                f.httfatjet_e, f.httfatjet_px, f.httfatjet_py, f.httfatjet_pz,
                cont_ones * mass,
            ]],
            axis=1,
        )
        cat_ones = np.ones_like(f.pair_type)
        cat_inputs = tf.concat(
            [t[..., None] for t in [
                f.pair_type, f.dau1_dm, f.dau2_dm, f.dau1_charge, f.dau2_charge, f.is_boosted, f.has_bjet_pair,
                cat_ones * era.value, cat_ones * spin,
            ]],
            axis=1,
        )

        # evaluate the model
        predictions = self.model([cont_inputs, cat_inputs], training=False)

        # insert into the output array
        pred[fold_mask] = predictions.numpy()

        return pred


class DotDict(dict):

    @classmethod
    def wrap(cls, *args, **kwargs) -> DotDict:
        wrap = lambda d: cls((k, wrap(v)) for k, v in d.items()) if isinstance(d, dict) else d  # type: ignore # noqa
        return wrap(dict(*args, **kwargs))

    def __getattr__(self, attr: str) -> Any:
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def __setattr__(self, attr: str, value: Any) -> None:
        self[attr] = value


if __name__ == "__main__":
    # helpers to artifically create a batch of events, repeating the same values
    ai = lambda v: np.array([v, v], dtype=np.int32)
    al = lambda v: np.array([v, v], dtype=np.int64)
    af = lambda v: np.array([v, v], dtype=np.float32)

    nn0 = NNInterface(
        fold_index=0,
        model_path="/nfs/dust/cms/user/riegerma/taunn_data/store/ExportEnsemble/prod3/hbtres_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_ED10_LU8x128_CTdense_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_RSv6_fi80_lbn_ft_lt20_lr1_LBdefault_daurot_fatjet_composite_FI0_SDx5",  # noqa
    )

    predictions = nn0(
        event_number=al(0),
        spin=0,
        mass=400.0,
        era=Era.e2016,
        pair_type=ai(0),
        dau1_dm=ai(0),
        dau2_dm=ai(0),
        dau1_charge=ai(1),
        dau2_charge=ai(-1),
        is_boosted=ai(0),
        has_bjet_pair=ai(1),
        met_px=af(230.0),
        met_py=af(-50.0),
        met_cov00=af(40000.0),
        met_cov01=af(35000.0),
        met_cov11=af(38800.0),
        dau1_e=af(50.0),
        dau1_px=af(10.0),
        dau1_py=af(90.0),
        dau1_pz=af(-30.0),
        dau2_e=af(75.0),
        dau2_px=af(20.0),
        dau2_py=af(-80.0),
        dau2_pz=af(5.0),
        bjet1_e=af(95.0),
        bjet1_px=af(50.0),
        bjet1_py=af(60.0),
        bjet1_pz=af(-10.0),
        bjet1_btag_df=af(0.9),
        bjet1_cvsb=af(0.1),
        bjet1_cvsl=af(0.5),
        bjet1_hhbtag=af(1.11),
        bjet2_e=af(150.0),
        bjet2_px=af(-50.0),
        bjet2_py=af(-60.0),
        bjet2_pz=af(40.0),
        bjet2_btag_df=af(0.8),
        bjet2_cvsb=af(0.05),
        bjet2_cvsl=af(0.4),
        bjet2_hhbtag=af(0.98),
        fatjet_e=af(0.0),
        fatjet_px=af(0.0),
        fatjet_py=af(0.0),
        fatjet_pz=af(0.0),
    )
    print(predictions)
