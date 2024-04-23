# coding: utf-8

from __future__ import annotations

import os
import functools
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from tautaunn.util import phi_mpi_to_pi, top_info, boson_info, match, calc_mass, calc_energy, calc_mt, hh


@dataclass
class ActivationSetting:

    # name of the activation as understood by tf.keras.layers.Activation
    name: str
    # name of the kernel initializer as understood by tf.keras.layers.Dense
    weight_init: str
    # whether to apply batch normalization before or after the activation (and if at all)
    batch_norm: tuple[bool, bool]
    # name of the dropout layer under tf.keras.layers
    dropout_name: str = "Dropout"


activation_settings = {
    "elu": ActivationSetting("ELU", "he_uniform", (True, False)),
    "relu": ActivationSetting("ReLU", "he_uniform", (False, True)),
    "prelu": ActivationSetting("PReLU", "he_normal", (True, False)),
    "selu": ActivationSetting("selu", "lecun_normal", (False, False), "AlphaDropout"),
    "tanh": ActivationSetting("tanh", "glorot_normal", (True, False)),
    "softmax": ActivationSetting("softmax", "glorot_normal", (True, False)),
    "swish": ActivationSetting("swish", "glorot_uniform", (True, False)),
}

skim_dirs = {
    "2016APV": os.environ["TN_SKIMS_2016APV"],
    "2016": os.environ["TN_SKIMS_2016"],
    "2017": os.environ["TN_SKIMS_2017"],
    "2018": os.environ["TN_SKIMS_2018"],
}


@functools.cache
def get_all_skim_names() -> dict[str, list[str]]:
    # note: VBF signals are skipped!
    return {
        year: [
            d for d in os.listdir(skim_dir)
            if (
                os.path.isdir(os.path.join(skim_dir, d))
            )
        ]
        for year, skim_dir in skim_dirs.items()
    }


luminosities = {
    "2016APV": 19_500.0,
    "2016": 16_800.0,
    "2017": 41_480.0,
    "2018": 59_830.0,
}

btag_wps = {
    "2016APV": {
        "loose": 0.0508,
        "medium": 0.2598,
    },
    "2016": {
        "loose": 0.0480,
        "medium": 0.2489,
    },
    "2017": {
        "loose": 0.0532,
        "medium": 0.3040,
    },
    "2018": {
        "loose": 0.0490,
        "medium": 0.2783,
    },
}

pnet_wps = {
    "2016APV": 0.9088,
    "2016": 0.9137,
    "2017": 0.9105,
    "2018": 0.9172,
}

metnomu_et_cuts = {
    "2016APV": 160,
    "2016": 160,
    "2017": 180,
    "2018": 180,
}

masses = [
    250, 260, 270, 280, 300, 320, 350, 400, 450, 500, 550, 600, 650,
    700, 750, 800, 850, 900, 1000, 1250, 1500, 1750, 2000, 2500, 3000,
]

spins = [0, 2]


@dataclass
class Sample:
    """
    Example:
    - name: "ggF_Radion_m350"
    - skim_name: "2016APV_ggF_Radion_m350"
    - directory_name: same as name, only existing for backwards compatibility
    - year: "2016APV"  (this is more like a "campaign")
    - year_int: 2016
    - year_flag: 0
    - spin: 0
    - mass: 350.0
    """

    name: str
    year: str
    label: int | None = None
    loss_weight: float = 1.0
    spin: int = -1
    mass: float = -1.0

    YEAR_FLAGS: ClassVar[dict[str, int]] = {
        "2016APV": 0,
        "2016": 1,
        "2017": 2,
        "2018": 3,
    }

    def __hash__(self) -> int:
        return hash(self.hash_values)

    @property
    def hash_values(self) -> tuple[Any]:
        return (self.skim_name, self.year, self.label, self.loss_weight, self.spin, self.mass)

    @property
    def skim_name(self) -> str:
        return f"{self.year}_{self.name}"

    @property
    def directory_name(self) -> str:
        return self.name

    @property
    def year_int(self) -> int:
        return int(self.year[:4])

    @property
    def year_flag(self) -> int:
        return self.YEAR_FLAGS[self.year]

    def with_label_and_loss_weight(self, label: int | None, loss_weight: float = 1.0) -> Sample:
        return self.__class__(
            name=self.name,
            year=self.year,
            label=label,
            loss_weight=loss_weight,
            spin=self.spin,
            mass=self.mass,
        )


all_samples = [
    *[
        Sample(f"{res_name}{mass}", year=year, spin=spin, mass=float(mass))
        for year in luminosities.keys()
        for spin, res_name in [(0, "Rad"), (2, "Grav")]
        for mass in masses
    ],
    *[
        Sample(f"TT_{tt_channel}Lep", year=year)
        for year in luminosities.keys()
        for tt_channel in ["Fully", "Semi"]
    ],
    *[
        Sample(f"DY_{dy_suffix}", year=year)
        for year in luminosities.keys()
        for dy_suffix in [
            "Incl",
            "0J", "1J", "2J",
            "PtZ0To50", "PtZ100To250", "PtZ250To400", "PtZ400To650", "PtZ50To100", "PtZ650ToInf",
        ]
    ],
    *[
        Sample("ttHToTauTau", year=year)
        for year in luminosities.keys()
    ],
]


# helper to get a single sample by name and year
def get_sample(skim_name: str, silent: bool = False) -> Sample | None:
    for sample in all_samples:
        if sample.skim_name == skim_name:
            return sample
    if silent:
        return None
    raise ValueError(f"sample with skim_name {skim_name} not found")


# helper to select samples with skim_name patterns
def select_samples(*patterns):
    samples = []
    for pattern in patterns:
        for sample in all_samples:
            if match(sample.skim_name, pattern) and sample not in samples:
                samples.append(sample)
    return samples


train_masses_central = "320|350|400|450|500|550|600|650|700|750|800|850|900|1000|1250|1500|1750"
train_masses_all = "250|260|270|280|300|320|350|400|450|500|550|600|650|700|750|800|850|900|1000|1250|1500|1750|2000|2500|3000"  # noqa
sample_sets = {
    "default_2016APV": (samples_default_2016APV := select_samples(
        rf"^2016APV_Rad({train_masses_all})$",
        rf"^2016APV_Grav({train_masses_all})$",
        r"^2016APV_DY_PtZ.*$",
        r"^2016APV_TT_(Fully|Semi)Lep$",
        r"^2016APV_ttHToTauTau*$",
    )),
    # TODO: adjust to actual 2016
    "default_2016": (samples_default_2016 := select_samples(
        rf"^2016_Rad({train_masses_all})$",
        rf"^2016_Grav({train_masses_all})$",
        r"^2016_DY_PtZ.*$",
        r"^2016_TT_(Fully|Semi)Lep$",
        r"^2016_ttHToTauTau*$",
    )),
    "default_2016all": samples_default_2016APV + samples_default_2016,
    "default_2017": (samples_default_2017 := select_samples(
        rf"^2017_Rad({train_masses_all})$",
        rf"^2017_Grav({train_masses_all})$",
        r"^2017_DY_PtZ.*$",
        r"^2017_TT_(Fully|Semi)Lep$",
        r"^2017_ttHToTauTau*$",
    )),
    "default_2018": (samples_default_2018 := select_samples(
        rf"^2018_Rad({train_masses_all})$",
        rf"^2018_Grav({train_masses_all})$",
        r"^2018_DY_PtZ.*$",
        r"^2018_TT_(Fully|Semi)Lep$",
        r"^2018_ttHToTauTau*$",
    )),
    "default_1617": samples_default_2016APV + samples_default_2016 + samples_default_2017,
    "default": samples_default_2016APV + samples_default_2016 + samples_default_2017 + samples_default_2018,
    "test": select_samples(
        "2017_ggF_BulkGraviton_m500",
        "2017_ggF_BulkGraviton_m550",
        "2017_DY_amc_PtZ_0To50",
        "2017_DY_amc_PtZ_100To250",
        "2017_TT_semiLep",
    ),
}

# label information
# (sample patterns are evaluated on top of those selected by sample_sets)
label_sets = {
    "binary": {
        0: {"name": "Signal", "sample_patterns": [r"^201\d.*_(Rad|Grav)\d+$"]},
        1: {"name": "Background", "sample_patterns": ["201*_DY*", "201*_TT*"]},
    },
    "multi3": {
        0: {"name": "HH", "sample_patterns": [r"^201\d.*_(Rad|Grav)\d+$"]},
        1: {"name": "TT", "sample_patterns": ["201*_TT*"]},
        2: {"name": "DY", "sample_patterns": ["201*_DY*"]},
    },
    "multi4": {
        0: {"name": "HH", "sample_patterns": [r"^201\d.*_(Rad|Grav)\d+$"]},
        1: {"name": "DY", "sample_patterns": ["201*_DY*"]},
        2: {"name": "TT", "sample_patterns": ["201*_TT*"]},
        3: {"name": "TTH", "sample_patterns": ["201*_ttHToTauTau*"]},
    },
}


def with_features(original, *, add=None, remove=None):
    features = list(original)
    if remove:
        remove = remove if isinstance(remove, list) else [remove]
        features = [f for f in features if not any(match(f, p) for p in remove)]
    if add:
        features += add if isinstance(add, list) else [add]
    return features


cont_feature_sets = {
    "reg": (cont_features_reg := [
        "met_px", "met_py", "dmet_resp_px", "dmet_resp_py", "dmet_reso_px",
        "met_cov00", "met_cov01", "met_cov11",
        "ditau_deltaphi", "ditau_deltaeta",
        *[
            f"dau{i}_{feat}"
            for i in [1, 2]
            for feat in ["px", "py", "pz", "e", "dxy", "dz", "iso"]
        ],
        *[
            f"bjet{i}_{feat}"
            for i in [1, 2]
            for feat in [
                "px", "py", "pz", "e", "btag_deepFlavor", "cID_deepFlavor",
                "pnet_bb", "pnet_cc", "pnet_b", "pnet_c", "pnet_g", "pnet_uds", "pnet_pu", "pnet_undef",
                "HHbtag",
            ]
        ],
    ]),
    "reg2": (cont_features_reg2 := [
        # order is important here since it is used as is for the tauNN
        "met_px", "met_py", "dmet_resp_px", "dmet_resp_py", "dmet_reso_px",
        "ditau_deltaphi", "ditau_deltaeta",
        "dau1_px", "dau1_py", "dau1_pz", "dau1_e", "dau1_iso",
        "dau2_px", "dau2_py", "dau2_pz", "dau2_e", "dau2_iso",
        "met_cov00", "met_cov01", "met_cov11",
        "bjet1_px", "bjet1_py", "bjet1_pz", "bjet1_e", "bjet1_btag_deepFlavor", "bjet1_cID_deepFlavor",
        "bjet2_px", "bjet2_py", "bjet2_pz", "bjet2_e", "bjet2_btag_deepFlavor", "bjet2_cID_deepFlavor",
    ]),
    "reg_nopnet": with_features(cont_features_reg, remove=["bjet*_pnet_*"]),
    "reg_nohhbtag": with_features(cont_features_reg, remove=["bjet*_HHbtag"]),
    "reg_nodf": with_features(cont_features_reg, remove=["bjet*_deepFlavor"]),
    "reg_nohl": with_features(cont_features_reg, remove=["ditau_*"]),
    "reg_nodau2iso": with_features(cont_features_reg, remove=["dau2_iso"]),
    "reg_nodaudxyz": with_features(cont_features_reg, remove=["dau*_dxy", "dau*_dz"]),
    "reg_nohhbtag_nohl": with_features(cont_features_reg, remove=["bjet*_HHbtag", "ditau_*"]),
    "post_meeting161": with_features(cont_features_reg, remove=["ditau_*", "dau*_iso", "dmet_*"]),
    "reg_reduced": (cont_features_reg_reduced := [
        "met_et", "met_cov00", "met_cov01", "met_cov11",
        *[
            f"dau{i}_{feat}"
            for i in [1, 2]
            for feat in ["px", "py", "pz", "e", "dxy", "dz"]
        ],
        *[
            f"bjet{i}_{feat}"
            for i in [1, 2]
            for feat in ["px", "py", "pz", "e", "btag_deepFlavor", "cID_deepFlavor", "HHbtag"]
        ],
    ]),
    "reg_reduced_cid": (cont_features_reg_reduced_cid := with_features(
        cont_features_reg_reduced,
        remove=["bjet*_cID_deepFlavor"],
        add=["bjet1_CvsL", "bjet1_CvsB", "bjet2_CvsL", "bjet2_CvsB"],
    )),
    "reg_reduced_cid_pnet": [
        "met_et", "met_cov00", "met_cov01", "met_cov11",
        *[
            f"dau{i}_{feat}"
            for i in [1, 2]
            for feat in ["px", "py", "pz", "e", "dxy", "dz"]
        ],
        *[
            f"bjet{i}_{feat}"
            for i in [1, 2]
            for feat in [
                "px", "py", "pz", "e",
                "btag_deepFlavor", "CvsB", "CvsL",
                "pnet_bb", "pnet_cc", "pnet_b", "pnet_c", "pnet_g", "pnet_uds", "pnet_pu", "pnet_undef",
                "HHbtag",
            ]
        ],
    ],
    "default_metrot": [
        "met_et", "met_cov00", "met_cov01", "met_cov11",
        *[
            f"dau{i}_{feat}"
            for i in [1, 2]
            for feat in ["px", "py", "pz", "e", "dxy", "dz"]
        ],
        *[
            f"bjet{i}_{feat}"
            for i in [1, 2]
            for feat in [
                "px", "py", "pz", "e",
                "btag_deepFlavor", "cID_deepFlavor", "CvsB", "CvsL",
                "pnet_bb", "pnet_cc", "pnet_b", "pnet_c", "pnet_g", "pnet_uds", "pnet_pu", "pnet_undef",
                "HHbtag",
            ]
        ],
    ],
    "default_daurot": [
        "met_px", "met_py", "dmet_resp_px", "dmet_resp_py", "dmet_reso_px", "dmet_reso_py",
        "met_cov00", "met_cov01", "met_cov11",
        *[
            f"dau{i}_{feat}"
            for i in [1, 2]
            for feat in ["px", "py", "pz", "e", "dxy", "dz"]
        ],
        *[
            f"bjet{i}_{feat}"
            for i in [1, 2]
            for feat in [
                "px", "py", "pz", "e",
                "btag_deepFlavor", "cID_deepFlavor", "CvsB", "CvsL",
                "HHbtag",
            ]
        ],
    ],
    "default_daurot_fatjet": [
        "met_px", "met_py",
        "met_cov00", "met_cov01", "met_cov11",
        *[
            f"dau{i}_{feat}"
            for i in [1, 2]
            for feat in ["px", "py", "pz", "e"]  # "dxy", "dz"
        ],
        *[
            f"bjet{i}_masked_{feat}"
            for i in [1, 2]
            for feat in [
                "px", "py", "pz", "e",
                "btag_deepFlavor", "CvsB", "CvsL",
                "HHbtag",
            ]
        ],
        *[
            f"fatjet_masked_{feat}"
            for feat in [
                "px", "py", "pz", "e",
            ]
        ],
    ],
    "full": (cont_features_full := cont_features_reg + [
        "tauH_e", "tauH_px", "tauH_py", "tauH_pz",
        "bH_e", "bH_px", "bH_py", "bH_pz",
        "HH_e", "HH_px", "HH_py", "HH_pz",
        "HHKin_mass",
        "top1_mass", "top2_mass", "W_distance", "Z_distance", "H_distance",
    ]),
    "full_svfit": cont_features_full + ["tauH_SVFIT_mass", "tauH_SVFIT_pt"],
    "reg_svfit": cont_features_reg + ["tauH_SVFIT_mass", "tauH_SVFIT_pt"],
    "class": [
        "bjet1_bID_deepFlavor", "bjet1_cID_deepFlavor", "bjet1_HHbtag",
        "bjet2_bID_deepFlavor", "bjet2_cID_deepFlavor", "bjet2_HHbtag",
        "dibjet_deltaR",
        "dau1_mt",
        "dau2_pt",
        "ditau_mt", "ditau_deltaR", "ditau_deltaeta",
        "tauH_SVFIT_E", "tauH_SVFIT_mt", "tauH_SVFIT_mass",
        "met_et",
        "top1_mass",
        "h_bb_mass",
        "hh_pt",
        "HHKin_mass_raw_chi2",
        "dphi_hbb_met",
        "deta_hbb_httvis",
        "HHKin_mass_raw",
        "diH_mass_met",
    ],
    "reg_v2": [
        "met_et",
        "met_cov00", "met_cov01", "met_cov11",
        "dau1_px", "dau1_py", "dau1_pz", "dau1_e", "dau1_dxy", "dau1_dz",
        "dau2_px", "dau2_py", "dau2_pz", "dau2_e", "dau2_dxy", "dau2_dz",
        "bjet1_px", "bjet1_py", "bjet1_pz", "bjet1_e", "bjet1_btag_deepFlavor", "bjet1_cID_deepFlavor", "bjet1_HHbtag",
        "bjet2_px", "bjet2_py", "bjet2_pz", "bjet2_e", "bjet2_btag_deepFlavor", "bjet2_cID_deepFlavor", "bjet2_HHbtag",
    ],
}

cat_feature_sets = {
    "reg": [
        # order is important here since it is used as is for the tauNN
        "pairType", "dau1_decayMode", "dau2_decayMode", "dau1_charge", "dau2_charge",
    ],
    "default": [
        "pairType", "dau1_decayMode", "dau2_decayMode", "dau1_charge", "dau2_charge", "isBoosted",
    ],
    "default_pnet": [
        "pairType", "dau1_decayMode", "dau2_decayMode", "dau1_charge", "dau2_charge", "pass_pnet",
    ],
    "default_extended": [
        "pairType", "dau1_decayMode", "dau2_decayMode", "dau1_charge", "dau2_charge", "isBoosted",
        "has_bjet1", "has_bjet2",
    ],
    "full": (cat_features_full := [
        "pairType", "dau1_decayMode", "dau2_decayMode", "dau1_charge", "dau2_charge", "isBoosted", "top_mass_idx",
    ]),
    "class": [
        "isBoosted", "pairType", "has_vbf_pair",
    ],
}

# selection sets can be strings, lists (which will be AND joined) or dictionaries with years mapping to strings or lists
# (in the latter case, the training script will choose the year automatically based on the sample)
selection_sets = {
    "baseline": (baseline_selection := [
        "nbjetscand > 1",
        "nleps == 0",
        "isOS == 1",
        "dau2_deepTauVsJet >= 5",
        (
            "((pairType == 0) & (dau1_iso < 0.15) & (isLeptrigger == 1)) | "
            "((pairType == 1) & (dau1_eleMVAiso == 1) & (isLeptrigger == 1)) | "
            "((pairType == 2) & (dau1_deepTauVsJet >= 5))"
        ),
    ]),
    "baseline_lbtag": {
        year: baseline_selection + [
            f"(bjet1_bID_deepFlavor > {w['loose']}) | (bjet2_bID_deepFlavor > {w['loose']})",
        ]
        for year, w in btag_wps.items()
    },
    "signal": {
        year: baseline_selection + [
            (
                f"(bjet1_bID_deepFlavor > {w['medium']}) | "
                f"(bjet2_bID_deepFlavor > {w['medium']}) | "
                f"((isBoosted == 1) & (bjet1_bID_deepFlavor > {w['loose']}) & (bjet2_bID_deepFlavor > {w['loose']}))"
            ),
        ]
        for year, w in btag_wps.items()
    },
    "new_baseline": [
        "nleps == 0",
        "isOS == 1",
        "dau2_deepTauVsJet >= 5",
        "((nbjetscand > 1) | (isBoosted == 1))",
        "((isLeptrigger == 1) | (isMETtrigger == 1) | (isSingleTautrigger == 1))",
        (
            "((pairType == 0) & (dau1_iso < 0.15)) | "
            "((pairType == 1) & (dau1_eleMVAiso == 1)) | "
            "((pairType == 2) & (dau1_deepTauVsJet >= 5))"
        ),
    ],
}

klub_aliases: dict[str, str] = {
    "bjet1_btag_deepFlavor": "bjet1_bID_deepFlavor",
    "bjet2_btag_deepFlavor": "bjet2_bID_deepFlavor",
    "dau1_charge": "dau1_flav / abs(dau1_flav)",
    "dau2_charge": "dau2_flav / abs(dau2_flav)",
}

klub_index_columns = [
    "EventNumber",
    "RunNumber",
    "lumi",
]

klub_category_columns = [
    "pairType",
    "nleps",
    "nbjetscand",
    "bjet1_bID_deepFlavor",
    "bjet2_bID_deepFlavor",
    "isBoosted",
    "isLeptrigger",
    "isMETtrigger",
    "isSingleTautrigger",
    "fatjet_particleNetMDJetTags_score",
]

dynamic_columns = {
    # columns needed for rotation
    (rot_phi := "dau_phi"): (
        ("dau1_pt", "dau1_phi", "dau2_pt", "dau2_phi"),
        (lambda pt1, phi1, pt2, phi2: np.arctan2(
            pt1 * np.sin(phi1) + pt2 * np.sin(phi2),
            pt1 * np.cos(phi1) + pt2 * np.cos(phi2),
        )),
    ),
    # actual columns
    "pass_pnet": (
        pass_pnet_cols := ("year_flag", "fatjet_particleNetMDJetTags_probXbb"),
        (lambda year_flag, pnet: (
            ((year_flag == 0) & (pnet >= pnet_wps["2016APV"])) |
            ((year_flag == 1) & (pnet >= pnet_wps["2016"])) |
            ((year_flag == 2) & (pnet >= pnet_wps["2017"])) |
            ((year_flag == 3) & (pnet >= pnet_wps["2018"]))
        )),
    ),
    "has_bjet1": (
        ("nbjetscand",),
        (lambda n: n >= 1),
    ),
    "has_bjet2": (
        ("nbjetscand",),
        (lambda n: n >= 2),
    ),
    "has_bjet_pair": (
        ("nbjetscand",),
        (lambda n: n >= 2),
    ),
    "dmet_resp_px": (
        ("DeepMET_ResponseTune_px", "DeepMET_ResponseTune_py", rot_phi),
        (lambda x, y, p: np.cos(-p) * x - np.sin(-p) * y),
    ),
    "dmet_resp_py": (
        ("DeepMET_ResponseTune_px", "DeepMET_ResponseTune_py", rot_phi),
        (lambda x, y, p: np.sin(-p) * x + np.cos(-p) * y),
    ),
    "dmet_reso_px": (
        ("DeepMET_ResolutionTune_px", "DeepMET_ResolutionTune_py", rot_phi),
        (lambda x, y, p: np.cos(-p) * x - np.sin(-p) * y),
    ),
    "dmet_reso_py": (
        ("DeepMET_ResolutionTune_px", "DeepMET_ResolutionTune_py", rot_phi),
        (lambda x, y, p: np.sin(-p) * x + np.cos(-p) * y),
    ),
    "met_dphi": (
        ("met_phi", rot_phi),
        (lambda a, b: phi_mpi_to_pi(a - b)),
    ),
    "met_px": (
        ("met_et", "met_dphi"),
        (lambda a, b: a * np.cos(b)),
    ),
    "met_py": (
        ("met_et", "met_dphi"),
        (lambda a, b: a * np.sin(b)),
    ),
    "dau1_dphi": (
        ("dau1_phi", rot_phi),
        (lambda a, b: phi_mpi_to_pi(a - b)),
    ),
    "dau2_dphi": (
        ("dau2_phi", rot_phi),
        (lambda a, b: phi_mpi_to_pi(a - b)),
    ),
    "genNu1_dphi": (
        ("genNu1_phi", rot_phi),
        (lambda a, b: phi_mpi_to_pi(a - b)),
    ),
    "genNu2_dphi": (
        ("genNu2_phi", rot_phi),
        (lambda a, b: phi_mpi_to_pi(a - b)),
    ),
    "dau1_px": (
        ("dau1_pt", "dau1_dphi"),
        (lambda a, b: a * np.cos(b)),
    ),
    "dau1_py": (
        ("dau1_pt", "dau1_dphi"),
        (lambda a, b: a * np.sin(b)),
    ),
    "dau1_pz": (
        ("dau1_pt", "dau1_eta"),
        (lambda a, b: a * np.sinh(b)),
    ),
    "dau1_m": (
        ("dau1_px", "dau1_py", "dau1_pz", "dau1_e"),
        (lambda x, y, z, e: np.sqrt(e ** 2 - (x ** 2 + y ** 2 + z ** 2))),
    ),
    "dau1_mt": (
        ("dau1_px", "dau1_py", "dau1_pz", "dau1_e", "met_et", "met_dphi"),
        (lambda a, b, c, d, e, f: calc_mt(a, b, c, d, e, np.zeros_like(a), f, np.zeros_like(a))),
    ),
    "dau2_px": (
        ("dau2_pt", "dau2_dphi"),
        (lambda a, b: a * np.cos(b)),
    ),
    "dau2_py": (
        ("dau2_pt", "dau2_dphi"),
        (lambda a, b: a * np.sin(b)),
    ),
    "dau2_pz": (
        ("dau2_pt", "dau2_eta"),
        (lambda a, b: a * np.sinh(b)),
    ),
    "dau2_m": (
        ("dau2_px", "dau2_py", "dau2_pz", "dau2_e"),
        (lambda x, y, z, e: np.sqrt(e ** 2 - (x ** 2 + y ** 2 + z ** 2))),
    ),
    "ditau_deltaphi": (
        ("dau1_dphi", "dau2_dphi"),
        (lambda a, b: np.abs(phi_mpi_to_pi(a - b))),
    ),
    "ditau_deltaeta": (
        ("dau1_eta", "dau2_eta"),
        (lambda a, b: np.abs(a - b)),
    ),
    "ditau_deltaR": (
        ("ditau_deltaphi", "ditau_deltaeta"),
        (lambda a, b: np.sqrt(a**2 + b**2)),
    ),
    "genNu1_px": (
        ("genNu1_pt", "genNu1_dphi"),
        (lambda a, b: a * np.cos(b)),
    ),
    "genNu1_py": (
        ("genNu1_pt", "genNu1_dphi"),
        (lambda a, b: a * np.sin(b)),
    ),
    "genNu1_pz": (
        ("genNu1_pt", "genNu1_eta"),
        (lambda a, b: a * np.sinh(b)),
    ),
    "genNu2_px": (
        ("genNu2_pt", "genNu2_dphi"),
        (lambda a, b: a * np.cos(b)),
    ),
    "genNu2_py": (
        ("genNu2_pt", "genNu2_dphi"),
        (lambda a, b: a * np.sin(b)),
    ),
    "genNu2_pz": (
        ("genNu2_pt", "genNu2_eta"),
        (lambda a, b: a * np.sinh(b)),
    ),
    "bjet1_dphi": (
        ("bjet1_phi", rot_phi),
        (lambda a, b: phi_mpi_to_pi(a - b)),
    ),
    "bjet1_px": (
        ("bjet1_pt", "bjet1_dphi"),
        (lambda a, b: a * np.cos(b)),
    ),
    "bjet1_py": (
        ("bjet1_pt", "bjet1_dphi"),
        (lambda a, b: a * np.sin(b)),
    ),
    "bjet1_pz": (
        ("bjet1_pt", "bjet1_eta"),
        (lambda a, b: a * np.sinh(b)),
    ),
    "bjet2_dphi": (
        ("bjet2_phi", rot_phi),
        (lambda a, b: phi_mpi_to_pi(a - b)),
    ),
    "bjet2_px": (
        ("bjet2_pt", "bjet2_dphi"),
        (lambda a, b: a * np.cos(b)),
    ),
    "bjet2_py": (
        ("bjet2_pt", "bjet2_dphi"),
        (lambda a, b: a * np.sin(b)),
    ),
    "bjet2_pz": (
        ("bjet2_pt", "bjet2_eta"),
        (lambda a, b: a * np.sinh(b)),
    ),
    # masked bjet features: when 1, features are set to "missing" values
    **{
        f"bjet{i}_masked_{f}": (
            (f"bjet{i}_{f}", "has_bjet_pair"),
            (lambda d: (lambda v, has_bjet: np.where(has_bjet, v, d)))(d),  # closure against context leak
        )
        for i in [1, 2]
        for f, d in [
            ("e", 0.0),
            ("px", 0.0),
            ("py", 0.0),
            ("pz", 0.0),
            ("btag_deepFlavor", -1.0),
            ("cID_deepFlavor", -1.0),
            ("CvsB", -1.0),
            ("CvsL", -1.0),
            ("HHbtag", -1.0),
        ]
    },
    # fatjet features
    "fatjet_dphi": (
        ("fatjet_phi", rot_phi),
        (lambda a, b: phi_mpi_to_pi(a - b)),
    ),
    "fatjet_px": (
        ("fatjet_pt", "fatjet_dphi"),
        (lambda a, b: a * np.cos(b)),
    ),
    "fatjet_py": (
        ("fatjet_pt", "fatjet_dphi"),
        (lambda a, b: a * np.sin(b)),
    ),
    "fatjet_pz": (
        ("fatjet_pt", "fatjet_eta"),
        (lambda a, b: a * np.sinh(b)),
    ),
    # masked fat jet features: when not 1, all features are set to 0
    **{
        f"fatjet_masked_{f}": (
            (f"fatjet_{f}", "isBoosted"),
            (lambda v, isBoosted: np.where(isBoosted, v, 0.0)),
        )
        for f in ["e", "px", "py", "pz"]
    },
    "dibjet_deltaR": (
        ("bjet1_phi", "bjet2_phi", "bjet1_eta", "bjet2_eta"),
        (lambda a, b, c, d: np.sqrt(np.abs(phi_mpi_to_pi(a - b))**2 + np.abs(c - d)**2)),
    ),
    "ditau_mt": (
        ("dau1_pt", "dau1_eta", "dau1_phi", "dau1_e", "dau2_pt", "dau2_eta", "dau2_phi", "dau2_e"),
        (lambda a, b, c, d, e, f, g, h: calc_mt(a, b, c, d, e, f, g, h)),
    ),
    "h_bb_mass": (
        ("bjet1_pt", "bjet1_eta", "bjet1_phi", "bjet1_e", "bjet2_pt", "bjet2_eta", "bjet2_phi", "bjet2_e"),
        (lambda a, b, c, d, e, f, g, h: calc_mass(a, b, c, d) + calc_mass(e, f, g, h)),
    ),
    "top1_mass": (
        top_info_fields := (
            "dau1_pt", "dau1_eta", "dau1_phi", "dau1_e", "dau2_pt", "dau2_eta", "dau2_phi", "dau2_e",
            "bjet1_pt", "bjet1_eta", "bjet1_phi", "bjet1_e", "bjet2_pt", "bjet2_eta", "bjet2_phi", "bjet2_e",
            "met_et", "met_phi",
        ),
        (lambda *args: top_info(*args, kind="top1_mass")),
    ),
    "top2_mass": (
        top_info_fields,
        (lambda *args: top_info(*args, kind="top2_mass")),
    ),
    "top_mass_idx": (
        top_info_fields,
        (lambda *args: top_info(*args, kind="indices")),
    ),
    "W_distance": (
        top_info_fields,
        (lambda *args: boson_info(*args, kind="W")),
    ),
    "Z_distance": (
        top_info_fields,
        (lambda *args: boson_info(*args, kind="Z")),
    ),
    "H_distance": (
        top_info_fields,
        (lambda *args: boson_info(*args, kind="H")),
    ),
    # "ditau_deltaR_x_sv_pt":(
    #     ("ditau_deltaR", "tauH_SVFIT_pt"),
    #     (lambda a, b: a*b)
    # )
    "tauH_dphi": (
        ("tauH_phi", rot_phi),
        (lambda a, b: phi_mpi_to_pi(a - b)),
    ),
    "tauH_px": (
        ("tauH_pt", "tauH_dphi"),
        (lambda a, b: a * np.cos(b)),
    ),
    "tauH_py": (
        ("tauH_pt", "tauH_dphi"),
        (lambda a, b: a * np.sin(b)),
    ),
    "tauH_pz": (
        ("tauH_pt", "tauH_eta"),
        (lambda a, b: a * np.sinh(b)),
    ),
    "bH_dphi": (
        ("bH_phi", rot_phi),
        (lambda a, b: phi_mpi_to_pi(a - b)),
    ),
    "bH_px": (
        ("bH_pt", "bH_dphi"),
        (lambda a, b: a * np.cos(b)),
    ),
    "bH_py": (
        ("bH_pt", "bH_dphi"),
        (lambda a, b: a * np.sin(b)),
    ),
    "bH_pz": (
        ("bH_pt", "bH_eta"),
        (lambda a, b: a * np.sinh(b)),
    ),
    "HH_e": (
        ("tauH_e", "bH_e"),
        (lambda a, b: a + b),
    ),
    "HH_px": (
        ("tauH_px", "bH_px"),
        (lambda a, b: a + b),
    ),
    "HH_py": (
        ("tauH_py", "bH_py"),
        (lambda a, b: a + b),
    ),
    "HH_pz": (
        ("tauH_pz", "bH_pz"),
        (lambda a, b: a + b),
    ),
    "tauH_SVFIT_E": (
        ("tauH_SVFIT_pt", "tauH_SVFIT_eta", "tauH_SVFIT_phi", "tauH_SVFIT_mass"),
        (lambda a, b, c, d: calc_energy(a, b, c, d)),
    ),
    "tauH_SVFIT_mt": (
        ("tauH_SVFIT_pt", "tauH_SVFIT_eta", "tauH_SVFIT_phi", "tauH_SVFIT_mass"),
        (lambda a, b, c, d: calc_energy(a, b, c, d)),
    ),
    "hh_pt": (
        hh_args := ("dau1_pt", "dau1_eta", "dau1_phi", "dau1_e", "dau2_pt", "dau2_eta", "dau2_phi", "dau2_e",
        "bjet1_pt", "bjet1_eta", "bjet1_phi", "bjet1_e", "bjet2_pt", "bjet2_eta", "bjet2_phi", "bjet2_e",
        "met_et", "met_phi", "tauH_SVFIT_pt", "tauH_SVFIT_eta", "tauH_SVFIT_phi", "tauH_SVFIT_mass",
        "HHKin_mass_raw", "HHKin_mass_raw_chi2"),
        (lambda *args: hh(*args, kind="hh_pt")),
    ),
    "deta_hbb_httvis": (
        hh_args,
        (lambda *args: hh(*args, kind="deta_hbb_httvis")),
    ),
    "dphi_hbb_met": (
        hh_args,
        (lambda *args: hh(*args, kind="dphi_hbb_met")),
    ),
    "diH_mass_met": (
        hh_args,
        (lambda *args: hh(*args, kind="dphi_hbb_met")),
    ),
    # "ditau_deltaR_x_sv_pt":(
    #     ("ditau_deltaR", "tauH_SVFIT_pt"),
    #     (lambda a, b: a*b)
    # ),
}

embedding_expected_inputs = {
    "pairType": [0, 1, 2],
    "dau1_decayMode": [-1, 0, 1, 10, 11],  # -1 for e/mu
    "dau2_decayMode": [0, 1, 10, 11],
    "dau1_charge": [-1, 1],
    "dau2_charge": [-1, 1],
    "spin": [0, 2],
    "year": [0, 1, 2, 3],
    "isBoosted": [0, 1],
    "pass_pnet": [0, 1],
    "top_mass_idx": [0, 1, 2, 3],
    "has_bjet1": [0, 1],
    "has_bjet2": [0, 1],
}


@dataclass
class RegressionSet:

    model_files: dict[int, str]
    cont_feature_set: str
    cat_feature_set: str
    parameterize_year: bool = False
    parameterize_spin: bool = True
    parameterize_mass: bool = True
    use_reg_outputs: bool = True
    use_reg_last_layer: bool = True
    use_cls_outputs: bool = True
    use_cls_last_layer: bool = True
    fade_in: tuple[int, int] = (0, 0)
    fine_tune: dict[str, Any] | None = None
    feed_lbn: bool = False

    def copy(self, **attrs) -> RegressionSet:
        kwargs = self.__dict__.copy()
        kwargs.update(attrs)
        return self.__class__(**kwargs)


regression_sets = {
    "default": (default_reg_set := RegressionSet(
        model_files={
            fold: os.path.join(os.getenv("TN_REG_MODEL_DIR"), "reg_mass_para_class_l2n400_removeMoreVars_addBkgs_addlast_set_1")  # noqa
            for fold in range(10)
        },
        cont_feature_set="reg2",
        cat_feature_set="reg",
        parameterize_year=False,
        parameterize_spin=True,
        parameterize_mass=True,
        use_reg_outputs=False,
        use_reg_last_layer=True,
        use_cls_outputs=False,
        use_cls_last_layer=True,
        fade_in=(150, 20),
        fine_tune=None,
        feed_lbn=False,
    )),
    "v2": (reg_set_v2 := RegressionSet(
        model_files={
            fold: os.path.join(os.getenv("TN_REG_MODEL_DIR"), "ttreg_ED5_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS4096_OPadam_LR3.0e-03_YEARy_SPINy_MASSy_FI0_SD1")  # noqa
            for fold in range(10)
        },
        cont_feature_set="reg_v2",
        cat_feature_set="default",
        parameterize_year=True,
        parameterize_spin=True,
        parameterize_mass=True,
        use_reg_outputs=False,
        use_reg_last_layer=True,
        use_cls_outputs=False,
        use_cls_last_layer=True,
        fade_in=(150, 20),
        fine_tune=None,
        feed_lbn=False,
    )),
    "v2_ft": reg_set_v2.copy(
        fine_tune={
            # use same norm as dnn
            "l2_norm": lambda dnn_l2_norm: dnn_l2_norm * 4,
            # use current learning rate, but with two reverse reduction steps
            "learning_rate": lambda dnn_initial_lr, current_lr: current_lr * 0.5**-2,
        },
    ),
    # "v2_lbn": reg_set_v2.copy(feed_lbn=True),
    # "v2_lbn_passall": reg_set_v2.copy(feed_lbn=True, use_reg_outputs=True, use_cls_outputs=True),
    "v3": (reg_set_v3 := RegressionSet(
        model_files={
            fold: os.path.join(os.getenv("TN_REG_MODEL_DIR_TOBI"), f"daurot_v3/tautaureg_PSbaseline_LSmulti4_SSdefault_FSdefault_daurot-default_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS4096_OPadam_LR3.0e-03_YEARy_SPINy_MASSy_FI{fold}_SD1_val_metric_sum")  # noqa
            for fold in range(5)
        },
        cont_feature_set="default_daurot",
        cat_feature_set="default",
        parameterize_year=True,
        parameterize_spin=True,
        parameterize_mass=True,
        use_reg_outputs=False,
        use_reg_last_layer=True,
        use_cls_outputs=False,
        use_cls_last_layer=True,
        fade_in=(150, 20),
        fine_tune=None,
        feed_lbn=False,
    )),
    "v3_lbn": reg_set_v3.copy(feed_lbn=True),
    "v3_lbn_ft_lt10_lr2": reg_set_v3.copy(
        feed_lbn=True,
        fine_tune={
            "l2_norm": lambda dnn_l2_norm: dnn_l2_norm * 10,
            "learning_rate": lambda dnn_initial_lr, current_lr: current_lr * 0.5**-2,
        },
    ),
    "v3_lbn_ft_lt20_lr1": reg_set_v3.copy(
        feed_lbn=True,
        fine_tune={
            "l2_norm": lambda dnn_l2_norm: dnn_l2_norm * 20,
            "learning_rate": lambda dnn_initial_lr, current_lr: current_lr * 0.5**-1,
        },
    ),
    "v3_ft_lt20_lr1": reg_set_v3.copy(
        fine_tune={
            "l2_norm": lambda dnn_l2_norm: dnn_l2_norm * 20,
            "learning_rate": lambda dnn_initial_lr, current_lr: current_lr * 0.5**-1,
        },
    ),
    "v4_1fold": (reg_set_v4pre := RegressionSet(
        model_files={
            # just one fold
            0: os.path.join(os.getenv("TN_REG_MODEL_DIR_TOBI"), "new_skims_all_samples/tautaureg_PSnew_baseline_LSmulti4_SSdefault_FSdefault_daurot_fatjet-default_pnet_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR3.0e-03_YEARy_SPINy_MASSy_FI0_SD1")  # noqa
        },
        cont_feature_set="default_daurot_fatjet",
        cat_feature_set="default_pnet",
        parameterize_year=True,
        parameterize_spin=True,
        parameterize_mass=True,
        use_reg_outputs=False,
        use_reg_last_layer=True,
        use_cls_outputs=False,
        use_cls_last_layer=True,
        fade_in=(150, 20),
        fine_tune=None,
        feed_lbn=False,
    )),
    "v4pre": (reg_set_v4pre := RegressionSet(
        model_files={
            # TODO: update path to pre model file
            fold: os.path.join(os.getenv("TN_REG_MODEL_DIR_TOBI"), f"new_skims_test/tautaureg_PSnew_baseline_LSmulti4_SSdefault_FSdefault_daurot_fatjet-default_pnet_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR3.0e-03_YEARy_SPINy_MASSy_FI0_SD1")  # noqa
            for fold in range(5)
        },
        cont_feature_set="default_daurot_fatjet",
        cat_feature_set="default_extended",
        parameterize_year=True,
        parameterize_spin=True,
        parameterize_mass=True,
        use_reg_outputs=False,
        use_reg_last_layer=True,
        use_cls_outputs=False,
        use_cls_last_layer=True,
        fade_in=(150, 20),
        fine_tune=None,
        feed_lbn=False,
    )),
    "v4pre_lbn_ft_lt20_lr1": reg_set_v4pre.copy(
        feed_lbn=True,
        fine_tune={
            "l2_norm": lambda dnn_l2_norm: dnn_l2_norm * 20,
            "learning_rate": lambda dnn_initial_lr, current_lr: current_lr * 0.5**-1,
        },
    ),
}


@dataclass
class LBNSet:

    input_features: list[str | None]
    output_features: list[str]
    boost_mode: str
    n_particles: int
    n_restframes: int | None = None

    def copy(self, **attrs) -> LBNSet:
        kwargs = self.__dict__.copy()
        kwargs.update(attrs)
        return self.__class__(**kwargs)


lbn_sets = {
    "test": LBNSet(
        input_features=[
            "dau1_e", "dau1_px", "dau1_py", "dau1_pz",
            "dau2_e", "dau2_px", "dau2_py", "dau2_pz",
            "bjet1_e", "bjet1_px", "bjet1_py", "bjet1_pz",
            "bjet2_e", "bjet2_px", "bjet2_py", "bjet2_pz",
            None, "met_px", "met_py", None,
            None, "dmet_resp_px", "dmet_resp_py", None,
        ],
        output_features=["E", "pt", "eta", "m", "pair_cos"],
        boost_mode="pairs",
        n_particles=7,
    ),
    "test2": LBNSet(
        input_features=[
            "dau1_e", "dau1_px", "dau1_py", "dau1_pz",
            "dau2_e", "dau2_px", "dau2_py", "dau2_pz",
            "bjet1_e", "bjet1_px", "bjet1_py", "bjet1_pz",
            "bjet2_e", "bjet2_px", "bjet2_py", "bjet2_pz",
            "tauH_e", "tauH_px", "tauH_py", "tauH_pz",
            "bH_e", "bH_px", "bH_py", "bH_pz",
            None, "met_px", "met_py", None,
            None, "dmet_resp_px", "dmet_resp_py", None,
        ],
        output_features=["E", "pt", "eta", "m", "pair_cos"],
        boost_mode="pairs",
        n_particles=7,
    ),
    "test3": (lbn_test3 := LBNSet(
        input_features=[
            "dau1_e", "dau1_px", "dau1_py", "dau1_pz",
            "dau2_e", "dau2_px", "dau2_py", "dau2_pz",
            "bjet1_e", "bjet1_px", "bjet1_py", "bjet1_pz",
            "bjet2_e", "bjet2_px", "bjet2_py", "bjet2_pz",
            "tauH_e", "tauH_px", "tauH_py", "tauH_pz",
            "bH_e", "bH_px", "bH_py", "bH_pz",
            "HH_e", "HH_px", "HH_py", "HH_pz",
            None, "met_px", "met_py", None,
            None, "dmet_resp_px", "dmet_resp_py", None,
        ],
        output_features=["E", "pt", "eta", "m", "pair_cos"],
        boost_mode="pairs",
        n_particles=7,
    )),
    "test4": (lbn_test4 := LBNSet(
        input_features=[
            "dau1_e", "dau1_px", "dau1_py", "dau1_pz",
            "dau2_e", "dau2_px", "dau2_py", "dau2_pz",
            "bjet1_e", "bjet1_px", "bjet1_py", "bjet1_pz",
            "bjet2_e", "bjet2_px", "bjet2_py", "bjet2_pz",
            "tauH_e", "tauH_px", "tauH_py", "tauH_pz",
            "bH_e", "bH_px", "bH_py", "bH_pz",
            "HH_e", "HH_px", "HH_py", "HH_pz",
            None, "met_px", "met_py", None,
        ],
        output_features=["E", "pt", "eta", "m", "pair_cos", "pair_dr"],
        boost_mode="pairs",
        n_particles=7,
    )),
    "test4_metfix": (lbn_test4_metfix := LBNSet(
        input_features=[
            "dau1_e", "dau1_px", "dau1_py", "dau1_pz",
            "dau2_e", "dau2_px", "dau2_py", "dau2_pz",
            "bjet1_e", "bjet1_px", "bjet1_py", "bjet1_pz",
            "bjet2_e", "bjet2_px", "bjet2_py", "bjet2_pz",
            "tauH_e", "tauH_px", "tauH_py", "tauH_pz",
            "bH_e", "bH_px", "bH_py", "bH_pz",
            "HH_e", "HH_px", "HH_py", "HH_pz",
            None, "met_et", None, None,
        ],
        output_features=["E", "pt", "eta", "m", "pair_cos", "pair_dr"],
        boost_mode="pairs",
        n_particles=7,
    )),
    "test5": lbn_test4_metfix.copy(boost_mode="product", n_restframes=4),
    "default_metrot": LBNSet(
        input_features=[
            "dau1_e", "dau1_px", "dau1_py", "dau1_pz",
            "dau2_e", "dau2_px", "dau2_py", "dau2_pz",
            "bjet1_e", "bjet1_px", "bjet1_py", "bjet1_pz",
            "bjet2_e", "bjet2_px", "bjet2_py", "bjet2_pz",
            "tauH_e", "tauH_px", "tauH_py", "tauH_pz",
            "bH_e", "bH_px", "bH_py", "bH_pz",
            "HH_e", "HH_px", "HH_py", "HH_pz",
            None, "met_et", None, None,
        ],
        output_features=["E", "pt", "eta", "m", "pair_cos"],
        boost_mode="pairs",
        n_particles=8,
    ),
    "default_daurot": LBNSet(
        input_features=[
            "dau1_e", "dau1_px", "dau1_py", "dau1_pz",
            "dau2_e", "dau2_px", "dau2_py", "dau2_pz",
            "bjet1_e", "bjet1_px", "bjet1_py", "bjet1_pz",
            "bjet2_e", "bjet2_px", "bjet2_py", "bjet2_pz",
            "tauH_e", "tauH_px", "tauH_py", "tauH_pz",
            "bH_e", "bH_px", "bH_py", "bH_pz",
            "HH_e", "HH_px", "HH_py", "HH_pz",
            None, "met_px", "met_py", None,
            None, "dmet_resp_px", "dmet_resp_py", None,
            None, "dmet_reso_px", "dmet_reso_py", None,
        ],
        output_features=["E", "pt", "eta", "m", "pair_cos"],
        boost_mode="pairs",
        n_particles=10,
    ),
    "default_daurot_fatjet": LBNSet(
        input_features=[
            "dau1_e", "dau1_px", "dau1_py", "dau1_pz",
            "dau2_e", "dau2_px", "dau2_py", "dau2_pz",
            "bjet1_masked_e", "bjet1_masked_px", "bjet1_masked_py", "bjet1_masked_pz",
            "bjet2_masked_e", "bjet2_masked_px", "bjet2_masked_py", "bjet2_masked_pz",
            "fatjet_masked_e", "fatjet_masked_px", "fatjet_masked_py", "fatjet_masked_pz",
            None, "met_px", "met_py", None,
        ],
        output_features=["E", "pt", "eta", "m", "pair_cos"],
        boost_mode="pairs",
        n_particles=10,
    ),
}
