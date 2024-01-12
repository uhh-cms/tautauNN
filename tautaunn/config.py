# coding: utf-8

from __future__ import annotations

import os
import functools
from dataclasses import dataclass
from typing import Any

import numpy as np

from tautaunn.util import phi_mpi_to_pi, top_info, boson_info, match


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
    return {
        year: [
            d[5:] for d in os.listdir(skim_dir)
            if d.startswith("SKIM_") and os.path.isdir(os.path.join(skim_dir, d))
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
    - directory_name: "SKIM_ggF_Radion_m350"
    - year: "2016APV"  (this is more like a "campaign")
    - year_int: 2016
    - spin: 0
    - mass: 350.0
    """

    name: str
    year: str
    label: int | None = None
    loss_weight: float = 1.0
    spin: int = -1
    mass: float = -1.0

    def __hash__(self) -> int:
        return hash(self.hash_values)

    @property
    def hash_values(self) -> tuple[Any]:
        return (self.skim_name, self.year, self.label, self.loss_weight, self.spin, self.mass)

    @property
    def skim_name(self) -> str:
        return f"{self.year}_{self.name}"

    @property
    def directory_name(self):
        return f"SKIM_{self.name}"

    @property
    def year_int(self) -> int:
        return int(self.year[:4])

    def with_label_and_loss_weight(self, label: int | None, loss_weight: float = 1.0) -> Sample:
        return self.__class__(
            name=self.name,
            year=self.year,
            label=label,
            loss_weight=loss_weight,
            spin=self.spin,
            mass=self.mass,
        )


# Note that two things are different in 2016APV and 2016 w.r.t. 2017 and 2018:
# - Graviton samples miss "Bulk" in the name
# - buggy DY suffix, "To" -> "to" in PtZ 250, 400 and 650
all_samples = [
    *[
        Sample(f"ggF_{res_name}_m{mass}", year=year, spin=spin, mass=float(mass))
        for year in ["2016APV", "2016"]
        for spin, res_name in [(0, "Radion"), (2, "Graviton")]
        for mass in masses
    ],
    *[
        Sample(f"ggF_{res_name}_m{mass}", year=year, spin=spin, mass=float(mass))
        for year in ["2017", "2018"]
        for spin, res_name in [(0, "Radion"), (2, "BulkGraviton")]
        for mass in masses
    ],
    *[
        Sample(f"TT_{tt_channel}Lep", year=year)
        for year in luminosities.keys()
        for tt_channel in ["fully", "semi"]
    ],
    *[
        Sample(f"DY_amc_{dy_suffix}", year=year)
        for year in ["2016APV", "2016"]
        for dy_suffix in [
            "incl", "0j", "1j", "2j",
            "PtZ_0To50", "PtZ_50To100", "PtZ_100To250", "PtZ_250to400", "PtZ_400to650", "PtZ_650toInf",
        ]
    ],
    *[
        Sample(f"DY_amc_{dy_suffix}", year=year)
        for year in ["2017", "2018"]
        for dy_suffix in [
            "incl", "0j", "1j", "2j",
            "PtZ_0To50", "PtZ_50To100", "PtZ_100To250", "PtZ_250To400", "PtZ_400To650", "PtZ_650ToInf",
        ]
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


# note that graviton samples in 2016APV and 2016 are different from 2017 and 2018
train_masses_central = "320|350|400|450|500|550|600|650|700|750|800|850|900|1000|1250|1500|1750"
train_masses_all = "250|260|270|280|300|320|350|400|450|500|550|600|650|700|750|800|850|900|1000|1250|1500|1750|2000|2500|3000"
sample_sets = {
    "default_2016APV": (samples_default_2016APV := select_samples(
        rf"^2016APV_ggF_Radion_m({train_masses_all})$",
        rf"^2016APV_ggF_Graviton_m({train_masses_all})$",
        r"^2016APV_DY_amc_PtZ_.*$",
        r"^2016APV_TT_(fully|semi)Lep$",
    )),
    "default_2016": (samples_default_2016 := select_samples(
        rf"^2016_ggF_Radion_m({train_masses_all})$",
        rf"^2016_ggF_Graviton_m({train_masses_all})$",
        r"^2016_DY_amc_PtZ_.*$",
        r"^2016_TT_(fully|semi)Lep$",
    )),
    "default_2016all": samples_default_2016APV + samples_default_2016,
    "default_2017": (samples_default_2017 := select_samples(
        rf"^2017_ggF_Radion_m({train_masses_all})$",
        rf"^2017_ggF_BulkGraviton_m({train_masses_all})$",
        r"^2017_DY_amc_PtZ_.*$",
        r"^2017_TT_(fully|semi)Lep$",
    )),
    "default_2018": (samples_default_2018 := select_samples(
        rf"^2018_ggF_Radion_m({train_masses_all})$",
        rf"^2018_ggF_BulkGraviton_m({train_masses_all})$",
        r"^2018_DY_amc_PtZ_.*$",
        r"^2018_TT_(fully|semi)Lep$",
    )),
    "default": samples_default_2016APV + samples_default_2016 + samples_default_2017 + samples_default_2018,
    "test": select_samples(
        "2017_ggF_BulkGraviton_m500",
        "2017_ggF_BulkGraviton_m550",
        "2017_DY_amc_PtZ_0To50",
        "2017_DY_amc_PtZ_100To250",
        "2017_TT_semiLep",
    ),
}

label_sets = {
    "binary": {
        0: {"name": "Signal", "sample_patterns": ["201*_ggF_Radion*", r"^201\d.*_ggF_(Bulk)?Graviton_m.+$"]},
        1: {"name": "Background", "sample_patterns": ["201*_DY*", "201*_TT*"]},
    },
    "multi3": {
        0: {"name": "HH", "sample_patterns": ["201*_ggF_Radion*", r"^201\d.*_ggF_(Bulk)?Graviton_m.+$"]},
        1: {"name": "TT", "sample_patterns": ["201*_TT*"]},
        2: {"name": "DY", "sample_patterns": ["201*_DY*"]},
    },
}

cont_feature_sets = {
    "reg": [
        "met_px", "met_py", "dmet_resp_px", "dmet_resp_py", "dmet_reso_px", "met_cov00", "met_cov01", "met_cov11",
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
                "px", "py", "pz", "e", "btag_deepFlavor", "cID_deepFlavor", "pnet_bb", "pnet_cc", "pnet_b", "pnet_c",
                "pnet_g", "pnet_uds", "pnet_pu", "pnet_undef", "HHbtag",
            ]
        ],
    ],
    "reg2": [
        "met_px", "met_py", "dmet_resp_px", "dmet_resp_py", "dmet_reso_px",
        "ditau_deltaphi", "ditau_deltaeta",
        "dau1_px", "dau1_py", "dau1_pz", "dau1_e", "dau1_iso",
        "dau2_px", "dau2_py", "dau2_pz", "dau2_e", "dau2_iso",
        "met_cov00", "met_cov01", "met_cov11",
        "bjet1_px", "bjet1_py", "bjet1_pz", "bjet1_e", "bjet1_btag_deepFlavor", "bjet1_cID_deepFlavor",
        "bjet2_px", "bjet2_py", "bjet2_pz", "bjet2_e", "bjet2_btag_deepFlavor", "bjet2_cID_deepFlavor",
    ],
}

cat_feature_sets = {
    "reg": [
        "pairType", "dau1_decayMode", "dau2_decayMode", "dau1_charge", "dau2_charge",
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

static_columns = ['isBoosted',
                  *[f"tauH_SVFIT_{i}" for i in ('pt', 'eta', 'phi', 'mass')],
                  'has_vbf_pair']

dynamic_columns = {
    "DeepMET_ResolutionTune_phi": (
        ("DeepMET_ResolutionTune_px", "DeepMET_ResolutionTune_py"),
        (lambda x, y: np.arctan2(y, x)),
    ),
    "met_dphi": (
        ("met_phi", "DeepMET_ResolutionTune_phi"),
        (lambda a, b: phi_mpi_to_pi(a - b)),
    ),
    "dmet_resp_px": (
        ("DeepMET_ResponseTune_px", "DeepMET_ResponseTune_py", "DeepMET_ResolutionTune_phi"),
        (lambda x, y, p: np.cos(-p) * x - np.sin(-p) * y),
    ),
    "dmet_resp_py": (
        ("DeepMET_ResponseTune_px", "DeepMET_ResponseTune_py", "DeepMET_ResolutionTune_phi"),
        (lambda x, y, p: np.sin(-p) * x + np.cos(-p) * y),
    ),
    "dmet_reso_px": (
        ("DeepMET_ResolutionTune_px", "DeepMET_ResolutionTune_py", "DeepMET_ResolutionTune_phi"),
        (lambda x, y, p: np.cos(-p) * x - np.sin(-p) * y),
    ),
    "dmet_reso_py": (
        ("DeepMET_ResolutionTune_px", "DeepMET_ResolutionTune_py", "DeepMET_ResolutionTune_phi"),
        (lambda x, y, p: np.sin(-p) * x + np.cos(-p) * y),
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
        ("dau1_phi", "DeepMET_ResolutionTune_phi"),
        (lambda a, b: phi_mpi_to_pi(a - b)),
    ),
    "dau2_dphi": (
        ("dau2_phi", "DeepMET_ResolutionTune_phi"),
        (lambda a, b: phi_mpi_to_pi(a - b)),
    ),
    "genNu1_dphi": (
        ("genNu1_phi", "DeepMET_ResolutionTune_phi"),
        (lambda a, b: phi_mpi_to_pi(a - b)),
    ),
    "genNu2_dphi": (
        ("genNu2_phi", "DeepMET_ResolutionTune_phi"),
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
        ("bjet1_phi", "DeepMET_ResolutionTune_phi"),
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
        ("bjet2_phi", "DeepMET_ResolutionTune_phi"),
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
    "dibjet_deltaR": (
        ("bjet1_phi", "bjet2_phi", "bjet1_eta", "bjet2_eta"),
        (lambda a, b, c, d: np.sqrt(np.abs(phi_mpi_to_pi(a - b))**2 + np.abs(c - d)**2)),
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
        (lambda *args: boson_info(*args, kind="W"))
    ),
    "Z_distance": (
        top_info_fields,
        (lambda *args: boson_info(*args, kind="Z"))
    ),
    "H_distance": (
        top_info_fields,
        (lambda *args: boson_info(*args, kind="H"))
    )
    #"ditau_deltaR_x_sv_pt":(
        #("ditau_deltaR", "tauH_SVFIT_pt"),
        #(lambda a, b: a*b)
    #)
}

embedding_expected_inputs = {
    "pairType": [0, 1, 2],
    "dau1_decayMode": [-1, 0, 1, 10, 11],  # -1 for e/mu
    "dau2_decayMode": [0, 1, 10, 11],
    "dau1_charge": [-1, 1],
    "dau2_charge": [-1, 1],
    "spin": [0, 2],
    "year": [2016, 2017, 2018],
    "isBoosted": [0, 1],
    "top_mass_idx": [0,1,2,3]
}


@dataclass
class RegressionSet:
    model_files: dict[int, str]
    cont_feature_set: str
    cat_feature_set: str
    parameterize_year: bool = False
    parameterize_spin: bool = True
    parameterize_mass: bool = True
    use_last_layers: bool = False
    fine_tune: bool = False

    def copy(self, **attrs) -> RegressionSet:
        kwargs = self.__dict__.copy()
        kwargs.update(attrs)
        return self.__class__(**kwargs)


regression_sets = {
    "default": (default_reg_set := RegressionSet(
        model_files={
            fold: os.path.join(os.getenv("TN_DATA_DIR"), "reg_models/reg_mass_para_class_l2n400_removeMoreVars_addBkgs_addlast_set_1")  # noqa
            for fold in range(10)
        },
        cont_feature_set="reg2",
        cat_feature_set="reg",
        parameterize_year=False,
        parameterize_spin=True,
        parameterize_mass=True,
        fine_tune=False,
        use_last_layers=True,
    )),
    "default_ft": default_reg_set.copy(fine_tune=True),
}
