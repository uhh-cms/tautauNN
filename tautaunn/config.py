# coding: utf-8

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tautaunn.util import phi_mpi_to_pi


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


@dataclass
class Sample:
    name: str
    label: int | None = None
    loss_weight: float = 1.0
    spin: int = -1
    mass: float = -1.0

    def with_label_and_loss_weight(self, label: int | None, loss_weight: float = 1.0) -> Sample:
        return self.__class__(
            name=self.name,
            label=label,
            loss_weight=loss_weight,
            spin=self.spin,
            mass=self.mass,
        )


sample_sets = {
    "default": [
        # Sample("SKIM_ggF_Radion_m250", spin=0, mass=250.0),
        # Sample("SKIM_ggF_Radion_m260", spin=0, mass=260.0),
        # Sample("SKIM_ggF_Radion_m270", spin=0, mass=270.0),
        # Sample("SKIM_ggF_Radion_m280", spin=0, mass=280.0),
        # Sample("SKIM_ggF_Radion_m300", spin=0, mass=300.0),
        Sample("SKIM_ggF_Radion_m320", spin=0, mass=320.0),
        Sample("SKIM_ggF_Radion_m350", spin=0, mass=350.0),
        Sample("SKIM_ggF_Radion_m400", spin=0, mass=400.0),
        Sample("SKIM_ggF_Radion_m450", spin=0, mass=450.0),
        Sample("SKIM_ggF_Radion_m500", spin=0, mass=500.0),
        Sample("SKIM_ggF_Radion_m550", spin=0, mass=550.0),
        Sample("SKIM_ggF_Radion_m600", spin=0, mass=600.0),
        Sample("SKIM_ggF_Radion_m650", spin=0, mass=650.0),
        Sample("SKIM_ggF_Radion_m700", spin=0, mass=700.0),
        Sample("SKIM_ggF_Radion_m750", spin=0, mass=750.0),
        Sample("SKIM_ggF_Radion_m800", spin=0, mass=800.0),
        Sample("SKIM_ggF_Radion_m850", spin=0, mass=850.0),
        Sample("SKIM_ggF_Radion_m900", spin=0, mass=900.0),
        Sample("SKIM_ggF_Radion_m1000", spin=0, mass=1000.0),
        Sample("SKIM_ggF_Radion_m1250", spin=0, mass=1250.0),
        Sample("SKIM_ggF_Radion_m1500", spin=0, mass=1500.0),
        Sample("SKIM_ggF_Radion_m1750", spin=0, mass=1750.0),
        # Sample("SKIM_ggF_BulkGraviton_m250", spin=2, mass=250.0),
        # Sample("SKIM_ggF_BulkGraviton_m260", spin=2, mass=260.0),
        # Sample("SKIM_ggF_BulkGraviton_m270", spin=2, mass=270.0),
        # Sample("SKIM_ggF_BulkGraviton_m280", spin=2, mass=280.0),
        # Sample("SKIM_ggF_BulkGraviton_m300", spin=2, mass=300.0),
        Sample("SKIM_ggF_BulkGraviton_m320", spin=2, mass=320.0),
        Sample("SKIM_ggF_BulkGraviton_m350", spin=2, mass=350.0),
        Sample("SKIM_ggF_BulkGraviton_m400", spin=2, mass=400.0),
        Sample("SKIM_ggF_BulkGraviton_m450", spin=2, mass=450.0),
        Sample("SKIM_ggF_BulkGraviton_m500", spin=2, mass=500.0),
        Sample("SKIM_ggF_BulkGraviton_m550", spin=2, mass=550.0),
        Sample("SKIM_ggF_BulkGraviton_m600", spin=2, mass=600.0),
        Sample("SKIM_ggF_BulkGraviton_m650", spin=2, mass=650.0),
        Sample("SKIM_ggF_BulkGraviton_m700", spin=2, mass=700.0),
        Sample("SKIM_ggF_BulkGraviton_m750", spin=2, mass=750.0),
        Sample("SKIM_ggF_BulkGraviton_m800", spin=2, mass=800.0),
        Sample("SKIM_ggF_BulkGraviton_m850", spin=2, mass=850.0),
        Sample("SKIM_ggF_BulkGraviton_m900", spin=2, mass=900.0),
        Sample("SKIM_ggF_BulkGraviton_m1000", spin=2, mass=1000.0),
        Sample("SKIM_ggF_BulkGraviton_m1250", spin=2, mass=1250.0),
        Sample("SKIM_ggF_BulkGraviton_m1500", spin=2, mass=1500.0),
        Sample("SKIM_ggF_BulkGraviton_m1750", spin=2, mass=1750.0),
        Sample("SKIM_DY_amc_incl"),
        Sample("SKIM_TT_fullyLep"),
        Sample("SKIM_TT_semiLep"),
        # Sample("SKIM_ttHToTauTau"),
    ],
}


label_sets = {
    "binary": {
        0: {"name": "Signal", "sample_patterns": ["*SKIM_ggF_Radion*", "*SKIM_ggF_BulkGraviton*"]},
        1: {"name": "Background", "sample_patterns": ["*SKIM_DY*", "*SKIM_TT*"]},
    },
    "multi3": {
        0: {"name": "HH", "sample_patterns": ["*SKIM_ggF_Radion*", "*SKIM_ggF_BulkGraviton*"]},
        1: {"name": "TT", "sample_patterns": ["*SKIM_TT*"]},
        2: {"name": "DY", "sample_patterns": ["*SKIM_DY*"]},
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
}

cat_feature_sets = {
    "reg": [
        "pairType", "dau1_decayMode", "dau2_decayMode", "dau1_charge", "dau2_charge",
    ],
}

masses = [
    250, 260, 270, 280, 300, 320, 350, 400, 450, 500, 550, 600, 650,
    700, 750, 800, 850, 900, 1000, 1250, 1500, 1750, 2000, 2500, 3000,
]

spins = [0, 2]

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
}

embedding_expected_inputs = {
    "pairType": [0, 1, 2],
    "dau1_decayMode": [-1, 0, 1, 10, 11],  # -1 for e/mu
    "dau2_decayMode": [0, 1, 10, 11],
    "dau1_charge": [-1, 1],
    "dau2_charge": [-1, 1],
    "spin": [0, 2],
}
