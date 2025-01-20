# coding: utf-8
from dataclasses import dataclass, field
from fnmatch import fnmatch


shape_nuisances = {}

@dataclass
class ShapeNuisance:
    name: str
    combine_name: str = ""
    processes: list[str] = field(default_factory=lambda: ["*"])
    weights: dict[str, tuple[str, str]] = field(default_factory=dict)  # original name mapped to (up, down) variations
    discriminator_suffix: tuple[str, str] = ("", "")  # name suffixes for (up, down) variations
    channels: set[str] = field(default_factory=set)
    skip: bool = False

    @classmethod
    def new(cls, *args, **kwargs):
        inst = cls(*args, **kwargs)
        shape_nuisances[inst.name] = inst
        return inst

    @classmethod
    def create_full_name(cls, name: str, *, year: str) -> str:
        return name.format(year=year)

    def __post_init__(self):
        # never skip nominal
        if self.is_nominal:
            self.skip = False

        # default combine name
        if not self.combine_name:
            self.combine_name = self.name

    @property
    def is_nominal(self) -> bool:
        return self.name == "nominal"

    def get_combine_name(self, *, year: str) -> str:
        return self.create_full_name(self.combine_name, year=year)

    def get_directions(self) -> list[str]:
        return [""] if self.is_nominal else ["up", "down"]

    def get_varied_weight(self, nominal_weight: str, direction: str) -> str:
        assert direction in ("", "up", "down")
        if direction:
            for nom, (up, down) in self.weights.items():
                if nom == nominal_weight:
                    return up if direction == "up" else down
        return nominal_weight

    def get_varied_full_weight(self, direction: str) -> str:
        assert direction in ("", "up", "down")
        # use the default weight field in case the nuisance is nominal or has no dedicated weight variations
        if not direction or not self.weights:
            return "full_weight_nominal"
        # compose the full weight field name
        return f"full_weight_{self.name}_{direction}"

    def get_varied_discriminator(self, nominal_discriminator: str, direction: str) -> str:
        assert direction in ("", "up", "down")
        suffix = ""
        if direction and (suffix := self.discriminator_suffix[direction == "down"]):
            suffix = f"_{suffix}"
        return nominal_discriminator + suffix

    def applies_to_process(self, process_name: str) -> bool:
        return any(fnmatch(process_name, pattern) for pattern in self.processes)

    def applies_to_channel(self, channel_name: str) -> bool:
        return not self.channels or channel_name in self.channels


ShapeNuisance.new(
    name="nominal",
)
ShapeNuisance.new(
    name="btag_hf",
    combine_name="CMS_btag_HF_2016_2017_2018",
    weights={"bTagweightReshape": ("bTagweightReshape_hf_up", "bTagweightReshape_hf_down")},
)
ShapeNuisance.new(
    name="btag_lf",
    combine_name="CMS_btag_LF_2016_2017_2018",
    weights={"bTagweightReshape": ("bTagweightReshape_lf_up", "bTagweightReshape_lf_down")},
)
ShapeNuisance.new(
    name="btag_lfstats1",
    combine_name="CMS_btag_lfstats1_{year}",
    weights={"bTagweightReshape": ("bTagweightReshape_lfstats1_up", "bTagweightReshape_lfstats1_down")},
)
ShapeNuisance.new(
    name="btag_lfstats2",
    combine_name="CMS_btag_lfstats2_{year}",
    weights={"bTagweightReshape": ("bTagweightReshape_lfstats2_up", "bTagweightReshape_lfstats2_down")},
)
ShapeNuisance.new(
    name="btag_hfstats1",
    combine_name="CMS_btag_hfstats1_{year}",
    weights={"bTagweightReshape": ("bTagweightReshape_hfstats1_up", "bTagweightReshape_hfstats1_down")},
)
ShapeNuisance.new(
    name="btag_hfstats2",
    combine_name="CMS_btag_hfstats2_{year}",
    weights={"bTagweightReshape": ("bTagweightReshape_hfstats2_up", "bTagweightReshape_hfstats2_down")},
)
ShapeNuisance.new(
    name="btag_cferr1",
    combine_name="CMS_btag_cfeff1_{year}",
    weights={"bTagweightReshape": ("bTagweightReshape_cferr1_up", "bTagweightReshape_cferr1_down")},
)
ShapeNuisance.new(
    name="btag_cferr2",
    combine_name="CMS_btag_cfeff2_{year}",
    weights={"bTagweightReshape": ("bTagweightReshape_cferr2_up", "bTagweightReshape_cferr2_down")},
)

for (name, add_year) in [
    ("stat0_DM0", True),
    ("stat1_DM0", True),
    ("systuncorrdmeras_DM0", True),
    ("stat0_DM1", True),
    ("stat1_DM1", True),
    ("systuncorrdmeras_DM1", True),
    ("stat0_DM10", True),
    ("stat1_DM10", True),
    ("systuncorrdmeras_DM10", True),
    ("stat0_DM11", True),
    ("stat1_DM11", True),
    ("systuncorrdmeras_DM11", True),
    ("systcorrdmeras", False),
    ("systcorrdmuncorreras", True),
    ("systcorrerasgt140", False),
    ("stat0gt140", True),
    ("stat1gt140", True),
    ("extrapgt140", False),
]:
    ShapeNuisance.new(
        name=f"id_tauid_2d_{name}",
        combine_name=f"CMS_eff_t_{name}" + (r"_{year}" if add_year else ""),
        weights={"dauSFs": (f"dauSFs_tauid_2d_{name}_up", f"dauSFs_tauid_2d_{name}_down")},
    )
# TODO: are we certain all of the following uncertainties should be uncorrelated across years?
ShapeNuisance.new(
    name="id_etauFR_barrel",
    combine_name="CMS_bbtt_etauFR_barrel_{year}",
    weights={"dauSFs": ("dauSFs_etauFR_barrel_up", "dauSFs_etauFR_barrel_down")},
)
ShapeNuisance.new(
    name="id_etauFR_endcap",
    combine_name="CMS_bbtt_etauFR_endcap_{year}",
    weights={"dauSFs": ("dauSFs_etauFR_endcap_up", "dauSFs_etauFR_endcap_down")},
)
ShapeNuisance.new(
    name="id_mutauFR_etaLt0p4",
    combine_name="CMS_bbtt_mutauFR_etaLt0p4_{year}",
    weights={"dauSFs": ("dauSFs_mutauFR_etaLt0p4_up", "dauSFs_mutauFR_etaLt0p4_down")},
)
ShapeNuisance.new(
    name="id_mutauFR_eta0p4to0p8",
    combine_name="CMS_bbtt_mutauFR_eta0p4to0p8_{year}",
    weights={"dauSFs": ("dauSFs_mutauFR_eta0p4to0p8_up", "dauSFs_mutauFR_eta0p4to0p8_down")},
)
ShapeNuisance.new(
    name="id_mutauFR_eta0p8to1p2",
    combine_name="CMS_bbtt_mutauFR_eta0p8to1p2_{year}",
    weights={"dauSFs": ("dauSFs_mutauFR_eta0p8to1p2_up", "dauSFs_mutauFR_eta0p8to1p2_down")},
)
ShapeNuisance.new(
    name="id_mutauFR_etaGt1p2to1p7",
    combine_name="CMS_bbtt_mutauFR_eta1p2to1p7_{year}",
    weights={"dauSFs": ("dauSFs_mutauFR_eta1p2to1p7_up", "dauSFs_mutauFR_eta1p2to1p7_down")},
)
ShapeNuisance.new(
    name="id_mutauFR_etaGt1p7",
    combine_name="CMS_bbtt_mutauFR_etaGt1p7_{year}",
    weights={"dauSFs": ("dauSFs_mutauFR_etaGt1p7_up", "dauSFs_mutauFR_etaGt1p7_down")},
)
ShapeNuisance.new(
    name="id_muid",
    combine_name="CMS_eff_m_id_{year}",
    weights={"dauSFs": ("dauSFs_muID_up", "dauSFs_muID_down")},
    channels={"mutau"},
)
ShapeNuisance.new(
    name="id_muiso",
    combine_name="CMS_eff_m_iso_{year}",
    weights={"dauSFs": ("dauSFs_muIso_up", "dauSFs_muIso_down")},
    channels={"mutau"},
)
ShapeNuisance.new(
    name="id_elereco",
    combine_name="CMS_eff_e_reco_{year}",
    weights={"dauSFs": ("dauSFs_eleReco_up", "dauSFs_eleReco_down")},
    channels={"etau"},
)
ShapeNuisance.new(
    name="id_eleid",
    combine_name="CMS_eff_e_id_{year}",
    weights={"dauSFs": ("dauSFs_eleID_up", "dauSFs_eleID_down")},
    channels={"etau"},
)
ShapeNuisance.new(
    name="pu_jet_id",
    combine_name="CMS_eff_j_PUJET_id_{year}",
    weights={"PUjetID_SF": ("PUjetID_SF_up", "PUjetID_SF_down")},
)
ShapeNuisance.new(
    name="trigSF_tau_DM0",
    combine_name="CMS_bbtt_{year}_trigSFTauDM0",
    weights={"trigSF": ("trigSF_tau_DM0_up", "trigSF_tau_DM0_down")},
)
ShapeNuisance.new(
    name="trigSF_tau_DM1",
    combine_name="CMS_bbtt_{year}_trigSFTauDM1",
    weights={"trigSF": ("trigSF_tau_DM1_up", "trigSF_tau_DM1_down")},
)
ShapeNuisance.new(
    name="trigSF_tau_DM10",
    combine_name="CMS_bbtt_{year}_trigSFTauDM10",
    weights={"trigSF": ("trigSF_tau_DM10_up", "trigSF_tau_DM10_down")},
)
ShapeNuisance.new(
    name="trigSF_DM11",
    combine_name="CMS_bbtt_{year}_trigSFTauDM11",
    weights={"trigSF": ("trigSF_tau_DM11_up", "trigSF_tau_DM11_down")},
)
ShapeNuisance.new(
    name="trigSF_met",
    combine_name="CMS_bbtt_{year}_trigSFMET",
    weights={"trigSF": ("trigSF_met_up", "trigSF_met_down")},
)
ShapeNuisance.new(
    name="trigSF_stau",
    combine_name="CMS_bbtt_{year}_trigSFSingleTau",
    weights={"trigSF": ("trigSF_stau_up", "trigSF_stau_down")},
)
ShapeNuisance.new(
    name="trigSF_ele",
    combine_name="CMS_bbtt_{year}_trigSFEle",
    weights={"trigSF": ("trigSF_ele_up", "trigSF_ele_down")},
    channels={"etau"},
)
ShapeNuisance.new(
    name="trigSF_mu",
    combine_name="CMS_bbtt_{year}_trigSFEMu",
    weights={"trigSF": ("trigSF_mu_up", "trigSF_mu_down")},
    channels={"mutau"},
)
ShapeNuisance.new(
    name="ees",
    combine_name="CMS_scale_e_{year}",
    discriminator_suffix=("ees_up", "ees_down"),
    channels={"etau"},
)
ShapeNuisance.new(
    name="eer",
    combine_name="CMS_res_e_{year}",
    discriminator_suffix=("eer_up", "eer_down"),
    channels={"etau"},
)
ShapeNuisance.new(
    name="fes_DM0",
    combine_name="CMS_scale_t_eFake_DM0_{year}",
    discriminator_suffix=("fes_DM0_up", "fes_DM0_down"),
)
ShapeNuisance.new(
    name="fes_DM1",
    combine_name="CMS_scale_t_eFake_DM1_{year}",
    discriminator_suffix=("fes_DM1_up", "fes_DM1_down"),
)
ShapeNuisance.new(
    name="tes_DM0",
    combine_name="CMS_scale_t_DM0_{year}",
    discriminator_suffix=("tes_DM0_up", "tes_DM0_down"),
)
ShapeNuisance.new(
    name="tes_DM1",
    combine_name="CMS_scale_t_DM1_{year}",
    discriminator_suffix=("tes_DM1_up", "tes_DM1_down"),
)
ShapeNuisance.new(
    name="tes_DM10",
    combine_name="CMS_scale_t_DM10_{year}",
    discriminator_suffix=("tes_DM10_up", "tes_DM10_down"),
)
ShapeNuisance.new(
    name="tes_DM11",
    combine_name="CMS_scale_t_DM11_{year}",
    discriminator_suffix=("tes_DM11_up", "tes_DM11_down"),
)
ShapeNuisance.new(
    name="mes",
    combine_name="CMS_scale_t_muFake_{year}",
    discriminator_suffix=("mes_up", "mes_down"),
)
ShapeNuisance.new(
    name="PUReweight",
    combine_name="CMS_pileup_{year}",
    weights={"PUReweight": ("PUReweight_up", "PUReweight_down")},
)
ShapeNuisance.new(
    name="l1_prefiring",
    combine_name="CMS_l1_prefiring_{year}",
    weights={"L1pref_weight": ("L1pref_weight_up", "L1pref_weight_down")},
)

jes_names = {
    1: "CMS_scale_j_Abs",
    2: "CMS_scale_j_Abs_{year}",
    3: "CMS_scale_j_BBEC1",
    4: "CMS_scale_j_BBEC1_{year}",
    5: "CMS_scale_j_EC2",
    6: "CMS_scale_j_EC2_{year}",
    7: "CMS_scale_j_FlavQCD",
    8: "CMS_scale_j_HF",
    9: "CMS_scale_j_HF_{year}",
    10: "CMS_scale_j_RelBal",
    11: "CMS_scale_j_RelSample_{year}",
}

for js in range(1, 12):
    ShapeNuisance.new(
        name=f"jes_{js}",
        combine_name=jes_names[js],
        discriminator_suffix=(f"jes_{js}_up", f"jes_{js}_down"),
        weights={"bTagweightReshape": (f"bTagweightReshape_jetup{js}", f"bTagweightReshape_jetdown{js}")},
    )


rate_nuisances = {}


@dataclass
class RateEffect:

    effect: str
    process: str = "*"
    year: str = "*"
    channel: str = "*"
    category: str = "*"

    def applies_to_process(self, process_name: str) -> bool:
        negate = self.process.startswith("!")
        return fnmatch(process_name, self.process.lstrip("!")) != negate

    def applies_to_year(self, year: str) -> bool:
        negate = self.year.startswith("!")
        return fnmatch(year, self.year.lstrip("!")) != negate

    def applies_to_channel(self, channel: str) -> bool:
        negate = self.channel.startswith("!")
        return fnmatch(channel, self.channel.lstrip("!")) != negate

    def applies_to_category(self, category: str) -> bool:
        negate = self.category.startswith("!")
        return fnmatch(category, self.category.lstrip("!")) != negate

    def get_up_effect(self) -> float:
        if "/" in up_effect:
            up_effect = float(self.effect.split("/")[1])
        else:
            up_effect = float(self.effect)
        return up_effect - 1

    def get_down_effect(self) -> float:
        if "/" in self.effect:
            down_effect = float(up_effect.split("/")[0])
        else:
            down_effect = 2 - float(self.effect)
        return 1 - down_effect


@dataclass
class RateNuisance:

    name: str
    rate_effects: list[RateEffect]

    @classmethod
    def new(cls, *args, **kwargs):
        inst = cls(*args, **kwargs)
        rate_nuisances[inst.name] = inst
        return inst


RateNuisance.new(
    name="BR_hbb",
    rate_effects=[
        RateEffect(process="*_hbb", effect="0.9874/1.0124"),
        RateEffect(process="*_hbbhtt", effect="0.9874/1.0124"),
    ],
)
RateNuisance.new(
    name="BR_htt",
    rate_effects=[
        RateEffect(process="*_htt", effect="0.9837/1.0165"),
        RateEffect(process="*_hbbhtt", effect="0.9837/1.0165"),
    ],
)
RateNuisance.new(
    name="pdf_gg",
    rate_effects=[RateEffect(process="TT", effect="1.042")],
)
RateNuisance.new(
    name="pdf_qqbar",
    rate_effects=[
        RateEffect(process="ST", effect="1.028"),  # conservatively from t-channel, also added to tW-channel
        RateEffect(process="WZ", effect="1.044"),
    ],
)
RateNuisance.new(
    name="pdf_Higgs_gg",
    rate_effects=[RateEffect(process="ggH_*", effect="1.019")],
)
RateNuisance.new(
    name="pdf_Higgs_qqbar",
    rate_effects=[
        RateEffect(process="qqH_*", effect="1.021"),
        RateEffect(process="WH_*", effect="1.017"),
        RateEffect(process="ZH_*", effect="1.013"),
    ],
)
RateNuisance.new(
    name="pdf_Higgs_ttH",
    rate_effects=[RateEffect(process="ttH_*", effect="1.030")],
)
RateNuisance.new(
    name="pdf_Higgs_ggHH",
    rate_effects=[RateEffect(process="ggHH_*", effect="1.030")],
)
RateNuisance.new(
    name="pdf_Higgs_qqHH",
    rate_effects=[RateEffect(process="qqHH_*", effect="1.021")],
)
RateNuisance.new(
    name="QCDscale_ttbar",
    rate_effects=[
        RateEffect(process="TT", effect="0.965/1.024"),
        RateEffect(process="ST", effect="0.979/1.031"),  # conservatively from t-channel
    ],
)
RateNuisance.new(
    name="QCDscale_VV",
    rate_effects=[RateEffect(process="WZ", effect="1.036")],
)
RateNuisance.new(
    name="QCDscale_ggH",
    rate_effects=[RateEffect(process="ggH_*", effect="1.039")],
)
RateNuisance.new(
    name="QCDscale_qqH",
    rate_effects=[RateEffect(process="qqH_*", effect="0.997/1.004")],
)
RateNuisance.new(
    name="QCDscale_VH",
    rate_effects=[
        RateEffect(process="WH_*", effect="0.993/1.005"),
        RateEffect(process="ZH_*", effect="0.970/1.038"),
    ],
)
RateNuisance.new(
    name="QCDscale_ttH",
    rate_effects=[RateEffect(process="ttH_*", effect="0.908/1.058")],
)
RateNuisance.new(
    name="QCDscale_ggHH",
    rate_effects=[RateEffect(process="ggHH_*", effect="0.770/1.060")],  # includes fully correlated mtop uncertainty

)
RateNuisance.new(
    name="QCDscale_qqHH",
    rate_effects=[RateEffect(process="qqHH_*", effect="0.9996/1.0003")],
)
RateNuisance.new(
    name="alpha_s",
    rate_effects=[
        RateEffect(process="ggH_*", effect="1.026"),
        RateEffect(process="qqH_*", effect="1.005"),
        RateEffect(process="ZH_*", effect="1.009"),
        RateEffect(process="WH_*", effect="1.009"),
        RateEffect(process="ttH_*", effect="1.020"),
    ],
)
RateNuisance.new(
    name="qqHH_pythiaDipoleOn",
    rate_effects=[RateEffect(process="qqHH_*", effect="0.781/1.219")],
)
RateNuisance.new(
    name="lumi_13TeV_2016",
    rate_effects=[RateEffect(process="!QCD", year="2016*", effect="1.010")],
)
RateNuisance.new(
    name="lumi_13TeV_2017",
    rate_effects=[RateEffect(process="!QCD", year="2017", effect="1.020")],
)
RateNuisance.new(
    name="lumi_13TeV_2018",
    rate_effects=[RateEffect(process="!QCD", year="2018", effect="1.015")],
)
RateNuisance.new(
    name="lumi_13TeV_1718",
    rate_effects=[
        RateEffect(process="!QCD", year="2017", effect="1.006"),
        RateEffect(process="!QCD", year="2018", effect="1.002"),
    ],
)
RateNuisance.new(
    name="lumi_13TeV_correlated",
    rate_effects=[
        RateEffect(process="!QCD", year="2016*", effect="1.006"),
        RateEffect(process="!QCD", year="2017", effect="1.009"),
        RateEffect(process="!QCD", year="2018", effect="1.020"),
    ],
)
RateNuisance.new(
    name="CMS_bbtt_m_wp_mismatch",
    rate_effects=[RateEffect(process="!QCD", channel="mutau", effect="1.05")],
)


def add_qcd_rate(name: str, year: str, channel: str, category: str, effect_percent: float) -> None:
    if effect_percent < 10:
        effect_str = f"{1 + effect_percent * 0.01}"
    else:
        effect_str = f"{max(1 - effect_percent * 0.01, 0.01)}/{1 + effect_percent * 0.01}"

    RateNuisance.new(
        name=f"CMS_bbtt_qcd_{name}_{year}_{channel}_{category}",
        rate_effects=[RateEffect(process="QCD", year=year, channel=channel, category=category + "*", effect=effect_str)],
    )


# taken from tables 36-39 in AN
add_qcd_rate("stat", "2016APV", "etau", "resolved1b", 8.02)
add_qcd_rate("stat", "2016APV", "mutau", "resolved1b", 3.96)
add_qcd_rate("stat", "2016APV", "tautau", "resolved1b", 2.44)
add_qcd_rate("stat", "2016APV", "mutau", "resolved2b", 33.33)
add_qcd_rate("stat", "2016APV", "tautau", "resolved2b", 33.33)
add_qcd_rate("stat", "2016APV", "tautau", "boosted", 12.2)

add_qcd_rate("stat", "2016", "etau", "resolved1b", 10.89)
add_qcd_rate("stat", "2016", "mutau", "resolved1b", 3.93)
add_qcd_rate("stat", "2016", "tautau", "resolved1b", 3.08)
add_qcd_rate("stat", "2016", "mutau", "resolved2b", 21.62)
add_qcd_rate("stat", "2016", "tautau", "resolved2b", 15.92)

add_qcd_rate("stat", "2017", "etau", "resolved1b", 9.16)
add_qcd_rate("stat", "2017", "mutau", "resolved1b", 2.72)
add_qcd_rate("stat", "2017", "tautau", "resolved1b", 2.28)
add_qcd_rate("stat", "2017", "mutau", "resolved2b", 6.59)
add_qcd_rate("stat", "2017", "tautau", "resolved2b", 11.5)
add_qcd_rate("stat", "2017", "etau", "boosted", 12.41)
add_qcd_rate("stat", "2017", "mutau", "boosted", 9.6)

add_qcd_rate("stat", "2018", "etau", "resolved1b", 6.24)
add_qcd_rate("stat", "2018", "mutau", "resolved1b", 2.17)
add_qcd_rate("stat", "2018", "tautau", "resolved1b", 1.71)
add_qcd_rate("stat", "2018", "etau", "resolved2b", 256.25)
add_qcd_rate("add", "2018", "etau", "resolved2b", 400.0)
add_qcd_rate("stat", "2018", "mutau", "resolved2b", 5.47)
add_qcd_rate("stat", "2018", "tautau", "resolved2b", 7.73)
add_qcd_rate("stat", "2018", "tautau", "boosted", 31.82)
