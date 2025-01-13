# coding: utf-8

from typing import Callable
from functools import wraps
import awkward as ak
from tautaunn.config import btag_wps, pnet_wps

channels = {
    "mutau": 0,
    "etau": 1,
    "tautau": 2,
}


def selector(
    needs: list | None = None,
    str_repr: str | None = None,
    **extra,
) -> Callable:
    def decorator(func: Callable) -> Callable:
        # declare func to be a selector
        func.is_selector = True

        # store extra data
        func.extra = extra

        # store raw list of required columns
        func.raw_columns = list(needs or [])

        # store recursive flat list of actual column names
        func.flat_columns = []
        for obj in func.raw_columns:
            if isinstance(obj, str):
                func.flat_columns.append(obj)
            elif getattr(obj, "is_selector", False):
                func.flat_columns.extend(obj.flat_columns)
            else:
                raise TypeError(f"cannot interpret columns '{obj}'")
        func.flat_columns = sorted(set(func.flat_columns), key=func.flat_columns.index)

        # store the string representation
        func.str_repr = str_repr

        @wraps(func)
        def wrapper(*args, **kwargs) -> ak.Array:
            return ak.values_astype(func(*args, **kwargs), bool)

        return wrapper
    return decorator


@selector(
    needs=["pairType", "dau1_deepTauVsJet", "dau1_iso", "dau1_eleMVAiso"],
    str_repr="((pairType == 0) & (dau1_iso < 0.15)) | ((pairType == 1) & (dau1_eleMVAiso == 1)) | ((pairType == 2) & (dau1_deepTauVsJet >= 5))",  # noqa
)
def sel_iso_first_lep(array: ak.Array, **kwargs) -> ak.Array:
    return (
        ((array.pairType == 0) & (array.dau1_iso < 0.15)) |
        ((array.pairType == 1) & (array.dau1_eleMVAiso == 1)) |
        ((array.pairType == 2) & (array.dau1_deepTauVsJet >= 5))
    )


@selector(
    needs=["isLeptrigger", "isMETtrigger", "isSingleTautrigger"],
    str_repr="((isLeptrigger == 1) | (isMETtrigger == 1) | (isSingleTautrigger == 1))",
)
def sel_trigger(array: ak.Array, **kwargs) -> ak.Array:
    return (
        (array.isLeptrigger == 1) | (array.isMETtrigger == 1) | (array.isSingleTautrigger == 1)
    )


@selector(
    needs=[sel_trigger, sel_iso_first_lep, "nleps", "nbjetscand", "isBoosted"],
    str_repr=f"({sel_trigger.str_repr}) & ({sel_iso_first_lep.str_repr}) & (nleps == 0) & ((nbjetscand > 1) | (isBoosted == 1))",  # noqa
)
def sel_baseline(array: ak.Array, **kwargs) -> ak.Array:
    return (
        sel_trigger(array, **kwargs) &
        # including cut on first isolated lepton to reduce memory footprint
        # (note that this is not called "baseline" anymore by KLUB standards)
        sel_iso_first_lep(array, **kwargs) &
        (array.nleps == 0) &
        ((array.nbjetscand > 1) | (array.isBoosted == 1))
    )


@selector(
    needs=["isOS", "dau2_deepTauVsJet", sel_iso_first_lep],
)
def sel_region_os_iso(array: ak.Array, **kwargs) -> ak.Array:
    return (
        sel_iso_first_lep(array, **kwargs) &
        (array.isOS == 1) &
        (array.dau2_deepTauVsJet >= 5)
    )


@selector(
    needs=["isOS", "dau2_deepTauVsJet", sel_iso_first_lep],
)
def sel_region_ss_iso(array: ak.Array, **kwargs) -> ak.Array:
    return (
        sel_iso_first_lep(array, **kwargs) &
        (array.isOS == 0) &
        (array.dau2_deepTauVsJet >= 5)
    )


@selector(
    needs=["isOS", "dau2_deepTauVsJet", sel_iso_first_lep],
)
def sel_region_os_noniso(array: ak.Array, **kwargs) -> ak.Array:
    return (
        sel_iso_first_lep(array, **kwargs) &
        (array.isOS == 1) &
        (array.dau2_deepTauVsJet < 5) &
        (array.dau2_deepTauVsJet >= 1)
    )


@selector(
    needs=["isOS", "dau2_deepTauVsJet", sel_iso_first_lep],
)
def sel_region_ss_noniso(array: ak.Array, **kwargs) -> ak.Array:
    return (
        sel_iso_first_lep(array, **kwargs) &
        (array.isOS == 0) &
        (array.dau2_deepTauVsJet < 5) &
        (array.dau2_deepTauVsJet >= 1)
    )


region_sels = [
    sel_region_os_iso,
    sel_region_ss_iso,
    sel_region_os_noniso,
    sel_region_ss_noniso,
]


region_sel_names = ["os_iso", "ss_iso", "os_noniso", "ss_noniso"]


def category_factory(channel: str) -> dict[str, Callable]:
    pair_type = channels[channel]

    @selector(needs=["pairType"])
    def sel_channel(array: ak.Array, **kwargs) -> ak.Array:
        return array.pairType == pair_type

    @selector(needs=["isBoosted"])
    def sel_ak8(array: ak.Array, **kwargs) -> ak.Array:
        return (
            (array.isBoosted == 1)
        )

    @selector(needs=["fatjet_particleNetMDJetTags_probXbb"])
    def sel_pnet(array: ak.Array, **kwargs) -> ak.Array:
        year = kwargs["year"]
        return (
            (array.fatjet_particleNetMDJetTags_probXbb >= pnet_wps[year])
        )

    @selector(needs=[sel_ak8, sel_pnet])
    def sel_boosted(array: ak.Array, **kwargs) -> ak.Array:
        return (
            sel_ak8(array, **kwargs) &
            sel_pnet(array, **kwargs)
        )

    def sel_combinations(main_sel, sub_sels):
        def create(sub_sel):
            @selector(
                needs=[main_sel, sub_sel],
                channel=channel,
            )
            def func(array: ak.Array, **kwargs) -> ak.Array:
                return main_sel(array, **kwargs) & sub_sel(array, **kwargs)
            return func

        return [create(sub_sel) for sub_sel in sub_sels]

    @selector(needs=["bjet1_bID_deepFlavor", "bjet2_bID_deepFlavor"])
    def sel_btag_m(array: ak.Array, **kwargs) -> ak.Array:
        year = kwargs["year"]
        return (
            (array.bjet1_bID_deepFlavor > btag_wps[year]["medium"]) &
            (array.bjet2_bID_deepFlavor <= btag_wps[year]["medium"])
        ) | (
            (array.bjet1_bID_deepFlavor <= btag_wps[year]["medium"]) &
            (array.bjet2_bID_deepFlavor > btag_wps[year]["medium"])
        )

    @selector(needs=["bjet1_bID_deepFlavor", "bjet2_bID_deepFlavor"])
    def sel_btag_mm(array: ak.Array, **kwargs) -> ak.Array:
        year = kwargs["year"]
        return (
            (array.bjet1_bID_deepFlavor > btag_wps[year]["medium"]) &
            (array.bjet2_bID_deepFlavor > btag_wps[year]["medium"])
        )

    @selector(needs=["bjet1_bID_deepFlavor", "bjet2_bID_deepFlavor"])
    def sel_btag_ll(array: ak.Array, **kwargs) -> ak.Array:
        year = kwargs["year"]
        return (
            (array.bjet1_bID_deepFlavor > btag_wps[year]["loose"]) &
            (array.bjet2_bID_deepFlavor > btag_wps[year]["loose"])
        )

    @selector(needs=["bjet1_bID_deepFlavor", "bjet2_bID_deepFlavor"])
    def sel_btag_m_first(array: ak.Array, **kwargs) -> ak.Array:
        year = kwargs["year"]
        return (
            (array.bjet1_bID_deepFlavor > btag_wps[year]["medium"]) |
            (array.bjet2_bID_deepFlavor > btag_wps[year]["medium"])
        )

    @selector(needs=["tauH_mass", "bH_mass"])
    def sel_mass_window_res(array: ak.Array, **kwargs) -> ak.Array:
        return (
            (array.tauH_mass >= 15.0) &
            (array.tauH_mass <= 130) &
            (array.bH_mass >= 40.0) &
            (array.bH_mass <= 270)
        )

    @selector(needs=["tauH_mass", "fatjet_softdropMass"])
    def sel_mass_window_boosted(array: ak.Array, **kwargs) -> ak.Array:
        return (
            (array.tauH_mass >= 15.0) &
            (array.tauH_mass <= 130) &
            (array.fatjet_softdropMass <= 450.0)
        )
    
    @selector(needs=["tauH_mass", "bH_mass"])
    def sel_mass_window_res_cr(array: ak.Array, **kwargs) -> ak.Array:
        return (
            (array.tauH_mass < 15.0) |
            (array.tauH_mass > 130) |
            (array.bH_mass < 40.0) |
            (array.bH_mass > 270)
        )
    
    @selector(needs=["tauH_mass", "fatjet_softdropMass"])
    def sel_mass_window_boosted_cr(array: ak.Array, **kwargs) -> ak.Array:
        return (
            (array.tauH_mass < 15.0) |
            (array.tauH_mass > 130) |
            (array.fatjet_softdropMass > 450.0)
        )

    @selector(
        needs=[sel_baseline],
        channel=channel,
    )
    def cat_baseline(array: ak.Array, **kwargs) -> ak.Array:
        return sel_baseline(array, **kwargs)

    @selector(
        needs=[sel_baseline, sel_channel, sel_boosted, sel_btag_m, sel_mass_window_res],
        channel=channel,
    )
    def cat_resolved_1b(array: ak.Array, **kwargs) -> ak.Array:
        return (
            sel_baseline(array, **kwargs) &
            sel_channel(array, **kwargs) &
            ~sel_boosted(array, **kwargs) &
            sel_btag_m(array, **kwargs) &
            sel_mass_window_res(array, **kwargs)
        )

    @selector(
        needs=[sel_baseline, sel_channel, sel_boosted, sel_btag_mm, sel_mass_window_res],
        channel=channel,
    )
    def cat_resolved_2b(array: ak.Array, **kwargs) -> ak.Array:
        return (
            sel_baseline(array, **kwargs) &
            sel_channel(array, **kwargs) &
            ~sel_boosted(array, **kwargs) &
            sel_btag_mm(array, **kwargs) &
            sel_mass_window_res(array, **kwargs)
        )

    @selector(
        needs=[sel_baseline, sel_channel, sel_boosted, sel_mass_window_boosted],
        channel=channel,
    )
    def cat_boosted(array: ak.Array, **kwargs) -> ak.Array:
        return (
            sel_baseline(array, **kwargs) &
            sel_channel(array, **kwargs) &
            sel_boosted(array, **kwargs) &
            sel_mass_window_boosted(array, **kwargs)
        )

    @selector(
        needs=[sel_baseline, sel_channel, sel_ak8, sel_btag_mm, sel_mass_window_res],
        channel=channel,
    )
    def cat_resolved_1b_no_ak8(array: ak.Array, **kwargs) -> ak.Array:
        return (
            sel_baseline(array, **kwargs) &
            sel_channel(array, **kwargs) &
            ~sel_ak8(array, **kwargs) &
            sel_btag_m(array, **kwargs) &
            sel_mass_window_res(array, **kwargs)
        )

    @selector(
        needs=[sel_baseline, sel_channel, sel_ak8, sel_btag_mm, sel_mass_window_res],
        channel=channel,
    )
    def cat_resolved_2b_no_ak8(array: ak.Array, **kwargs) -> ak.Array:
        return (
            sel_baseline(array, **kwargs) &
            sel_channel(array, **kwargs) &
            ~sel_ak8(array, **kwargs) &
            sel_btag_mm(array, **kwargs) &
            sel_mass_window_res(array, **kwargs)
        )

    @selector(
        needs=[sel_baseline, sel_channel, sel_btag_mm, sel_mass_window_res],
        channel=channel,
    )
    def cat_resolved_2b_first(array: ak.Array, **kwargs) -> ak.Array:
        return (
            sel_baseline(array, **kwargs) &
            sel_channel(array, **kwargs) &
            sel_btag_mm(array, **kwargs) &
            sel_mass_window_res(array, **kwargs)
        )

    @selector(
        needs=[cat_boosted, cat_resolved_2b_first],
        channel=channel,
    )
    def cat_boosted_not_res2b(array: ak.Array, **kwargs) -> ak.Array:
        return (
            cat_boosted(array, **kwargs) &
            ~cat_resolved_2b_first(array, **kwargs)
        )


    @selector(
        needs=[sel_baseline, sel_channel, sel_ak8, sel_btag_mm, sel_mass_window_res_cr],
        channel=channel,
    )
    def cat_resolved_1b_no_ak8_cr(array: ak.Array, **kwargs) -> ak.Array:
        return (
            sel_baseline(array, **kwargs) &
            sel_channel(array, **kwargs) &
            ~sel_ak8(array, **kwargs) &
            sel_btag_m(array, **kwargs) &
            sel_mass_window_res_cr(array, **kwargs)
        )
    
    @selector(
        needs=[sel_baseline, sel_channel, sel_btag_mm, sel_mass_window_res_cr],
        channel=channel,
    )
    def cat_resolved_2b_first_cr(array: ak.Array, **kwargs) -> ak.Array:
        return (
            sel_baseline(array, **kwargs) &
            sel_channel(array, **kwargs) &
            sel_btag_mm(array, **kwargs) &
            sel_mass_window_res_cr(array, **kwargs)
        )

    @selector(
        needs=[cat_boosted, cat_resolved_2b_first_cr, sel_mass_window_boosted_cr],
        channel=channel,
    )
    def cat_boosted_not_res2b_cr(array: ak.Array, **kwargs) -> ak.Array:
        return (
            cat_boosted(array, **kwargs) &
            ~cat_resolved_2b_first_cr(array, **kwargs) &
            sel_mass_window_boosted_cr(array, **kwargs)
        )

    # create a dict of all selectors, but without subdivision into regions
    selectors = {
        "baseline": cat_baseline,
        "resolved1b": cat_resolved_1b,
        "resolved2b": cat_resolved_2b,
        "boosted": cat_boosted,
        "resolved1b_noak8": cat_resolved_1b_no_ak8,  # to use with boosted & resolved2b_no_ak8 or resolved2b_first & boosted_not_res2b
        "resolved2b_noak8": cat_resolved_2b_no_ak8,  # to use with boosted & resolved1b_no_ak8
        "resolved2b_first": cat_resolved_2b_first,  # to use with boosted_not_res2b & resolved1b_no_ak8
        "boosted_notres2b": cat_boosted_not_res2b,  # to use with resolved2b_first & resolved1b_no_ak8
        "resolved1b_noak8_cr": cat_resolved_1b_no_ak8_cr,
        "resolved2b_first_cr": cat_resolved_2b_first_cr,
        "boosted_notres2b_cr": cat_boosted_not_res2b_cr,
    }

    # add all region combinations
    for name, sel in list(selectors.items()):
        selectors.update({
            f"{name}_{region_name}": combined_sel
            for region_name, combined_sel in zip(
                region_sel_names,
                sel_combinations(sel, region_sels),
            )
        })

    return selectors
