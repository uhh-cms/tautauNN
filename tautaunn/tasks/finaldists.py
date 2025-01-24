
import luigi
import law
import os
import itertools
from tqdm import tqdm

from pathlib import Path

from tautaunn.tasks.base import Task
from tautaunn.tasks.datacards import WriteDatacards, _default_categories



class PlotDists(Task):
    datacards = law.CSVParameter(
        description=f"comma-separated patterns to match datacard location",
        brace_expand=True,
    )

    limits_file = luigi.Parameter(
        default=law.NO_STR,
        description="path to a limits.npz file; default: ''",
    )

    unblind_edge = luigi.FloatParameter(
        default=0.8,
        description="unblinding edge; default: 0.8 -> unblind all bins with edge < 0.8", 
    )

    control_region = luigi.BoolParameter(
        default=False,
        description="whether to plot the control region; default: False",
    )

    file_type = luigi.ChoiceParameter(
        default="png",
        choices=("png", "pdf"),
        description="type of the plot files, choices: png, pdf; default: png",
    )


    def get_card_dir(self, card):
        if any([s in card for s in ("first", "noak8", "notres2b")]): 
            if "cr_" in card:
                _, _, year, channel, cat, cat_suffix, region, sign, isolation, _, spin, _, mass = card.split("_")
            else:
                _, _, year, channel, cat, cat_suffix, sign, isolation, _, spin, _, mass = card.split("_")
        else:
            if "cr_" in card:
                _, _, year, channel, cat, region, sign, isolation, _, spin, _, mass = card.split("_")
            else:
                _, _, _, channel, cat, sign, isolation, _, spin, _, mass = card.split("_")
        return f"{year}/{channel}/{cat}/"


    def get_data_dir(self, card):
        if any([s in card for s in ("first", "noak8", "notres2b")]): 
            if "cr_" in card:
                _, _, year, channel, cat, cat_suffix, region, sign, isolation, _, spin, _, mass = card.split("_")
                data_dir = f"cat_{year}_{channel}_{cat}_{cat_suffix}_{region}_{sign}_{isolation}"
            else:
                _, _, year, channel, cat, cat_suffix, sign, isolation, _, spin, _, mass = card.split("_")
                data_dir = f"cat_{year}_{channel}_{cat}_{cat_suffix}_{sign}_{isolation}"
        else:
            if "cr_" in card:
                _, _, year, channel, cat, region, sign, isolation, _, spin, _, mass = card.split("_")
                data_dir = f"cat_{year}_{channel}_{cat}_{region}_{sign}_{isolation}"
            else:
                _, _, _, channel, cat, sign, isolation, _, spin, _, mass = card.split("_")
                data_dir = f"cat_{year}_{channel}_{cat}_{sign}_{isolation}"
        return data_dir, channel, cat, spin, mass, year


    def get_signal_name_and_dir(self, card):
        if len(card.split("_")) == 11:
            _, _, _, channel, cat, sign, isolation, _, spin, _, mass = card.split("_")
            return f"cat_{year}_{channel}_{cat}_{sign}_{isolation}", f"ggf_spin_{spin}_mass_{mass}_{year}_hbbhtt", year, mass
        if len(card.split("_")) == 12:
            _, _, year, channel, cat, cat_suffix, sign, isolation, _, spin, _, mass = card.split("_")
            return f"cat_{year}_{channel}_{cat}_{cat_suffix}_{sign}_{isolation}", f"ggf_spin_{spin}_mass_{mass}_{year}_hbbhtt", year, mass


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.file_type in ("png", "pdf")
        import re
        # Regular expression pattern to match the years 2016, 2016APV, 2017, and 2018
        year_pattern = r"2016(?:APV)?|2017|2018"
        from glob import glob
        # glob the datacards
        self.matched_cards = []
        for pattern in self.datacards:
            self.matched_cards.append(glob(pattern.replace("datacard_", "shapes_").replace(".txt", ".root")))
        self.matched_cards = list(itertools.chain(*self.matched_cards))
        try:
            years = set([re.search(year_pattern, card).group() for card in self.matched_cards])
        except AttributeError as e:
            raise ValueError(f"Couldn't match any cards with provided pattern {self.datacards}")
        if len(years) > 1 and self.limits_file != law.NO_STR:
            print(("\n WARNING: \n"
                  f"Datacard pattern includes multiple years ({years}) "
                   "and a limits file was passed (just for one year ?).\n"
                  f"This might not work as expected."))


    def output(self):
        # prepare the output directory
        d = self.local_target("", dir=True)
        # hotfix location in case TN_STORE_DIR is set to Marcel's
        output_path = d.path
        path_user = (pathlist := d.absdirname.split("/"))[int(pathlist.index("user")+1)]
        if path_user != os.environ["USER"]:
            new_path = output_path.replace(path_user, os.environ["USER"])
            print(f"replacing {path_user} with {os.environ['USER']} in output path.")
            d = self.local_target(new_path, dir=True)

        return law.FileCollection({
            card: d.child(f"{self.get_card_dir(stem:=Path(card).stem)}/{stem}.{self.file_type}", type="f")
            for card in self.matched_cards
        })


    def run(self):

        from tautaunn.plot_dists import plot_mc_data_sig, load_hists, load_reslim

        fc = self.output()
        for card, path in tqdm(fc.targets.items()):
            card_name = Path(card).stem
            data_dir, channel, cat, spin, mass, year = self.get_data_dir(card_name)
            signal_name = f"ggf_spin_{spin}_mass_{mass}_{year}_hbbhtt" if not self.control_region else None
            #stack, stack_err, sig, data, bin_edges = load_hists(card, data_dir, signal_name, year)
            stack, bkgd_errors_up, bkgd_errors_down, sig, data, bin_edges = load_hists(card, data_dir, channel, cat, signal_name, year)
            # define the signal name
            signal_name = None
            if not self.control_region:
                signal_name = " ".join(signal_name.split("_")[0:5])
                signal_name = signal_name.replace("ggf", "ggf;").replace("spin ", 's:').replace("mass ", "m:")
            plot_mc_data_sig(
                data_hist=data,
                signal_hist=sig,
                bkgd_stack=stack,
                bkgd_errors_up=bkgd_errors_up,
                bkgd_errors_down=bkgd_errors_down,
                bin_edges=bin_edges,
                year=year,
                channel=channel,
                cat=cat,
                signal_name=signal_name,
                savename=path.path,
                limit_value=None if self.limits_file == law.NO_STR else load_reslim(self.limits_file, mass),
                unblind_edge=self.unblind_edge if self.unblind_edge > 0.0 else None,
            )
