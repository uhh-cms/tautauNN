from collections import OrderedDict
import argparse
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm
from functools import reduce
from operator import add
import os

import uproot
import hist
import re
from hist import Hist, Stack
from hist.intervals import ratio_uncertainty 


import matplotlib.pyplot as plt
import mplhep as hep

from tautaunn.write_datacards_stack import br_hh_bbtt
from tautaunn.nuisances import rate_nuisances
hep.style.use("CMS")


luminosities = {  # in /fb
    '2016': 36.310,
    '2017': 41.480,
    '2018': 59.830,
}


def make_parser():
    parser = argparse.ArgumentParser(description="Plot the shapes that are fed into combine.")
    parser.add_argument("-i", "--input_dir",
                        type=str, help='directory, where the datacards & shapes are stored')
    parser.add_argument("-o", "--output_dir",
                        type=str, help='output dir to store plots (default ./input_dir.stem)',
                        default="")
    parser.add_argument("-y", "--year", required=True, type=str, help='2016, 2016APV, 2017 or 2018')
    parser.add_argument("-s", "--spin", required=False, default=None, type=str, help="spin of the signal sample")
    parser.add_argument("-l", "--limits_file", required=False, default=None,
                        type=str, help='/full/path/to/reslimits.npz file')
    parser.add_argument("-e", "--unblind_edge", required=False, default=0.8, type=float,
                        help="unblind all bins up to this edge")
    return parser


def histo_equalwidth(h):
    """ Return a histogram with equal bin widths."""
    xedges = h.axes[0].edges
    values = h.values()
    variances = h.variances()

    newh = hist.Hist.new.Reg(len(xedges)-1, xedges[0], xedges[-1], name="x").Weight()
    newh.label = h.label
    newh.view().value = values
    newh.view().variance = variances
    return newh, xedges


def loose_year(str) -> str:
    return re.sub(r"_2016APV|_2016|_2017|_2018", "", str)


def reduce_stack(stack: Stack,) -> Stack:
    main_bkgds = ['TT', 'ST', 'DY', 'W', 'QCD']
    bkgd_dict  = {loose_year(h.name): h for h in stack if any(loose_year(h.name) == s for s in main_bkgds)}
    others = reduce(add, (h for h in stack if loose_year(h.name) not in bkgd_dict))
    bkgd_dict["Others"] = others
    bkgd_dict = {k: histo_equalwidth(v)[0] for k, v in bkgd_dict.items()}
    # impose a fixed order of "Others", "QCD", "W", "ST", "DY", "TT"
    sorted_bkgd_dict = OrderedDict()
    for name in ["Others", "QCD", "W", "ST", "DY", "TT"]:
        if name in bkgd_dict:
            sorted_bkgd_dict[name] = bkgd_dict[name]
    return Stack.from_dict(sorted_bkgd_dict)


def load_hists(filename: str | Path,
               dirname: str,
               channel: str, 
               category: str,
               signal_name: str | None,
               year: str) -> tuple[Stack, Hist]:

    with uproot.open(filename) as f:
        objects = f[dirname].classnames()
        nominal_bkgd_names = [o.strip(";1") for o in objects
                            if not any(s in o for s in ["Up", "Down"])
                            and ("data_obs" not in o)]
        if not (signal_name is None):
            nominal_bkgd_names = [o for o in objects if signal_name not in o]
        shift_names = list(set([o.split("__")[1].replace("Up;1", "")
                            for o in objects
                            if "Up" in o and not any(s in o for s in ["ggf_", "vbf_" 'data_obs'])]))
        

        sig = histo_equalwidth(f[dirname][signal_name].to_hist())[0] if signal_name is not None else None
        data = histo_equalwidth(f[dirname]['data_obs'].to_hist())[0]
        hists = {"nominal": {bkgd: f[dirname][f"{bkgd}"].to_hist() for bkgd in nominal_bkgd_names}}
        for shift in shift_names:
            for direction in ["Up", "Down"]:
                for bkgd in nominal_bkgd_names:
                    try: 
                        h_ = f[dirname][f"{bkgd}__{shift}{direction}"].to_hist()
                    except uproot.exceptions.KeyInFileError:
                        # use an empty hist with the same binning
                        print(f"Warning: {bkgd}__{shift}{direction} not found in {filename}")
                        print(f"Replacing with an empty hist")
                        h_ = f[dirname][f"TT_{year}__{shift}{direction}"].to_hist()
                        hval = h_.values()
                        hvar = h_.variances()
                        hval[..., :] = 1e-5
                        hvar[..., :] = 1e-5
                    hists.setdefault(f"{shift}{direction}", {})[bkgd] = h_

    nominal_stack = Stack.from_dict(hists["nominal"])
    # shape nuisances 
    bkgd_erros_shapes = [
        sum(nominal_stack).values() - sum(Stack.from_dict(hists[f"{shape}"])).values()
        for shape in hists if shape != "nominal" and ("Up" in shape or "Down" in shape)
    ]

    # wherever the delta is negative, the shift contributes to the "up" part of the errorbar
    bkgd_errors_shape_up = [
        np.abs(np.where(delta < 0, delta, 0)) for delta in bkgd_erros_shapes
    ]

    # vice versa
    bkgd_errors_shape_down = [
        np.abs(np.where(delta > 0, delta, 0)) for delta in bkgd_erros_shapes
    ]

    # rate nuisances
    bkgd_errors_rate_up = []
    bkgd_errors_rate_down = []

    for nuisance in rate_nuisances.values():
        for process in hists["nominal"]:
            for rate_effect in nuisance.rate_effects:
                if (rate_effect.applies_to_channel(channel)
                    and rate_effect.applies_to_process(process)
                    and rate_effect.applies_to_category(category)
                    and rate_effect.applies_to_year(year)):
                    bkgd_errors_rate_up.append((rate_effect.get_up_effect() * hists["nominal"][process]).values())
                    bkgd_errors_rate_down.append((rate_effect.get_down_effect() * hists["nominal"][process]).values())

    bkgd_errors_shape_up = np.asarray(bkgd_errors_shape_up)
    bkgd_errors_shape_down = np.asarray(bkgd_errors_shape_down)
    bkgd_errors_rate_up = np.asarray(bkgd_errors_rate_up)
    bkgd_errors_rate_down = np.asarray(bkgd_errors_rate_down)
    mc_stat_error = sum(nominal_stack).variances()/4

    bkgd_errors_up = np.sqrt(np.sum(bkgd_errors_shape_up**2, axis=0) + np.sum(bkgd_errors_rate_up**2, axis=0) + mc_stat_error)
    bkgd_errors_down = np.sqrt(np.sum(bkgd_errors_shape_down**2, axis=0) + np.sum(bkgd_errors_rate_down**2, axis=0) + mc_stat_error)

    bin_edges = nominal_stack.axes[0].edges
    nominal_stack = reduce_stack(nominal_stack)
    return nominal_stack, bkgd_errors_up, bkgd_errors_down, sig, data, bin_edges


def load_reslim(file: str | Path,
                mass: float | int):
    limits = np.load(file)
    masses = limits['data']['mhh']
    exp_lim = limits['data']['limit']#*1000
    return exp_lim[np.where(masses==int(mass))][0]


def plot_asym_errorband(mc: Stack,
                        err_up: np.ndarray,
                        err_down: np.ndarray,
                        ax: plt.Axes,
                        mode: str = "errorbar",
                        label: str | None = None) -> None:
    assert mode in ("errorbar", "ratio")

    errps =  {'hatch':'////', 'facecolor':'none', 'lw': 0, 'edgecolor': 'k', 'alpha': 0.5}
    sum_mc = sum(mc)
    if mode == "errorbar":
        bottom = sum_mc.values()
        normalization = 1
    elif mode == "ratio":
        bottom = 1.
        # normalise the errors to the sum of the stack
        normalization = sum_mc.values()
    for sign, err in zip([1, -1], [err_up/normalization, err_down/normalization]): 
        ax.hist([sum_mc.axes[0].edges[:-1]],
                bins=sum_mc.axes[0].edges,
                weights=sign*err,
                bottom=bottom,
                histtype="stepfilled",
                label=label if sign == 1 else None, # to avoid double labeling
                **errps) 


def plot_mc_data_sig(data_hist: Hist,
                     signal_hist: Hist | None,
                     bkgd_stack: Stack,
                     bkgd_errors_up: np.ndarray,
                     bkgd_errors_down: np.ndarray,
                     bin_edges: list,
                     year: str,
                     channel: str,
                     cat: str,
                     savename: str | Path | None = None,
                     signal_name: str | None = None,
                     limit_value = None,
                     unblind: bool = False,
                     ) -> None:

    if not signal_name is None:
        if limit_value is None:
            label = (f"{signal_name} $\\times$ 1")
            signal_hist *= 1
        else:
            label = (f"{signal_name}\n"
                    #"$\cdot\,\sigma(\mathrm{pp}\rightarrow\mathrm{X}\rightarrow{HH})$"
                    f"$\\times$ exp. limit: {limit_value*1000:.1f} [fb]\n"
                    "$\\times$ BR($HH \\rightarrow bb\\tau\\tau$)")
            signal_hist *= limit_value * br_hh_bbtt
    # mask = (signal_hist.values()/ sum(bkgd_stack).values()) < sb_limit 
    # unblind all bins up to 0.8 
    if unblind: 
        # unblind everywhere, where s/sqrt(b) <  
        s_sqrt_b = signal_hist.values()/np.sqrt(sum(bkgd_stack).values())
        mask = s_sqrt_b < 0.05
    else:
        # don't unblind
        mask = np.zeros_like(data_hist.values(), dtype=bool)
    # blind data
    data_hist.values()[~mask] = np.nan
    data_hist.variances()[~mask] = np.nan
    
    color_map = {
        "DY": "#7a21dd",
        "TT": "#9c9ca1",
        "ST": "#e42536",
        "W": "#964a8b",
        "QCD": "#f89c20",
        "Others":"#5790fc",
    }
    lumi = {"2016APV": "19.5", "2016": "16.8", "2017": "41.5", "2018": "59.7"}[year]

    fig, (ax1, ax2) = plt.subplots(2, 1,
                                figsize=(10, 12),
                                sharex=True,
                                gridspec_kw={'height_ratios': [4, 1]},)
    fig.subplots_adjust(hspace=0.05)
    hep.cms.text(" Preliminary", fontsize=20, ax=ax1)
    mu, tau = '\u03BC','\u03C4'
    chn_map = {"etau": r"$bbe$"+tau, "tautau":r"$bb$"+tau+tau, "mutau": r"$bb$"+mu+tau}
    hep.cms.lumitext(r"{} $fb^{{-1}}$ (13 TeV)".format(lumi), fontsize=20, ax = ax1)
    ax1.text(0.05, .91, f"{chn_map[channel]}\n{cat}", fontsize=15,transform=ax1.transAxes)
    
    bkgd_stack.plot(stack=True, ax=ax1, color=[color_map[i.name] for i in bkgd_stack], histtype='fill')
    errps =  {'hatch':'////', 'facecolor':'none', 'lw': 0, 'edgecolor': 'k', 'alpha': 0.5}
    data_hist.plot(ax=ax1, color='black', label="Data", histtype='errorbar')
    plot_asym_errorband(mc=bkgd_stack,
                        err_up=bkgd_errors_up,
                        err_down=bkgd_errors_down,
                        ax=ax1,
                        mode="errorbar",
                        label="Unc. (stat. & syst.)")
    # hep.histplot(stack_error_hist, ax=ax1, histtype="band", **errps, label="Unc. (stat. & syst.)")
    if not signal_hist is None:
        signal_hist.plot(color='black', ax=ax1, label=label if signal_name is not None else None) #signal_name)
    
    if any(mask):
        idx = np.where(mask)[0][-1] + 1
        x = data_hist.axes[0].edges[idx]
        y = sum(bkgd_stack).values()[idx-1]*2 
        ax1.vlines(x, 0, y,
                color='red', linestyle='--', label=f"unblinding edge")
        
    lgd = ax1.legend( fontsize = 12,bbox_to_anchor = (0.99, 0.99), loc="upper right", ncols=2,
                    frameon=True, facecolor='white', edgecolor='black')
    lgd.get_frame().set_boxstyle("Square", pad=0.0)
    ax1.set_yscale("log")
    max_y = sum(bkgd_stack).values().max()
    ax1.set_ylim((1e-2, 10 * max_y))
    ax1.set_xlabel("")
    ax1.set_ylabel("Events")
    
    ax2.set_xlabel("pDNN Score")
    ax2.set_ylabel("Data/MC")

    ax2.hlines(1, 0, 1, color='black', linestyle='--')
    if not signal_hist is None:
        ax2.hlines([0.75, 1.25], 0, 1, color='grey', linestyle='--')
    else:
        ax2.hlines([0.5, 1.5], 0, 1, color='grey', linestyle='--')
    hep.histplot(data_hist.values()/sum(bkgd_stack).values(),
                 data_hist.axes[0].edges,
                 yerr=np.sqrt(data_hist.variances())/sum(bkgd_stack).values(),
                 ax=ax2, histtype='errorbar', color='black')
    
    plot_asym_errorband(mc=bkgd_stack,
                        err_up=bkgd_errors_up,
                        err_down=bkgd_errors_down,
                        ax=ax2,
                        mode="ratio")
    if not signal_hist is None:
        ax2.set_ylim(0.7, 1.3)
    else:
        ax2.set_ylim(0.4, 1.6)
    ax2.set_xlim(0, 1)
    ax2.set_xticks(data_hist.axes[0].edges, [round(i, 4) for i in bin_edges], rotation=60)
    if not savename is None:
        if not Path(savename).parent.exists():
            os.makedirs(Path(savename).parent)
        plt.savefig(savename, bbox_inches='tight', pad_inches=0.05)
        plt.close()
    else:
        plt.show()
    

def make_plots(input_dir: str | Path,
               output_dir: str | Path,
               year: str,
               spin: str,
               limits_file: str | Path | None = None,
               unblind: bool = False, 
               control_region: bool = False) -> None:
    if output_dir == "":
        output_dir = f"./{Path(input_dir).parent.stem}"
    if not os.path.exists(output_dir):
        yn = input(f"Output dir {output_dir} does not exist. Create now? [y/n]")
        if yn.lower().strip() == "y":
            os.makedirs(output_dir)
        else:
            raise ValueError(f"Output dir {output_dir} does not exist and wasn't created")
    datashapes = glob(f'{input_dir}/shapes_cat_{year}_*_spin_{spin}_*.root')
    datashapes = [shape for shape in datashapes
                  if not any(s in shape for s in ['mwc', 'mdnn', 'mhh'])]
    for file in tqdm(datashapes):
        filename = Path(file)
        #_, _, _, channel, cat, sign, isolation, _, spin, _, mass = filename.stem.split("_")
        _, _, _, channel, cat, cat_suffix, sign, isolation, _, spin, _, mass = filename.stem.split("_")
        dirname = f"cat_{year}_{channel}_{cat}_{cat_suffix}_{sign}_{isolation}"
        signal_name = f"ggf_spin_{spin}_mass_{mass}_{year}_hbbhtt" if not control_region else None
        stack, bkgd_errors_up, bkgd_errors_down, sig, data, bin_edges = load_hists(filename, dirname, channel, cat, signal_name, year)
        unblind = unblind if not control_region else True 
        if limits_file is not None:
            lim = load_reslim(limits_file, mass)
            signal_name = " ".join(signal_name.split("_")[0:5]).replace("ggf", "ggf;").replace("spin ", 's:').replace("mass ", "m:")
            plot_mc_data_sig(data_hist=data,
                             signal_hist=sig,
                             bkgd_stack=stack,
                             bkgd_errors_up=bkgd_errors_up,
                             bkgd_errors_down=bkgd_errors_down,
                             bin_edges=bin_edges,
                             year=year,
                             channel=channel,
                             cat=cat,
                             signal_name=signal_name,
                             savename=f"{output_dir}/{year}/{channel}/{cat}/{filename.stem}.pdf",
                             limit_value=lim,
                             unblind=unblind,)
        else: 
            signal_name = " ".join(signal_name.split("_")[0:5]).replace("ggf", "ggf;").replace("spin ", 's:').replace("mass ", "m:")
            plot_mc_data_sig(data_hist=data,
                             signal_hist=sig,
                             bkgd_stack=stack,
                             bkgd_errors_up=bkgd_errors_up,
                             bkgd_errors_down=bkgd_errors_down,
                             bin_edges=bin_edges,
                             year=year,
                             channel=channel,
                             cat=cat,
                             signal_name=signal_name,
                             savename=f"{output_dir}/{year}/{channel}/{cat}/{filename.stem}.pdf",
                             limit_value=None,
                             unblind=unblind)


def main(input_dir: str | Path,
         output_dir: str | Path,
         year: str,
         spin: str,
         limits_file: str | Path | None,
         unblind_edge: float | None = 0.8) -> None:
    make_plots(input_dir=input_dir,
               output_dir=output_dir,
               year=year,
               spin=spin,
               limits_file=limits_file,
               unblind_edge=unblind_edge)


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    main(input_dir=args.input_dir,
         output_dir=args.output_dir,
         year=args.year,
         spin=args.spin,
         limits_file=args.limits_file,
         unblind_edge=args.unblind_edge
         )



