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
from hist import Hist, Stack
from hist.intervals import ratio_uncertainty 


import matplotlib.pyplot as plt
import mplhep as hep

from tautaunn.write_datacards_stack import br_hh_bbtt
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


def load_hists(filename: str | Path,
               dirname: str,
               signal_name: str,
               year: str) -> tuple[Stack, Hist]:
    with uproot.open(filename) as f:
        objects = f[dirname].classnames()
        nominal_hists = [o.strip(";1") for o in objects if not any(s in o for s in ["Up", "Down"])]
        hists = {o: f[dirname][o].to_hist() for o in nominal_hists if o != signal_name and o != 'data_obs'}

        shape_nuisances = list(set([o.split("__")[-1].strip("Up;1") for o in objects if "Up" in o]))
        bin_edges = hists[list(hists.keys())[0]].axes[0].edges
        equal_width_hists = {name.replace(f"_{year}", ""): histo_equalwidth(hists[name])[0] for name in hists}

        sig = histo_equalwidth(f[dirname][signal_name].to_hist())[0]
        data = histo_equalwidth(f[dirname]['data_obs'].to_hist())[0]
        main_bkgds = ['TT', 'ST', 'DY', 'W', 'QCD']
        bkgd_dict  = {name: h for name, h in equal_width_hists.items() if any(name == s for s in main_bkgds)}
        others = reduce(add, (h for name, h in equal_width_hists.items() if name not in bkgd_dict))
        bkgd_dict["Others"] = others
        #sorted_bkgd_dict = dict(sorted(bkgd_dict.items(), key=lambda x: x[1].sum().value, reverse=False))
        # instead impose a fixed order of "Others", "QCD", "W", "ST", "DY", "TT"
        sorted_bkgd_dict = OrderedDict()
        for name in ["Others", "QCD", "W", "ST", "DY", "TT"]:
            if name in bkgd_dict:
                sorted_bkgd_dict[name] = bkgd_dict[name]
        bkgd_stack = hist.Stack.from_dict(sorted_bkgd_dict)
    return bkgd_stack, sig, data, bin_edges


def load_reslim(file: str | Path,
                mass: float | int):
    limits = np.load(file)
    masses = limits['data']['mhh']
    exp_lim = limits['data']['limit']#*1000
    return exp_lim[np.where(masses==int(mass))][0]


def plot_single_dist(mc_stack: Stack,
                     title: str,
                     savename: str | Path,
                     sig: Hist | None = None,
                     lim: float | None = None) -> None:
    mc_stack.plot(stack=True, histtype='fill')
    if sig is not None: 
        if lim is not None:
            sig = sig/sig.sum().value
            sig = sig*lim
            sig.plot1d(label=f'sig. norm. to limit {lim:.1f} fb', color='black')
        sig.plot1d(label=f'sig.', color='black')
    plt.yscale('log')
    lgd = plt.legend(bbox_to_anchor=(1.06, 1.02))
    plt.ylabel("N")
    plt.xlabel("DNN out")
    plt.title(title)
    plt.savefig(savename, bbox_extra_artists=(lgd,), bbox_inches='tight', pad_inches=0.2)
    plt.close()


def plot_hist_cms_style(bkgd_stack: hist.Stack,
                        signal_hist: hist,
                        bin_edges: list,
                        signal_name: str,
                        year: str,
                        channel: str,
                        cat: str,
                        savename: str | Path) -> None:

    # map those hexcodes to the bkgds:
    color_map = {
        "DY": "#7a21dd",
        "TT": "#9c9ca1",
        "ST": "#e42536",
        "W": "#964a8b",
        "QCD": "#f89c20",
        "Others":"#5790fc",
    }
    fig, ax = plt.subplots()
    hep.cms.text(" Preliminary", fontsize=20, ax=ax)
    lumi = {"2016APV": "19.5", "2016": "16.8", "2017": "41.5", "2018": "59.7"}[year]
    mu, tau = '\u03BC','\u03C4'
    chn_map = {"etau": r"$bbe$"+tau, "tautau":r"$bb$"+tau+tau, "mutau": r"$bb$"+mu+tau}
    hep.cms.lumitext(r"{} $fb^{{-1}}$ (13 TeV)".format(lumi), fontsize=20, ax = ax)
    ax.text(0.32, 1.013, chn_map[channel], fontsize=13,transform=ax.transAxes)
    ax.text(0.39, 1.013, cat, fontsize=13, transform=ax.transAxes)
    bkgd_stack.plot(stack=True, ax=ax, color=[color_map[i.name] for i in bkgd_stack], histtype='fill')
    signal_hist.plot(color='black', ax=ax, label=signal_name)
    lgd = ax.legend( fontsize = 12,bbox_to_anchor = (0.99, 0.99), loc="upper right", ncols=2,
                    frameon=True, facecolor='white', edgecolor='black')
    lgd.get_frame().set_boxstyle("Square", pad=0.0)
    ax.set_xticks(signal_hist.axes[0].edges, [round(i, 4) for i in bin_edges], rotation=60)
    ax.set_yscale("log")
    # find the right y-axis limit
    min_y_tt_dy = min([bkgd_stack[h].values().min() for h in ["DY", "TT"]])
    min_y_sig = signal_hist.values().min()
    min_y = min(min_y_tt_dy, min_y_sig)
    summed_hist = reduce(add, [h for h in bkgd_stack])
    max_y = summed_hist.values().max() 

    ax.set_ylim((min_y, 2 * max_y))

    ax.set_xlabel("pDNN Score")
    ax.set_ylabel("Events")
    if not Path(savename).parent.exists():
        os.makedirs(Path(savename).parent)
    plt.savefig(savename, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def plot_mc_stat(mc: Stack,
                 ax: plt.Axes,
                 mode: str = "errorbar",):
    assert mode in ("errorbar", "ratio")

    mc_stat = Hist.new.Var(sum(mc).axes[0].edges, name="mc_stat").Weight()
    mc_stat.view().value = np.sqrt(sum(mc).variances())
    if mode == "errorbar":
        bottom = sum(mc).values()
        mc_stat.view().value = np.sqrt(sum(mc).variances())
    elif mode == "ratio":
        bottom = 1.
        mc_stat.view().value = np.nan_to_num(np.divide(np.sqrt(sum(mc).variances()), sum(mc).values()), nan=0.)
    for sign in (-1., 1.):
        ax.hist([mc_stat.axes[0].edges[:-1]], bins=mc_stat.axes[0].edges, weights=sign*mc_stat.values(),
                        bottom=bottom, histtype="stepfilled", color="gainsboro",
                        hatch='///', edgecolor="gray", linewidth=0., alpha=0.6)


def plot_mc_data_sig(data_hist: Hist,
                     signal_hist: Hist,
                     bkgd_stack: Stack,
                     bin_edges: list,
                     year: str,
                     channel: str,
                     cat: str,
                     savename: str | Path,
                     signal_name: str,
                     limit_value = None,
                     ) -> None:

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
    if len(bin_edges) > 2:
        mask = bin_edges[1:] < 0.8
        if all(~mask):
            # unblind just the first bin
            mask = np.zeros_like(signal_hist.values(), dtype=bool)
            mask[0] = True
    else:
        # don't unblind
        mask = np.zeros_like(signal_hist.values(), dtype=bool)
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
    hep.histplot(sum(bkgd_stack), histtype="band", ax=ax1, **errps)
    hep.histplot(data_hist, ax=ax1, label="data", histtype='errorbar', color='k', capsize=2, yerr=True)
    signal_hist.plot(color='black', ax=ax1, label=label) #signal_name)
    
    if any(mask):
        idx = np.where(mask)[0][-1] + 1
        x = signal_hist.axes[0].edges[idx]
        y = sum(bkgd_stack).values()[idx-1]*2 
        ax1.vlines(x, 0, y,
                color='red', linestyle='--', label=f"unblinding edge")
        
    lgd = ax1.legend( fontsize = 12,bbox_to_anchor = (0.99, 0.99), loc="upper right", ncols=2,
                    frameon=True, facecolor='white', edgecolor='black')
    lgd.get_frame().set_boxstyle("Square", pad=0.0)
    ax1.set_yscale("log")
    min_y_tt_dy = min([bkgd_stack[h].values().min() for h in ["DY", "TT"]])
    min_y_sig = signal_hist.values().min()
    min_y = min(min_y_tt_dy, min_y_sig)
    max_y = sum(bkgd_stack).values().max()
    #ax1.set_ylim((0.1*min_y, 100 * max_y))
    #ax1.set_ylim((min_y, 10 * max_y))
    ax1.set_ylim((1e-4, 100 * max_y))
    ax1.set_xlabel("")
    ax1.set_ylabel("Events")
    
    ax2.set_xlabel("pDNN Score")
    ax2.set_ylabel("Data/MC")
    ratio_hist = Hist.new.Var(data_hist.axes[0].edges, name="ratio").Weight()
    ratio_hist.view().value = data_hist.values()/sum(bkgd_stack).values()
    ratio_hist.view().variance = np.divide(np.sqrt(data_hist.variances()), data_hist.values())

    ratio_hist.view().value[~mask] = np.nan
    ratio_hist.variances()[~mask] = np.nan

    ax2.hlines(1, 0, 1, color='black', linestyle='--')
    ax2.hlines([0.5, 1.5], 0, 1, color='grey', linestyle='--')
    hep.histplot(data_hist.values()/sum(bkgd_stack).values(),
                 data_hist.axes[0].edges,
                 yerr=np.sqrt(data_hist.values())/sum(bkgd_stack).values(),
                 ax=ax2, histtype='errorbar', color='black')
    #ax2.errorbar(ratio_hist.axes[0].centers, ratio_hist.values(), yerr=ratio_hist.variances(), fmt='o', color='black')
    #plot_mc_stat(bkgd_stack, ax2, mode="ratio")
    yerr = ratio_uncertainty(data_hist.values(),sum(bkgd_stack).values(), "poisson-ratio")
    ax2.stairs(1+yerr[1], edges=data_hist.axes[0].edges, baseline=1-yerr[0], **errps)
    ax2.set_ylim(0.4, 1.6)
    ax2.set_xlim(0, 1)
    ax2.set_xticks(signal_hist.axes[0].edges, [round(i, 4) for i in bin_edges], rotation=60)
    if not Path(savename).parent.exists():
        os.makedirs(Path(savename).parent)
    plt.savefig(savename, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    

def make_plots(input_dir: str | Path,
               output_dir: str | Path,
               year: str,
               spin: str,
               limits_file: str | Path | None = None):
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
        signal_name = f"ggf_spin_{spin}_mass_{mass}_{year}_hbbhtt"
        stack, sig, data, bin_edges = load_hists(filename, dirname, signal_name, year)
        if limits_file is not None:
            lim = load_reslim(limits_file, mass)
            signal_name = " ".join(signal_name.split("_")[0:5]).replace("ggf", "ggf;").replace("spin ", 's:').replace("mass ", "m:")
            plot_mc_data_sig(data_hist=data,
                             signal_hist=sig,
                             bkgd_stack=stack,
                             bin_edges=bin_edges,
                             year=year,
                             channel=channel,
                             cat=cat,
                             signal_name=signal_name,
                             savename=f"{output_dir}/{year}/{channel}/{cat}/{filename.stem}.pdf",
                             limit_value=lim)
        else: 
            signal_name = " ".join(signal_name.split("_")[0:5]).replace("ggf", "ggf;").replace("spin ", 's:').replace("mass ", "m:")
            plot_mc_data_sig(data_hist=data,
                             signal_hist=sig,
                             bkgd_stack=stack,
                             bin_edges=bin_edges,
                             year=year,
                             channel=channel,
                             cat=cat,
                             signal_name=signal_name,
                             savename=f"{output_dir}/{year}/{channel}/{cat}/{filename.stem}.pdf",
                             limit_value=None)
            #lim = None
            #plot_hist_cms_style(bkgd_stack=stack,
                                #signal_hist=sig,
                                #bin_edges=bin_edges,
                                #year=year,
                                #channel=channel,
                                #signal_name=" ".join(signal_name.split("_")[0:5]).replace("ggf", "ggf;").replace("spin ", 's:').replace("mass ", "m:"),
                                #cat=cat,
                                #savename=f"{output_dir}/{year}/{channel}/{cat}/{filename.stem}.png")


def main(input_dir: str | Path,
         output_dir: str | Path,
         year: str,
         spin: str,
         limits_file: str | Path | None) -> None:
    make_plots(input_dir=input_dir,
               output_dir=output_dir,
               year=year,
               spin=spin,
               limits_file=limits_file)


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    main(input_dir=args.input_dir,
         output_dir=args.output_dir,
         year=args.year,
         spin=args.spin,
         limits_file=args.limits_file)



