from collections import OrderedDict
import argparse
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm
import os

import uproot
import hist
from hist import Hist, Stack


import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)


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
        nominal_objects = [o.strip(";1") for o in objects if not any(s in o for s in ["Up", "Down"])]
        hists = {o: f[dirname][o].to_hist() for o in nominal_objects if o != signal_name and o != 'data_obs'}
        bin_edges = hists[list(hists.keys())[0]].axes[0].edges
        equal_width_hists = {name: histo_equalwidth(hists[name])[0] for name in hists}

        bkgd_stack = hist.Stack.from_dict({name.replace(f"_{year}", ""): hist for name, hist in equal_width_hists.items() if name != signal_name and name != 'data_obs'})
        sig = histo_equalwidth(f[dirname][signal_name].to_hist())[0]
        #sig = file[dirname][signal_name].to_hist()
        #mc_bkgds = ['TT', 'ST', 'DY', 'W', 'EWK']
        #stack_dict = {i: file[dirname][f"{i}_{year}"].to_hist() for i in mc_bkgds}
        ## merge
        #vv = Hist.new.Variable(sig.axes.edges[0], name=sig.axes.name[0], label=sig.axes.label[0])
        #vv = file[dirname][f'WW_{year}'].to_hist()+file[dirname][f'WZ_{year}'].to_hist()+file[dirname][f'ZZ_{year}'].to_hist()
        #sm_h = Hist.new.Variable(sig.axes.edges[0], name=sig.axes.name[0], label=sig.axes.label[0])
        #sm_h = file[dirname][f'ggH_{year}_htt'].to_hist()+file[dirname][f'qqH_{year}_htt'].to_hist()
        #vh = Hist.new.Variable(sig.axes.edges[0], name=sig.axes.name[0], label=sig.axes.label[0])
        #vh = file[dirname][f'ZH_{year}_htt'].to_hist()+file[dirname][f'WH_{year}_htt'].to_hist()
        #tth = Hist.new.Variable(sig.axes.edges[0], name=sig.axes.name[0], label=sig.axes.label[0])
        #tth = file[dirname][f'ttH_{year}_htt'].to_hist()+file[dirname][f'ttH_{year}_hbb'].to_hist()
        #other = Hist.new.Variable(sig.axes.edges[0], name=sig.axes.name[0], label=sig.axes.label[0])
        #other = file[dirname][f'VVV_{year}'].to_hist()+file[dirname][f'TTV_{year}'].to_hist()+file[dirname][f'TTVV_{year}'].to_hist()

        #stack_dict['VV'] = vv
        #stack_dict['SM H'] = sm_h
        #stack_dict['VH'] = vh
        #stack_dict['tth'] = tth
        #stack_dict['VVV & TTV & TTVV'] = other
        #stack_dict = dict(sorted(stack_dict.items(), key=lambda item: item[1].sum().value, reverse=True))
        #stack = hist.Stack.from_dict(stack_dict)
    return bkgd_stack, sig, bin_edges


def load_reslim(file: str | Path,
                mass: float | int):
    limits = np.load(file)
    masses = limits['data']['mhh']
    exp_lim = limits['data']['limit']*1000
    return exp_lim[np.where(masses==int(mass))][0]


def get_exclusion_idx(widths: list,
                      thres: float = 50.):
    ratios = widths/widths[-1]
    check = ratios>thres
    if np.all(~check):
        return -1
    else:
        boolchange = check[:-1] != check[1:]
        if boolchange.sum()>1:
            print("Binning seems to decrease and then drastically increase again!")
            print(f"Check the widths: {widths}")
        return np.where(boolchange==True)[0][0]+1


def plot_double_dist(mc_stack: Stack,
                     exclusion_idx: int,
                     title: str,
                     savename: str | Path,
                     sig: Hist | None = None,
                     lim: float | None = None) -> None:

    fig, (ax_0, ax_1) = plt.subplots(1, 2, figsize = (24, 8))
    fig.suptitle(title)
    ax_0.set_title("Full")
    mc_stack.plot(stack=True, histtype='fill', ax=ax_0)
    if sig is not None:
        if lim is not None:
            # normalise to limit 
            sig = (sig/sig.sum().value)*lim
            sig.plot1d(label=f'sig. norm. to limit: {lim:.1f} fb', color='black', ax=ax_1)
        sig.plot1d(label=f'sig.', color='black', ax=ax_0)
    ax_0.set_yscale('log')
    ax_0.set_ylabel('N')
    ax_0.set_xlabel("DNN Out")
    lgd = ax_0.legend(bbox_to_anchor=(2.8, 1.01))

    if exclusion_idx == 1:
        ax_1.set_title("Excluding first bin")
    else:
        ax_1.set_title(f"Excluding first {exclusion_idx} bins")
    mc_stack.plot(stack=True, histtype='fill', ax=ax_1)
    if sig is not None:
        if lim is not None:
            sig = (sig/sig.sum().value)*lim
            sig.plot1d(label=f'sig. norm. to limit: {lim:.1f} fb', color='black', ax=ax_1)
        sig.plot1d(label=f'sig.', color='black', ax=ax_1)
    ax_1.set_yscale('log')
    ax_1.set_ylabel('')
    ax_1.set_xlabel('DNN Out')
    ax_1.set_xlim([mc_stack.axes.edges[0][exclusion_idx],1])
    plt.savefig(savename, bbox_extra_artists=(lgd,), bbox_inches='tight', pad_inches=0.8)
    plt.close()


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
    fig, ax = plt.subplots()
    hep.cms.text(" Preliminary", fontsize=20, ax=ax)
    lumi = {"2016APV": "19.5", "2016": "16.8", "2017": "41.5", "2018": "59.7"}[year]
    mu, tau = '\u03BC','\u03C4'
    chn_map = {"etau": r"$bb\;e$"+tau, "tautau":r"$bb\;$"+tau+tau, "mutau": r"$bb\;$"+mu+tau}
    hep.cms.lumitext(r"{} $fb^{{-1}}$ (13 TeV)".format(lumi), fontsize=20, ax = ax)
    ax.text(0.32, 1.02, chn_map[channel], fontsize=13,transform=ax.transAxes)
    ax.text(0.39, 1.02, cat, fontsize=13, transform=ax.transAxes)
    bkgd_stack.plot(stack=True, ax=ax, color=[plt.cm.tab20.colors[i] for i in range(len(bkgd_stack))], histtype='fill')
    signal_hist.plot(color='black', ax=ax, label=signal_name)
    lgd = ax.legend( fontsize = 12,bbox_to_anchor = (0.99, 0.99), loc="upper right", ncols=4,
                    frameon=True, facecolor='white', edgecolor='black')
    lgd.get_frame().set_boxstyle("Square", pad=0.0)
    ax.set_xticks(signal_hist.axes[0].edges, [round(i, 4) for i in bin_edges], rotation=60)
    ax.set_yscale("log")
    ax.set_xlabel("pDNN Score")
    ax.set_ylabel("N")
    if not Path(savename).parent.exists():
        os.makedirs(Path(savename).parent)
    plt.savefig(savename, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    


def make_plots(input_dir: str | Path,
               output_dir: str | Path,
               limits_file: str | Path | None = None):
    if output_dir == "":
        output_dir = f"./{Path(input_dir).parent.stem}"
    if not os.path.exists(output_dir):
        yn = input(f"Output dir {output_dir} does not exist. Create now? [y/n]")
        if yn.lower().strip() == "y":
            os.makedirs(output_dir)
        else:
            raise ValueError(f"Output dir {output_dir} does not exist and wasn't created")
    datashapes = glob(f'{input_dir}/shapes_*.root')
    datashapes = [shape for shape in datashapes
                  if not any(s in shape for s in ['mwc', 'mdnn', 'mhh'])]
    for file in tqdm(datashapes):
        filename = Path(file)
        _, _, year, channel, cat, sign, isolation, _, spin, _, mass = filename.stem.split("_")
        dirname = f"cat_{year}_{channel}_{cat}_{sign}_{isolation}"
        signal_name = f"ggf_spin_{spin}_mass_{mass}_{year}_hbbhtt"
        stack, sig, bin_edges = load_hists(filename, dirname, signal_name, year)
        if limits_file is not None:
            lim = load_reslim(limits_file, mass)
        else: 
            lim = None
        #title = f"cat: {cat}, spin: {spin} mass: {mass}"
        #widths = stack.axes.widths[0]
        #exclusion_idx = get_exclusion_idx(widths)
        #if exclusion_idx == -1:
            #plot_single_dist(mc_stack=stack,
                             #title=title,
                             #savename=f"{output_dir}/{filename.stem}.png",
                             #sig=sig,
                             #lim=lim)
        #else:
            #plot_double_dist(mc_stack=stack,
                             #exclusion_idx=exclusion_idx,
                             #title=title,
                             #savename=f"{output_dir}/{filename.stem}.png",
                             #sig=sig,
                             #lim=lim)
        plot_hist_cms_style(bkgd_stack=stack,
                            signal_hist=sig,
                            bin_edges=bin_edges,
                            year=year,
                            channel=channel,
                            signal_name=" ".join(signal_name.split("_")[0:5]).replace("ggf", "ggf;").replace("spin ", 's:').replace("mass ", "m:"),
                            cat=cat,
                            savename=f"{output_dir}/{year}/{channel}/{cat}/{filename.stem}.png")


def main(input_dir: str | Path,
         output_dir: str | Path,
         limits_file: str | Path | None) -> None:
    make_plots(input_dir=input_dir,
               output_dir=output_dir,
               limits_file=limits_file)


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    main(input_dir=args.input_dir,
         output_dir=args.output_dir,
         limits_file=args.limits_file)



