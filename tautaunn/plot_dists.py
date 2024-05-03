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


def load_hists(filename: str | Path,
               dirname: str,
               signal_name: str) -> tuple[Stack, Hist]:
    with uproot.open(filename) as file:
        objects = file[dirname].classnames()
        sig = file[dirname][signal_name].to_hist()
        mc_bkgds = ['TT', 'ST', 'DY', 'W', 'EWK']
        stack_dict = {i: file[dirname][i].to_hist() for i in mc_bkgds}
        # merge
        vv = Hist.new.Variable(sig.axes.edges[0], name=sig.axes.name[0], label=sig.axes.label[0])
        vv = file[dirname]['WW'].to_hist()+file[dirname]['WZ'].to_hist()+file[dirname]['ZZ'].to_hist()
        sm_h = Hist.new.Variable(sig.axes.edges[0], name=sig.axes.name[0], label=sig.axes.label[0])
        sm_h = file[dirname]['ggH_htt'].to_hist()+file[dirname]['qqH_htt'].to_hist()
        vh = Hist.new.Variable(sig.axes.edges[0], name=sig.axes.name[0], label=sig.axes.label[0])
        vh = file[dirname]['ZH_htt'].to_hist()+file[dirname]['WH_htt'].to_hist()
        tth = Hist.new.Variable(sig.axes.edges[0], name=sig.axes.name[0], label=sig.axes.label[0])
        tth = file[dirname]['ttH_htt'].to_hist()+file[dirname]['ttH_hbb'].to_hist()
        other = Hist.new.Variable(sig.axes.edges[0], name=sig.axes.name[0], label=sig.axes.label[0])
        other = file[dirname]['VVV'].to_hist()+file[dirname]['TTV'].to_hist()+file[dirname]['TTVV'].to_hist()

        stack_dict['VV'] = vv
        stack_dict['SM H'] = sm_h
        stack_dict['VH'] = vh
        stack_dict['tth'] = tth
        stack_dict['VVV & TTV & TTVV'] = other
        stack_dict = dict(sorted(stack_dict.items(), key=lambda item: item[1].sum().value, reverse=True))
        stack = hist.Stack.from_dict(stack_dict)
    return stack, sig


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
        signal_name = f"ggf_spin_{spin}_mass_{mass}_hbbhtt"
        stack, sig = load_hists(filename, dirname, signal_name)
        if limits_file is not None:
            lim = load_reslim(limits_file, mass)
        else: 
            lim = None
        title = f"cat: {cat}, spin: {spin} mass: {mass}"
        widths = stack.axes.widths[0]
        exclusion_idx = get_exclusion_idx(widths)
        if exclusion_idx == -1:
            plot_single_dist(mc_stack=stack,
                             title=title,
                             savename=f"{output_dir}/{filename.stem}.png",
                             sig=sig,
                             lim=lim)
        else:
            plot_double_dist(mc_stack=stack,
                             exclusion_idx=exclusion_idx,
                             title=title,
                             savename=f"{output_dir}/{filename.stem}.png",
                             sig=sig,
                             lim=lim)


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



