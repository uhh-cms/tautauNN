from collections import OrderedDict
import argparse
import numpy as np
from pathlib2 import Path
from glob import glob
from tqdm import tqdm
import os

import uproot
import hist
from hist import Hist


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
    parser.add_argument("-m", "--model_name",
                        type=str, help='model name i.e. baseline_nonparam')
    parser.add_argument("-i", "--input_dir",
                        type=str, help='directory, where the datacards & shapes are stored')
    parser.add_argument("-o", "--output_dir",
                        type=str, help='output dir to store plots (default ./input_dir.stem)',
                        default="")
    return parser


def load_hists(filename, dirname, signal_name):
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


def load_reslim(file, mass):
    limits = np.load(file)
    masses = limits['data']['mhh']
    exp_lim = limits['data']['limit']*1000
    return exp_lim[np.where(masses==int(mass))][0]


def get_exclusion_idx(widths, thres=50):
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


def plot_double_dist(mc_stack, sig, lim, exclusion_idx, title, savename):

    fig, (ax_0, ax_1) = plt.subplots(1, 2, figsize = (24, 8))
    fig.suptitle(title)
    ax_0.set_title("Full")
    mc_stack.plot(stack=True, histtype='fill', ax=ax_0)
    sig = (sig/sig.sum().value)*lim
    sig.plot1d(label=f'sig. norm. to limit: {lim:.1f} fb', color='black', ax=ax_0)
    #hep.cms.label(f"Work in Progress {channel} {cat}",
                  #lumi=luminosities[year],
                  #year=year,
                  #loc=2)
    ax_0.set_yscale('log')
    ax_0.set_ylabel('N')
    ax_0.set_xlabel("DNN Out")
    lgd = ax_0.legend(bbox_to_anchor=(2.8, 1.01))

    if exclusion_idx == 1:
        ax_1.set_title("Excluding first bin")
    else:
        ax_1.set_title(f"Excluding first {exclusion_idx} bins")
    mc_stack.plot(stack=True, histtype='fill', ax=ax_1)
    sig = (sig/sig.sum().value)*lim
    sig.plot1d(label=f'sig. norm. to limit: {lim:.1f} fb', color='black', ax=ax_1)
    ax_1.set_yscale('log')
    ax_1.set_ylabel('')
    ax_1.set_xlabel('DNN Out')
    ax_1.set_xlim([mc_stack.axes.edges[0][exclusion_idx],1])
    plt.savefig(savename, bbox_extra_artists=(lgd,), bbox_inches='tight', pad_inches=0.8)
    plt.close()

def plot_single_dist(mc_stack, sig, lim, title, savename):
    mc_stack.plot(stack=True, histtype='fill')
    sig = sig/sig.sum().value
    sig = sig*lim
    sig.plot1d(label=f'sig. norm. to limit {lim:.1f} fb', color='black')
    plt.yscale('log')
    lgd = plt.legend(bbox_to_anchor=(1.06, 1.02))
    plt.ylabel("N")
    plt.xlabel("DNN out")
    plt.title(title)
    plt.savefig(savename, bbox_extra_artists=(lgd,), bbox_inches='tight', pad_inches=0.2)
    plt.close()


def main(model_name, input_dir, output_dir):
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
        limits_file= f'./reslimits_{model_name}_{year}.npz'
        lim = load_reslim(limits_file, mass)
        title = f"{model_name.replace('_', '-')}, cat: {cat}, spin: {spin} mass: {mass}"
        widths = stack.axes.widths[0]
        exclusion_idx = get_exclusion_idx(widths)
        if exclusion_idx == -1:
            plot_single_dist(stack, sig, lim, title, f"{output_dir}/{filename.stem}.png")
        else:
            plot_double_dist(stack,sig, lim,
                             exclusion_idx,
                             title, f"{output_dir}/{filename.stem}.png")


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    main(model_name=args.model_name,
         input_dir=args.input_dir,
         output_dir=args.output_dir)


