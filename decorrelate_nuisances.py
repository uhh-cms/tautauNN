from fnmatch import filter
from optparse import OptionParser, OptionGroup
import os
import sys
import ROOT
import json
cmssw_base = os.environ["CMSSW_BASE"]
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

if not os.path.exists(cmssw_base):
    raise ValueError("Could not find CMSSW base!")

ROOT.TH1.AddDirectory(False)
try:
    print("importing CombineHarvester")
    import CombineHarvester.CombineTools.ch as ch
    print("done")
except:
    msg = " ".join("""Could not find package 'CombineHarvester'. 
            Are you sure you installed it?""".split())
    raise ImportError(msg)

to_freeze = """kfactor_wjets kfactor_zjets
                                .*bgscale_MCCR.* .*bgnorm_ttcc_FH""".split()

def copy_and_rename(syst, suffix):
    copy_func = getattr(syst, "Copy", None)
    if not callable(copy_func):
        msg = " ".join("""This version of CombineHarvester does not include the 
                ch.Systematic.Copy function, cannot copy systematics!
                """.split())
        raise NotImplementedError(msg)
    new = syst.Copy()
    name = new.name()
    name = "_".join([name, suffix])
    new.set_name(name)
    return new

def match_proc_and_syst(proc, syst):
    return (proc.era() == syst.era() and
            proc.bin() == syst.bin() and
            proc.process() == syst.process()
    )

def split_nuisance_per_histobin(harvester, process, syst):
    if not match_proc_and_syst(process, syst):
        return
    # first, reconstruct real varied shapes using the nominal rate
    # get the normalized varied shapes for the current systematic
    shape_u = syst.ShapeUAsTH1F()
    val_u = syst.value_u()
    shape_d = syst.ShapeDAsTH1F()
    val_d = syst.value_d()

    # get nominal shape
    orig_nom = process.ShapeAsTH1F()
    shape_nom = orig_nom.Clone("cloned_{}".format(orig_nom.GetName()))
    nom_rate = process.rate()

    if syst.type() == "lnN":
        shape_u = orig_nom.Clone("{}_{}Up".format(orig_nom.GetName(), syst.name()))
        shape_d = orig_nom.Clone("{}_{}Down".format(orig_nom.GetName(), syst.name()))

    # scale the normalized templates
    shape_nom.Scale(nom_rate)

    # now loop through the bins and set them to zero where appropriate
    # get number of bins
    nbins = shape_nom.GetNbinsX()

    # get list of bins
    bin_list = list(range(1, nbins+1))

    # prepare the chunks that are to be mapped to STXS bins
    stxs_bins = "STXS_0 STXS_1 STXS_2 STXS_3 STXS_4".split()

    # we don't know how many bins there are, so map the bins explicitly
    bin_chunks = [[bin_list[0]], [bin_list[1]], [bin_list[2]], [bin_list[3]], bin_list[4:]]

    for suffix, bin_chunk in zip(stxs_bins, bin_chunks):
        # first, clone the varied shapes
        new_up = shape_u.Clone(shape_u.GetName().replace("Up", "_{}Up".format(suffix)))
        new_down = shape_d.Clone(shape_d.GetName().replace("Down", "_{}Down".format(suffix)))
        new_up.Scale(val_u*nom_rate)
        new_down.Scale(val_d*nom_rate)
        scale_var_up=0
        scale_var_down=0
        scale_nom=0
        # loop through the bins
        for i in bin_list:
            # if i is in the current bin_chunk associated with the stxs bin, don't do anything
            if not i in bin_chunk:
                # if it's not in the list, set it to zero
                new_up.SetBinContent(i, shape_nom.GetBinContent(i))
                new_down.SetBinContent(i, shape_nom.GetBinContent(i))
            else:
                scale_var_up +=new_up.GetBinContent(i)
                scale_var_down += new_down.GetBinContent(i)
                scale_nom += shape_nom.GetBinContent(i)

        name = syst.name()
        name = "_".join([name, suffix])

        if not scale_nom == 0:
            scale_var_up /= float(scale_nom)
            scale_var_down /= float(scale_nom)
        else:
            print("bogus norm encountered, will skip nuisance {}/{}/{}".format(syst.bin(), syst.process(), name))
            continue
        if all(x.Integral() > 0 for x in [new_up, new_down]):
            # build new systematic
            new_syst = syst.Copy()
            new_syst.set_type("shape")

            # normalize the new varied shapes such that they retain the original
            # normalization change in the bin
            # new_up.Scale(scale_var_up*nom_rate/new_up.Integral())
            # new_down.Scale(scale_var_down*nom_rate/new_down.Integral())
            # set the shapes
            new_syst.set_shapes(new_up, new_down, shape_nom)

            # build the new name
            # name = new_syst.name()
            # name = "_".join([name, suffix])
            new_syst.set_name(name)
            # from IPython import embed; embed()
            # finally, add this new systematic to the harvester instance
            harvester.InsertSystematic(new_syst)
        else:
            print("will not create syst {}/{}/{}".format(syst.bin(), syst.process(), name))

def partially_decorrelate_on_template_level(
    harvester,
    process_list,
    nuisance_list,
    syst_type=["lnN", "shape"],
):
    # loop through the processes in the harvester instance and
    # perform the actual construction of the templates

    harvester.cp().process(process_list).ForEachProc(
        lambda proc: (harvester.cp().bin([proc.bin()]).process([proc.process()])
                      .syst_name(nuisance_list).syst_type(syst_type)
                      .ForEachSyst(
                        lambda syst: split_nuisance_per_histobin(harvester, proc, syst)
                        )
                      )
    )



def freeze_nuisances(harvester):
        harvester.SetFlag("filters-use-regex", True)
        for par in to_freeze:
            systs = harvester.cp().syst_name([par]).syst_name_set()
            for p in systs:
                harvester.GetParameter(p).set_frozen(True)

def partially_decorrelate_nuisances(
    harvester,
    channel_suffixes,
    problematic_nps,
    processes,
    syst_types=["lnN", "shape"],
):
    if "rateParam" in syst_types:
        raise ValueError("You cannot partially decorrelate rateParams! Please use the add function.")
    for c in channel_suffixes:
        suffix = channel_suffixes[c]
        sub_harvester = harvester.cp().bin([c])
        syst_harvester = sub_harvester.cp().syst_type(
            syst_types).syst_name(problematic_nps)
        to_decorrelate = syst_harvester.syst_name_set()
 #       for param in to_decorrelate:
#            these_pars = filter(harvester_params, param)
        bins = sub_harvester.cp().bin_set()
        
        print("partially decorrelating bin '{}'".format(bins))
        print("\tprocesses: '{}'".format(", ".join(processes)))
        print("\tsystematics: '{}'".format(to_decorrelate))
        syst_harvester.cp().bin(bins).process(processes).ForEachSyst(
            lambda syst: harvester.InsertSystematic(copy_and_rename(syst, suffix))
        )

def decorrelate_nuisances(
    harvester,
    channel_suffixes,
    problematic_nps,
    processes,
    syst_types = ["lnN", "shape"],
):
    for c in channel_suffixes:
        sub_harvester = harvester.cp().bin([c])
        to_decorrelate = sub_harvester.cp().syst_type(
            syst_types).syst_name(problematic_nps).syst_name_set()
 #       for param in to_decorrelate:
#            these_pars = filter(harvester_params, param)
        for p in to_decorrelate:
            new_parname = "_".join([p, channel_suffixes[c]])
            print("decorrelating bin '{}'".format(sub_harvester.cp().bin_set()))
            print("\tprocesses: '{}'".format(", ".join(processes)))
            print("\tnew name: '{}'".format(new_parname))
            sub_harvester.cp().process(processes)\
                .RenameSystematic(harvester, p, new_parname)

            # current_bins = harvester.cp().bin([c]).bin_set()
            # current_bins = [x.replace(y, ".*") for x in current_bins \
            #                 for y in "ttH_2016_ ttH_2017_ ttH_2018_".split() \
            #                 if x.startswith(y)]
            # # print(current_bins)
            # current_bins = list(set(current_bins))
            # print(current_bins)
            # exit()
            # for b in current_bins:
            #     ultisplit = "_".join([p, b.replace(".*", "")])
            #     print("decorrelating bin '{}'".format(b))
            #     print("\tprocesses: '{}'".format(", ".join(processes)))
            #     print("\tnew name: '{}'".format(ultisplit))
            #     harvester.cp().bin([b]).process(processes).\
            #         RenameSystematic(harvester, p, ultisplit)
#                     pnames = []
            # harvester.cp().bin(current_bins)\
            #     .ForEachProc(lambda x: pnames.append(x.process()))
            # pnames = list(set(pnames))
            # for p in pnames:
            #     harvester.cp().bin(current_bins)\
            #     .process([p])\
            #     .RenameSystematic(harvester, new_parname, "_".join([new_parname, p]))
            #     if p == "ttlf":
            #         for b in current_bins:
            #             ultisplit = "_".join([new_parname, p, b])
            #             harvester.cp().bin([b])\
            #             .process([p])\
            #             .RenameSystematic(harvester, "_".join([new_parname, p]), ultisplit)

def decorrelate_nuisances_per_process(
    harvester,
    problematic_nps,
    processes,
    syst_types = ["lnN", "shape"],
):
    
    for proc in processes:
        # resolve possible wildcards for processes
        resolved_processes = harvester.cp().process([proc]).process_set()
        print("Will decorrelate nuisances for processes")
        print(resolved_processes)
        for resolved_proc in resolved_processes:
            to_decorrelate = (
                harvester.cp().syst_type(syst_types).
                process([resolved_proc]).syst_name(problematic_nps).
                syst_name_set()
            )
 
            for par in to_decorrelate:
                new_parname = "_".join([par, resolved_proc])
                print("decorrelating bin '{}'".format(", ".join(harvester.cp().bin_set())))
                print("\tprocesses: '{}'".format(", ".join([resolved_proc])))
                print("\tnew name: '{}'".format(new_parname))
                harvester.cp().process([resolved_proc])\
                    .RenameSystematic(harvester, par, new_parname)


def merge_systs_across_years(harvester, problematic_nps, processes, syst_types = ["lnN", "shape"], delete_suffix= "2016"):
    to_decorrelate = harvester.cp().syst_type(
        syst_types).syst_name(problematic_nps).syst_name_set()
#       for param in to_decorrelate:
#            these_pars = filter(harvester_params, param)
    for p in to_decorrelate:
        # new_parname = "_".join([p, channel_suffixes[c]])
        new_parname = p.replace(delete_suffix, "")
        print("decorrelating bin '{}'".format(harvester.cp().bin_set()))
        print("\tprocesses: '{}'".format(", ".join(processes)))
        print("\tnew name: '{}'".format(new_parname))
        harvester.cp().process(processes).RenameSystematic(harvester, p, new_parname)

def rename_systematics(harvester, nps_to_rename, processes=[".*"], syst_types=["lnN", "shape"]):
    
    for wildcard, new_parname in nps_to_rename.items():
        parameters = harvester.cp().syst_type(syst_types).syst_name([wildcard]).process(processes).syst_name_set()

        for p in parameters:
            print("renaming in bin '{}'".format(harvester.cp().bin_set()))
            print("\tprocesses: '{}'".format(", ".join(processes)))
            print("\tnew name: '{}'".format(new_parname))
            harvester.cp().process(processes).RenameSystematic(harvester, p, new_parname)

def remove_nuisances(harvester, problematic_nps, syst_types = ["lnN", "shape"], processes=[".*"]):
    to_remove = harvester.cp().process(processes).syst_type(
        syst_types).syst_name(problematic_nps).syst_name_set()
    process_set = harvester.cp().process(processes).process_set()
    if len(to_remove) > 0:
        print("removing following nuisances completely")
        print(to_remove)
        harvester.FilterSysts(lambda syst: any(syst.name() == p for p in to_remove if syst.process() in process_set))
    else:
        print("could not find any parameters from following list")
        print(problematic_nps)

def add_nuisances(harvester, nuisance_dictionary):
    for parname in nuisance_dictionary:
        this_dict = nuisance_dictionary[parname]
        syst_type = this_dict.get("type", "lnN")
        value = this_dict.get("value", 1.5)
        processes = this_dict.get("processes", [])
        channels = this_dict.get("channels", [])
        harvester.cp().bin(channels).process(processes)\
                        .AddSyst(harvester, parname, syst_type, ch.SystMap()(value))
        if parname in harvester.cp().syst_type(["rateParam"]).syst_name_set():
            harvester.GetParameter(parname).set_range(-5, 5)

def write_cards(
    harvester,
    cardpath,
    prefix,
    output_rootfile,
    outdir,
):
    freeze_nuisances(harvester=harvester)
    
    # harvester.PrintSysts()
    
    basename = os.path.basename(cardpath)
    basename = "{}_{}".format(prefix, basename) if prefix else basename
    newpath = os.path.join(outdir, "datacards", basename)
    if output_rootfile is None:
        output_rootfile = ".".join(basename.split(".")[:-1] + ["root"])
    output_rootfile = "{}_{}".format(prefix, output_rootfile) \
        if prefix else output_rootfile
    output_rootfile = os.path.join(outdir, output_rootfile)

    # harvester.WriteDatacard(newpath)
    writer = ch.CardWriter(newpath, output_rootfile)
    writer.SetWildcardMasses([])
    writer.SetVerbosity(1)
    writer.WriteCards("cmb", harvester)

def main(**kwargs):

    harvester = ch.CombineHarvester()
    cardpath = kwargs.get("datacard")
    harvester.SetFlag("filters-use-regex", True)
    print(cardpath)
    harvester.ParseDatacard(cardpath, "test", "13TeV", "")

    # harvester.PrintAll()
    channels = harvester.bin_set()
    channel_suffixes = {
        "{}".format(ch): "TT_{}".format(ch) for ch in channels
    }
    to_decorrelate = """
        CMS_btag_LF_2016_2017_2018
    """.split()
   
    processes = ["TT"]
    
    # partially_decorrelate_nuisances(
    #     harvester = harvester,
    #     channel_suffixes=channel_suffixes,
    #     problematic_nps=to_decorrelate,
    #     processes=processes,
    # )
    # partially_decorrelate_on_template_level(
    #     harvester=harvester,
    #     process_list=processes,
    #     nuisance_list=to_decorrelate
    # )
    nuisance_dict = {
        "CMS_ttHbb_bgnorm_ttcc_{}".format(suffix): {
            "channels": [channels],
            "processes": ["ttcc.*"],
            "type": "rateParam",
            "value": float(1)
        } for channels, suffix in channel_suffixes.items()
    }
    
    # rename_dict = {
    #     'CMS_16_L1PreFiring': 'CMS_L1PreFiring_2016',
    #     'CMS_ttHbb_L1PreFiring_2016': 'CMS_L1PreFiring_2016',
    #     'CMS_17_L1PreFiring': 'CMS_L1PreFiring_2017',
    #     'CMS_ttHbb_L1PreFiring_2017': 'CMS_L1PreFiring_2017', 
    #     'CMS_HEM': 'CMS_scaleHEM1516_j',
    #     'CMS_ttHbb_PU': "CMS_pileup",
    #     'CMS_ttHbb_scaleMuF_tHq': 'QCDscale_muF_tHq',
    #     'CMS_ttHbb_scaleMuF_ttH': 'QCDscale_muF_ttH',
    #     'CMS_ttHbb_scaleMuR_tHq': 'QCDscale_muR_tHq',
    #     'CMS_ttHbb_scaleMuR_ttH': 'QCDscale_muR_ttH',
    #     'hzz_br': 'BR_hzz',
    #     'CMS_hgg_JER_TTH_2016': 'CMS_res_j_2016',
    #     'CMS_hgg_JER_TTH_2017': 'CMS_res_j_2017',
    #     'CMS_hgg_JER_TTH_2018': 'CMS_res_j_2018',
    # }
    decorrelate_nuisances(
        harvester=harvester, channel_suffixes=channel_suffixes,
        problematic_nps=to_decorrelate, processes=processes,
        
    )
    decorrelate_nuisances_per_process(
        harvester=harvester, problematic_nps=to_decorrelate, processes=processes,
    )

    outdir = kwargs.get("outdir")
    prefix = kwargs.get("prefix", None)
    output_rootfile = kwargs.get("output_rootpath")

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    # this_prefix = "_".join([prefix, "STXS_preferred"]) if prefix else "STXS_preferred"
    # write_cards(harvester=harvester,
    #     cardpath=cardpath,
    #     prefix=this_prefix,
    #     output_rootfile=output_rootfile,
    #     outdir=outdir,
    # )
    # bu_harvester = harvester.cp()

    # now additionally decorrelate scale uncertainties
    # to_decorrelate = ".*bgnorm_ttbb".split()
    # partially_decorrelate_nuisances(
    #     harvester = harvester,
    #     channel_suffixes=channel_suffixes,
    #     problematic_nps=to_decorrelate,
    #     processes=processes,
    #     syst_types=["rateParam"],
    # )

    # this_prefix = "_".join([prefix, "STXS_preferred_bgnorm_ttbb"]) if prefix else "STXS_preferred_bgnorm_ttbb"
    # write_cards(harvester=harvester,
    #     cardpath=cardpath,
    #     prefix=this_prefix,
    #     output_rootfile=output_rootfile,
    #     outdir=outdir,
    # )

    # finally, also decorrelate UE
    # to_decorrelate = [".*_glusplit"]
    # partially_decorrelate_nuisances(
    #     harvester = bu_harvester,
    #     channel_suffixes=channel_suffixes,
    #     problematic_nps=to_decorrelate,
    #     processes=processes,
    # )

    # this_prefix = "_".join([prefix, "STXS_preferred_glusplit"]) if prefix else "STXS_preferred_glusplit"
    # write_cards(harvester=bu_harvester,
    #     cardpath=cardpath,
    #     prefix=this_prefix,
    #     output_rootfile=output_rootfile,
    #     outdir=outdir,
    # )

    # to_decorrelate = [".*_glusplit"]
    # partially_decorrelate_nuisances(
    #     harvester = harvester,
    #     channel_suffixes=channel_suffixes,
    #     problematic_nps=to_decorrelate,
    #     processes=processes,
    # )

    # this_prefix = "_".join([prefix, "STXS_preferred_glusplit_bgnorm_ttbb_ttcc"]) if prefix else "STXS_preferred_glusplit_bgnorm_ttbb_ttcc"
    this_prefix = prefix
    write_cards(harvester=harvester,
        cardpath=cardpath,
        prefix=this_prefix,
        output_rootfile=output_rootfile,
        outdir=outdir,
    )

def parse_arguments():
    usage = " ".join("""
    Tool to change inputs for combine based on output of
    'ValidateDatacards.py'. This tool employs functions
    from the CombineHarvester package. Please make sure
    that you have installed it!
    """.split())
    parser = OptionParser(usage=usage)
    parser.add_option("-d", "--datacard",
                      help=" ".join(
                            """
                            path to datacard to change
                            """.split()
                      ),
                      dest="datacard",
                      metavar="path/to/datacard",
                      type="str"
                      )

    parser.add_option("-o", "--outputrootfile",
                      help=" ".join(
                            """
                            use this root file name for the varied
                            inputs. Default is the original root file
                            name
                            """.split()
                      ),
                      dest="output_rootpath",
                      metavar="path/for/output",
                      # default = ".",
                      type="str"
                      )

    optional_group = OptionGroup(parser, "optional options")
    optional_group.add_option("--directory",
                              help=" ".join(
                                  """
                            save new datacards in this directory.
                            Default = "."
                            """.split()
                              ),
                              dest="outdir",
                              metavar="path/for/output",
                              default=".",
                              type="str"
                              )
    optional_group.add_option("-p", "--prefix",
                              help=" ".join(
                                  """
                            prepend this string to the datacard names.
                            Output will have format 'PREFIX_DATACARDNAME'
                            """.split()
                              ),
                              dest="prefix",
                              type="str"
                              )
    parser.add_option_group(optional_group)
    options, files = parser.parse_args()

    cardpath = options.datacard
    if not cardpath:
        parser.error("You need to provide the path to a datacard!")
    elif not os.path.exists(cardpath):
        parser.error("Could not find datacard in '{}'".format(cardpath))

    return options, files


if __name__ == "__main__":
    options, files = parse_arguments()
    main(**vars(options))
