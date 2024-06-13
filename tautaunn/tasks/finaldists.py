
import luigi
import law
import os

from tautaunn.tasks.base import Task
from tautaunn.tasks.datacards import WriteDatacards, _default_categories


class MakeFinalDistPlots(WriteDatacards, Task):

    output_suffix = luigi.Parameter(
        default=law.NO_STR,
        description="suffix to append to the output directory; default: ''",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        # hotfix location in case TN_STORE_DIR is set to Marcel's
        self.output_dir = ""
        
        path_user = (pathlist := self.input().dir.path.split("/"))[int(pathlist.index("user")+1)]
        if path_user != os.environ["USER"]: 
            old_path = self.output().targets[self.card_names[0]].abs_dirname
            user_path = old_path.replace(path_user, os.environ["USER"])
            print(f"replacing {path_user} with {os.environ['USER']} in output path.")
            yn = input("continue? [y/n] ")
            if yn.lower() != "y":
                user_path = input(f"enter the correct path (should point to your $TN_STORE_DIR/{self.__class__.__name__}): ")
        self.output_dir = user_path

    def requires(self):
        return WriteDatacards.req(self,
                                  version=self.version,
                                  categories=self.categories,
                                  qcd_estimation=self.qcd_estimation,
                                  binning=self.binning,
                                  n_bins=self.n_bins,
                                  variable=self.variable,
                                  output_suffix=self.output_suffix)

    def output(self):
        # TODO: change this such that they are separated by year and channel (and maybe cat)
        # prepare the output directory
        dirname = f"{self.output_dir}"
        if self.output_suffix not in ("", law.NO_STR):
            dirname += f"_{self.output_suffix.lstrip('_')}"
        d = self.local_target(dirname, dir=True)
        return law.FileCollection({name: d.child(f"{name}.png", type="f")
                                   for name in self.card_names})

    @law.decorator.safe_output
    def run(self):
        # load the datacard creating function
        from tautaunn.plot_dists import make_plots

        inp = self.input()
        # create the cards
        make_plots(
            input_dir=inp.dir.path,
            output_dir=self.output().targets[self.card_names[0]].abs_dirname,
        )
