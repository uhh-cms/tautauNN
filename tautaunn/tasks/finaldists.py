
import luigi
import law

from tautaunn.tasks.base import Task
from tautaunn.tasks.datacards import WriteDatacards, _default_categories


class MakeFinalDistPlots(WriteDatacards, Task):

    output_dir = luigi.Parameter(default=law.NO_STR,
                                 description="output directory for the plots") 

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def requires(self):
        return WriteDatacards.req(self,
                                  version=self.version,
                                  categories=self.categories,
                                  qcd_estimation=self.qcd_estimation,
                                  binning=self.binning,
                                  n_bins=self.n_bins,
                                  uncertainty=self.uncertainty,
                                  signal_uncertainty=self.signal_uncertainty,
                                  variable=self.variable,
                                  parallel_read=self.parallel_read,
                                  parallel_write=self.parallel_write,
                                  output_suffix=self.output_suffix,
                                  rewrite_existing=self.rewrite_existing)

    def output(self):
        # prepare the output directory
        dirname = f"{self.output_dir}"
        if self.output_suffix not in ("", law.NO_STR):
            dirname += f"_{self.output_suffix.lstrip('_')}"
        return self.local_target(dirname, dir=True)

    @law.decorator.safe_output
    def run(self):
        # load the datacard creating function
        from tautaunn.plot_dists import make_plots

        inp = self.input()
        # create the cards
        input_dir = inp[inp.keys()[0]].abspath
        from IPython import embed; embed()
        make_plots(
            input_dir=input_dir,
            output_dir=self.output_dir,
        )
