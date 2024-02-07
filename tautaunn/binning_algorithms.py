import awkward as ak
import numpy as np

def uncertainty_driven(signal_values: ak.Array,
                       bkgd_values: ak.Array,
                       bkgd_uncertainty: float,
                       signal_uncertainty: float | None = None,
                       N_signal: int | None = None,
                       x_min: float=0.,
                       x_max: float=1.,):
    if N_signal is None and signal_uncertainty is None:
        raise ValueError("Either N_signal or signal_uncertainty must be provided")
    if N_signal is not None and signal_uncertainty is not None:
        raise ValueError("Only one of N_signal or signal_uncertainty must be provided")

    N_bkgd = int(np.ceil(1/(bkgd_uncertainty)**2))
    if N_signal is None:
        N_signal = int(np.ceil(1/(signal_uncertainty)**2))
    # sort the signal values ascending
    signal_values = ak.sort(signal_values)
    # sort the bkgd values ascending
    bkgd_values = ak.sort(bkgd_values)
    bin_edges = [x_max,]
    while True:
        # calculate the min of 
        min_bkgd = bkgd_values[int(-1*N_bkgd)]
        min_sig = signal_values[int(-1*N_signal)]
        # the rightmost bin should contain at least min_N events
        next_edge = ak.min([min_sig, min_bkgd])
        # update vals
        bkgd_values = bkgd_values[bkgd_values < next_edge]
        signal_values = signal_values[signal_values < next_edge]
        # check remaining stats
        if any(len(vals) < limit for vals, limit in zip([signal_values, bkgd_values], [N_signal, N_bkgd])):
            # close bin edges with x_min
            bin_edges.append(x_min)
            break
        bin_edges.append(next_edge)
    bin_edges = sorted(set(round(edge, 5) for edge in bin_edges))
    return bin_edges

def flat_signal(signal_values: ak.Array,
                signal_weights: ak.Array,
                x_min: float=0.,
                x_max: float=1.,
                n_bins: int=10,):
    if n_bins == 1:
        return [x_min, x_max]
    else:
        # calculate the signal yield per bin
        bin_yield = ak.sum(signal_weights)/n_bins
        # reverse sort the signal values
        sort_indices = ak.argsort(signal_values, ascending=False)
        signal_values = signal_values[sort_indices]
        signal_weights = signal_weights[sort_indices]
        cumulative_yield = np.cumsum(signal_weights)
        bin_edges = [x_max]
        for i in range(n_bins-1):
            # find the index such that the cumulative yield up to signal_values[index] is bin_yield*(i+1) 
            bin_index = np.searchsorted(cumulative_yield, bin_yield*(i+1))
            # check if remaining cumulative yield is less than half of the bin_yield
            if ak.sum(signal_weights[bin_index:]) < (0.5 * bin_yield):
                # if so, merge the last two bins -> close with x_min
                break
            bin_edges.append(signal_values[bin_index])
        bin_edges.append(x_min)
        bin_edges = sorted(set(round(edge, 5) for edge in bin_edges))
        return bin_edges
    
def flat_signal_ud(signal_values: ak.Array,
                   signal_weights: ak.Array,
                   bkgd_values: ak.Array,
                   uncertainty: float,
                   x_min: float=0.,
                   x_max: float=1.,
                   n_bins: int=10):
    if n_bins == 1:
        return [x_min, x_max]
    else:
        N_min = int(np.ceil(1/(uncertainty**2)))
        bkgd_values = ak.sort(bkgd_values)
        # reverse sort the signal values and weights
        sort_indices = ak.argsort(signal_values, ascending=False)
        signal_values = signal_values[sort_indices]
        signal_weights = signal_weights[sort_indices]
        cumulative_yield = np.cumsum(signal_weights)
        bin_edges = [x_max]
        # rightmost bin edge is derived by requiring at least N_min bkgd events in the last bin
        bin_edges.append(bkgd_values[int(-1*N_min)])
        # calculate the signal yield in that bin
        bin_yield = ak.sum(signal_weights[signal_values >= bin_edges[-1]])
        # update bkgd values
        bkgd_values = bkgd_values[bkgd_values < bin_edges[-1]]
        # now we can calculate the remaining bin edges
        for i in range(n_bins-2):
            # find the index such that the cumulative yield up to signal_values[index] is bin_yield*(i+1) 
            bin_index = np.searchsorted(cumulative_yield, bin_yield*(i+1))
            # if this index is equal to the length of the signal array this means we can only create one bin
            if bin_index == len(signal_values):
                print(f"Reducing n_bins to 1 due to low signal statistics")
                break
            # check bkgd stats for new bin (bkgd_values has been updated to exclude
            # anything above the last bin edge already)
            new_edge = signal_values[bin_index]
            mask = bkgd_values >= new_edge 
            if len(bkgd_values[mask]) < N_min:
                new_edge = bkgd_values[int(-1*N_min)]
            # check if remaining cumulative yield is less than half of the bin_yield
            if ak.sum(signal_weights[bin_index:]) < (0.5 * bin_yield):
                # if so, merge the last two bins -> close with x_min
                break
            bin_edges.append(signal_values[bin_index])
            bkgd_values = bkgd_values[bkgd_values < bin_edges[-1]]
        bin_edges.append(x_min)
        bin_edges = sorted(set(round(edge, 5) for edge in bin_edges))
        return bin_edges

def tt_dy_driven(signal_values: ak.Array,
                 tt_values: ak.Array,
                 dy_values: ak.Array,
                 uncertainty: float,
                 signal_uncertainty: float | None = None,
                 x_min: float=0.,
                 x_max: float=1.,):
    if signal_uncertainty is None:
        signal_uncertainty = uncertainty
    # sort the tt and dy values
    signal_values = ak.sort(signal_values)
    tt_values = ak.sort(tt_values)
    dy_values = ak.sort(dy_values)
    # the rightmost bin should contain at least min_N events
    bin_edges = [x_max]
    min_N = int(np.ceil(1/(uncertainty)**2))
    min_N_signal = int(np.ceil(1/(signal_uncertainty)**2))
    edge_num = 1
    while True:
        # calculate the max of these two
        min_tt = tt_values[int(-1*min_N)]
        min_dy = dy_values[int(-1*min_N)]
        # we want at least one signal event in the bin
        bkgd_driven = ak.max([min_tt, min_dy])
        min_signal = signal_values[int(-1*min_N_signal)]
        next_edge = ak.min([bkgd_driven, min_signal])
        bin_edges.append(next_edge)
        tt_values = tt_values[tt_values < next_edge]
        dy_values = dy_values[dy_values < next_edge]
        signal_values = signal_values[signal_values < next_edge]
        # check remaining stats
        if len(signal_values) < min_N_signal: 
            # remove previous edge
            bin_edges.pop()
            # close bin edges with x_min
            bin_edges.append(x_min)
            break
        if any([len(vals) < min_N for vals in [tt_values, dy_values]]):
            # remove previous edge
            bin_edges.pop()
            # close bin edges with x_min
            bin_edges.append(x_min)
            break
    bin_edges.append(x_min)
    bin_edges = sorted(set(round(edge, 5) for edge in bin_edges))
    return bin_edges



