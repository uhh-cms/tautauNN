import awkward as ak
import numpy as np

def uncertainty_driven(signal_values: ak.Array,
                       bkgd_values: ak.Array,
                       uncertainty: float,
                       n_bins: int=10):
    # sort the bkgd values ascending
    bkgd_values = ak.sort(bkgd_values)
    # sort signal values ascending
    signal_values = ak.sort(signal_values)
    # the rightmost bin should contain at least 400 bkgd events
    bin_edges = [1,]
    min_N = int(np.ceil(1/(uncertainty)**2))
    edge_num = 1
    while True:
        # calculate the min of 
        min_sig = signal_values[int(-1*min_N)]
        min_bkgd = bkgd_values[int(-1*min_N)]
        next_edge = ak.min([min_sig, min_bkgd])
        edge_num += 1
        if any([next_edge == ak.min(vals) for vals in [signal_values, bkgd_values]]):
            # check remaining stats
            if any([len(vals) < min_N for vals in [signal_values, bkgd_values]]):
                # remove previous edge
                bin_edges.pop()
            # close bin edges with 0
            bin_edges.append(0)
            break
        if edge_num == n_bins-1:
            # check remaining stats
            if any([len(vals) < min_N for vals in [signal_values, bkgd_values]]):
                # remove previous edge
                bin_edges.pop()
            # close bin edges with 0
            bin_edges.append(0)
            break
        bin_edges.append(next_edge)
        signal_values = signal_values[signal_values < next_edge]
        bkgd_values = bkgd_values[bkgd_values < next_edge]
    bin_edges = sorted(set(round(edge, 5) for edge in bin_edges))
    return bin_edges

def flat_signal(signal_values: ak.Array,
                signal_weights: ak.Array,
                n_bins: int=10):
    # calculate the signal yield per bin
    bin_yield = ak.sum(signal_weights)/n_bins
    # reverse sort the signal values
    sort_indices = ak.argsort(signal_values, ascending=False)
    signal_values = signal_values[sort_indices]
    signal_weights = signal_weights[sort_indices]
    cumulative_yield = ak.cumsum(signal_weights)
    bin_edges = [1]
    for i in range(n_bins-1):
        # find the index such that the cumulative yield up to signal_values[index] is bin_yield*(i+1) 
        bin_index = np.searchsorted(cumulative_yield, bin_yield*(i+1))
        # check if remaining cumulative yield is less than half of the bin_yield
        if ak.sum(signal_weights[bin_index:]) < (0.5 * bin_yield):
            # if so, merge the last two bins -> close with 0
            break
        bin_edges.append(signal_values[bin_index])
    bin_edges.append(0)
    bin_edges = sorted(set(round(edge, 5) for edge in bin_edges))
    return bin_edges
    
def flat_signal_ud(signal_values: ak.Array,
                   signal_weights: ak.Array,
                   bkgd_values: ak.Array,
                   uncertainty: float,
                   n_bins: int=10):
    N_min = int(np.ceil(1/(uncertainty)**2))
    bkgd_values = ak.sort(bkgd_values)
    # reverse sort the signal values and weights
    sort_indices = ak.argsort(signal_values, ascending=False)
    signal_values = signal_values[sort_indices]
    signal_weights = signal_weights[sort_indices]
    cumulative_yield = ak.cumsum(signal_weights)
    bin_edges = [1]
    # rightmost bin edge is derived by requiring at least N_min signal events in the last bin
    bin_edges.append(bkgd_values[int(-1*N_min)])
    # calculate the signal yield in that bin
    bin_yield = ak.sum(signal_weights[signal_values > bin_edges[-1]])
    # now we can calculate the remaining bin edges
    for i in range(n_bins-2):
        # find the index such that the cumulative yield up to signal_values[index] is bin_yield*(i+1) 
        bin_index = np.searchsorted(cumulative_yield, bin_yield*(i+1))
        # check bkgd stats for new bin (bkgd_values has been updated to exclude
        # anything above the last bin edge already)
        new_edge = signal_values[bin_index]
        mask = bkgd_values >= new_edge 
        if len(bkgd_values[mask]) < N_min:
            new_edge = bkgd_values[int(-1*N_min)]
        # check if remaining cumulative yield is less than half of the bin_yield
        if ak.sum(signal_weights[bin_index:]) < (0.5 * bin_yield):
            # if so, merge the last two bins -> close with 0
            break
        bin_edges.append(signal_values[bin_index])
        bkgd_values = bkgd_values[bkgd_values < bin_edges[-1]]
    bin_edges.append(0)
    bin_edges = sorted(set(round(edge, 5) for edge in bin_edges))
    return bin_edges



