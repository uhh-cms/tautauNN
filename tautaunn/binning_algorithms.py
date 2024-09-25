import awkward as ak
import numpy as np
import itertools
from typing import List, Tuple

def uncertainty_driven(signal_values: ak.Array,
                       bkgd_values: ak.Array,
                       dy_values: ak.Array,
                       tt_values: ak.Array,
                       bkgd_uncertainty: float,
                       signal_uncertainty: float | None = None,
                       n_bins: int=10,
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
    tt_values = ak.sort(tt_values)
    dy_values = ak.sort(dy_values)
    bin_edges = [x_max,]
    inner_edge_num = 0
    while True:
        # check remaining stats
        if any(len(vals) < limit for vals, limit in zip([signal_values, bkgd_values], [N_signal, N_bkgd])):
            # close bin edges with x_min
            bin_edges.append(x_min)
            break
        inner_edge_num += 1
        # one needs n-1 inner edges (+2 outer edges) for n bins
        if inner_edge_num > n_bins-1:
            # close bin edges with x_min
            bin_edges.append(x_min)
            break
        # the rightmost bin should contain at least min_N events and also at least one dy and tt event
        next_edge = ak.min([signal_values[int(-1*N_signal)], bkgd_values[int(-1*N_bkgd)], tt_values[-1], dy_values[-1]])
        # update vals
        bkgd_values = bkgd_values[bkgd_values < next_edge]
        tt_values = tt_values[tt_values < next_edge]
        dy_values = dy_values[dy_values < next_edge]
        signal_values = signal_values[signal_values < next_edge]
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
                   tt_values: ak.Array,
                   dy_values: ak.Array,
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
        tt_values = ak.sort(tt_values)
        dy_values = ak.sort(dy_values)
        bin_edges = [x_max]
        # rightmost bin edge is derived by requiring at least N_min bkgd events in the last bin
        # additionally check after each bin edge if there's at least one dy&tt event
        bin_edges.append(ak.min([dy_values[-1], tt_values[-1], bkgd_values[int(-1*N_min)]]))
        # calculate the signal yield in that bin
        bin_yield = ak.sum(signal_weights[signal_values >= bin_edges[-1]])
        # update bkgd values
        bkgd_values = bkgd_values[bkgd_values < bin_edges[-1]]
        dy_values = dy_values[dy_values < bin_edges[-1]]
        tt_values = tt_values[tt_values < bin_edges[-1]]
        # now we can calculate the remaining bin edges
        for i in range(n_bins-2):
            # find the index such that the cumulative yield up to signal_values[index] is bin_yield*(i+1) 
            bin_index = np.searchsorted(cumulative_yield, bin_yield*(i+1))
            # if this index is equal to the length of the signal array the signal yield has been 'used up' 
            if bin_index == len(signal_values):
                print(f"Reducing n_bins to {len(bin_edges)-1} due to low signal statistics")
                # check if remaining cumulative yield is less than 80 percent of the bin_yield
                mask = signal_values < bin_edges[-1]
                if ak.sum(signal_weights[mask]) < (0.8 * bin_yield):
                    # if so, merge the last two bins -> close with x_min
                    bin_edges.pop()
                break
            flat_s_edge = signal_values[bin_index]
            # choose the min out of the following to fulfill all criteria
            new_edge = ak.min([dy_values[-1], tt_values[-1], bkgd_values[int(-1*N_min)], flat_s_edge])
            bin_edges.append(new_edge)
            bkgd_values = bkgd_values[bkgd_values < new_edge] 
            dy_values = dy_values[dy_values < new_edge] 
            tt_values = tt_values[tt_values < new_edge] 
        bin_edges.append(x_min)
        epsilon = 5e-6
        bin_edges = sorted(set(round(edge-epsilon, 5) for edge in bin_edges))
        return bin_edges

def tt_dy_driven(signal_values: ak.Array,
                 tt_values: ak.Array,
                 dy_values: ak.Array,
                 uncertainty: float,
                 signal_uncertainty: float | None = None,
                 mode: str = "min", # if min, the unct. requirement is fulfilled by both tt and dy 
                 n_bins: int=10,
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
    inner_edge_num = 0
    while True:
        # check remaining stats
        if len(signal_values) < min_N_signal: 
            # close bin edges with x_min
            bin_edges.append(x_min)
            break
        if any([len(vals) < min_N for vals in [tt_values, dy_values]]):
            # close bin edges with x_min
            bin_edges.append(x_min)
            break
        inner_edge_num += 1
        if inner_edge_num > n_bins-1:
            # close bin edges with x_min
            bin_edges.append(x_min)
            break
        # tt & dy requirements 
        min_tt = tt_values[int(-1*min_N)]
        min_dy = dy_values[int(-1*min_N)]
        if mode == "min":
            bkgd_driven = ak.min([min_tt, min_dy])
        elif mode == "max":
            bkgd_driven = ak.max([min_tt, min_dy])
        else:
            raise ValueError(f"mode must be either 'min' or 'max', got {mode}")
        # further require that the signal has at least min_N_signal events
        min_signal = signal_values[int(-1*min_N_signal)]
        next_edge = ak.min([bkgd_driven, min_signal])
        bin_edges.append(next_edge)
        tt_values = tt_values[tt_values < next_edge]
        dy_values = dy_values[dy_values < next_edge]
        signal_values = signal_values[signal_values < next_edge]
    bin_edges.append(x_min)
    epsilon = 5e-6
    bin_edges = sorted(set(round(edge-epsilon, 5) for edge in bin_edges))
    return bin_edges



def flatsguarded(hh_values: ak.Array,
                 tt_values: ak.Array,
                 dy_values: ak.Array,
                 tt_weights: ak.Array,
                 hh_weights: ak.Array,
                 dy_weights: ak.Array,
                 n_bins: int=10,
                 x_min: float=0.,
                 x_max: float=1.,):

    # create a record array with eight entries:
    # - value
    # - process (0: hh, 1: tt, 2: dy)
    # - hh_count_cs, tt_count_cs, dy_count_cs (cumulative sums of raw event counts)
    # - hh_weight_cs, tt_weight_cs, dy_weight_cs (cumulative sums of weights)
    all_values_list = [hh_values, tt_values, dy_values]
    rec = np.core.records.fromarrays(
        [
            # value
            (all_values := np.concatenate(all_values_list, axis=0)),
            # process
            np.concatenate([i * np.ones(len(v), dtype=np.int8) for i, v in enumerate(all_values_list)], axis=0),
            # counts and weights per process
            (izeros := np.zeros(len(all_values), dtype=np.int32)),
            (fzeros := np.zeros(len(all_values), dtype=np.float32)),
            izeros,
            fzeros,
            izeros,
            fzeros,
        ],
        names="value,process,hh_count_cs,hh_weight_cs,tt_count_cs,tt_weight_cs,dy_count_cs,dy_weight_cs",
    )
    # insert counts and weights into columns for correct processes
    # (faster than creating arrays above which then get copied anyway when the recarray is created)
    HH, TT, DY = range(3)
    rec.hh_count_cs[rec.process == HH] = 1
    rec.tt_count_cs[rec.process == TT] = 1
    rec.dy_count_cs[rec.process == DY] = 1
    rec.hh_weight_cs[rec.process == HH] = hh_weights
    rec.tt_weight_cs[rec.process == TT] = tt_weights
    rec.dy_weight_cs[rec.process == DY] = dy_weights
    # sort by decreasing value to start binning from "the right" later on
    rec.sort(order="value")
    rec = np.flip(rec, axis=0)
    # replace counts and weights with their cumulative sums
    rec.hh_count_cs[:] = np.cumsum(rec.hh_count_cs)
    rec.tt_count_cs[:] = np.cumsum(rec.tt_count_cs)
    rec.dy_count_cs[:] = np.cumsum(rec.dy_count_cs)
    rec.hh_weight_cs[:] = np.cumsum(rec.hh_weight_cs)
    rec.tt_weight_cs[:] = np.cumsum(rec.tt_weight_cs)
    rec.dy_weight_cs[:] = np.cumsum(rec.dy_weight_cs)
    # eager cleanup
    del all_values, izeros, fzeros
    del hh_values, hh_weights
    del tt_values, tt_weights
    del dy_values, dy_weights
    # now, between any two possible discriminator values, we can easily extract the hh, tt and dy integrals,
    # as well as raw event counts without the need for additional, costly accumulation ops (sum, count, etc.),
    # but rather through simple subtraction of values at the respective indices instead

    #
    # step 2: binning
    #

    # determine the approximate hh yield per bin
    hh_yield_per_bin = rec.hh_weight_cs[-1] / n_bins
    # keep track of bin edges and the hh yield accumulated so far
    bin_edges = [x_max]
    hh_yield_binned = 0.0
    min_hh_yield = 1.0e-5
    # during binning, do not remove leading entries, but remember the index that denotes the start of the bin
    offset = 0
    # helper to extract a cumulative sum between the start offset (included) and the stop index (not included)
    get_integral = lambda cs, stop: cs[stop - 1] - (0 if offset == 0 else cs[offset - 1])
    # bookkeep reasons for stopping binning
    stop_reason = ""
    # start binning
    while len(bin_edges) < n_bins:
        # stopping condition 1: reached end of events
        if offset >= len(rec):
            stop_reason = "no more events left"
            break
        # stopping condition 2: remaining hh yield too small, so cause a background bin to be created
        remaining_hh_yield = rec.hh_weight_cs[-1] - hh_yield_binned
        if remaining_hh_yield < min_hh_yield:
            stop_reason = "remaining signal yield insufficient"
            # remove the last bin edge and stop
            if len(bin_edges) > 1:
                bin_edges.pop()
            break
        # find the index of the event that would result in a hh yield increase of more than the expected
        # per-bin yield; this index would mark the start of the next bin given all constraints are met
        if remaining_hh_yield >= hh_yield_per_bin:
            threshold = hh_yield_binned + hh_yield_per_bin
            next_idx = offset + np.where(rec.hh_weight_cs[offset:] > threshold)[0][0]
        else:
            # special case: remaining hh yield smaller than the expected per-bin yield, so find the last event
            next_idx = offset + np.where(rec.process[offset:] == HH)[0][-1] + 1
        # advance the index until backgrounds constraints are met
        while next_idx < len(rec):
            # get the number of tt events and their yield
            n_tt = get_integral(rec.tt_count_cs, next_idx)
            y_tt = get_integral(rec.tt_weight_cs, next_idx)
            # get the number of dy events and their yield
            n_dy = get_integral(rec.dy_count_cs, next_idx)
            y_dy = get_integral(rec.dy_weight_cs, next_idx)
            # evaluate constraints
            # TODO: potentially relax constraints here, e.g when there are 3 (4?) tt events, drop the constraint
            #       on dy, and vice-versa
            constraints_met = (
                # tt and dy events
                n_tt >= 1 and
                n_dy >= 1 and
                n_tt + n_dy >= 4 and
                # yields must be positive to avoid negative sums of weights per process
                y_tt > 0 and
                y_dy > 0
            )
            if constraints_met:
                # TODO: maybe also check if the background conditions are just barely met and advance next_idx
                # to the middle between the current value and the next one that would change anything about the
                # background predictions; this might be more stable as the current implementation can highly
                # depend on the exact value of a single event (the one that tips the constraints over the edge
                # to fulfillment)

                # bin found, stop
                break
            # constraints not met, advance index to include the next tt or dy event and try again
            next_bkg_indices = np.where(rec.process[next_idx:] != HH)[0]
            if len(next_bkg_indices) == 0:
                # no more background events left, move to the last position and let the stopping condition 3
                # below handle the rest
                next_idx = len(rec)
            else:
                next_idx += next_bkg_indices[0] + 1
        else:
            # stopping condition 3: no more events left, so the last bin (most left one) does not fullfill
            # constraints; however, this should practically never happen
            stop_reason = "no more events left while trying to fulfill constraints"
            if len(bin_edges) > 1:
                bin_edges.pop()
            break
        # next_idx found, update values
        edge_value = x_min if next_idx == 0 else float(rec.value[next_idx - 1:next_idx + 1].mean())
        bin_edges.append(max(min(edge_value, x_max), x_min))
        hh_yield_binned += get_integral(rec.hh_weight_cs, next_idx)
        offset = next_idx

    # make sure the minimum is included
    if bin_edges[-1] != x_min:
        if len(bin_edges) > n_bins:
            raise RuntimeError(f"number of bins reached and initial bin edge is not x_min (edges: {bin_edges})")
        bin_edges.append(x_min)

    # reverse edges and optionally re-set n_bins
    bin_edges = sorted(set(bin_edges))
    return bin_edges, stop_reason


def flatsguarded_systs(hh_values: ak.Array,
                       hh_weights: ak.Array,
                       dy_shifts: dict[str, ak.Array], # (including nominal)
                       tt_shifts: dict[str, ak.Array],
                       n_bins: int=10,
                       x_min: float=0.,
                       x_max: float=1.,):
    
    # create a record array with eight entries:
    # - value
    # - process (0: hh, 1: tt, 2: dy)
    # - hh_count_cs, tt_count_cs, dy_count_cs (cumulative sums of raw event counts)
    # - hh_weight_cs, tt_weight_cs, dy_weight_cs (cumulative sums of weights)

    tt_values = [tt_shifts[key][0] for key in tt_shifts]
    dy_values = [dy_shifts[key][0] for key in dy_shifts]

    process_map = {
        "hh_nominal": 0,
        **{"tt_"+key: i for i,key in enumerate(tt_shifts.keys())},
        **{"dy_"+key: i+len(tt_shifts.keys()) for i,key in enumerate(dy_shifts.keys())}
    }

    all_values_list = [hh_values, *tt_values, *dy_values]
    rec_list = [
                # value
                (all_values := np.concatenate(all_values_list, axis=0)),
                # process
                np.concatenate([i * np.ones(len(v), dtype=np.int8) for i, v in enumerate(all_values_list)], axis=0),
               ]
    izeros = np.zeros(len(all_values), dtype=np.int32)
    fzeros = np.zeros(len(all_values), dtype=np.float32)
    for i in range(len(all_values_list)):
        rec_list.append(izeros)
        rec_list.append(fzeros)
    rec_names = "value,process,hh_count_cs,hh_weight_cs,"
    rec_names += ",".join([f"tt_{key}_count_cs,tt_{key}_weight_cs"
                           if key != "nominal" else "tt_count_cs,tt_weight_cs"
                           for key in tt_shifts])
    rec_names += ",".join([f"dy_{key}_count_cs,dy_{key}_weight_cs"
                           if key != "nominal" else "dy_count_cs,dy_weight_cs"
                           for key in dy_shifts])

    rec = np.core.records.fromarrays(rec_list, names=rec_names)
    # insert counts and weights into columns for correct processes
    # (faster than creating arrays above which then get copied anyway when the recarray is created)
    rec.hh_count_cs[rec.process == process_map["hh_nominal"]] = 1
    rec.hh_weight_cs[rec.process == process_map["hh_nominal"]] = hh_weights
    for key in rec_names.split(",")[4:]:
        process_name = "_".join(key.split("_")[:2])
        if "counts" in key:
            rec[key][rec.process == process_map[process_name]] = 1
        if "weight" in key:
            if key.startswith("dy"):
                rec[key][rec.process == process_map[process_name]] = dy_shifts[key.split("_")[1]]
            elif key.startswith("tt"):
                rec[key][rec.process == process_map[process_name]] = tt_shifts[key.split("_")[1]]
            else:
                raise ValueError(f"Unknown process {key}")
    # sort by decreasing value to start binning from "the right" later on
    rec.sort(order="value")
    rec = np.flip(rec, axis=0)
    # replace counts and weights with their cumulative sums
    for key in rec_names.split(",")[2:]:
        rec[key][:] = np.cumsum(rec[key])
    # eager cleanup
    del all_values_list, izeros, fzeros
    del hh_values, hh_weights
    del dy_shifts, tt_shifts
    del rec_names, rec_list
    # now, between any two possible discriminator values, we can easily extract the hh, tt and dy integrals,
    # as well as raw event counts without the need for additional, costly accumulation ops (sum, count, etc.),
    # but rather through simple subtraction of values at the respective indices instead

    #
    # step 2: binning
    #

    # determine the approximate hh yield per bin
    hh_yield_per_bin = rec.hh_weight_cs[-1] / n_bins
    # keep track of bin edges and the hh yield accumulated so far
    bin_edges = [x_max]
    hh_yield_binned = 0.0
    min_hh_yield = 1.0e-5
    # during binning, do not remove leading entries, but remember the index that denotes the start of the bin
    offset = 0
    # helper to extract a cumulative sum between the start offset (included) and the stop index (not included)
    get_integral = lambda cs, stop: cs[stop - 1] - (0 if offset == 0 else cs[offset - 1])
    # bookkeep reasons for stopping binning
    stop_reason = ""
    # start binning
    while len(bin_edges) < n_bins:
        # stopping condition 1: reached end of events
        if offset >= len(rec):
            stop_reason = "no more events left"
            break
        # stopping condition 2: remaining hh yield too small, so cause a background bin to be created
        remaining_hh_yield = rec.hh_weight_cs[-1] - hh_yield_binned
        if remaining_hh_yield < min_hh_yield:
            stop_reason = "remaining signal yield insufficient"
            # remove the last bin edge and stop
            if len(bin_edges) > 1:
                bin_edges.pop()
            break
        # find the index of the event that would result in a hh yield increase of more than the expected
        # per-bin yield; this index would mark the start of the next bin given all constraints are met
        if remaining_hh_yield >= hh_yield_per_bin:
            threshold = hh_yield_binned + hh_yield_per_bin
            next_idx = offset + np.where(rec.hh_weight_cs[offset:] > threshold)[0][0]
        else:
            # special case: remaining hh yield smaller than the expected per-bin yield, so find the last event
            next_idx = offset + np.where(rec.process[offset:] == process_map["hh_nominal"])[0][-1] + 1
        # advance the index until backgrounds constraints are met
        while next_idx < len(rec):
            counts_per_shift = {
                shift: [get_integral(rec[f"dy_{shift}_count_cs"], next_idx), # n_dy
                        get_integral(rec[f"tt_{shift}_count_cs"], next_idx), # n_tt
                        get_integral(rec[f"dy_{shift}_weight_cs"], next_idx), # y_dy
                        get_integral(rec[f"tt_{shift}_weight_cs"], next_idx)] # y_tt
                for shift in dy_shifts
            }
            # evaluate constraints
            # TODO: potentially relax constraints here, e.g when there are 3 (4?) tt events, drop the constraint
            #       on dy, and vice-versa
            constraints_met = {shift: (counts_per_shift[shift][1] >= 1 and
                                       counts_per_shift[shift][0] >= 1 and
                                       counts_per_shift[shift][1] + counts_per_shift[shift][0] >= 3 and
                                       counts_per_shift[shift][2] > 0 and
                                       counts_per_shift[shift][3] > 0)
                               for shift in dy_shifts}
            if all(constraints_met.values()):
                # TODO: maybe also check if the background conditions are just barely met and advance next_idx
                # to the middle between the current value and the next one that would change anything about the
                # background predictions; this might be more stable as the current implementation can highly
                # depend on the exact value of a single event (the one that tips the constraints over the edge
                # to fulfillment)

                # bin found, stop
                break
            # constraints not met, check which shifts cause the problem
            failing_shifts = [shift for shift, met in constraints_met.items() if not met]
            
            process_names = f"dy_{failing_shifts[0]}_count_cs",f"tt_{failing_shifts[0]}_count_cs"

            next_bkg_indices = np.where(
                ((rec.process[next_idx:] == process_names[0]) | (rec.process[next_idx:] == process_names[1]))
            )[0]
            if len(next_bkg_indices) == 0:
                # no more background events left, move to the last position and let the stopping condition 3
                # below handle the rest
                next_idx = len(rec)
            else:
                next_idx += next_bkg_indices[0] + 1
        else:
            # stopping condition 3: no more events left, so the last bin (most left one) does not fullfill
            # constraints; however, this should practically never happen
            stop_reason = "no more events left while trying to fulfill constraints"
            if len(bin_edges) > 1:
                bin_edges.pop()
            break
        # next_idx found, update values
        edge_value = x_min if next_idx == 0 else float(rec.value[next_idx - 1:next_idx + 1].mean())
        bin_edges.append(max(min(edge_value, x_max), x_min))
        hh_yield_binned += get_integral(rec.hh_weight_cs, next_idx)
        offset = next_idx

    # make sure the minimum is included
    if bin_edges[-1] != x_min:
        if len(bin_edges) > n_bins:
            raise RuntimeError(f"number of bins reached and initial bin edge is not x_min (edges: {bin_edges})")
        bin_edges.append(x_min)

    # reverse edges and optionally re-set n_bins
    bin_edges = sorted(set(bin_edges))
    return bin_edges, stop_reason


def flats_systs(hh_values: ak.Array,
                hh_weights: ak.Array,
                dy_shifts: dict[str, (ak.Array, ak.Array)], # (values, weights)
                tt_shifts: dict[str, (ak.Array, ak.Array)], # (values, weights)
                n_bins: int=10,
                x_min: float=0.,
                x_max: float=1.,):

    # assert that dy_shifts and tt_shifts have the same keys
    assert dy_shifts.keys() == tt_shifts.keys()
    # binning should be done in the range [x_min, x_max]
    assert x_min < x_max <= 1

    assert len(hh_values) == len(hh_weights)
    assert all(len(dy_shifts[key][0]) == len(dy_shifts[key][1]) for key in dy_shifts.keys())
    assert all(len(tt_shifts[key][0]) == len(tt_shifts[key][1]) for key in tt_shifts.keys())
    

    def edge_to_offset(values, edge):
        return np.where(values <= edge)[0][0]

    # sort hh 
    sort_indices = ak.argsort(hh_values, ascending=False)
    hh_values = hh_values[sort_indices]
    hh_weights = hh_weights[sort_indices]
    # sort dy and tt & replace weights with cumulative sums
    for key in dy_shifts.keys():
        dy_sort_indices = ak.argsort(dy_shifts[key][0], ascending=False)
        dy_shifts[key] = dy_shifts[key][0][dy_sort_indices], np.cumsum(dy_shifts[key][1][dy_sort_indices])
        tt_sort_indices = ak.argsort(tt_shifts[key][0], ascending=False)
        tt_shifts[key] = tt_shifts[key][0][tt_sort_indices], np.cumsum(tt_shifts[key][1][tt_sort_indices])

    # create a combined array of dy and tt values
    dy_tt_values = {key: ak.sort(
                         ak.concatenate([dy_shifts[key][0], tt_shifts[key][0]], axis=0),
                         ascending=False)
                    for key in dy_shifts.keys()}
    # create a combined array of dy and tt weights
    dy_tt_weights = {key: np.cumsum(
                            ak.concatenate([dy_shifts[key][1], tt_shifts[key][1]], axis=0))
                        for key in dy_shifts.keys()}    
    # create a combined array of dy and tt squared weights 
    dy_tt_squared_weights = {key: np.cumsum(
                            ak.concatenate([(dy_shifts[key][1])**2, (tt_shifts[key][1])**2], axis=0))
                    for key in dy_shifts.keys()}    
        

    bin_edges = [x_max]
    hh_weights_cumsum = np.cumsum(hh_weights)
    del hh_weights
    signal_per_bin = np.round(hh_weights_cumsum[-1] / n_bins, 5)
    hh_offset, next_hh_offset = 0, 0
    dy_offset, next_dy_offset = 0, 0
    tt_offset, next_tt_offset = 0, 0
    dy_tt_offset, next_dy_tt_offset = 0, 0
    edge_count = 0
    while True:
        # this would be the next bin edge if we would only consider the signal
        pushover_idx = np.where(hh_weights_cumsum[hh_offset:] > (edge_count+1)*signal_per_bin)[0]
        if len(pushover_idx) == 0:
            # no more hh events left
            stop_reason = "no more events left"
            bin_edges.append(x_min)
            break
        bin_idx_by_signal = hh_offset + pushover_idx[0]
        bin_edge_by_signal = hh_values[bin_idx_by_signal]
        # N_DY & N_TT requirements:
        # we want at least 1 dy and 1 tt event in the bin
        dy_edges = [dy_shifts[key][0][dy_offset] for key in dy_shifts.keys()] 
        tt_edges = [tt_shifts[key][0][tt_offset] for key in tt_shifts.keys()]
        # SUM (N_DY,N_TT) requirement:
        # we want the sum of dy and tt events to be at least 4 
        dy_tt_edges = [dy_tt_values[key][dy_tt_offset+3] for key in dy_tt_values.keys()]
        # find the next bin edge that fulfills all requirements
        next_edge = np.min([bin_edge_by_signal, *dy_edges, *tt_edges, *dy_tt_edges])
        if next_edge <= np.min(hh_values):
            # no more hh events left
            bin_edges.append(x_min)
            stop_reason = "no more events left"
            break
        # calculate the next offsets
        next_hh_offset = edge_to_offset(hh_values, next_edge)
        next_dy_offset = edge_to_offset(dy_shifts["nominal"][0], next_edge)
        next_tt_offset = edge_to_offset(tt_shifts["nominal"][0], next_edge)
        next_dy_tt_offset = edge_to_offset(dy_tt_values["nominal"], next_edge)

        if any(next_offset >= len(vals) for next_offset, vals in zip([next_dy_offset, next_tt_offset, next_dy_tt_offset],
                                                          [dy_shifts["nominal"][0], tt_shifts["nominal"][0], dy_tt_values["nominal"]])): 
            # close bin edges with x_min
            bin_edges.append(x_min)
            stop_reason = ("no more dy or tt events left."
                           "I think this shouldn't happen often...")
            break

        ###### YIELD REQUIREMENTS ######
        # now about the yields... we want to make sure that the dy and tt yields are positive
        # calculate the bin yield for dy and tt (dy_shifts["nominal"][1] is already the cumulative sum)
        dy_bin_yields = {key: dy_shifts[key][1][next_dy_offset:] - dy_shifts[key][1][dy_offset] for key in dy_shifts.keys()}
        tt_bin_yields = {key: tt_shifts[key][1][next_tt_offset:] - tt_shifts[key][1][tt_offset] for key in tt_shifts.keys()}
        dy_min_offset = {key: np.where(dy_bin_yields[key] > 0)[0][0] for key in dy_bin_yields}
        tt_min_offset = {key: np.where(tt_bin_yields[key] > 0)[0][0] for key in tt_bin_yields}
        next_dy_offset = next_dy_offset + np.max([dy_min_offset[key] for key in dy_min_offset])
        next_tt_offset = next_tt_offset + np.max([tt_min_offset[key] for key in tt_min_offset])
        # convert to bin edges
        next_dy_edge = dy_shifts["nominal"][0][next_dy_offset]
        next_tt_edge = tt_shifts["nominal"][0][next_tt_offset]
        next_edge = np.min([next_edge, next_dy_edge, next_tt_edge])
        # convert edge to offset
        next_hh_offset = edge_to_offset(hh_values, next_edge)
        next_dy_tt_offset = edge_to_offset(dy_tt_values["nominal"], next_edge)
        
        
        ###### ERROR REQUIREMENTS ######
        # now make sure the combined error is below < 0.5
        # only for nominal for now
        #    dy_tt_errs = {key: (np.sqrt(dy_tt_squared_weights[key][next_dy_tt_offset] - dy_tt_squared_weights[key][dy_tt_offset])/
        #                   (dy_tt_weights[key][next_dy_tt_offset] - dy_tt_weights[key][dy_tt_offset]))
        #                   for key in dy_tt_squared_weights.keys()}
        #    dy_tt_err_bins = list(itertools.chain.from_iterable([np.where(dy_tt_errs[key] < 0.5)[0] for key in dy_tt_errs]))

        dy_tt_errs = np.sqrt(dy_tt_squared_weights["nominal"][next_dy_tt_offset:] - dy_tt_squared_weights["nominal"][dy_tt_offset]) / (dy_tt_weights["nominal"][next_dy_tt_offset:] - dy_tt_weights["nominal"][dy_tt_offset]) #noqa
        dy_tt_err_edge = dy_tt_values["nominal"][next_dy_tt_offset + np.where(dy_tt_errs < 0.5)[0][0]]
        next_edge = np.min([next_edge, dy_tt_err_edge])
    
        # update offsets 
        next_hh_offset = edge_to_offset(hh_values, next_edge)
        next_dy_offset = edge_to_offset(dy_shifts["nominal"][0], next_edge)
        next_tt_offset = edge_to_offset(tt_shifts["nominal"][0], next_edge)
        next_dy_tt_offset = edge_to_offset(dy_tt_values["nominal"], next_edge)
        assert all((next_offset > offset for next_offset, offset in zip((next_hh_offset, next_dy_offset, next_tt_offset, next_dy_tt_offset),
                                                                          (hh_offset, dy_offset, tt_offset, dy_tt_offset)))), "offsets not updated correctly"
        hh_offset, dy_offset, tt_offset, dy_tt_offset = next_hh_offset, next_dy_offset, next_tt_offset, next_dy_tt_offset
        edge_count += 1
        # check remaining stats
        if any(offset >= vals for offset, vals in
                zip([next_hh_offset, next_dy_offset, next_tt_offset, next_dy_tt_offset],
                     [len(hh_values), len(dy_shifts["nominal"][0]), len(tt_shifts["nominal"][0]), len(dy_tt_values["nominal"])])):
            # close bin edges with x_min
            stop_reason = "no more events left"
            bin_edges.append(x_min)
            break
        if (hh_weights_cumsum[-1] - hh_weights_cumsum[hh_offset] ) < edge_count*signal_per_bin:
            # close bin edges with x_min
            stop_reason = "remaining signal yield insufficient"
            bin_edges.append(x_min)
            break
        if edge_count > n_bins-1:
            # close bin edges with x_min
            stop_reason = "reached maximum number of bins"
            bin_edges.append(x_min)
            break
        if not (round(next_edge, 6) == round(bin_edges[-1], 6)):
            bin_edges.append(next_edge)
    bin_edges = sorted([round(float(i), 6) for i in bin_edges])
    return bin_edges, stop_reason

                    
def non_res_like(hh: Tuple[ak.Array, ak.Array],
                 dy: dict[str, (ak.Array, ak.Array)], # dict containing the nominal and jes & tes shifts
                 tt: dict[str, (ak.Array, ak.Array)], # dict containing the nominal and jes & tes shifts
                 others: dict[str, (ak.Array, ak.Array)], # dict containing the nominal and jes & tes shifts
                 n_bins: int=10,) -> list[float]:


    if len(hh[0]) == 0:
        return [0., 1.], "no hh events"
    assert len(hh[0]) == len(hh[1])
    assert all(len(dy[key][0]) == len(dy[key][1]) for key in dy.keys())
    assert all(len(tt[key][0]) == len(tt[key][1]) for key in tt.keys())
    assert all(len(others[key][0]) == len(others[key][1]) for key in others.keys())
    
    def calc_rel_err(cs_w: ak.Array, # cumulative sum of weights
                     cs_w_2: ak.Array, # cumulative sum of squared weights
                     idx: int,) -> ak.Array:
        return np.sqrt(cs_w_2[idx+1:] - cs_w_2[idx]) / (cs_w[idx+1:] - cs_w[idx])

    def edge_to_idx(values: ak.Array, edge: float) -> int:
        return np.where(values < edge)[0][0]

    def get_sorted_values_and_weights(values: ak.Array, weights: ak.Array) -> Tuple[ak.Array, ak.Array]:
        sort_indices = ak.argsort(values, ascending=False)
        return values[sort_indices], weights[sort_indices]
    
    
    hh_values, hh_weights = get_sorted_values_and_weights(*hh) 
    hh_weights_cumsum = np.cumsum(hh_weights)
    signal_per_bin = np.round(hh_weights_cumsum[-1] / n_bins, 5)

    tt = {key: get_sorted_values_and_weights(*tt[key]) for key in tt.keys()}
    dy = {key: get_sorted_values_and_weights(*dy[key]) for key in dy.keys()}

    all_bkgds = {key: get_sorted_values_and_weights(ak.concatenate([tt[key][0], dy[key][0], others[key][0]], axis=0), #values
                                                    ak.concatenate([tt[key][1], dy[key][1], others[key][1]], axis=0)) #weights
                    for key in tt.keys()}
    all_bkgds_cs = {key: np.cumsum(all_bkgds[key][1]) for key in all_bkgds.keys()}

    all_bkgds_weights_cs = np.cumsum(all_bkgds['nominal'][1])
    all_bkgds_squared_weights_cs = np.cumsum(all_bkgds['nominal'][1]**2)

    # start at 1
    bin_edges = [1.]
    hh_offset = 0
    dy_offset = 0
    tt_offset = 0
    all_bkgds_offset = 0
    stop_reason = ""

    while True:
        # bin edge if we would only consider the flat-signal requirement (only nominal)
        flat_s_edge = hh_values[np.where(hh_weights_cumsum[hh_offset:] > len(bin_edges)*signal_per_bin)[0][0]]
        # N_DY & N_TT requirements:
        # we want at least 1 dy and 1 tt event in the bin (for all shifts)
        dy_edges = {key: dy[key][0][dy_offset] for key in dy.keys()}
        tt_edges = {key: tt[key][0][tt_offset] for key in tt.keys()}
        dy_shift, dy_edge = min(dy_edges.items(), key=lambda x: x[1])
        tt_shift, tt_edge = min(tt_edges.items(), key=lambda x: x[1])
        # we want the sum of the bkgd weights to be at least 0.03 (for all shifts)
        all_bkgds_edges = {key:
            all_bkgds[key][0][np.where(all_bkgds_cs[key][all_bkgds_offset:] > 0.03)[0][0]]
            for key in all_bkgds.keys()}
        all_bkgds_shift, all_bkgds_edge = min(all_bkgds_edges.items(), key=lambda x: x[1])
        # we want the relative error on all bkgds to be below 1 (just nominal)
        rel_err_edge = all_bkgds['nominal'][0][np.where(calc_rel_err(all_bkgds_weights_cs,
                                             all_bkgds_squared_weights_cs,
                                             all_bkgds_offset) < 1)[0][0] + 1] # +1 because rel_err is calculated starting from the second element 
        # find the next bin edge that fulfills all requirements
        next_edge_dict = {"flat_s": flat_s_edge,
                          "dy": dy_edge,
                          "tt": tt_edge,
                          "all_bkgds": all_bkgds_edge,
                          "rel_err": rel_err_edge}
        next_edge_reason, next_edge = min(next_edge_dict.items(), key=lambda x: x[1])
        bin_edges.append(next_edge)
        if len(bin_edges) == n_bins:
            # close bin edges with 0
            bin_edges.append(0.)
            stop_reason = "reached maximum number of bins"
            break
        # convert edges to offsets and update them
        if next_edge <= np.min(hh_values):
            # close bin edges with 0
            bin_edges.append(0.)
            stop_reason = "no more hh events left"
            break
        hh_offset = edge_to_idx(hh_values, next_edge)

        if next_edge_reason in ["flat_s", "rel_err"]:
            dy_offset = edge_to_idx(dy["nominal"][0], next_edge)
            tt_offset = edge_to_idx(tt["nominal"][0], next_edge)
            all_bkgds_offset = edge_to_idx(all_bkgds["nominal"][0], next_edge)
        elif next_edge_reason == "dy":
            dy_offset = edge_to_idx(dy[dy_shift][0], next_edge)
            tt_offset = edge_to_idx(tt["nominal"][0], next_edge)
            all_bkgds_offset = edge_to_idx(all_bkgds["nominal"][0], next_edge)
        elif next_edge_reason == "tt":
            dy_offset = edge_to_idx(dy["nominal"][0], next_edge)
            tt_offset = edge_to_idx(tt[tt_shift][0], next_edge)
            all_bkgds_offset = edge_to_idx(all_bkgds["nominal"][0], next_edge)
        elif next_edge_reason == "all_bkgds":
            all_bkgds_offset = edge_to_idx(all_bkgds[all_bkgds_shift][0], next_edge)
            
        if len(bin_edges) == 1:
            # update signal per bin such that for all next bins the signal yield is the same
            signal_per_bin = hh_weights_cumsum[hh_offset]
        
        if hh_offset > len(hh_values):
            # close bin edges with 0
            bin_edges.append(0.)
            stop_reason = "no more hh events left"
            break

        if any(offset >= vals for offset, vals in zip([dy_offset, tt_offset, all_bkgds_offset],
                                                            [len(dy["nominal"][0]),
                                                             len(tt["nominal"][0]),
                                                             len(all_bkgds["nominal"][0])])):
            # close bin edges with 0
            bin_edges.append(0.)
            stop_reason = "no more bkgd events left"
            break
    bin_edges = sorted(list(set([round(float(i), 6) for i in bin_edges])))
    return bin_edges, stop_reason
        


        




        
        
        
        