import awkward as ak
import numpy as np
import itertools

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
                n_tt + n_dy >= 3 and
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
    

    # sort hh 
    sort_indices = ak.argsort(hh_values, ascending=False)
    hh_values = hh_values[sort_indices]
    hh_weights = hh_weights[sort_indices]
    # sort dy and tt 
    for key in dy_shifts.keys():
        dy_sort_indices = ak.argsort(dy_shifts[key][0], ascending=False)
        dy_shifts[key] = dy_shifts[key][0][dy_sort_indices], np.cumsum(dy_shifts[key][1][dy_sort_indices])
        tt_sort_indices = ak.argsort(tt_shifts[key][0], ascending=False)
        tt_shifts[key] = tt_shifts[key][0][tt_sort_indices], np.cumsum(tt_shifts[key][1][tt_sort_indices])

    # create a combined array of dy and tt values
    dy_tt_shifts = {key: ak.sort(
                         ak.concatenate([dy_shifts[key][0], tt_shifts[key][0]], axis=0),
                         ascending=False)
                    for key in dy_shifts.keys()}
        

    # the rightmost bin should contain at least min_N events
    bin_edges = [x_max]
    #min_N = int(np.ceil(1/(uncertainty)**2))
    hh_weights_cumsum = np.cumsum(hh_weights)
    del hh_weights
    signal_per_bin = np.round(hh_weights_cumsum[-1] / n_bins, 5)
    offset = 0
    edge_count = 0
    while True:
        # this would be the next bin edge if we would only consider the signal
        pushover_idx = np.where(hh_weights_cumsum[offset:] > (edge_count+1)*signal_per_bin)[0]
        if len(pushover_idx) == 0:
            # no more hh events left
            stop_reason = "no more events left"
            bin_edges.append(x_min)
            break
        bin_idx_by_signal = offset + pushover_idx[0]
        bin_edge_by_signal = hh_values[bin_idx_by_signal]
        # we want at least 1 dy and 1 tt event in the bin
        dy_edges = [dy_shifts[key][0][offset] for key in dy_shifts.keys()] 
        tt_edges = [tt_shifts[key][0][offset] for key in tt_shifts.keys()]
        # we want the sum of dy and tt events to be at least 3 
        dy_tt_edges = [dy_tt_shifts[key][offset+2] for key in dy_tt_shifts.keys()]
        # find the next bin edge that fulfills all requirements
        next_edge = np.min([bin_edge_by_signal, *dy_edges, *tt_edges, *dy_tt_edges])
        if next_edge <= np.min(hh_values):
            # no more hh events left
            bin_edges.append(x_min)
            stop_reason = "no more events left"
            break
        # convert edge into offset
        next_offset = np.where(hh_values <= next_edge)[0][0]
        if any((next_offset >= vals for vals in (len(dy_shifts["nominal"][0]),
                                                 len(tt_shifts["nominal"][0])))):
            # close bin edges with x_min
            bin_edges.append(x_min)
            stop_reason = ("no more dy or tt events left before checking yield constraints. "
                           "(this shouldn't happen often)")
            break

        # now about the yields... we want to make sure that the dy and tt yields are positive
        # but this requirement should only be checked for the nominal values (?)
        # TODO: maybe even for all shifts? 
        bin_yield_dy = dy_shifts["nominal"][1][next_offset] - dy_shifts["nominal"][1][offset]
        bin_yield_tt = tt_shifts["nominal"][1][next_offset] - tt_shifts["nominal"][1][offset]
        if bin_yield_dy <= 0 or bin_yield_tt <= 0: 
            # now we need to go further 
            dy_yield_bin = np.where(dy_shifts["nominal"][1][offset:] > 0)[0][0]
            tt_yield_bin = np.where(tt_shifts["nominal"][1][offset:] > 0)[0][0]
            next_offset = offset + np.max([dy_yield_bin, tt_yield_bin])
            next_edge = np.min([dy_shifts["nominal"][0][next_offset],
                                tt_shifts["nominal"][0][next_offset]])
            
        offset = next_offset 
        edge_count += 1
        # check remaining stats
        if ( hh_weights_cumsum[-1] - hh_weights_cumsum[offset] ) < edge_count*signal_per_bin:
            # close bin edges with x_min
            stop_reason = "remaining signal yield insufficient"
            bin_edges.append(x_min)
            break
        if any(
            (offset >= vals for vals in
            [len(hh_values), len(dy_shifts["nominal"][0]), len(tt_shifts["nominal"][0])]
            )): 
            # close bin edges with x_min
            stop_reason = "no more events left after adding yield constraints." 
            bin_edges.append(x_min)
            break
        if edge_count > n_bins-1:
            # close bin edges with x_min
            stop_reason = "reached maximum number of bins"
            bin_edges.append(x_min)
            break
        bin_edges.append(next_edge)
    bin_edges = sorted(bin_edges)
    return bin_edges, stop_reason