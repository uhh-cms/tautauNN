import awkward as ak
import numpy as np
import itertools
from typing import List, Tuple


def yield_requirement(bkgd_values,bkgd_weights, target_val=1e-5):
    bkgd_weights_cs = np.cumsum(bkgd_weights) 
    mask = bkgd_weights_cs>target_val
    if not any(mask):
        # for this particular shift, the requirement cannot be reached
        return 1 # ignore this shift for req.
    else:
        return bkgd_values[mask][0]


def calc_rel_error(weights):
    return np.sqrt(np.sum(weights**2))/np.sum(weights)


def error_requirement(bkgd_values,bkgd_weights, target_val=1.):
    bkgd_w_cs = np.cumsum(bkgd_weights)
    bkgd_w_cs_2 = np.cumsum(bkgd_weights**2)
    rel_error = np.sqrt(bkgd_w_cs_2)/bkgd_w_cs
    mask = np.logical_and(rel_error<target_val, rel_error>0) 
    if not any(mask):
        return 1 # ignore this shift for req. 
    else:
        return bkgd_values[mask][0]


def sort_vals_and_weights(values, weights):
    sort_indices = ak.argsort(values, ascending=False)
    values = values[sort_indices]
    weights = weights[sort_indices]
    return values, weights


def update_vals_and_weights(vals, weights, edge):
    mask = vals < edge
    return vals[mask], weights[mask]


def fill_counts(counts, next_edge, hh_shifts, dy_shifts, tt_shifts):
    hh_mask = hh_shifts["nominal"][0]>=next_edge
    dy_mask = dy_shifts["nominal"][0]>=next_edge
    tt_mask = tt_shifts["nominal"][0]>=next_edge
    counts["HH"][0].append(len(hh_shifts["nominal"][0][hh_mask]))
    counts["HH"][1].append(np.sum(hh_shifts["nominal"][1][hh_mask]).astype("float64"))
    counts["HH"][2].append(calc_rel_error(hh_shifts["nominal"][1][hh_mask]).astype("float64"))
    counts["DY"][0].append(len(dy_shifts["nominal"][0][dy_mask]))
    counts["DY"][1].append(np.sum(dy_shifts["nominal"][1][dy_mask]).astype("float64"))
    counts["DY"][2].append(calc_rel_error(dy_shifts["nominal"][1][dy_mask]).astype("float64"))
    counts["TT"][0].append(len(tt_shifts["nominal"][0][tt_mask]))
    counts["TT"][1].append(np.sum(tt_shifts["nominal"][1][tt_mask]).astype("float64"))
    counts["TT"][2].append(calc_rel_error(tt_shifts["nominal"][1][tt_mask]).astype("float64"))


def check_yield_requirement(bkgd_values, bkgd_weights, next_edge, target_val=1e-5):
    mask = bkgd_values >= next_edge
    return np.sum(bkgd_weights[mask]) > target_val


def check_error_requirement(bkgd_values, bkgd_weights, next_edge, target_val=1.):
    mask = bkgd_values >= next_edge
    return calc_rel_error(bkgd_weights[mask]) < target_val


def get_conditions(dy_vals_weights, tt_vals_weights, next_edge, yield_target=1e-5, error_target=1.):
    dy_conds = (check_yield_requirement(*dy_vals_weights, next_edge, yield_target)
                and check_error_requirement(*dy_vals_weights, next_edge, error_target))
    tt_conds = (check_yield_requirement(*tt_vals_weights, next_edge, yield_target)
                and check_error_requirement(*tt_vals_weights, next_edge, error_target))
    return dy_conds, tt_conds


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


def flats_systs(hh_shifts: dict[str, Tuple[ak.Array, ak.Array]],
                dy_shifts: dict[str, Tuple[ak.Array, ak.Array]],
                tt_shifts: dict[str, Tuple[ak.Array, ak.Array]],
                all_bkgds:  Tuple[ak.Array, ak.Array],
                yield_target: float=1e-5,
                error_target: float=1.,
                n_bins: int=10,
                x_min: float=0.,
                x_max: float=1.,):
    

    def add_bkgd_driven_bins(bin_edges,
                             all_bkgds_scores,
                             all_bkgds_weights,
                             x_min=0.):
        if bin_edges[-1] == x_min:
            bin_edges = bin_edges[:-1] 
        bin_edges_arr = np.asarray(bin_edges)
        if np.sum(bin_edges_arr<0.8) >= 3:
            bin_edges.append(x_min)
            return bin_edges
        else:
            binmask = np.logical_and(all_bkgds_scores>=bin_edges_arr[-1], all_bkgds_scores<bin_edges_arr[-2])
            last_yield = np.sum(all_bkgds_weights[binmask])
            additional_edges = 3 - np.sum(bin_edges_arr<0.8)
            remaining_scores = all_bkgds_scores[all_bkgds_scores<bin_edges_arr[-1]]
            remaining_weights = all_bkgds_weights[all_bkgds_scores<bin_edges_arr[-1]]
            remaining_w_cs = np.cumsum(remaining_weights)
            #additional_edges = min(additional_edges, int(np.floor(np.log2(remaining_yield // last_yield))))
            while additional_edges > 0:
                if np.sum(all_bkgds_weights[all_bkgds_scores<bin_edges_arr[-1]]) < 2*last_yield:
                    break
                yield_edge = remaining_scores[np.searchsorted(remaining_w_cs, last_yield*2)]
                error_edge = error_requirement(remaining_scores, remaining_weights, target_val=0.05)
                if error_edge == 1:
                    break
                next_edge = min([0.75,yield_edge, error_edge])
                last_yield = np.sum(remaining_weights[remaining_scores>=next_edge])
                remaining_scores = all_bkgds_scores[all_bkgds_scores<next_edge]
                remaining_weights = all_bkgds_weights[all_bkgds_scores<next_edge]
                remaining_w_cs = np.cumsum(remaining_weights)
                if remaining_w_cs[-1] > 2*last_yield:
                    bin_edges_arr = np.append(bin_edges_arr, next_edge)
                    additional_edges -= 1
                else:
                    break
            bin_edges_arr = np.append(bin_edges_arr, x_min)
            return list(bin_edges_arr)
        

    assert dy_shifts.keys() == tt_shifts.keys() == hh_shifts.keys() 
    assert all([len(dy_shifts[key][0]) == len(dy_shifts[key][1]) for key in dy_shifts.keys()])
    assert all([len(tt_shifts[key][0]) == len(tt_shifts[key][1]) for key in tt_shifts.keys()])
    assert all([len(hh_shifts[key][0]) == len(hh_shifts[key][1]) for key in hh_shifts.keys()])
    # sort dy and tt values and weights 
    for key in dy_shifts.keys():
        hh_shifts[key] = sort_vals_and_weights(hh_shifts[key][0], hh_shifts[key][1])
        dy_shifts[key] = sort_vals_and_weights(dy_shifts[key][0], dy_shifts[key][1]) 
        tt_shifts[key] = sort_vals_and_weights(tt_shifts[key][0], tt_shifts[key][1]) 
    all_bkgds = sort_vals_and_weights(*all_bkgds)

    required_signal_yield = np.max([np.sum(hh_shifts["nominal"][1])/n_bins,1e-5])
    # create a combined array of dy and tt values and squared weights cumulatively summed
    dy_tt_shifts = {key: sort_vals_and_weights(ak.concatenate([dy_shifts[key][0], tt_shifts[key][0]], axis=0),
                                            ak.concatenate([dy_shifts[key][1], tt_shifts[key][1]], axis=0),)
                    for key in dy_shifts.keys()}

    counts = {"HH": [[],[],[]], "DY": [[],[],[]], "TT": [[],[],[]]}
    bin_edges = [1]
    while True:
        # apply the conditions on n_mc for dy, tt and dy_tt
        # at least 1 event of dy and tt in each bin
        dy_edge = np.min([dy_shifts[key][0][0] for key in dy_shifts.keys()])
        tt_edge = np.min([tt_shifts[key][0][0] for key in tt_shifts.keys()])
        # at least 4 events of dy and tt combined in each bin
        dy_tt_edge = np.min([dy_tt_shifts[key][0][3] for key in dy_tt_shifts.keys()])
        next_edge = np.min([dy_edge, tt_edge, dy_tt_edge])
        # now check if the required signal yield is reached
        hh_yields = [np.sum(hh_shifts[key][1][hh_shifts[key][0]>=next_edge]) for key in hh_shifts.keys()]
        if any([y < required_signal_yield for y in hh_yields]): 
            # advance the next edge such that we have the required signal yield
            hh_yield_cs = [np.cumsum(hh_shifts[key][1]) for key in hh_shifts.keys()]
            # make sure that we're checking beyond the next edge (due to negative weights)
            hh_vals = [hh_shifts[key][0] for key in hh_shifts.keys()]
            hh_yield_cs = [cs[vals<next_edge] for cs, vals in zip(hh_yield_cs, hh_vals)] 
            hh_vals = [vals[vals<next_edge] for vals in hh_vals]
            passing_yields = [y_cs>=required_signal_yield for y_cs in hh_yield_cs]
            if not all([any(y) for y in passing_yields]):
                stop_reason = "no more signal events left"
                fill_counts(counts, x_min, hh_shifts, dy_shifts, tt_shifts)
                bin_edges.append(x_min)
                break
            else:
                # I think we can just take the min in this case because the signal samples don't have negative weights
                next_edge = np.min([vals[y][0] for y, vals in zip(passing_yields, hh_vals)])
        # now apply the yield requirements
        yield_conds_not_met = True
        while yield_conds_not_met:
            shifts_conds = {key: get_conditions(dy_shifts[key],tt_shifts[key],
                                                next_edge, yield_target, error_target)
                            for key in dy_shifts.keys()}
            if all([all(conds) for conds in shifts_conds.values()]):
                # current edge is fine
                yield_conds_not_met = False
            else:
                failing_shifts = [key for key, conds in shifts_conds.items() if not all(conds)]
                shift = failing_shifts[0]
                # advance the edge
                dy_conds, tt_conds = shifts_conds[shift]
                if ((not dy_conds) and (not tt_conds)):
                    mask = dy_tt_shifts[shift][0]<next_edge
                    if not any(mask):
                        stop_reason = "no more dy_tt events left (yield/error requirements)"
                        bin_edges.append(x_min)
                        yield_conds_not_met = False
                    else:
                        next_edge = dy_tt_shifts[shift][0][mask][0]
                elif not dy_conds:
                    mask = dy_shifts[shift][0]<next_edge
                    if not any(mask):
                        stop_reason = "no more dy events left (yield/error requirements)"
                        bin_edges.append(x_min)
                        yield_conds_not_met = False
                    else:
                        next_edge = dy_shifts[shift][0][mask][0]
                elif not tt_conds:
                    mask = tt_shifts[shift][0]<next_edge
                    if not any(mask):
                        stop_reason = "no more tt events left (yield/error requirements)"
                        bin_edges.append(x_min)
                        yield_conds_not_met = False
                    else:
                        next_edge = tt_shifts[shift][0][mask][0]
        if len(bin_edges) == 1:
            # set the required signal yield to the yield in the first bin
            required_signal_yield = np.sum(hh_shifts["nominal"][1][hh_shifts["nominal"][0]>=next_edge]) 
        if x_min in bin_edges:
            # stopping reason found due to yield / error requirements
            break
        fill_counts(counts, next_edge, hh_shifts, dy_shifts, tt_shifts)
        bin_edges.append(next_edge)
        # stopping conditions
        if len(bin_edges) == n_bins:
            stop_reason = "n_bins reached"
            fill_counts(counts, x_min, hh_shifts, dy_shifts, tt_shifts)
            bin_edges.append(x_min)
            break
        hh_vals = [hh_shifts[key][0] for key in hh_shifts.keys()]
        hh_vals_masks = [v<next_edge for v in hh_vals]
        if not all([any(v) for v in hh_vals_masks]):
            stop_reason = "no more signal events left"
            fill_counts(counts, x_min, hh_shifts, dy_shifts, tt_shifts)
            bin_edges.append(x_min)
            break
        # update the values and weights
        for key in dy_shifts.keys():
            hh_shifts[key] = update_vals_and_weights(hh_shifts[key][0], hh_shifts[key][1], next_edge)
            dy_shifts[key] = update_vals_and_weights(dy_shifts[key][0], dy_shifts[key][1], next_edge)
            tt_shifts[key] = update_vals_and_weights(tt_shifts[key][0], tt_shifts[key][1], next_edge)
            dy_tt_shifts[key] = update_vals_and_weights(dy_tt_shifts[key][0], dy_tt_shifts[key][1], next_edge)

    bin_edges = add_bkgd_driven_bins(bin_edges, *all_bkgds)

    format_bins = lambda x: float(x) #round(x - 1e-6, 6)
    bin_edges = sorted([format_bins(i) if ((i != x_min) and (i != x_max)) else i for i in bin_edges])
    return bin_edges, stop_reason, counts 


def flats(hh: Tuple[ak.Array, ak.Array],
          dy: Tuple[ak.Array, ak.Array],
          tt: Tuple[ak.Array, ak.Array],
          all_bkgds: Tuple[ak.Array, ak.Array],
          n_bins: int=10,
          x_min: float=0.,
         x_max: float=1.):

    """ 
        just a flat-s without taking into account the systematics 
        (for debugging purposes)
    """

    def add_bkgd_driven_bins(bin_edges,
                             all_bkgds_scores,
                             all_bkgds_weights,
                             x_min=0.):
        if len(bin_edges) < 2:
            if x_min not in bin_edges:
                bin_edges.append(x_min)
            return bin_edges
        if bin_edges[-1] == x_min:
            bin_edges = bin_edges[:-1] 
        bin_edges_arr = np.asarray(bin_edges)
        if np.sum(bin_edges_arr<0.5) >= 3:
            bin_edges.append(x_min)
            return bin_edges
        else:
            binmask = np.logical_and(all_bkgds_scores>=bin_edges_arr[-1], all_bkgds_scores<bin_edges_arr[-2])
            last_yield = np.sum(all_bkgds_weights[binmask])
            additional_edges = 3 - np.sum(bin_edges_arr<0.5)
            remaining_scores = all_bkgds_scores[all_bkgds_scores<bin_edges_arr[-1]]
            remaining_weights = all_bkgds_weights[all_bkgds_scores<bin_edges_arr[-1]]
            remaining_w_cs = np.cumsum(remaining_weights)
            #additional_edges = min(additional_edges, int(np.floor(np.log2(remaining_yield // last_yield))))
            while additional_edges > 0:
                if np.sum(all_bkgds_weights[all_bkgds_scores<bin_edges_arr[-1]]) < 2*last_yield:
                    break
                yield_edge = remaining_scores[np.searchsorted(remaining_w_cs, last_yield*2)]
                error_edge = error_requirement(remaining_scores, remaining_weights, target_val=0.05)
                if error_edge == 1:
                    break
                next_edge = min([0.5,yield_edge, error_edge])
                last_yield = np.sum(remaining_weights[remaining_scores>=next_edge])
                remaining_scores = all_bkgds_scores[all_bkgds_scores<next_edge]
                remaining_weights = all_bkgds_weights[all_bkgds_scores<next_edge]
                remaining_w_cs = np.cumsum(remaining_weights)
                if remaining_w_cs[-1] > 2*last_yield:
                    bin_edges_arr = np.append(bin_edges_arr, next_edge)
                    additional_edges -= 1
                else:
                    break
            bin_edges_arr = np.append(bin_edges_arr, x_min)
            return list(bin_edges_arr)

    hh = sort_vals_and_weights(*hh)
    dy = sort_vals_and_weights(*dy)
    tt = sort_vals_and_weights(*tt)
    dy_tt = sort_vals_and_weights(ak.concatenate([dy[0], tt[0]]),
                                  ak.concatenate([dy[1], tt[1]]))
    all_bkgds = sort_vals_and_weights(*all_bkgds)
    
    bin_edges = [1.]
    stop_reason = ""
    signal_cs = np.cumsum(hh[1])
    signal_yield_target = signal_cs[-1]/n_bins
    while True:
        error_edges = [error_requirement(dy[0],dy[1]),
                       error_requirement(tt[0],tt[1]),
                       error_requirement(dy_tt[0],dy_tt[1],target_val=0.5)]
        if any([i == 1 for i in error_edges]):
            # not enough events to fulfill error requirements
            stop_reason = "not enough events to fulfill error requirements"
            break
        yield_edges = [yield_requirement(dy[0],dy[1]),
                       yield_requirement(tt[0],tt[1]),
                       yield_requirement(dy_tt[0],dy_tt[1])]
        if any([i == 1 for i in yield_edges]):
            # not enough events to fulfill yield requirements
            stop_reason = "not enough events to fulfill yield requirements"
            break
        signal_edge = yield_requirement(hh[0],hh[1],target_val=signal_yield_target)
        if signal_edge == 1:
            # not enough events to fulfill signal yield requirements
            stop_reason = "not enough events to fulfill signal yield requirements"
            break
        next_edge = np.min([*error_edges,*yield_edges,signal_edge])
        if len(bin_edges) == 1:
            signal_yield_target = np.sum(hh[1][hh[0]>=next_edge])
        bin_edges.append(next_edge)
        if len(bin_edges) == n_bins:
            stop_reason = "n_bins reached"
            break
        # update the values and weights
        hh = update_vals_and_weights(*hh, next_edge)
        dy = update_vals_and_weights(*dy, next_edge)
        tt = update_vals_and_weights(*tt, next_edge)
        dy_tt = update_vals_and_weights(*dy_tt, next_edge)

    bin_edges = add_bkgd_driven_bins(bin_edges, *all_bkgds)

    format_bins = lambda x: round(x - 1e-6, 6)
    bin_edges = sorted([format_bins(i) if ((i != x_min) and (i != x_max)) else i for i in bin_edges])
    return bin_edges, stop_reason


