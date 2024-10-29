import awkward as ak
import numpy as np
import itertools
from typing import List, Tuple


def yield_requirement(bkgd_values,bkgd_weights, target_val=1e-4):
    bkgd_weights_cs = np.cumsum(bkgd_weights) 
    mask = bkgd_weights_cs>target_val
    if not any(mask):
        #print(f"yield requirement cannot be reached. returnning the min")
        return 1 # ignore this shift for req.
    else:
        return bkgd_values[mask][0]


def error_requirement(bkgd_values,bkgd_weights, target_val=1.):
    bkgd_w_cs = np.cumsum(bkgd_weights)
    bkgd_w_cs_2 = np.cumsum(bkgd_weights**2)
    neg_mask = bkgd_w_cs>0
    bkgd_values = bkgd_values[neg_mask]
    bkgd_w_cs = bkgd_w_cs[neg_mask]
    bkgd_w_cs_2 = bkgd_w_cs_2[neg_mask]
    rel_error = np.sqrt(bkgd_w_cs_2/bkgd_w_cs)
    mask = rel_error<target_val
    if not any(mask):
        #print(f"error requirement cannot be reached. returning the min")
        return 1 # ignore this shift for req. 
    else:
        return bkgd_values[mask][0]


def sort_vals_and_weights(values, weights, square_weights=False):
    sort_indices = ak.argsort(values, ascending=False)
    values = values[sort_indices]
    weights = weights[sort_indices]
    return values, weights
    

def update_vals_and_weights(vals, weights, edge):
    mask = vals < edge
    return vals[mask], weights[mask]



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


def flats_systs(hh_values: ak.Array,
                hh_weights: ak.Array,
                dy_shifts: dict[str, (ak.Array, ak.Array)], # (values, weights)
                tt_shifts: dict[str, (ak.Array, ak.Array)], # (values, weights)
                st_shifts: dict[str, (ak.Array, ak.Array)] | None = None,
                n_bins: int=10,
                x_min: float=0.,
                x_max: float=1.,):

    continuous_st = False
    if not st_shifts is None:
        continuous_st = True
        st_already_binned = False
    # assert that dy_shifts and tt_shifts have the same keys
    assert dy_shifts.keys() == tt_shifts.keys()
    # binning should be done in the range [x_min, x_max]
    assert x_min < x_max <= 1

    assert len(hh_values) == len(hh_weights)
    assert all(len(dy_shifts[key][0]) == len(dy_shifts[key][1]) for key in dy_shifts.keys())
    assert all(len(tt_shifts[key][0]) == len(tt_shifts[key][1]) for key in tt_shifts.keys())
    if continuous_st:
        assert all(len(st_shifts[key][0]) == len(st_shifts[key][1]) for key in st_shifts.keys())
        assert st_shifts.keys() == dy_shifts.keys()
    bin_edges = [x_max]
    edge_count = 1

    # sort hh 
    sort_indices = ak.argsort(hh_values, ascending=False)
    hh_values = hh_values[sort_indices]
    hh_weights = hh_weights[sort_indices]
    required_signal_yield = np.max([np.sum(hh_weights)/n_bins,1e-4])
    # sort dy and tt & replace weights with cumulative sums
    for key in dy_shifts.keys():
        dy_shifts[key] = sort_vals_and_weights(dy_shifts[key][0], dy_shifts[key][1]) 
        tt_shifts[key] = sort_vals_and_weights(tt_shifts[key][0], tt_shifts[key][1]) 
        if continuous_st:
            st_shifts[key] = sort_vals_and_weights(st_shifts[key][0], st_shifts[key][1])

    # create a combined array of dy and tt values and squared weights cumulatively summed
    dy_tt_shifts = {key: sort_vals_and_weights(ak.concatenate([dy_shifts[key][0], tt_shifts[key][0]], axis=0),
                                            ak.concatenate([dy_shifts[key][1], tt_shifts[key][1]], axis=0),)
                    for key in dy_shifts.keys()}


    def fill_counts(counts, next_edge, last_edge, continuous_st,
                    hh_values, hh_weights, dy_shifts, tt_shifts, st_shifts):
        def get_mask(values, upper_bound, lower_bound):
            return np.logical_and(values>=lower_bound, values<upper_bound)

        hh_mask = get_mask(hh_values, last_edge, next_edge)
        dy_mask = get_mask(dy_shifts["nominal"][0], last_edge, next_edge)
        tt_mask = get_mask(tt_shifts["nominal"][0], last_edge, next_edge) 
        counts["HH"][0].append(len(hh_values[hh_mask]))
        counts["HH"][1].append(np.sum(hh_weights[hh_mask]).astype("float64"))
        counts["DY"][0].append(len(dy_shifts["nominal"][0][dy_mask]))
        counts["DY"][1].append(np.sum(dy_shifts["nominal"][1][dy_mask]).astype("float64"))
        counts["TT"][0].append(len(tt_shifts["nominal"][0][tt_mask]))
        counts["TT"][1].append(np.sum(tt_shifts["nominal"][1][tt_mask]).astype("float64"))
        if continuous_st:
            st_mask = get_mask(st_shifts["nominal"][0], last_edge, next_edge) 
            counts["ST"][0].append(len(st_shifts["nominal"][0][st_mask]))
            counts["ST"][1].append(np.sum(st_shifts["nominal"][1][st_mask]).astype("float64"))

    counts = {"HH": ([], []),
              "DY": ([], []),
              "TT": ([], []),}
    if continuous_st:
        counts["ST"] = ([], [])
    while True:
        # let's just get the first bin
        # 1st requirements: 1 dy & tt bar event 
        dy_edge = min([dy_shifts[key][0][0] for key in dy_shifts])
        tt_edge = min([tt_shifts[key][0][0] for key in tt_shifts])
        # at least 4 of dy tt
        dy_tt_edge = min([dy_tt_shifts[key][0][3] for key in dy_tt_shifts])
        # now add the bkgd yield requirement
        dy_yield_edges = [yield_requirement(*dy_shifts[key]) for key in dy_shifts]
        tt_yield_edges = [yield_requirement(*tt_shifts[key]) for key in tt_shifts]
        mask_dy = [i == 1 for i in dy_yield_edges]
        mask_tt = [i == 1 for i in tt_yield_edges]
        problem_shifts = []
        if any(mask_dy) or any(mask_tt):
            # also print which shifts are causing the problem
            problem_shifts += [key for key, mask in zip(dy_shifts.keys(), mask_dy) if mask]
            problem_shifts += [key for key, mask in zip(tt_shifts.keys(), mask_tt) if mask]
            stop_reason = f"cannot guarantee dy/tt yield for shifts {set(problem_shifts)}"
            fill_counts(counts, 0.1, bin_edges[-1], continuous_st, hh_values, hh_weights, 
                        dy_shifts, tt_shifts, st_shifts)
            bin_edges.append(0.1) # doing this just to make it easy to spot what was the reason
            fill_counts(counts, x_min, bin_edges[-1], continuous_st, hh_values, hh_weights, 
                        dy_shifts, tt_shifts, st_shifts)
            bin_edges.append(x_min)
            break
        # error requirements
        dy_error_edges = [error_requirement(*dy_shifts[key], target_val=0.5)
                          if key == "nominal"
                          else error_requirement(*dy_shifts[key], target_val=0.5)
                          for key in dy_shifts]

        tt_error_edges = [error_requirement(*dy_shifts[key], target_val=0.5)
                          if key == "nominal"
                          else error_requirement(*dy_shifts[key], target_val=0.5)
                          for key in dy_shifts]
        mask_dy = [i == 1 for i in dy_error_edges]
        mask_tt = [i == 1 for i in tt_error_edges]
        if any(mask_dy) or any(mask_tt):
            # also print which shifts are causing the problem
            problem_shifts += [key for key, mask in zip(dy_shifts.keys(), mask_dy) if mask]
            problem_shifts += [key for key, mask in zip(tt_shifts.keys(), mask_tt) if mask]
            stop_reason = f"cannot guarantee dy/tt error req. for shifts {set(problem_shifts)}"
            fill_counts(counts, 0.15, bin_edges[-1], continuous_st, hh_values, hh_weights, 
                        dy_shifts, tt_shifts, st_shifts)
            bin_edges.append(0.15) # same as above
            fill_counts(counts, x_min, bin_edges[-1], continuous_st, hh_values, hh_weights, 
                        dy_shifts, tt_shifts, st_shifts)
            bin_edges.append(x_min)
            break
        next_edge = np.min([dy_edge, tt_edge, dy_tt_edge,
                        np.min([dy_yield_edges]),
                        np.min([tt_yield_edges]),
                        np.min([dy_error_edges]),
                        np.min([tt_error_edges])])
        if continuous_st:
            if st_already_binned:
                # st has already been binned for some shift: just add the yield requirement for st
                st_edges = [yield_requirement(*st_shifts[key]) for key in st_shifts]
                next_edge = np.min([next_edge, np.min(st_edges)])
            else:
                masks_st = [st_shifts[key][0]>next_edge for key in st_shifts]
                # we only change the next_edge if st is binned for some shift 
                if any([any(mask) for mask in masks_st]):
                    # okay so for some shift we have already binned the st
                    st_already_binned = True
                    binned_masks = [mask for mask in masks_st if any(mask)]
                    # let's find the largest idx out of all shifts
                    max_idx = max([np.where(mask)[0][0] for mask in binned_masks])
                    # this way we make sure that for this bin, st is included for all shifts
                    next_edge = min([next_edge, *[st_shifts[key][0][max_idx] for key in st_shifts]])
        # now calculate the signal  yield up to there
        if edge_count == 1:
            first_bin_yield = np.sum(hh_weights[np.logical_and((hh_values>=next_edge), (hh_values<1))])
            if first_bin_yield < required_signal_yield:
                yieldsum = np.cumsum(hh_weights)
                mask = yieldsum>=required_signal_yield
                if not any(mask):
                    stop_reason = "reached end of signal yield"
                    fill_counts(counts, np.min(hh_values), bin_edges[-1], continuous_st, 
                                hh_values, hh_weights, dy_shifts, tt_shifts, 
                                st_shifts)
                    bin_edges.append(np.min(hh_values))
                    fill_counts(counts, x_min, bin_edges[-1], continuous_st, hh_values, 
                                hh_weights, dy_shifts, tt_shifts, st_shifts)
                    bin_edges.append(x_min)
                    break
                else:
                    next_edge = hh_values[mask][0]
                    binned_signal_yield = np.sum(hh_weights[hh_values>=next_edge])
                    assert binned_signal_yield > required_signal_yield
                    required_signal_yield = binned_signal_yield 
            else:
                required_signal_yield = first_bin_yield
        else:
            signal_yield = np.sum(hh_weights[hh_values>=next_edge])
            if signal_yield < required_signal_yield:
                yieldsum = np.cumsum(hh_weights)
                mask = yieldsum>=required_signal_yield
                if not any(mask):
                    stop_reason = "reached end of signal yield"
                    next_edge = np.min([*hh_values, next_edge])
                    fill_counts(counts, next_edge, bin_edges[-1], continuous_st, hh_values, 
                                hh_weights, dy_shifts, tt_shifts, st_shifts)
                    bin_edges.append(next_edge)
                    fill_counts(counts, x_min, bin_edges[-1], continuous_st, hh_values, 
                                hh_weights, dy_shifts, tt_shifts, st_shifts)
                    bin_edges.append(x_min)
                    break
                else:
                    next_edge = hh_values[mask][0]
                    assert np.sum(hh_weights[hh_values>=next_edge]) > required_signal_yield

        if not any(hh_values<next_edge):
            stop_reason = "no signal events left"
            next_edge = np.min([*hh_values, next_edge])
            fill_counts(counts, next_edge, bin_edges[-1], continuous_st, hh_values, 
                        hh_weights, dy_shifts, tt_shifts, st_shifts)
            bin_edges.append(next_edge)
            fill_counts(counts, x_min, bin_edges[-1], continuous_st, hh_values, hh_weights, 
                        dy_shifts, tt_shifts, st_shifts)
            bin_edges.append(x_min)
            break
        fill_counts(counts, next_edge, bin_edges[-1], continuous_st, hh_values, hh_weights,
                    dy_shifts, tt_shifts, st_shifts)
        hh_values, hh_weights = update_vals_and_weights(hh_values, hh_weights, next_edge)
        # maybe this way it's slow but otherwise we'd have to keep track of offsets for all shifts..
        for key in dy_shifts:
            dy_shifts[key] = update_vals_and_weights(*dy_shifts[key],next_edge)
            tt_shifts[key] = update_vals_and_weights(*tt_shifts[key],next_edge)
            dy_tt_shifts[key] = update_vals_and_weights(dy_tt_shifts[key][0],dy_tt_shifts[key][1],next_edge)
            if continuous_st:
                st_shifts[key] = update_vals_and_weights(*st_shifts[key],next_edge)

        bin_edges.append(next_edge)
        edge_count += 1
        if edge_count == n_bins:
            bin_edges.append(x_min)
            stop_reason = "reached maximum number of bins"
            break
    format_bins = lambda x: round(x - 1e-6, 6)
    bin_edges = sorted([format_bins(i) if ((i != x_min) and (i != x_max)) else i for i in bin_edges])
    return bin_edges, stop_reason, problem_shifts, counts


def non_res_like(hh_values: ak.Array,
                hh_weights: ak.Array,
                dy_shifts: dict[str, (ak.Array, ak.Array)], # (values, weights)
                tt_shifts: dict[str, (ak.Array, ak.Array)], # (values, weights)
                others_shifts: dict[str, (ak.Array, ak.Array)], # (values, weights)
                n_bins: int=10,
                x_min: float=0.,
                x_max: float=1.,):

    # assert that dy_shifts and tt_shifts have the same keys
    assert dy_shifts.keys() == tt_shifts.keys() == others_shifts.keys()
    # binning should be done in the range [x_min, x_max]
    assert x_min < x_max <= 1

    assert len(hh_values) == len(hh_weights)
    assert all(len(dy_shifts[key][0]) == len(dy_shifts[key][1]) for key in dy_shifts.keys())
    assert all(len(tt_shifts[key][0]) == len(tt_shifts[key][1]) for key in tt_shifts.keys())
    assert all(len(others_shifts[key][0]) == len(others_shifts[key][1]) for key in others_shifts.keys())
    bin_edges = [x_max]
    edge_count = 1

    # sort hh 
    sort_indices = ak.argsort(hh_values, ascending=False)
    hh_values = hh_values[sort_indices]
    hh_weights = hh_weights[sort_indices]
    required_signal_yield = np.max([np.sum(hh_weights)/n_bins,0.001])
    # all bkgds
    all_bkgds = {key: sort_vals_and_weights(ak.concatenate([dy_shifts[key][0], tt_shifts[key][0], others_shifts[key][0]], axis=0),
                                      ak.concatenate([dy_shifts[key][1], tt_shifts[key][1], others_shifts[key][1]], axis=0))
                    for key in dy_shifts.keys()}
    # sort dy and tt & replace weights with cumulative sums
    for key in dy_shifts.keys():
        dy_shifts[key] = sort_vals_and_weights(dy_shifts[key][0], dy_shifts[key][1]) 
        tt_shifts[key] = sort_vals_and_weights(tt_shifts[key][0], tt_shifts[key][1]) 

    while True:
        # let's just get the first bin
        # 1st requirements: 1 dy & tt bar event 
        dy_edge = min([dy_shifts[key][0][0] for key in dy_shifts])
        tt_edge = min([tt_shifts[key][0][0] for key in tt_shifts])
        # 4 events combined
        # dy_tt_edge = min([dy_tt_shifts[key][0][3] for key in dy_tt_shifts])
        # now add the bkgd yield requirement
        all_bkgd_edges = [yield_requirement(*all_bkgds[key], target_val=0.03) for key in dy_shifts]
        if any([i == 1 for i in all_bkgd_edges]):
            stop_reason = "cannot guarantee all bkgds error req. for all shifts"
            bin_edges.append(np.min(hh_values))
            bin_edges.append(x_min)
            break
        # add relative error req. on nominal bkgd yield
        rel_error_edge = error_requirement(*all_bkgds["nominal"])
        if rel_error_edge == 1:
            stop_reason = "cannot reach rel. error req. on nominal bkgd."
            bin_edges.append(np.min(hh_values))
            bin_edges.append(x_min)
            break
        next_edge = np.min([dy_edge, tt_edge, np.min([all_bkgd_edges]), rel_error_edge])
        # now calculate the signal  yield up to there
        if edge_count == 1:
            first_bin_yield = np.sum(hh_weights[np.logical_and((hh_values>=next_edge), (hh_values<1))])
            if first_bin_yield < required_signal_yield:
                yieldsum = np.cumsum(hh_weights)
                mask = yieldsum>=required_signal_yield
                if not any(mask):
                    stop_reason = "reached end of signal yield"
                    bin_edges.append(np.min(hh_values))
                    bin_edges.append(x_min)
                    break
                else:
                    next_edge = hh_values[mask][0]
            else:
                required_signal_yield = first_bin_yield
        else:
            signal_yield = np.sum(hh_weights[hh_values>=next_edge])
            if signal_yield < required_signal_yield:
                yieldsum = np.cumsum(hh_weights)
                mask = yieldsum>=required_signal_yield
                if not any(mask):
                    stop_reason = "reached end of signal yield"
                    bin_edges.append(np.min(hh_values))
                    bin_edges.append(x_min)
                    break
                else:
                    next_edge = hh_values[mask][0]

        if not any(hh_values<next_edge):
            stop_reason = "no signal events left"
            bin_edges.append(np.min(hh_values))
            bin_edges.append(x_min)
            break
        hh_values, hh_weights = update_vals_and_weights(hh_values, hh_weights, next_edge)
        # maybe this way it's slow but otherwise we'd have to keep track of offsets for all shifts..
        for key in dy_shifts:
            dy_shifts[key] = update_vals_and_weights(*dy_shifts[key],next_edge)
            tt_shifts[key] = update_vals_and_weights(*tt_shifts[key],next_edge)
            all_bkgds[key] = update_vals_and_weights(*all_bkgds[key],next_edge)

        bin_edges.append(next_edge)
        edge_count += 1
        if edge_count == n_bins:
            bin_edges.append(x_min)
            stop_reason = "reached maximum number of bins"
            break
    bin_edges = sorted([round(float(i), 6) for i in bin_edges])
    return bin_edges, stop_reason


# this shit didn't work because the rec array takes waay too much memory with all shifts
# apart from the fact that there's probably still bugs in there
#def flatsguarded_systs(hh_values: ak.Array,
#                       hh_weights: ak.Array,
#                       dy_shifts: dict[str, ak.Array], # (including nominal)
#                       tt_shifts: dict[str, ak.Array],
#                       n_bins: int=10,
#                       x_min: float=0.,
#                       x_max: float=1.,):
#    
#    # create a record array with eight entries:
#    # - value
#    # - process (0: hh, 1: tt, 2: dy)
#    # - hh_count_cs, tt_count_cs, dy_count_cs (cumulative sums of raw event counts)
#    # - hh_weight_cs, tt_weight_cs, dy_weight_cs (cumulative sums of weights)
#
#    tt_values = [tt_shifts[key][0] for key in tt_shifts]
#    dy_values = [dy_shifts[key][0] for key in dy_shifts]
#
#    process_map = {
#        "hh_nominal": 0,
#        **{"tt_"+key: i for i,key in enumerate(tt_shifts.keys())},
#        **{"dy_"+key: i+len(tt_shifts.keys()) for i,key in enumerate(dy_shifts.keys())}
#    }
#
#    all_values_list = [hh_values, *tt_values, *dy_values]
#    rec_list = [
#                # value
#                (all_values := np.concatenate(all_values_list, axis=0)),
#                # process
#                np.concatenate([i * np.ones(len(v), dtype=np.int8) for i, v in enumerate(all_values_list)], axis=0),
#               ]
#    izeros = np.zeros(len(all_values), dtype=np.int32)
#    fzeros = np.zeros(len(all_values), dtype=np.float32)
#    for i in range(len(all_values_list)):
#        rec_list.append(izeros)
#        rec_list.append(fzeros)
#    rec_names = "value,process,hh_count_cs,hh_weight_cs,"
#    rec_names += ",".join([f"tt_{key}_count_cs,tt_{key}_weight_cs"
#                           if key != "nominal" else "tt_count_cs,tt_weight_cs"
#                           for key in tt_shifts])
#    rec_names += ",".join([f"dy_{key}_count_cs,dy_{key}_weight_cs"
#                           if key != "nominal" else "dy_count_cs,dy_weight_cs"
#                           for key in dy_shifts])
#
#    rec = np.core.records.fromarrays(rec_list, names=rec_names)
#    # insert counts and weights into columns for correct processes
#    # (faster than creating arrays above which then get copied anyway when the recarray is created)
#    rec.hh_count_cs[rec.process == process_map["hh_nominal"]] = 1
#    rec.hh_weight_cs[rec.process == process_map["hh_nominal"]] = hh_weights
#    for key in rec_names.split(",")[4:]:
#        process_name = "_".join(key.split("_")[:2])
#        if "counts" in key:
#            rec[key][rec.process == process_map[process_name]] = 1
#        if "weight" in key:
#            if key.startswith("dy"):
#                rec[key][rec.process == process_map[process_name]] = dy_shifts[key.split("_")[1]]
#            elif key.startswith("tt"):
#                rec[key][rec.process == process_map[process_name]] = tt_shifts[key.split("_")[1]]
#            else:
#                raise ValueError(f"Unknown process {key}")
#    # sort by decreasing value to start binning from "the right" later on
#    rec.sort(order="value")
#    rec = np.flip(rec, axis=0)
#    # replace counts and weights with their cumulative sums
#    for key in rec_names.split(",")[2:]:
#        rec[key][:] = np.cumsum(rec[key])
#    # eager cleanup
#    del all_values_list, izeros, fzeros
#    del hh_values, hh_weights
#    del dy_shifts, tt_shifts
#    del rec_names, rec_list
#    # now, between any two possible discriminator values, we can easily extract the hh, tt and dy integrals,
#    # as well as raw event counts without the need for additional, costly accumulation ops (sum, count, etc.),
#    # but rather through simple subtraction of values at the respective indices instead
#
#    #
#    # step 2: binning
#    #
#
#    # determine the approximate hh yield per bin
#    hh_yield_per_bin = rec.hh_weight_cs[-1] / n_bins
#    # keep track of bin edges and the hh yield accumulated so far
#    bin_edges = [x_max]
#    hh_yield_binned = 0.0
#    min_hh_yield = 1.0e-5
#    # during binning, do not remove leading entries, but remember the index that denotes the start of the bin
#    offset = 0
#    # helper to extract a cumulative sum between the start offset (included) and the stop index (not included)
#    get_integral = lambda cs, stop: cs[stop - 1] - (0 if offset == 0 else cs[offset - 1])
#    # bookkeep reasons for stopping binning
#    stop_reason = ""
#    # start binning
#    while len(bin_edges) < n_bins:
#        # stopping condition 1: reached end of events
#        if offset >= len(rec):
#            stop_reason = "no more events left"
#            break
#        # stopping condition 2: remaining hh yield too small, so cause a background bin to be created
#        remaining_hh_yield = rec.hh_weight_cs[-1] - hh_yield_binned
#        if remaining_hh_yield < min_hh_yield:
#            stop_reason = "remaining signal yield insufficient"
#            # remove the last bin edge and stop
#            if len(bin_edges) > 1:
#                bin_edges.pop()
#            break
#        # find the index of the event that would result in a hh yield increase of more than the expected
#        # per-bin yield; this index would mark the start of the next bin given all constraints are met
#        if remaining_hh_yield >= hh_yield_per_bin:
#            threshold = hh_yield_binned + hh_yield_per_bin
#            next_idx = offset + np.where(rec.hh_weight_cs[offset:] > threshold)[0][0]
#        else:
#            # special case: remaining hh yield smaller than the expected per-bin yield, so find the last event
#            next_idx = offset + np.where(rec.process[offset:] == process_map["hh_nominal"])[0][-1] + 1
#        # advance the index until backgrounds constraints are met
#        while next_idx < len(rec):
#            counts_per_shift = {
#                shift: [get_integral(rec[f"dy_{shift}_count_cs"], next_idx), # n_dy
#                        get_integral(rec[f"tt_{shift}_count_cs"], next_idx), # n_tt
#                        get_integral(rec[f"dy_{shift}_weight_cs"], next_idx), # y_dy
#                        get_integral(rec[f"tt_{shift}_weight_cs"], next_idx)] # y_tt
#                for shift in dy_shifts
#            }
#            # evaluate constraints
#            # TODO: potentially relax constraints here, e.g when there are 3 (4?) tt events, drop the constraint
#            #       on dy, and vice-versa
#            constraints_met = {shift: (counts_per_shift[shift][1] >= 1 and
#                                       counts_per_shift[shift][0] >= 1 and
#                                       counts_per_shift[shift][1] + counts_per_shift[shift][0] >= 3 and
#                                       counts_per_shift[shift][2] > 0 and
#                                       counts_per_shift[shift][3] > 0)
#                               for shift in dy_shifts}
#            if all(constraints_met.values()):
#                # TODO: maybe also check if the background conditions are just barely met and advance next_idx
#                # to the middle between the current value and the next one that would change anything about the
#                # background predictions; this might be more stable as the current implementation can highly
#                # depend on the exact value of a single event (the one that tips the constraints over the edge
#                # to fulfillment)
#
#                # bin found, stop
#                break
#            # constraints not met, check which shifts cause the problem
#            failing_shifts = [shift for shift, met in constraints_met.items() if not met]
#            
#            process_names = f"dy_{failing_shifts[0]}_count_cs",f"tt_{failing_shifts[0]}_count_cs"
#
#            next_bkg_indices = np.where(
#                ((rec.process[next_idx:] == process_names[0]) | (rec.process[next_idx:] == process_names[1]))
#            )[0]
#            if len(next_bkg_indices) == 0:
#                # no more background events left, move to the last position and let the stopping condition 3
#                # below handle the rest
#                next_idx = len(rec)
#            else:
#                next_idx += next_bkg_indices[0] + 1
#        else:
#            # stopping condition 3: no more events left, so the last bin (most left one) does not fullfill
#            # constraints; however, this should practically never happen
#            stop_reason = "no more events left while trying to fulfill constraints"
#            if len(bin_edges) > 1:
#                bin_edges.pop()
#            break
#        # next_idx found, update values
#        edge_value = x_min if next_idx == 0 else float(rec.value[next_idx - 1:next_idx + 1].mean())
#        bin_edges.append(max(min(edge_value, x_max), x_min))
#        hh_yield_binned += get_integral(rec.hh_weight_cs, next_idx)
#        offset = next_idx
#
#    # make sure the minimum is included
#    if bin_edges[-1] != x_min:
#        if len(bin_edges) > n_bins:
#            raise RuntimeError(f"number of bins reached and initial bin edge is not x_min (edges: {bin_edges})")
#        bin_edges.append(x_min)
#
#    # reverse edges and optionally re-set n_bins
#    bin_edges = sorted(set(bin_edges))
#    return bin_edges, stop_reason
#




        
        
        
        
