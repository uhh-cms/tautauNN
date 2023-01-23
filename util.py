import glob
import numpy as np
import numpy.lib.recfunctions as rfn 
import tensorflow as tf
import math

def load_data(basepath, samples, features, selections, maxevents = 1000000):
    feature_vecs = []
    weights = []
    for sample in samples.keys():
        nevents = 0
        filenames = glob.glob(f"{basepath}/{sample}/output*.npz")
        for filename in filenames:
            with np.load(filename) as f:
                e = f["events"]
                mask = [True] * len(e)
                for (varnames, func) in selections:
                    variables = [e[v] for v in varnames]
                    mask = mask & func(*variables)
                feature_vecs.append(e[features][mask])
                nevents += len(feature_vecs[-1])
                if nevents > maxevents:
                    break
        print(f"{sample}: {nevents} events")
        weights.append([samples[sample]/nevents] * nevents)    
    feature_vecs = np.concatenate(feature_vecs, axis = 0)
    weights = np.concatenate(weights, axis = 0, dtype="float32")
    weights /= np.mean(weights)
    return feature_vecs, weights

def load_sample(basepath, sample, weight, features, selections, maxevents = 1000000):
    feature_vecs = []
    nevents = 0
    filenames = glob.glob(f"{basepath}/{sample}/output*.npz")
    for filename in filenames:
        with np.load(filename) as f:
            e = f["events"]
            mask = [True] * len(e)
            for (varnames, func) in selections:
                variables = [e[v] for v in varnames]
                mask = mask & func(*variables)
            feature_vecs.append(e[features][mask])
            nevents += len(feature_vecs[-1])
            if nevents > maxevents:
                break
    print(f"{sample}: {nevents} events")
    weights = np.array([weight] * nevents, dtype="float32")
    feature_vecs = np.concatenate(feature_vecs, axis = 0)
    return feature_vecs, weights

def calc_new_columns(data, rules):
    # columns = []
    for name, (input_columns, func) in rules.items():
        input_values = [data[c] for c in input_columns]
        column = func(*input_values)
        column[np.isnan(column)] = 0
        # columns.append(column)
        data = rfn.rec_append_fields(data, name, column, dtypes=["<f4"])
    # data = rfn.rec_append_fields(data, list(rules.keys()), columns, dtypes=["<f4"]*len(columns))
    return data

def split_train_validation_mask(nevents, fraction = 0.75, seed = 0):
    np.random.seed(seed)
    mask = np.random.rand(nevents) <= fraction
    return mask

def calc_4vec_sum(pt1, eta1, phi1, e1, pt2, eta2, phi2, e2):
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    
    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2
    e = e1 + e2
    
    pt = np.sqrt(px**2 + py**2)
    p = np.sqrt(pt**2 + pz**2)
    theta = np.arccos(pz/p)
    eta = -np.log(np.tan(theta/2))
    phi = np.arccos(px/pt)
    phi[py==0] = 0
    phi[py<0] = -np.arccos(px/pt)[py<0]
    
    return pt, eta, phi, e

def calc_energy(pt, eta, phi, m):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    
    energy = np.sqrt(m**2 + px**2 + py**2 + pz**2)
    
    return energy

def calc_mass(pt, eta, phi, e):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    mass = np.sqrt(e**2 - px**2 - py**2 - pz**2)
    mass[np.isnan(mass)] = 0

    return mass

def phi_mpi_to_pi(phi):
    larger_pi = phi > math.pi
    smaller_pi = phi < -math.pi
    while np.any(larger_pi) or np.any(smaller_pi):
        phi[larger_pi] -= 2 * math.pi
        phi[smaller_pi] += 2 * math.pi
        larger_pi = phi > math.pi
        smaller_pi = phi < -math.pi
    return phi

def create_tensorboard_callbacks(log_dir):
    add = flush = lambda *args, **kwargs: None
    if log_dir:
        writer = tf.summary.create_file_writer(log_dir)
        flush = writer.flush
        def add(attr, *args, **kwargs):
            with writer.as_default():
                getattr(tf.summary, attr)(*args, **kwargs)
    return add, flush
