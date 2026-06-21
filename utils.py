import os
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from wnetdeconv.spectrum import Spectrum_1D, Spectrum

def load_1d_spectrum(path,
                            max_peak_fraction=None, #0.1
                            intensity_threshold=0, #0.01
                            verbose=False,
                            ):
    label = path.split("/")[-1].split(".")[0]
    df = pd.read_csv(path, header=None, names=['1H', 'i'])
    df = df[df['i'] > intensity_threshold]

    if verbose:
        s = df.i.sum()
        print("{}\nnumber of peaks: {}, total signal: {}".format(label, df.shape[0], round(s, 2)))
    if max_peak_fraction: 
        df = df[df['i'] > df.i.max()*max_peak_fraction]
        if verbose: 
            print("Peaks with intensities higher than max_intensity * {}".format(max_peak_fraction))
            print("number of peaks: {}, max_intensity * max_peak_fraction: {}, % of signal left: {}".format(df.shape[0], round(df.i.max()*max_peak_fraction, 5), round(100*df.i.sum()/s, 2)))
    elif intensity_threshold:
        df = df[df['i'] > intensity_threshold]
        if verbose: 
            print("Peaks with intensities > {}".format(intensity_threshold))
            print("number of peaks: {}, intensity_threshold: {}, % of signal left: {}".format(df.shape[0], intensity_threshold, round(100*df.i.sum()/s, 2)))
    if verbose: print()

    positions = df['1H'].to_numpy()
    intensities = df['i'].to_numpy()
    return Spectrum_1D(positions = positions,
                    intensities = intensities,
                    label = label,
                    )

def load_2d_spectrum(path, 
                  dim=2, 
                  scale_nucl={}, #{'15N':10} - 2D, {'C':10} - 4D
                  max_peak_fraction=None, #0.1
                  intensity_threshold=0, #0.01
                  verbose=False,
                  ):
    label = path.split("/")[-1].split(".")[0]
    df = pd.read_csv(path)
    if 'i' not in df.columns:
        if verbose: print("{}\nnumber of peaks: {}, no signal intensities available - setting 'i' to 1".format(label, df.shape[0]))
        verbose=False
    if verbose:
        s = df.i.sum()
        print("{}\nnumber of peaks: {}, total signal: {}".format(label, df.shape[0], round(s, 2)))
    if max_peak_fraction: 
        df = df[df['i'] > df.i.max()*max_peak_fraction]
        if verbose: 
            print("Peaks with intensities higher than max_intensity * {}".format(max_peak_fraction))
            print("number of peaks: {}, max_intensity * max_peak_fraction: {}, % of signal left: {}".format(df.shape[0], round(df.i.max()*max_peak_fraction, 3), round(100*df.i.sum()/s, 2)))
    elif intensity_threshold:
        df = df[df['i'] > intensity_threshold]
        if verbose: 
            print("Peaks with intensities > {}".format(intensity_threshold))
            print("number of peaks: {}, intensity_threshold: {}, % of signal left: {}".format(df.shape[0], intensity_threshold, round(100*df.i.sum()/s, 2)))
    if verbose: print()

    for nucl, scale in scale_nucl.items():
        for c in df.columns:
            if c.startswith(nucl): df[c] = df[c]/scale
    positions = df[df.columns[:dim]].T.to_numpy()

    if 'i' in df.columns: intensities = df['i'].to_numpy()
    else: intensities = np.ones(positions.shape[1])

    return Spectrum(positions = positions,
                        intensities = intensities,
                        label = label,
                        )

def plot_temperatures(data_path, 
                      max_peak_fraction=0.1, 
                      intensity_threshold=0.01,
                      cmap_name="plasma", 
                      figsize=(9,6), 
                      dpi=200,
                      alpha=0.8,
                      s=3,
                      hlim = (6, 11), 
                      nlim = (100, 136),
                      save_path = None,
                     ):

    if isinstance(data_path, list):
        for path in data_path:
            if not os.path.isfile(path): raise Exception("Path in a list is not a file path.")
            if path[-4:] != ".csv": raise Exception("Incorrect file extention,. Must be csv")
            paths = data_path
    elif os.path.isdir(data_path):
        paths = sorted(glob(data_path + "/*.csv"))
    else:
        raise Exception("Incorrect path. Should be 1) directory path with csv files 2) list of paths to csv files")
        
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, len(paths)))

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True) #constrained_layout=True, squeeze=False, 
    for i, path in enumerate(paths):
        label = path.split("/")[-1].split(".")[0].split("_")[-1]
        df = pd.read_csv(path)
        if max_peak_fraction: 
            df = df[df['i'] > df.i.max()*max_peak_fraction]
        else: 
            df = df[df['i'] > intensity_threshold]
        ax.scatter(df['1H'], df['15N'], s=s, alpha=alpha, color=colors[i], label=label)
    
    plt.xlabel('1H (ppsrcm)')
    plt.xlim(hlim[0], hlim[1])
    plt.ylabel('15N (ppm)')
    plt.ylim(nlim[0], nlim[1])
    plt.title('$^1$H $^{15}$N HSQC spectra of GB1 protein')
    plt.grid(True)
    plt.gca().invert_xaxis()  # Often used for NMR spectra
    plt.gca().invert_yaxis()  # Often used for NMR spectra
    plt.legend(loc="upper left")

    if save_path: plt.savefig(save_path)
    return ax