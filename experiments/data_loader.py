"""
data_loader.py
==============
Load experimental and theoretical spectra for wnetdeconv experiments, sourced
from the magnetstein and masserstein example datasets.

Each loader returns a DatasetResult with:
  .experimental                      wnetdeconv.Spectrum (1D)
  .theoretical                       list[wnetdeconv.Spectrum]
  .suggested_max_transport_distance  float  (same units as spectrum positions)
  .description                       str

The optional ``approx_runtime`` parameter (seconds) trims spectra so that a
single wnetdeconv gradient-descent step (set_point + gradient) takes roughly
that long on a modern laptop.  Trimming keeps the highest-intensity peaks.
Scaling is linear in peak count (1D chain factory); reference times were
measured and are embedded in each loader.
"""

from __future__ import annotations

import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from wnetdeconv.spectrum import Spectrum
from wnetdeconv import Spectrum_1D

# Sibling package directories:  wnet_stuff/{magnetstein,masserstein}
_HERE = Path(__file__).resolve().parent          # wnetdeconv/experiments/
_WNET_STUFF = _HERE.parent.parent                # wnet_stuff/
_MAGNETSTEIN = _WNET_STUFF / "magnetstein"
_PBTTT_DATA = (
    _WNET_STUFF / "masserstein" / "Tutorials" / "PBTTT_polymer_analysis" / "Data"
)


@dataclass
class DatasetResult:
    experimental: Spectrum
    theoretical: list   # list[Spectrum]
    suggested_max_transport_distance: float
    description: str


# ──────────────────────────── internal helpers ────────────────────────────

def _load_csv_spectrum(path: Path, n_trim: Optional[int] = None) -> Spectrum:
    arr = np.loadtxt(path, delimiter=",")
    pos, ints = arr[:, 0], arr[:, 1]
    if n_trim is not None and n_trim < len(ints):
        idx = np.argpartition(ints, -n_trim)[-n_trim:]
        pos, ints = pos[np.sort(idx)], ints[np.sort(idx)]
    return Spectrum_1D(pos, ints, label=path.stem)


def _masser_to_wnet(s, n_trim: Optional[int] = None) -> Spectrum:
    """Convert a masserstein Spectrum to a wnetdeconv Spectrum_1D."""
    confs = s.confs
    if n_trim is not None and n_trim < len(confs):
        confs = sorted(confs, key=lambda x: -x[1])[:n_trim]
        confs.sort(key=lambda x: x[0])
    mzs = np.fromiter((m for m, _ in confs), float, len(confs))
    ints = np.fromiter((i for _, i in confs), float, len(confs))
    return Spectrum_1D(mzs, ints, label=getattr(s, "label", None))


def _nmr_n_trim(n_full: int, approx_runtime: Optional[float], t_full: float) -> Optional[int]:
    """Scale n linearly so that runtime ≈ approx_runtime. Floor at 50 peaks."""
    if approx_runtime is None:
        return None
    return max(50, min(n_full, int(n_full * approx_runtime / t_full)))


# ──────────────────────────── NMR loaders (magnetstein) ────────────────────────────

def load_pinene_benzyl(approx_runtime: Optional[float] = None) -> DatasetResult:
    """
    1H NMR deconvolution: Pinene + Benzyl benzoate mixture.

    Source: magnetstein/examples/estimation.ipynb.
    Full profile: 70 340 points per spectrum.  Reference step time: ~0.10 s.

    Parameters
    ----------
    approx_runtime : float, optional
        Target seconds per gradient step.  Trims all spectra to the top-n
        peaks by intensity, where n is chosen so that total peak count scales
        proportionally.
    """
    base = _MAGNETSTEIN / "examples"
    n_trim = _nmr_n_trim(70_340, approx_runtime, 0.10)
    return DatasetResult(
        experimental=_load_csv_spectrum(base / "preprocessed_mix.csv", n_trim),
        theoretical=[
            _load_csv_spectrum(base / "preprocessed_comp0.csv", n_trim),
            _load_csv_spectrum(base / "preprocessed_comp1.csv", n_trim),
        ],
        suggested_max_transport_distance=0.25,
        description=(
            "1H NMR mixture of Pinene (16 protons) and Benzyl benzoate (12 protons). "
            "Profile-mode spectrum, 70 340 points per spectrum. "
            "Suggested max_distance / trash_cost: 0.25 ppm (L1 metric); "
            "asymmetric: MTD_mix=0.25, MTD_components=0.22. "
            "Source: magnetstein/examples/estimation.ipynb."
        ),
    )


def load_overlapping_intensity(approx_runtime: Optional[float] = None) -> DatasetResult:
    """
    1H NMR deconvolution: Benzyl benzoate + Anisaldehyde (overlapping, intensity mismatch).

    Source: magnetstein/visualization_package/data_examples/overlapping_and_intensity_difference.
    Full profile: 131 072 points per spectrum.  Reference step time: ~0.22 s.
    """
    base = (
        _MAGNETSTEIN / "visualization_package" / "data_examples"
        / "overlapping_and_intensity_difference"
    )
    n_trim = _nmr_n_trim(131_072, approx_runtime, 0.22)
    return DatasetResult(
        experimental=_load_csv_spectrum(base / "preprocessed_mix.csv", n_trim),
        theoretical=[
            _load_csv_spectrum(base / "preprocessed_comp0.csv", n_trim),
            _load_csv_spectrum(base / "preprocessed_comp1.csv", n_trim),
        ],
        suggested_max_transport_distance=0.25,
        description=(
            "1H NMR mixture of Benzyl benzoate (12 protons) and Anisaldehyde (8 protons). "
            "Challenging: strongly overlapping peaks with a large intensity difference. "
            "Profile-mode spectrum, 131 072 points per spectrum. "
            "Suggested max_distance / trash_cost: 0.25 ppm (L1 metric); "
            "asymmetric: MTD_mix=0.25, MTD_components=0.22. "
            "Source: magnetstein visualization_package/overlapping_and_intensity_difference."
        ),
    )


def load_perfumes(approx_runtime: Optional[float] = None) -> DatasetResult:
    """
    1H NMR deconvolution: 4-component perfume mixture (some components absent).

    Source: magnetstein/visualization_package/data_examples/perfumes_and_absent_components.
    Components: Isopropyl myristate (34H), Benzyl benzoate (12H),
                Alpha pinene (16H), Limonene (16H).
    Full profile: 12 449 points per spectrum.  Reference step time: ~0.029 s.
    """
    base = (
        _MAGNETSTEIN / "visualization_package" / "data_examples"
        / "perfumes_and_absent_components"
    )
    n_trim = _nmr_n_trim(12_449, approx_runtime, 0.029)
    return DatasetResult(
        experimental=_load_csv_spectrum(base / "preprocessed_mix.csv", n_trim),
        theoretical=[
            _load_csv_spectrum(base / f"preprocessed_comp{i}.csv", n_trim)
            for i in range(4)
        ],
        suggested_max_transport_distance=0.22,
        description=(
            "1H NMR mixture of Isopropyl myristate (34H), Benzyl benzoate (12H), "
            "Alpha pinene (16H), Limonene (16H). "
            "Some library components are absent from the actual mixture. "
            "Profile-mode spectrum, 12 449 points per spectrum. "
            "Suggested max_distance / trash_cost: 0.22 ppm (L1 metric). "
            "Source: magnetstein visualization_package/perfumes_and_absent_components."
        ),
    )


def load_shim(approx_runtime: Optional[float] = None) -> DatasetResult:
    """
    1H NMR deconvolution: 5-component metabolite shim mixture.

    Source: magnetstein/visualization_package/data_examples/shim.
    Components: Lactate (4H), Alanine (4H), Creatine (5H),
                Creatinine (5H), Choline chloride (13H).
    Full profile: 27 129 points per spectrum.  Reference step time: ~0.077 s.
    """
    base = _MAGNETSTEIN / "visualization_package" / "data_examples" / "shim"
    n_trim = _nmr_n_trim(27_129, approx_runtime, 0.077)
    return DatasetResult(
        experimental=_load_csv_spectrum(base / "preprocessed_mix.csv", n_trim),
        theoretical=[
            _load_csv_spectrum(base / f"preprocessed_comp{i}.csv", n_trim)
            for i in range(5)
        ],
        suggested_max_transport_distance=0.22,
        description=(
            "1H NMR metabolite shim: Lactate (4H), Alanine (4H), Creatine (5H), "
            "Creatinine (5H), Choline chloride (13H). "
            "Profile-mode spectrum, 27 129 points per spectrum. "
            "Suggested max_distance / trash_cost: 0.22 ppm (L1 metric). "
            "Source: magnetstein visualization_package/shim."
        ),
    )


# ──────────────────────────── MS loader (masserstein) ────────────────────────────

_HAEMOGLOBIN_A = (
    "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHG"
    "KKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTP"
    "AVHASLDKFLASVSTVLTSKYR"
)
_HAEMOGLOBIN_B = (
    "VHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPK"
    "VKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFG"
    "KEFTPPVQAAYQKVVAGVANALAHKYH"
)
_MYOGLOBIN = (
    "GLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKFKHLKSEDEMKASE"
    "DLKKHGATVLTALGGILKKKGHHEAEIKPLAQSHATKHKIPVKYLEFISECIIQVLQSKH"
    "PGDFGADAQGAMNKALELFRKDMASNYKELGFQG"
)


def load_hemoglobin(approx_runtime: Optional[float] = None, seed: int = 42) -> DatasetResult:
    """
    ESI-MS deconvolution: haemoglobin A/B + myoglobin charge-state mixture.

    Reproduces the masserstein Package-presentation notebook.  Ten theoretical
    spectra (hA 19-21+, hB 20-22+, myo 21-24+) are mixed with known
    proportions, chemical noise is added, and the result is centroided.

    Reference step time at full scale (~2 800 empirical, ~1 200 peaks per
    theoretical spectrum, 10 spectra): ~0.004 s.  Runtime scales linearly in
    total peak count (empirical + sum of theoretical).

    Parameters
    ----------
    approx_runtime : float, optional
        Target seconds per gradient step.  Empirical and theoretical spectra
        are trimmed to their highest-intensity peaks proportionally.
    seed : int
        Random seed for chemical and Gaussian noise (default 42).
    """
    try:
        from masserstein import Spectrum as MS, peptides
    except ImportError as exc:
        raise ImportError(
            f"load_hemoglobin requires masserstein.  "
            f"Install it from {_WNET_STUFF / 'masserstein'}."
        ) from exc

    np.random.seed(seed)
    random.seed(seed)

    hA_f = peptides.get_protein_formula(_HAEMOGLOBIN_A)
    hB_f = peptides.get_protein_formula(_HAEMOGLOBIN_B)
    m_f  = peptides.get_protein_formula(_MYOGLOBIN)

    _specs_meta = [
        (hA_f, 19, "hA 19+"), (hA_f, 20, "hA 20+"), (hA_f, 21, "hA 21+"),
        (hB_f, 20, "hB 20+"), (hB_f, 21, "hB 21+"), (hB_f, 22, "hB 22+"),
        (m_f,  21, "myo 21+"), (m_f,  22, "myo 22+"), (m_f,  23, "myo 23+"),
        (m_f,  24, "myo 24+"),
    ]
    masser_spectra = [MS(f, charge=c, adduct="H", label=lbl) for f, c, lbl in _specs_meta]
    for s in masser_spectra:
        s.normalize()

    proportions_raw = [1, 2, 1.2, 0.5, 0.9, 0.6, 0.2, 0.3, 0.4, 0.0]
    total = sum(proportions_raw)
    proportions = [p / total for p in proportions_raw]

    convolved = MS(label="experimental")
    for s, p in zip(masser_spectra, proportions):
        convolved += s * p
    convolved.add_chemical_noise(100, 0.1)
    convolved.gaussian_smoothing(0.01, 0.001)
    convolved.add_gaussian_noise(0.01)

    peaks, _ = convolved.centroid(peak_height_fraction=0.5, max_width=0.03)
    centroided = MS(confs=peaks)
    centroided.normalize()

    # Reference: n_emp≈2794, avg_theo≈1200, 10 spectra → 0.004 s
    _n_ref_emp  = 2794
    _avg_theo   = 1200
    _t_full     = 0.004
    n_emp_full  = len(centroided.confs)

    if approx_runtime is not None:
        n_trim_emp  = max(20, min(n_emp_full, int(_n_ref_emp * approx_runtime / _t_full)))
        n_trim_theo = max(10, int(_avg_theo * n_trim_emp / _n_ref_emp))
    else:
        n_trim_emp  = None
        n_trim_theo = None

    return DatasetResult(
        experimental=_masser_to_wnet(centroided, n_trim_emp),
        theoretical=[_masser_to_wnet(s, n_trim_theo) for s in masser_spectra],
        suggested_max_transport_distance=0.025,
        description=(
            "ESI-MS deconvolution: haemoglobin A (19-21+), haemoglobin B (20-22+), "
            "myoglobin (21-24+). Simulated centroided spectrum with chemical and Gaussian noise. "
            "10 theoretical isotope-envelope spectra (~1 000–1 400 peaks each). "
            "Suggested max_distance / trash_cost: 0.025 Da (L∞ metric). "
            "Source: masserstein/Tutorials/Package presentation.ipynb."
        ),
    )


# ──────────────────────────── PBTTT polymer loaders (masserstein + pyteomics) ──────

def _import_polymers():
    """
    Import masserstein.polymers, injecting a stub pyteomics when the real
    package is not installed (the stub satisfies the top-level import but
    load_mzxml will still fail without the real package).
    """
    needs_stub = False
    if "pyteomics" not in sys.modules:
        try:
            import pyteomics  # noqa: F401
        except ImportError:
            needs_stub = True

    if needs_stub:
        stub = types.ModuleType("pyteomics")
        stub.mzxml = types.ModuleType("pyteomics.mzxml")
        sys.modules["pyteomics"] = stub
        sys.modules["pyteomics.mzxml"] = stub.mzxml
        sys.modules.pop("masserstein.polymers", None)

    try:
        from masserstein import polymers
        from masserstein.polymers import MCounter
    finally:
        if needs_stub:
            sys.modules.pop("pyteomics", None)
            sys.modules.pop("pyteomics.mzxml", None)
            sys.modules.pop("masserstein.polymers", None)

    return polymers, MCounter


def _pbttt_theoretical(n_trim: Optional[int]):
    """
    Generate the full PBTTT theoretical library (BT/TT co-polymers, 3000–4500 Da).
    Does not require pyteomics; masserstein.polymers is imported with a stub if needed.
    """
    polymers, MCounter = _import_polymers()

    BT = ("BT", MCounter(C=36, H=60, S=2))
    TT = ("TT", MCounter(C=6,  H=2,  S=2))
    end_groups = dict(
        Stannyl=MCounter(C=3, H=9, Sn=1),
        Br=MCounter(Br=1),
        H=MCounter(H=1),
        Methyl=MCounter(C=1, H=3),
        Phenyl=MCounter(C=6, H=5),
    )
    specs = polymers.get_possible_compounds(
        heavier_monomer=BT,
        lighter_monomer=TT,
        end_groups=end_groups,
        min_mz=3000,
        max_mz=4500,
        max_count_diff=5,
        verbose=False,
    )
    for s in specs:
        s.normalize()

    if n_trim is not None:
        specs = specs[:n_trim]

    return [_masser_to_wnet(s) for s in specs]


def _pbttt_n_trim_theo(approx_runtime: Optional[float]) -> Optional[int]:
    """
    Number of theoretical PBTTT spectra to keep.
    Reference: ~400 spectra, ~0.5 s/step (gradient dominated by many subgraphs).
    """
    if approx_runtime is None:
        return None
    return max(20, min(400, int(400 * approx_runtime / 0.5)))


def _load_pbttt_sample(label: str, approx_runtime: Optional[float]) -> DatasetResult:
    try:
        import pyteomics  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PBTTT loaders require pyteomics for mzXML parsing. "
            "Install it with: pip install pyteomics"
        ) from exc

    polymers, _ = _import_polymers()

    filename = "P3_7p.mzXML" if label == "P3 7p" else f"{label}.mzXML"
    s = polymers.load_mzxml(str(_PBTTT_DATA / filename), huge_tree=True)
    s = polymers.restrict(s, 2999, 4501)

    if label == "P3":
        s = polymers.correct_baseline(s, 3000)
    elif label == "P3 7p":
        s = polymers.correct_baseline(s, 1500)
    else:
        s = polymers.correct_baseline(s, 1000)

    s.gaussian_smoothing(sd=0.1)
    s = polymers.centroided(s, max_width=0.75, peak_height_fraction=0.5)
    s = polymers.remove_low_signal(s, signal_proportion=0.005)
    s.label = label

    n_trim_theo = _pbttt_n_trim_theo(approx_runtime)
    theoretical = _pbttt_theoretical(n_trim_theo)

    return DatasetResult(
        experimental=_masser_to_wnet(s),
        theoretical=theoretical,
        suggested_max_transport_distance=0.6,
        description=(
            f"MALDI-ToF MS PBTTT polymer analysis, sample {label!r}. "
            "Centroided spectrum in 3 000–4 500 Da range. "
            "Theoretical library: BT/TT co-polymer isotope envelopes with five "
            "end-group types (H, Methyl, Br, Phenyl, Stannyl), |nBT−nTT| ≤ 5. "
            "Suggested max_distance / trash_cost: 0.6 Da (L∞ metric); "
            "asymmetric: MTD_mix=0.6, MTD_components=0.7. "
            "Source: masserstein/Tutorials/PBTTT_polymer_analysis/PBTTT_analysis.ipynb."
        ),
    )


def load_pbttt_p1(approx_runtime: Optional[float] = None) -> DatasetResult:
    """PBTTT polymer analysis, sample P1. Requires pyteomics and masserstein."""
    return _load_pbttt_sample("P1", approx_runtime)


def load_pbttt_p2(approx_runtime: Optional[float] = None) -> DatasetResult:
    """PBTTT polymer analysis, sample P2. Requires pyteomics and masserstein."""
    return _load_pbttt_sample("P2", approx_runtime)


def load_pbttt_p3(approx_runtime: Optional[float] = None) -> DatasetResult:
    """PBTTT polymer analysis, sample P3. Requires pyteomics and masserstein."""
    return _load_pbttt_sample("P3", approx_runtime)


def load_pbttt_p3_7p(approx_runtime: Optional[float] = None) -> DatasetResult:
    """PBTTT polymer analysis, sample P3 7p. Requires pyteomics and masserstein."""
    return _load_pbttt_sample("P3 7p", approx_runtime)
