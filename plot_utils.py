"""
Plot styling utilities.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def set_publication_style(font_family="Times New Roman", dpi=300):
    """
    Set Matplotlib/Seaborn style for publication-quality figures.

    Args:
        font_family (str): Preferred font family ("Times New Roman" or "Arial")
        dpi (int): Figure DPI for on-screen rendering and saving
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": font_family,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
