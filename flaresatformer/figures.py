"""Paper figures. Each model keeps a fixed color from the validated categorical palette plus its own marker."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter  # noqa: E402

MODEL_ORDER = ["unet", "segformer", "segformer_scratch", "swin_unet", "transunet", "prithvi"]
MODEL_LABELS = {"unet": "UNet", "segformer": "SegFormer-B2", "segformer_scratch": "SegFormer-B2 (scratch)",
                "swin_unet": "Swin-UNet", "transunet": "TransUNet", "prithvi": "Prithvi-EO-2.0"}
MODEL_COLORS = {"unet": "#2a78d6", "segformer": "#eb6834", "segformer_scratch": "#1baf7a",
                "swin_unet": "#eda100", "transunet": "#e87ba4", "prithvi": "#008300"}
MODEL_MARKERS = {"unet": "o", "segformer": "s", "segformer_scratch": "D", "swin_unet": "^", "transunet": "v", "prithvi": "P"}
INK, INK_MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def _axes(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def _marker_style(model):
    return {"color": MODEL_COLORS[model], "marker": MODEL_MARKERS[model], "label": MODEL_LABELS[model]}


def learning_curve(per_run, path):
    df = per_run[(per_run.fold == "all") & (per_run.bands == "b267") & per_run.variant.isin(MODEL_ORDER)]
    fig, ax = _axes((5.5, 3.6))
    for model in MODEL_ORDER:
        g = df[df.variant == model].groupby("frac").pixel_f1.mean()
        if len(g) < 2:
            continue
        x = g.index.to_numpy() * 100
        ax.plot(x, g.to_numpy(), linewidth=2, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, **_marker_style(model))
        ax.annotate(MODEL_LABELS[model], (x[-1], g.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100], ["10%", "25%", "50%", "100%"])
    ax.minorticks_off()
    ax.set_xlabel("Training data used", color=INK)
    ax.set_ylabel("Pixel F1 (test, all folds)", color=INK)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    _save(fig, path)


def f1_per_continent(continent_df, path):
    df = continent_df[(continent_df.bands == "b267") & continent_df.variant.isin(MODEL_ORDER)]
    continents = sorted(df.continent.unique())
    fig, ax = _axes((6.0, 0.8 + 0.6 * len(continents)))
    offsets = np.linspace(-0.3, 0.3, len(MODEL_ORDER))
    for offset, model in zip(offsets, MODEL_ORDER):
        g = df[df.variant == model].set_index("continent").reindex(continents)
        if g["mean"].isna().all():
            continue
        ax.scatter(g["mean"], np.arange(len(continents)) + offset, s=45, edgecolors=SURFACE, linewidths=1.5,
                   zorder=3, **_marker_style(model))
    ax.set_yticks(range(len(continents)), continents)
    ax.set_xlabel("Pixel F1 (test)", color=INK)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=8)
    _save(fig, path)


def cost_vs_f1(cost_df, summary_df, path):
    s = summary_df[(summary_df.fold == "all") & (summary_df.bands == "b267") & (summary_df.frac == 1.0)].set_index("variant")
    c = cost_df.set_index("variant")
    fig, ax = _axes((5.5, 3.6))
    for model in MODEL_ORDER:
        if model not in s.index or model not in c.index:
            continue
        x, y = c.loc[model, "inference_patches_per_s"], s.loc[model, "pixel_f1_mean"]
        ax.scatter(x, y, s=70, edgecolors=SURFACE, linewidths=1.5, zorder=3, **_marker_style(model))
        ax.annotate(MODEL_LABELS[model], (x, y), xytext=(6, 4), textcoords="offset points", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Inference throughput (patches/s, RTX 4070 Laptop)", color=INK)
    ax.set_ylabel("Pixel F1 (test, all folds)", color=INK)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    _save(fig, path)
