"""De figuren voor de softwaremetriek en de deployment (taak 6.5), uit de CSV's in resultaten/.

python -m src.figuren

Maakt in resultaten/figuren/:

software_fases.png      waar de tijd naartoe gaat: mediane tijd per fase van de kern, per workload
software_schaal.png     latency tegenover het aantal boeking-nachten (kern en vers proces)
modus.png               batch, online, model laden per verzoek (les 07 p. 8, met het eigen model)
serving_belasting.png   p95 tegenover de aangeboden belasting, 1, 2 en 4 containers
sizing_pareto.png       CPU per advies tegenover p95, per infrastructuur

Daarnaast de tekenfuncties voor src.evaluate (teken_winstcurve, teken_ijking): evaluate rekent,
deze module tekent. De andere figuren maken de opdrachten die ook de cijfers maken:
src.robuustheid (risico per maand, rolling origin, complexiteit, bootstrap, ijking met intervallen)
en src.business (scenario's, ROI, MAE tegenover euro's).
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402

from .config import PROJECT, laad_config  # noqa: E402
from .logboek import zet_logging_op  # noqa: E402

RES = PROJECT / "resultaten"
UIT = RES / "figuren"

logger = logging.getLogger(__spec__.name if __name__ == "__main__" else __name__)
GRENS_MS = 1000

# Eén kleur per model, in een vaste volgorde. Met land: dezelfde kleur, een stippellijn.
KLEUR = {"vuistregel": "#2a78d6", "logreg": "#eb6834", "boosting": "#1baf7a"}
INKT, INKT_ZACHT, RASTER = "#1f1f1e", "#5f5e58", "#e4e3df"


def bron(ax, tekst: str) -> None:
    ax.annotate(f"bron: {tekst}", (0, -0.2), xycoords="axes fraction", fontsize=7, color="#666")


def opmaak(ax) -> None:
    """Rustige assen: alleen een licht raster, geen kader rechts en boven."""
    ax.grid(True, color=RASTER, linewidth=0.8)
    ax.set_axisbelow(True)
    for kant in ["top", "right"]:
        ax.spines[kant].set_visible(False)
    for kant in ["left", "bottom"]:
        ax.spines[kant].set_color(RASTER)
    ax.tick_params(colors=INKT_ZACHT, labelsize=9)


def teken_winstcurve(curve: pd.DataFrame, zekerheid: float, pad) -> None:
    fig, assen = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, periode in zip(assen, ["validatie", "test"], strict=True):
        deel = curve[curve["periode"] == periode]
        for soort, lijn in deel.groupby("soort", sort=False):
            ax.plot(lijn["zekerheid"], lijn["euro"], color=KLEUR[soort], linewidth=2, label=soort)
            top = lijn.loc[lijn["euro"].idxmax()]
            ax.plot(top["zekerheid"], top["euro"], "o", color=KLEUR[soort], markersize=8)
            ax.annotate(
                soort,
                (lijn["zekerheid"].iloc[-1], lijn["euro"].iloc[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                color=INKT,
                fontsize=9,
                va="center",
            )
        ax.axvline(zekerheid, color=INKT_ZACHT, linewidth=1, linestyle="--")
        ax.text(zekerheid, ax.get_ylim()[1], f" {zekerheid:.2f}", color=INKT_ZACHT, fontsize=9)
        ax.set_title(periode.capitalize(), color=INKT, fontsize=11, loc="left")
        ax.set_xlabel("zekerheid: P(minstens k annulaties)", color=INKT_ZACHT, fontsize=9)
        opmaak(ax)
    assen[0].set_ylabel("€ per volgeboekte nacht", color=INKT_ZACHT, fontsize=9)
    assen[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(
        "Winst per volgeboekte nacht naargelang de zekerheid (bolletje: hoogste punt)",
        color=INKT,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(pad, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def teken_ijking(uitkomst: dict, pad) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 0.6], [0, 0.6], color=INKT_ZACHT, linewidth=1, linestyle="--")
    ax.text(0.45, 0.47, "perfect geijkt", color=INKT_ZACHT, fontsize=9, rotation=40)
    for (soort, land), u in uitkomst["per_variant"].items():
        if land:
            continue
        echt, voorspeld = calibration_curve(
            u["validatie"]["y"], u["validatie"]["p"], n_bins=10, strategy="quantile"
        )
        # Bij weinig verschillende kansen (de vuistregel) vallen tienden samen: sorteer.
        orde = np.argsort(voorspeld)
        voorspeld, echt = voorspeld[orde], echt[orde]
        ax.plot(voorspeld, echt, "-o", color=KLEUR[soort], linewidth=2, markersize=5, label=soort)
    ax.set_xlabel("voorspelde kans (gemiddelde per tiende)", color=INKT_ZACHT, fontsize=9)
    ax.set_ylabel("echt aandeel annulaties", color=INKT_ZACHT, fontsize=9)
    ax.set_title("IJking op de validatie: is 20 % echt 20 %?", color=INKT, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    opmaak(ax)
    fig.tight_layout()
    fig.savefig(pad, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


# --------------------------------------------------------------------- figuren van src.robuustheid


def teken_risico(tabel: pd.DataFrame, grens: float, pad: Path):
    """Te veel per hotel en maand, per model, met de 90 %-blokbootstrap en de risicogrens."""
    modellen = ["vuistregel", "logreg", "boosting"]
    fig, assen = plt.subplots(2, 3, figsize=(13, 6.5), sharey=True)
    for i, hotel in enumerate(["City Hotel", "Resort Hotel"]):
        for j, model in enumerate(modellen):
            ax = assen[i, j]
            g = tabel[(tabel["hotel"] == hotel) & (tabel["model"] == model)].sort_values("maand")
            x = np.arange(len(g))
            kleur = np.where(g["periode"] == "validatie", "#2b6cb0", "#b7791f")
            ax.errorbar(
                x,
                g["te_veel"],
                yerr=[
                    g["te_veel"] - g["blok7_laag"].fillna(g["te_veel"]),
                    g["blok7_hoog"].fillna(g["te_veel"]) - g["te_veel"],
                ],
                fmt="none",
                ecolor="#999999",
                capsize=2,
            )
            ax.scatter(x, g["te_veel"], c=kleur, zorder=3, s=22)
            ax.axhline(grens, color="#c53030", lw=1, ls="--")
            ax.set_xticks(x, g["maand"], rotation=60, fontsize=7)
            ax.set_title(f"{model} · {hotel}", fontsize=9)
            if j == 0:
                ax.set_ylabel("aandeel nachten met te veel kamers")
    fig.suptitle(
        "Te veel per hotel en maand (blauw: validatie, oranje: test, alleen diagnose)."
        " Lijn: risicogrens 20 %. Balkjes: 90 %-blokbootstrap (7 dagen)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(pad, dpi=130)
    return fig


def teken_bootstrap(tabel: pd.DataFrame, pad: Path):
    """Per vergelijking het 90 %-interval over losse nachten en over blokken van 1 tot 28 dagen."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    lengtes = ["losse_nachten", "blok1", "blok7", "blok14", "blok28"]
    for i, rij in tabel.iterrows():
        for j, lengte in enumerate(lengtes):
            y = i + (j - 2) * 0.13
            ax.plot(
                [rij[f"{lengte}_laag"], rij[f"{lengte}_hoog"]],
                [y, y],
                color=f"C{j}",
                lw=2,
                label=lengte if i == 0 else None,
            )
        ax.plot(rij["verschil"], i, "k|", ms=12)
    ax.axvline(0, color="#555555", lw=1)
    ax.set_yticks(
        range(len(tabel)),
        [f"{v} ({e})" for v, e in zip(tabel["vergelijking"], tabel["eenheid"], strict=True)],
        fontsize=8,
    )
    ax.set_xscale("symlog", linthresh=1)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(
        "90 %-interval van het verschil per nacht: losse nachten tegenover blokken van dagen"
        " (validatie)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(pad, dpi=130)
    return fig


def teken_rolling(tabel: pd.DataFrame, pad: Path):
    """De MAE per kwartaal van de rolling origin, per model, voor beide hotels en per hotel."""
    fig, assen = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, hotel in zip(assen, ["beide", "City Hotel", "Resort Hotel"], strict=True):
        for soort in ["vuistregel", "logreg", "boosting"]:
            g = tabel[(tabel["hotel"] == hotel) & (tabel["model"] == soort)]
            ax.plot(g["fold"], g["mae"], marker="o", label=soort)
        ax.set_title(hotel, fontsize=9)
        ax.set_xlabel("meetblok (kwartaal)")
    assen[0].set_ylabel("MAE per hotel-nacht")
    assen[0].legend(fontsize=8)
    fig.suptitle("Rolling origin: elk kwartaal geleerd op alles vóór het embargo", fontsize=9)
    fig.tight_layout()
    fig.savefig(pad, dpi=130)
    return fig


def teken_complexiteit(tabel: pd.DataFrame, grens_ms: float, pad: Path):
    """De complexiteitsladder: MAE tegenover de p95-latency, met de latencygrens."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for _, rij in tabel.iterrows():
        kleur = "#2f855a" if rij["gekozen"] else ("#2b6cb0" if rij["binnen_grenzen"] else "#c53030")
        ax.scatter(rij["kern_p95_ms"], rij["mae"], s=50, color=kleur, zorder=3)
        ax.annotate(
            rij["trede"].split(" ", 1)[0],
            (rij["kern_p95_ms"], rij["mae"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvline(grens_ms, color="#c53030", ls="--", lw=1, label="grens p95 = 1 000 ms")
    ax.set_xscale("log")
    ax.set_xlabel("p95-latency van de kern, normale workload (ms, log)")
    ax.set_ylabel("MAE per hotel-nacht (validatie)")
    ax.set_title(
        "Complexiteitsladder (les 06 p. 23): de MAE optimaliseren, de latency als grens", fontsize=9
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(pad, dpi=130)
    return fig


def teken_kalibratie(tabel: pd.DataFrame, pad: Path):
    """IJking per hotel: gemiddelde kans tegenover het aandeel per bak, met 95 %-Wilson-interval."""
    fig, assen = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for (soort, hotel), g in tabel.groupby(["model", "hotel"], sort=False):
        ax = assen[["City Hotel", "Resort Hotel"].index(hotel)]
        xs, ys = g["gem_kans"].to_numpy(), g["aandeel"].to_numpy()
        ax.errorbar(
            xs,
            ys,
            yerr=[ys - g["wilson95_laag"].to_numpy(), g["wilson95_hoog"].to_numpy() - ys],
            marker="o",
            ms=4,
            capsize=2,
            label=f"{soort} (ECE {g['ece'].iloc[0]:.3f})",
        )
        ax.set_title(hotel, fontsize=9)
    for ax in assen:
        ax.plot([0, 0.6], [0, 0.6], color="#888888", lw=1, ls="--")
        ax.set_xlabel("gemiddelde voorspelde kans (10 kwantielbakken)")
        ax.legend(fontsize=7)
    assen[0].set_ylabel("waargenomen aandeel, met 95 %-Wilson-interval")
    fig.suptitle("IJking per hotel op de validatie", fontsize=9)
    fig.tight_layout()
    fig.savefig(pad, dpi=130)
    return fig


def software_fases() -> Path:
    ruw = pd.read_csv(RES / "software" / "raw_runs.csv", low_memory=False)
    kern = ruw[(ruw["sessie"] == "fase1") & (ruw["doel"] == "kern")]
    fases = ["schema", "scope", "voorspellen", "aggregeren", "advies", "advies_valideren"]
    volgorde = ["smoke", "normaal", "kort_verblijf", "lang_verblijf", "schaal_10k", "schaal_100k"]
    tabel = kern.groupby("workload")[[f"fase_{f}_ms" for f in fases]].median().reindex(volgorde)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    onder = np.zeros(len(tabel))
    for fase in fases:
        waarden = tabel[f"fase_{fase}_ms"].to_numpy()
        ax.bar(tabel.index, waarden, bottom=onder, label=fase)
        onder += waarden
    ax.set_yscale("log")
    ax.set_ylabel("mediane tijd per fase (ms, log)")
    ax.set_title("Waar de tijd naartoe gaat in een adviesjob (kern, zonder HTTP)", fontsize=10)
    ax.legend(fontsize=8, ncols=3)
    bron(ax, "resultaten/software/raw_runs.csv, sessie fase1")
    fig.tight_layout()
    pad = UIT / "software_fases.png"
    fig.savefig(pad, dpi=130)
    plt.close(fig)
    return pad


def software_schaal() -> Path:
    s = pd.read_csv(RES / "software" / "summary.csv")
    s = s[s["sessie"] == "fase1"]
    import json

    wl = json.loads((RES / "software" / "workloads.json").read_text(encoding="utf-8"))["workloads"]
    s = s.assign(nachten=s["workload"].map(lambda w: wl[w]["n_boeking_nachten"]))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for doel, kleur in [("kern", "#2b6cb0"), ("cli", "#b7791f")]:
        g = s[s["doel"] == doel].sort_values("nachten")
        ax.plot(g["nachten"], g["p95_ms"], marker="o", color=kleur, label=f"{doel}: p95")
        ax.plot(g["nachten"], g["p50_ms"], ls="--", color=kleur, alpha=0.6, label=f"{doel}: p50")
    ax.axhline(GRENS_MS, color="#c53030", ls=":", label="grens 1 000 ms (primair: HTTP)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("boeking-nachten in de invoer (log)")
    ax.set_ylabel("latency (ms, log)")
    ax.set_title("Schalen: van 76 tot 100 000 boekingen", fontsize=10)
    ax.legend(fontsize=8)
    bron(ax, "resultaten/software/summary.csv")
    fig.tight_layout()
    pad = UIT / "software_schaal.png"
    fig.savefig(pad, dpi=130)
    plt.close(fig)
    return pad


def modus() -> Path:
    m = pd.read_csv(RES / "software" / "modus.csv")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(m["variant"], m["p50_ms"], color=["#2f855a", "#b7791f", "#c53030"])
    ax.axvline(GRENS_MS, color="#c53030", ls=":", label="grens 1 000 ms")
    ax.set_xscale("log")
    ax.set_xlabel("tijd per adviesnacht, 358 boekingen (ms, log)")
    for i, v in enumerate(m["p50_ms"]):
        ax.annotate(
            f"{v:,.0f} ms".replace(",", " "),
            (v, i),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    ax.set_title("Batch tegenover online (les 07 p. 8, met het eigen model)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    bron(ax, "resultaten/software/modus.csv")
    fig.tight_layout()
    pad = UIT / "modus.png"
    fig.savefig(pad, dpi=130)
    plt.close(fig)
    return pad


def serving_belasting() -> Path:
    b = pd.read_csv(RES / "serving" / "belasting.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for variant, g in b.groupby("variant"):
        ax.plot(g["aangeboden_rps"], g["p95_ms"], marker="o", label=variant)
        fout = g[g["foutrate"] >= 0.01]
        ax.scatter(
            fout["aangeboden_rps"], fout["p95_ms"], marker="x", s=80, color="#c53030", zorder=3
        )
    ax.axhline(GRENS_MS, color="#c53030", ls=":", label="grens 1 000 ms")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("aangeboden verzoeken per seconde (log)")
    ax.set_ylabel("p95-latency (ms, log)")
    ax.set_title(
        "Wat breekt eerst? Containers van 0,5 vCPU achter nginx (× = ≥ 1 % fouten)", fontsize=10
    )
    ax.legend(fontsize=8)
    bron(ax, "resultaten/serving/belasting.csv")
    fig.tight_layout()
    pad = UIT / "serving_belasting.png"
    fig.savefig(pad, dpi=130)
    plt.close(fig)
    return pad


def sizing_pareto() -> Path:
    s = pd.read_csv(RES / "serving" / "sizing.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, r in s.iterrows():
        label = f"{r['cpus']} vCPU, {r['geheugen']}, {int(r['threads'])} thread(s)"
        ax.scatter(r["cpu_s_per_advies"], r["p95_ms"], s=50)
        ax.annotate(
            label,
            (r["cpu_s_per_advies"], r["p95_ms"]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=7,
        )
    ax.axhline(GRENS_MS, color="#c53030", ls=":", label="grens 1 000 ms")
    ax.set_xlabel("CPU-seconden per advies (de kost)")
    ax.set_ylabel("p95-latency (ms)")
    ax.set_ylim(0, 1100)
    ax.set_title(
        "Infrastructuur: elke combinatie haalt de grens; de kleinste volstaat", fontsize=10
    )
    ax.legend(fontsize=8)
    bron(ax, "resultaten/serving/sizing.csv")
    fig.tight_layout()
    pad = UIT / "sizing_pareto.png"
    fig.savefig(pad, dpi=130)
    plt.close(fig)
    return pad


def main() -> None:
    """Teken de figuren van de softwaremetriek en zeg waar elke figuur staat."""
    zet_logging_op(laad_config(), "figuren")
    UIT.mkdir(parents=True, exist_ok=True)
    for maak in (software_fases, software_schaal, modus, serving_belasting, sizing_pareto):
        logger.info("Figuur weggeschreven naar %s", maak())


if __name__ == "__main__":
    main()
