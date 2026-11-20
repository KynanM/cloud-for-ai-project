"""De figuren voor de softwaremetriek en de deployment (taak 6.5), uit de CSV's in resultaten/.

python -m src.figuren

Maakt in resultaten/figuren/:

software_fases.png      waar de tijd naartoe gaat: mediane tijd per fase van de kern, per workload
software_schaal.png     latency tegenover het aantal boeking-nachten (kern en vers proces)
modus.png               batch, online, model laden per verzoek (les 07 p. 8, met het eigen model)
serving_belasting.png   p95 tegenover de aangeboden belasting, 1, 2 en 4 containers
sizing_pareto.png       CPU per advies tegenover p95, per infrastructuur

De andere figuren maken de opdrachten die ook de cijfers maken: src.evaluate (winstcurve, ijking),
src.robuustheid (risico per maand, rolling origin, complexiteit, bootstrap, ijking met intervallen)
en src.business (scenario's, ROI, MAE tegenover euro's).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import PROJECT  # noqa: E402

RES = PROJECT / "resultaten"
UIT = RES / "figuren"
GRENS_MS = 1000


def bron(ax, tekst: str) -> None:
    ax.annotate(f"bron: {tekst}", (0, -0.2), xycoords="axes fraction", fontsize=7, color="#666")


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
    UIT.mkdir(parents=True, exist_ok=True)
    for maak in (software_fases, software_schaal, modus, serving_belasting, sizing_pareto):
        print(maak())


if __name__ == "__main__":
    main()
