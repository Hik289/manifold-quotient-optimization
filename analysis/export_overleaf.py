"""Create a compact, self-contained Overleaf source upload for the current paper."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
STAGE = ROOT / "overleaf_upload"
ARCHIVE = ROOT / "manifold_quotient_overleaf_2026-11-04.zip"


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_core_data() -> dict:
    audit = json.loads((ROOT / "experiments/full_proposal/evidence/icml_core_losses_audit.json").read_text())
    sources = audit["sources"]
    assert len(sources) == 32
    core = []
    methods = ("adamw", "sgdm", "balance_only", "quotientopt")
    for index, source in enumerate(sources):
        path = ROOT / source["path"]
        assert sha(path) == source["sha256"], path
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if index < 12:
            panel, method, seed_index, key = "cora", methods[index // 3], index % 3, "loss"
            values = [row["validation"][key] for row in rows]
        elif index < 24:
            panel, method, seed_index = "sdf", methods[(index - 12) // 3], (index - 12) % 3
            values = [row["validation"]["mse"] for row in rows]
        elif index < 28:
            panel, method, seed_index = "citeseer", methods[index - 24], 0
            values = [row["validation"]["loss"] for row in rows]
        else:
            panel, method, seed_index = "matrix", methods[index - 28], 0
            values = [row["train_loss"] for row in rows]
        core.append({"panel": panel, "method": method, "seed_index": seed_index,
                     "run_id": source["run_id"], "epoch": [row["epoch"] for row in rows],
                     "loss": values, "source_sha256": source["sha256"]})

    poiss_audit = json.loads((ROOT / "experiments/full_proposal/focused_poisson_v1/paper_audit.json").read_text())
    poisson = []
    for row in poiss_audit["rows"]:
        path = ROOT / row["trajectory_path"]
        assert sha(path) == row["trajectory_sha256"], path
        trajectory = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(trajectory) == 2000
        poisson.append({"method": row["method"], "seed": row["initialization_seed"],
                        "run_id": row["run_id"], "epoch": [r["epoch"] for r in trajectory],
                        "loss": [r["validation"]["mse"] for r in trajectory],
                        "source_sha256": row["trajectory_sha256"]})
    assert len(poisson) == 12
    return {"core": core, "poisson": poisson,
            "provenance": {"core_audit_sha256": sha(ROOT / "experiments/full_proposal/evidence/icml_core_losses_audit.json"),
                           "poisson_audit_sha256": sha(ROOT / "experiments/full_proposal/focused_poisson_v1/paper_audit.json")}}


def check_tex_dependencies() -> None:
    text = "\n".join(path.read_text() for path in STAGE.rglob("*.tex"))
    missing = []
    for name in re.findall(r"\\(?:input|include)\{([^}]+)\}", text):
        if not (STAGE / f"{name}.tex").exists():
            missing.append(f"input: {name}")
    for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text):
        if not (STAGE / name).exists():
            missing.append(f"image: {name}")
    for name in re.findall(r"\\bibliography\{([^}]+)\}", text):
        for item in name.split(","):
            if not (STAGE / f"{item.strip()}.bib").exists():
                missing.append(f"bibliography: {item}")
    for name in re.findall(r"\\bibliographystyle\{([^}]+)\}", text):
        if not (STAGE / f"{name}.bst").exists():
            missing.append(f"bibliography style: {name}")
    for name in re.findall(r"\\usepackage(?:\[[^]]*\])?\{([^}]+)\}", (STAGE / "main.tex").read_text()):
        if name.startswith("icml") and not (STAGE / f"{name}.sty").exists():
            missing.append(f"local package: {name}")
    if missing:
        raise RuntimeError("Missing packaged dependencies:\n" + "\n".join(missing))


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir()
    for name in ("main.tex", "references.bib", "icml2027_draft.sty", "icml2026.sty",
                 "icml2026.bst", "TEMPLATE_PROVENANCE.json"):
        copy(PAPER / name, STAGE / name)
    for folder in ("sections", "figures"):
        for path in (PAPER / folder).glob("*.tex"):
            copy(path, STAGE / folder / path.name)
    for path in (PAPER / "figures").glob("*.pdf"):
        copy(path, STAGE / "figures" / path.name)

    script_sources = {
        "plot_core_figures_original.py": ROOT / "experiments/full_proposal/build_icml_publication.py",
        "plot_poisson_original.py": ROOT / "experiments/full_proposal/focused_poisson_v1/build_paper.py",
        "plot_supporting_losses_original.py": ROOT / "experiments/full_proposal/build_core_loss.py",
        "plot_convergence_profiles_original.py": ROOT / "experiments/full_proposal/build_convergence_profiles.py",
        "plot_initialization_original.py": ROOT / "experiments/full_proposal/build_initialization_paper.py",
        "plot_new_tasks_original.py": ROOT / "experiments/full_proposal/build_new_campaign_paper.py",
        "plot_replacement_original.py": ROOT / "analysis/plot_replacement.py",
        "plot_fresh_evidence_original.py": ROOT / "experiments/full_proposal/build_evidence_paper.py",
        "plot_kappa_archive_original.py": ROOT / "figures/fig1_kappa_trajectory_code.py",
        "plot_gauge_archive_original.py": ROOT / "figures/fig2_gauge_stress_code.py",
        "plot_null_archive_original.py": ROOT / "figures/fig3_null_probes_code.py",
    }
    for name, path in script_sources.items():
        copy(path, STAGE / "figure_code" / "original_scripts" / name)
    copy(ROOT / "analysis/revision_rigid_math_20261101/figure_prompt.txt",
         STAGE / "figure_code" / "method_illustration_prompt.txt")
    with gzip.open(STAGE / "figure_code" / "main_plot_data.json.gz", "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(make_core_data(), handle, separators=(",", ":"), allow_nan=False)

    copy(ROOT / "analysis/plot_overleaf_core_figures.py", STAGE / "figure_code" / "plot_core_figures.py")
    (STAGE / "README_OVERLEAF.md").write_text(
        "# Manifold quotient paper: Overleaf upload\n\n"
        "Upload `manifold_quotient_overleaf_2026-11-04.zip` as a **new project** in Overleaf. "
        "Set `main.tex` as the main document and choose pdfLaTeX. The root directory is already arranged for upload. "
        "The PDF figures are included; Overleaf needs only the LaTeX files and figure PDFs to compile. "
        "Use a normal full compile so BibTeX resolves `references.bib`.\n\n"
        "`icml2027_draft.sty` is a working-draft adapter to the bundled official ICML 2026 style; "
        "it is not an official ICML 2027 release.\n\n"
        "## Figure source code\n\n"
        "`figure_code/plot_core_figures.py` reproduces the two principal loss figures from "
        "`figure_code/main_plot_data.json.gz` without experiment logs. Run it locally with "
        "`python3 figure_code/plot_core_figures.py` after installing NumPy and Matplotlib. "
        "The exact plotting scripts for the other figures are in `figure_code/original_scripts/`; "
        "they document the full plotting and selection logic and expect the separate original experiment records. "
        "The method illustration was generated from `figure_code/method_illustration_prompt.txt`; "
        "it is not referenced in this table-led manuscript.\n"
    )
    check_tex_dependencies()
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(STAGE))
    with zipfile.ZipFile(ARCHIVE) as zf:
        assert zf.testzip() is None
        assert "main.tex" in zf.namelist()
        print(json.dumps({"archive": str(ARCHIVE), "file_count": len(zf.namelist()),
                          "bytes": ARCHIVE.stat().st_size, "sha256": sha(ARCHIVE)}, indent=2))


if __name__ == "__main__":
    main()
