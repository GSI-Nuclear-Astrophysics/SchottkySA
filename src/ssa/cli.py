"""``ssa`` command-line entry point: GUI launcher plus the ``run`` subcommand.

``ssa`` with no subcommand launches the GUI. ``ssa run`` is a headless
analysis command reading named-column CSV/TSV/whitespace tables and writing
a JSON summary plus a per-bin curve CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from ssa import __version__
from ssa.config import FitConfig, TemplateBuildConfig, UncertaintyConfig
from ssa.constants import EDGE_MODES, LOSS_CHOICES
from ssa.exceptions import InputFormatError, SSAError
from ssa.io import load_table, select_frequency_range, write_curve_csv
from ssa.pipeline import build_run_summary, build_template_bank, run_fit_with_uncertainty
from ssa.session import load_session, replay_session


def _parse_range(text: str | None) -> tuple[float | None, float | None]:
    if text is None:
        return None, None
    if ":" not in text:
        raise argparse.ArgumentTypeError(f"Range must be 'LO:HI', got {text!r}.")
    lo_s, hi_s = text.split(":", 1)
    return float(lo_s), float(hi_s)


def _parse_mu_bounds(text: str | None) -> list[tuple[float, float]] | None:
    if not text:
        return None
    parts = [p.strip() for p in text.split(";") if p.strip()]
    bounds = []
    for part in parts:
        if ":" not in part:
            raise argparse.ArgumentTypeError(f"Each mu-bounds range must be 'lo:hi'; got {part!r}.")
        lo_s, hi_s = part.split(":", 1)
        bounds.append((float(lo_s), float(hi_s)))
    return bounds


def _parse_init_mus(text: str | None) -> list[float] | None:
    if not text:
        return None
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def _parse_block_size(text: str) -> str | int:
    return "auto" if text.strip().lower() == "auto" else int(text)


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ssa",
        description="SchottkySA -- empirical-template Schottky peak analysis.",
    )
    parser.add_argument("--version", action="version", version=f"SchottkySA {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help="Headless template-fit run: read reference/target tables, write a JSON summary and curve CSV.",
        description=(
            "Build an empirical template from a reference table/region, fit a target table/region, "
            "propagate uncertainty (residual bootstrap + template-shape propagation), and write a "
            "JSON summary plus a per-bin curve CSV. No GUI/display required."
        ),
    )
    run_parser.add_argument("--reference", required=True, type=Path, help="Reference table file.")
    run_parser.add_argument("--reference-range", default=None, help="Restrict the reference table to 'LO:HI' Hz.")
    run_parser.add_argument("--target", required=True, type=Path, help="Target table file.")
    run_parser.add_argument("--target-range", default=None, help="Restrict the target table to 'LO:HI' Hz.")
    run_parser.add_argument("--freq-col", default="frequency_hz")
    run_parser.add_argument("--power-col", default="power")
    run_parser.add_argument("--sigma-col", default="sigma")
    run_parser.add_argument("--label", default="fit_1")

    run_parser.add_argument("--n-peaks", type=int, default=1)
    run_parser.add_argument("--background-order", type=int, default=0, help="-1 disables the background term.")
    run_parser.add_argument("--allow-scale", action=argparse.BooleanOptionalAction, default=True)
    run_parser.add_argument("--common-scale", action=argparse.BooleanOptionalAction, default=True)
    run_parser.add_argument("--loss", choices=list(LOSS_CHOICES), default="linear")
    run_parser.add_argument("--min-separation", type=float, default=0.0, help="Minimum centroid separation, Hz.")
    run_parser.add_argument("--mu-bounds", default=None, help="'lo0:hi0;lo1:hi1;...' centroid search ranges, Hz.")
    run_parser.add_argument("--init-mus", default=None, help="'m0,m1,...' initial centroid guesses, Hz.")

    run_parser.add_argument("--template-name", default="reference_peak")
    run_parser.add_argument("--smooth-template", action=argparse.BooleanOptionalAction, default=False)
    run_parser.add_argument("--clip-negative", action=argparse.BooleanOptionalAction, default=True)
    run_parser.add_argument("--resample-factor", type=int, default=16)
    run_parser.add_argument("--edge-mode", choices=list(EDGE_MODES), default="none")
    run_parser.add_argument("--edge-width-hz", type=float, default=0.0)
    run_parser.add_argument("--edge-fraction", type=float, default=0.08)
    run_parser.add_argument("--n-template-boot", type=int, default=300)

    run_parser.add_argument("--n-boot", type=int, default=1000)
    run_parser.add_argument("--n-template-prop", type=int, default=300)
    run_parser.add_argument("--block-size", default="auto", help="'auto' or an integer block length.")
    run_parser.add_argument("--seed", type=int, default=12345)
    run_parser.add_argument("--run-bootstrap", action=argparse.BooleanOptionalAction, default=True)
    run_parser.add_argument("--run-template-propagation", action=argparse.BooleanOptionalAction, default=True)

    run_parser.add_argument("--out-dir", type=Path, default=Path("."))
    run_parser.add_argument("--out-json", type=Path, default=None, help="Default: <out-dir>/<label>_ssa_summary.json")
    run_parser.add_argument("--out-csv", type=Path, default=None, help="Default: <out-dir>/<label>_ssa_curve.csv")

    replay_parser = subparsers.add_parser(
        "replay",
        help="Recompute a saved session from its TOML file and verify the numbers reproduce.",
        description=(
            "Load a session saved by the GUI (or ssa.session), rerun every saved template build and "
            "fit from its recorded parameters, and compare the result against the values recorded "
            "when it was saved. Exits 0 if everything reproduces within tolerance, 1 otherwise."
        ),
    )
    replay_parser.add_argument("session", type=Path, help="Session TOML file.")
    replay_parser.add_argument(
        "--data", type=Path, default=None, help="Override the input NPZ path recorded in the session."
    )

    return parser


def _run_command(args: argparse.Namespace) -> int:
    try:
        ref_lo, ref_hi = _parse_range(args.reference_range)
        tgt_lo, tgt_hi = _parse_range(args.target_range)

        f_ref, y_ref, _sigma_ref = load_table(args.reference, args.freq_col, args.power_col, args.sigma_col)
        f_ref, y_ref, _ = select_frequency_range(f_ref, y_ref, None, ref_lo, ref_hi)

        f_tgt, y_tgt, sigma_tgt = load_table(args.target, args.freq_col, args.power_col, args.sigma_col)
        f_tgt, y_tgt, sigma_tgt = select_frequency_range(f_tgt, y_tgt, sigma_tgt, tgt_lo, tgt_hi)
        sigma_source = "supplied" if sigma_tgt is not None else "robust_second_difference"

        template_config = TemplateBuildConfig(
            name=args.template_name,
            n_template_boot=args.n_template_boot,
            block_size=_parse_block_size(args.block_size),
            random_seed=args.seed,
            clip_negative=args.clip_negative,
            smooth=args.smooth_template,
            resample_factor=args.resample_factor,
            edge_mode=args.edge_mode,
            edge_width_hz=args.edge_width_hz,
            edge_fraction=args.edge_fraction,
        )
        fit_config = FitConfig(
            n_peaks=args.n_peaks,
            init_mus=_parse_init_mus(args.init_mus),
            mu_bounds=_parse_mu_bounds(args.mu_bounds),
            background_order=args.background_order,
            allow_scale=args.allow_scale,
            common_scale=args.common_scale,
            loss=args.loss,
            min_separation=args.min_separation,
        )
        unc_config = UncertaintyConfig(
            block_size=_parse_block_size(args.block_size),
            random_seed=args.seed,
            run_bootstrap=args.run_bootstrap,
            n_boot=args.n_boot,
            run_template_propagation=args.run_template_propagation,
            n_template_prop=args.n_template_prop,
        )

        print(f"Building template from {args.reference} ({len(f_ref)} points)...", file=sys.stderr)
        bank = build_template_bank(f_ref, y_ref, template_config)

        print(f"Fitting {args.target} ({len(f_tgt)} points, n_peaks={args.n_peaks})...", file=sys.stderr)
        result = run_fit_with_uncertainty(
            f_tgt, y_tgt, bank["template_nominal"], bank["templates"], fit_config, unc_config, sigma=sigma_tgt
        )

        summary = build_run_summary(
            label=args.label,
            reference_source={
                "path": str(args.reference),
                "sha256": _sha256_of(args.reference),
                "range_hz": [ref_lo, ref_hi],
            },
            target_source={
                "path": str(args.target),
                "sha256": _sha256_of(args.target),
                "range_hz": [tgt_lo, tgt_hi],
            },
            template_config=template_config,
            fit_config=fit_config,
            unc_config=unc_config,
            template_bank_result=bank,
            analysis_result=result,
            sigma_source=sigma_source,
        )

        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_json = args.out_json or (args.out_dir / f"{args.label}_ssa_summary.json")
        out_csv = args.out_csv or (args.out_dir / f"{args.label}_ssa_curve.csv")

        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        fit = result["fit"]
        write_curve_csv(out_csv, fit["x"], fit["y"], fit["y_model"], fit["background"], fit["components"])

        print(f"quality_flag = {summary['quality_flag']}", file=sys.stderr)
        print(f"Wrote {out_json}", file=sys.stderr)
        print(f"Wrote {out_csv}", file=sys.stderr)
        return 0
    except (InputFormatError, SSAError) as exc:
        print(f"ssa run failed: {exc}", file=sys.stderr)
        return 1


def _replay_command(args: argparse.Namespace) -> int:
    try:
        session = load_session(args.session)
        report = replay_session(
            session,
            data_path_override=args.data,
            progress_cb=lambda msg: print(msg, file=sys.stderr),
        )
        for warning in report.input_warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
        for template in session.templates:
            status = "ok" if not any(f"template {template.id}:" in m for m in report.mismatches) else "MISMATCH"
            print(f"template {template.id}: {status}", file=sys.stderr)
        for fit in session.fits:
            status = "ok" if not any(m.startswith(f"{fit.label} component ") for m in report.mismatches) else "MISMATCH"
            print(f"fit '{fit.label}': {status}", file=sys.stderr)
        for mismatch in report.mismatches:
            print(f"  {mismatch}", file=sys.stderr)
        if report.ok:
            print(
                f"All {len(session.templates)} template(s) and {len(session.fits)} fit(s) reproduced.",
                file=sys.stderr,
            )
            return 0
        print(f"{len(report.mismatches)} mismatch(es) found.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ssa replay failed: {exc}", file=sys.stderr)
        return 1


def _launch_gui() -> int:
    try:
        from ssa.gui.main_window import MainWindow
    except ImportError as exc:
        print(
            "The SchottkySA GUI requires the optional 'gui' extra.\n"
            "Install it with:  pip install 'SchottkySA[gui]'\n"
            f"(import failed: {exc})",
            file=sys.stderr,
        )
        return 1

    from pyqtgraph.Qt import QtWidgets

    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("SchottkySA")
    win = MainWindow()
    win.show()
    return app.exec()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run_command(args)
    if args.command == "replay":
        return _replay_command(args)
    return _launch_gui()


if __name__ == "__main__":
    raise SystemExit(main())
