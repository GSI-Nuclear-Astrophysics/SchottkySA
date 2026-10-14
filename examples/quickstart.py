"""Fit the public two-component fixture and save a verified replay recipe.

Run ``python examples/quickstart.py --output example-results`` to retain the
tables and session. The fixtures are constructed inputs, not measurements.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ssa import __version__
from ssa.config import FitConfig, TemplateBuildConfig, UncertaintyConfig
from ssa.io import (
    export_history_rows,
    load_spectrum_npz,
    make_export_rows_for_result,
    select_frequency_range,
    write_curve_csv,
)
from ssa.pipeline import build_template_bank, run_fit_with_uncertainty
from ssa.session import (
    SCHEMA_VERSION,
    FitRecord,
    FormState,
    InputRef,
    Session,
    TemplateRecord,
    load_session,
    replay_session,
    save_session,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "tests" / "data"


def run_example(output: Path, draws: int = 100):
    """Return the saved session and results after checking their replay."""
    output.mkdir(parents=True, exist_ok=True)
    source = DATA_DIR / "synthetic_full_spectrum.npz"
    x, y, fkey, ykey = load_spectrum_npz(source)
    reference_region = (1937300.0, 1937360.0)
    target_region = (1937370.0, 1937420.0)
    template_config = TemplateBuildConfig(name="isolated reference", n_template_boot=draws)
    fit_config = FitConfig(n_peaks=2, init_mus=[1937390.0, 1937397.5])
    uncertainty_config = UncertaintyConfig(n_boot=draws, n_template_prop=draws)

    fx, fy, _ = select_frequency_range(x, y, None, *reference_region)
    bank = build_template_bank(fx, fy, template_config)
    tx, ty, _ = select_frequency_range(x, y, None, *target_region)
    result = run_fit_with_uncertainty(
        tx, ty, bank["template_nominal"], bank["templates"], fit_config, uncertainty_config
    )
    registry = [
        {
            "template": bank["template_nominal"],
            "template_bank": bank["templates"],
            "region_lo": reference_region[0],
            "region_hi": reference_region[1],
            "settings": {
                "smooth_nominal_template": template_config.smooth,
                "resample_factor": template_config.resample_factor,
            },
        }
    ]
    rows = make_export_rows_for_result(
        result, label="doublet", template_registry=registry, notes="Constructed public example"
    )
    template = bank["template_nominal"]
    q16, q50, q84 = template.quantile([0.16, 0.50, 0.84])
    template_checkpoints = {
        "raw_area": float(template.raw_area),
        "raw_cog": float(template.raw_cog),
        "raw_std": float(template.raw_std),
        "q16": float(q16),
        "q50": float(q50),
        "q84": float(q84),
        "n_template_bank": len(bank["templates"]),
    }
    session = Session(
        schema_version=SCHEMA_VERSION,
        ssa_version=__version__,
        created_utc=datetime.now(timezone.utc).isoformat(),
        input=InputRef(str(source.resolve()), hashlib.sha256(source.read_bytes()).hexdigest(), fkey, ykey),
        form_state=FormState(
            template_config, fit_config, "doublet", True, uncertainty_config, target_region, active_template_id=0
        ),
        templates=[TemplateRecord(0, reference_region, template_config, template_checkpoints)],
        fits=[
            FitRecord(
                "doublet",
                "Constructed public example",
                0,
                target_region,
                fit_config,
                uncertainty_config,
                verify_rows=rows,
            )
        ],
    )
    session_path = output / "session.toml"
    save_session(session_path, session)
    report = replay_session(load_session(session_path))
    if not report.ok:
        raise RuntimeError(f"Example replay failed: {report.mismatches}")
    fit = result["fit"]
    write_curve_csv(
        output / "doublet_curve.csv", fit["x"], fit["y"], fit["y_model"], fit["background"], fit["components"]
    )
    export_history_rows(rows, output / "components.csv")
    for name, xx, yy in (("reference.csv", fx, fy), ("target.csv", tx, ty)):
        np.savetxt(output / name, np.column_stack([xx, yy]), delimiter=",", header="frequency_hz,power", comments="")
    return session, result, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Retain the example files in this directory.")
    parser.add_argument("--draws", type=int, default=100, help="Draws per uncertainty stream.")
    args = parser.parse_args()
    with TemporaryDirectory() as temporary:
        output = args.output or Path(temporary)
        _, result, report = run_example(output, args.draws)
        fit = result["fit"]
        for index, (mu, area) in enumerate(zip(fit["info"]["mus"], fit["info"]["areas"], strict=True)):
            print(f"C{index + 1}: centroid={mu:.6f} Hz; area={area:.6g}")
        print(f"Replay verified: {report.ok}")
        if args.output:
            print(f"Example files: {output}")


if __name__ == "__main__":
    main()
