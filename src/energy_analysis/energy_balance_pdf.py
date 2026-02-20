from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

__all__ = ["plot_energybalance_csv_to_pdf", "batch_plot_directory"]


@dataclass(frozen=True)
class EnergyBalanceColumns:
    # Column indices (0-based)
    time: int = 0
    direct_consumption: int = 1
    battery_discharging: int = 2
    grid_import: int = 3
    total_consumption: int = 4
    grid_export: int = 5
    direct_consumption_bis: int = 6
    battery_charging: int = 7
    pv_power: int = 8


def _to_numeric_clean(
    series: pd.Series,
    *,
    col_label: str,
    csv_name: str,
    fillna_value: float | None = 0.0,
) -> np.ndarray:
    """
    Robust conversion:
    - trims whitespace
    - converts to numeric (errors -> NaN)
    - optionally fills NaN with a value (default 0.0)
    - logs how many values were 'dirty' (non-numeric after stripping)
    """
    # Keep original for dirtiness detection
    s_raw = series

    # Normalize to string, strip whitespace
    s_stripped = s_raw.astype(str).str.strip()

    # Convert to numeric; invalid -> NaN
    s_num = pd.to_numeric(s_stripped, errors="coerce")

    # Determine "dirty" entries:
    # non-empty after strip, but became NaN after conversion
    dirty_mask = (s_stripped != "") & s_num.isna()
    dirty_count = int(dirty_mask.sum())

    # Also catch "blank-ish" values that were strings but stripped to empty
    blank_mask = (s_stripped == "")
    blank_count = int(blank_mask.sum())

    if dirty_count or blank_count:
        msg_parts = []
        if dirty_count:
            msg_parts.append(f"{dirty_count} non-numeric")
        if blank_count:
            msg_parts.append(f"{blank_count} blank/whitespace")
        logger.warning(
            "CSV '%s': column '%s' had %s value(s) coerced to NaN.",
            csv_name,
            col_label,
            " and ".join(msg_parts),
        )

    if fillna_value is not None:
        s_num = s_num.fillna(fillna_value)

    return s_num.to_numpy(dtype=float)


def plot_energybalance_csv_to_pdf(
    csv_path: str | Path,
    *,
    out_dir: str | Path | None = None,
    sep: str = ";",
    thousands: str = ",",
    fillna_value: float | None = None,  # set to None if you prefer NaNs to propagate
    columns: EnergyBalanceColumns = EnergyBalanceColumns(),
) -> Path:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    if out_dir is None:
        out_dir = csv_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir / f"{csv_path.stem}.pdf"

    # Treat blanks/spaces as NA already during read (extra safety)
    df = pd.read_csv(
        csv_path,
        sep=sep,
        thousands=thousands,
        na_values=["", " "],
        keep_default_na=True,
    )

    csv_name = csv_path.name

    # Robust numeric extraction
    pv_power = _to_numeric_clean(
        df.iloc[:, columns.pv_power],
        col_label="pv_power",
        csv_name=csv_name,
        fillna_value=fillna_value,
    )
    dir_consumption = _to_numeric_clean(
        df.iloc[:, columns.direct_consumption],
        col_label="direct_consumption",
        csv_name=csv_name,
        fillna_value=fillna_value,
    )
    battery_charging = _to_numeric_clean(
        df.iloc[:, columns.battery_charging],
        col_label="battery_charging",
        csv_name=csv_name,
        fillna_value=fillna_value,
    )
    battery_discharging = _to_numeric_clean(
        df.iloc[:, columns.battery_discharging],
        col_label="battery_discharging",
        csv_name=csv_name,
        fillna_value=fillna_value
    )
    grid_export = _to_numeric_clean(
        df.iloc[:, columns.grid_export],
        col_label="grid_export",
        csv_name=csv_name,
        fillna_value=fillna_value
    )

    # Derived
    surplus = pv_power - dir_consumption

    # X-axis
    time_index = np.arange(len(df))

    # ---- Plot to PDF (single page, two stacked charts)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11.69, 8.27), sharex=True)
    fig.suptitle(csv_path.name, fontsize=14)

    ax1, ax2 = axes

    ax1.plot(time_index, pv_power, label="PV-power")
    ax1.plot(time_index, dir_consumption, label="direct consumption")
    ax1.set_ylabel("power, W")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax1.legend()

    ax2.plot(time_index, surplus, label="surplus")
    ax2.plot(time_index, battery_charging, label="battery charging")
    ax2.plot(time_index, grid_export, label="grid export")
    ax2.plot(time_index, battery_discharging, label="battery discharging", linestyle="--")
    ax2.set_xlabel("time index")
    ax2.set_ylabel("power, W")
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax2.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)

    return pdf_path


def batch_plot_directory(
    csv_dir: str | Path,
    *,
    out_dir: str | Path | None = None,
    pattern: str = "*.csv",
) -> list[Path]:
    """
    Convert all CSV files in a directory to PDFs.

    Returns a list with created PDF paths.
    """
    csv_dir = Path(csv_dir)
    if not csv_dir.exists():
        raise FileNotFoundError(csv_dir)

    created: list[Path] = []
    for csv_path in sorted(csv_dir.glob(pattern)):
        if csv_path.is_file():
            created.append(plot_energybalance_csv_to_pdf(csv_path, out_dir=out_dir))
    return created
