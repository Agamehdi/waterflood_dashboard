"""Rumaila Pattern Waterflood | upload-driven Streamlit dashboard.

Run locally: streamlit run streamlit_app.py
GitHub/Streamlit Cloud: deploy with streamlit_app.py + requirements.txt.
No field data is bundled in this source code.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Pattern Waterflood | Rumaila", page_icon="💧", layout="wide", initial_sidebar_state="expanded")

# -----------------------------------------------------------------------------
# Input definitions — these are matched without regard to case or punctuation.
# -----------------------------------------------------------------------------
FIELDS = {
    "Pattern": ["Pattern Name", "Pattern", "Pattern ID"],
    "Reservoir": ["Reservoir", "Reservoir Name", "Resevoir", "Resevoir Name", "Reservior", "Reservior Name", "Reservoir Type"],
    "Oil": ["Produced Oil (stb)", "Produced Oil", "Oil Production (stb)", "Oil Volume"],
    "Water": ["Produced Water (stb)", "Produced Water", "Water Production (stb)"],
    "Injection": ["Injected Water (stb)", "Injected Water", "Water Injection (stb)", "Water Injected"],
    "VRR": ["VRR", "Voidage Replacement Ratio"],
    "Res Pressure": ["Average of Res P", "Average Reservoir Pressure", "Reservoir Pressure"],
    "Psat": ["Average of P Sat", "Average Saturation Pressure", "Bubble Point Pressure"],
    "Delta P": ["Delta P from P Sat", "Delta P from Psat", "Pressure Margin"],
    "FBHP Trend": ["Avg. FBHP Trend (psi/day)", "FBHP Trend (psi/day)", "Average FBHP Trend"],
    "Watercut Input": ["Watercut %", "Water Cut", "Watercut", "Water Cut %"],
    "Agreed Target": ["Agreed Target (stb)", "Agreed Target"],
    "RMT Target": ["RMT Target (stb)", "RMT Target"],
    "Actual Supported Oil": ["Oil Supported (stb)", "Supported Oil (stb)", "Actual Supported Oil (stb)", "Incremental Oil (stb)"],
    "Support Fraction": ["Oil Support Fraction", "Supported Fraction", "Support Factor", "Support Fraction"],
}
REQUIRED = ["Pattern", "Oil", "Water", "Injection"]
NUMERIC_FIELDS = [k for k in FIELDS if k not in ("Pattern", "Reservoir")]
VOL_STB_TO_M3 = 0.158987294928
UNIT_OPTIONS = {
    "stb": (1.0, "stb"),
    "Mstb": (1e-3, "Mstb"),
    "MMstb": (1e-6, "MMstb"),
    "m³": (VOL_STB_TO_M3, "m³"),
    "10³ m³": (VOL_STB_TO_M3 / 1_000, "10³ m³"),
}
DEFAULT_CPS_COLORS = {
    "CPS1": "#1677B9", "CPS2": "#DA6D24", "CPS3": "#159579",
    "CPS4": "#8E62A6", "CPS5": "#C43B5F", "CPS6": "#807026",
    "CPS7": "#18A9B4", "Unassigned": "#8A929C",
}


@dataclass
class Design:
    font: str = "Arial"
    text: str = "#182536"
    paper: str = "#FFFFFF"
    plot: str = "#FFFFFF"
    grid: str = "#E7EDF5"
    title_size: int = 19
    axis_size: int = 14
    tick_size: int = 11
    legend_size: int = 11
    line_width: float = 2.5
    marker_size: int = 13
    opacity: float = 0.83
    height: int = 570
    legend_pos: str = "Bottom"
    legend_x: float = 0.5
    legend_y: float = -0.27
    legend_anchor: str = "center"
    grid_on: bool = True


def keyify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def col_lookup(frame: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    dic = {keyify(x): x for x in frame.columns}
    for term in aliases:
        if keyify(term) in dic:
            return dic[keyify(term)]
    return None


# Canonical labels used only for recognising a misnamed reservoir column.
# Other reservoir names are preserved when the workbook contains a valid header.
RESERVOIR_LABELS = {
    "mishrif": "Mishrif",
    "uppershale": "Upper Shale",
    "mainpay": "Main Pay",
}


def canonical_reservoir(value: object) -> str:
    if pd.isna(value) or not str(value).strip():
        return "Unspecified"
    original = str(value).strip()
    return RESERVOIR_LABELS.get(keyify(original), original)


def detect_reservoir_column(frame: pd.DataFrame) -> Optional[str]:
    """Find a mislabeled reservoir column by values, not just the header.

    Some Rumaila files have B1='Mishrif' instead of B1='Reservoir', although
    the cells below contain Mishrif, Upper Shale and Main Pay. Requiring a
    strong concentration of *known reservoir labels* avoids mistaking Pattern
    names or other text columns for a reservoir.
    """
    candidates: list[tuple[float, int, str]] = []
    for col in frame.columns:
        if keyify(col) in {"pattern", "patternname", "patternid"}:
            continue
        values = frame[col].dropna()
        if values.empty:
            continue
        strings = values.astype(str).str.strip()
        strings = strings.loc[strings.ne("")]
        if len(strings) < 3:
            continue
        recognized = strings.map(lambda v: keyify(v) in RESERVOIR_LABELS)
        confidence = float(recognized.mean())
        known_kinds = strings.loc[recognized].map(keyify).nunique()
        is_reservoir_header = keyify(col) in RESERVOIR_LABELS
        if confidence >= 0.80 and (is_reservoir_header or known_kinds >= 2):
            candidates.append((confidence, int(recognized.sum()), str(col)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (series.astype("string").str.replace(",", "", regex=False)
               .str.replace("%", "", regex=False)
               .str.replace("−", "-", regex=False).str.strip())
    return pd.to_numeric(cleaned, errors="coerce")


def get_cps(pattern: str) -> tuple[str, str]:
    text = str(pattern).upper()
    match = re.search(r"(?<![A-Z0-9])CPS[\s_-]*(\d+)(?!\d)", text)
    if match:
        return f"CPS{int(match.group(1))}", ""
    # One known typo in the October source: B-CSP4-i9-R-856.
    typo = re.search(r"(?<![A-Z0-9])CSP[\s_-]*(\d+)(?!\d)", text)
    if typo:
        return f"CPS{int(typo.group(1))}", "CSP/CPS spelling normalised; check source"
    return "Unassigned", "CPS not found in pattern name"


@st.cache_data(show_spinner=False)
def sheet_names(file_bytes: bytes, extension: str) -> list[str]:
    if extension == "csv":
        return ["CSV"]
    excel = pd.ExcelFile(io.BytesIO(file_bytes))
    return excel.sheet_names


@st.cache_data(show_spinner=False)
def read_sheet(file_bytes: bytes, extension: str, name: str) -> pd.DataFrame:
    if extension == "csv":
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=name)


def standardise(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    # Detect before discarding unnamed columns: the reservoir header is
    # sometimes overwritten (e.g., B1='Mishrif') or left blank in Excel.
    matched = {k: col_lookup(raw, aliases) for k, aliases in FIELDS.items()}
    if matched["Reservoir"] is None:
        matched["Reservoir"] = detect_reservoir_column(raw)
    missing = [k for k in REQUIRED if matched[k] is None]
    if missing:
        return pd.DataFrame(), missing, {k: v for k, v in matched.items() if v}
    df = pd.DataFrame(index=raw.index)
    for key in FIELDS:
        if matched[key]:
            df[key] = raw[matched[key]]
        else:
            # Missing Reservoir must never silently become Mishrif.
            df[key] = "Unspecified" if key == "Reservoir" else np.nan
    df["Pattern"] = df["Pattern"].astype("string").str.strip()
