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
    "Manifold": ["Manifold", "Manifold Name", "Injection Manifold", "Production Manifold", "Manifold ID", "Manifold Code"],
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
NUMERIC_FIELDS = [k for k in FIELDS if k not in ("Pattern", "Reservoir", "Manifold")]
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
