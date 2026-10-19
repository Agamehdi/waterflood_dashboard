# Pattern Waterflood Analytics — Streamlit

Interactive, upload-driven Streamlit app for Rumaila Pattern / CPS waterflood surveillance.

## Deploy to GitHub + Streamlit Cloud

1. Create a GitHub repository. **Upload `streamlit_app.py` and `requirements.txt` to the repository root** (`README.md` is optional).
2. In Streamlit Community Cloud, select the repository, branch, and **main file: `streamlit_app.py`**.
3. Deploy. Open the app and use **Browse** to upload `Pattern_Data_Oct2026.xlsx`.
4. Keep field datasets **out of public repositories**. The uploaded workbook is not bundled into the app.

### Local run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## What is calculated

- Technical Value of Water (VoW): `Oil Supported (stb) / Injected Water (stb)`.
- CPS VoW: **sum of supported oil across Patterns / sum of injection across Patterns** (NOT average of Pattern ratios).
- Optional economic VoW: `technical VoW × oil netback (USD/stb oil) − injection variable cost (USD/stb water)`.
- Oil Supported is not in the original October file. The default `Oil Supported = Produced Oil` is an **explicit screening proxy**, not causal/incremental oil due to injection. An optional upload column `Oil Supported (stb)` is used when available; or apply a stated support fraction.
- Display raw 3-month total volumes or average volume per month; choose stb, Mstb, MMstb, m³, 10³m³. Ratios do not change with consistent units or averaging.
- Zero-injection Patterns have undefined VoW and stay visible under QC. For Produced Oil proxy modes the default excludes their oil from the supported-oil numerator; this assumption is user-configurable.
- Name-based CPS detection handles `A-CPS2-...`, `CPS-1-N`, and flags `B-CSP4-...` for review.
- Pressure and VRR data are imported from source. Averaged source VRR is *not* a proper recalculated reservoir-condition CPS VRR.

## Chart / analysis features

- Interactive **Value of Water (X)** vs **Oil Supported (Y)** cross-plot: by CPS or Pattern, custom labels, bubble sizes, trend line, median quadrants.
- CPS volumes / watercut, Pattern and CPS ranking, surveillance plots and customizable X/Y crossplot lab.
- Global style tools: font family/sizes, marker opacity/size, line width, plot and paper background colors, series colors, grid, legend positions (including custom X and Y), plot height.
- Downloads: Pattern CSV, CPS CSV, Excel with calculation methodology + QC, interactive cross-plot HTML; Plotly chart toolbar supports PNG exports.

## Notes

Data fields `Produced Oil`, `Produced Water`, and `Injected Water` should all be total volumes over the **same period**; current workbook is described as three months. Derived economics uses operator-supplied netback and cost assumptions. Waterflood response requires connectivity/lag/decline analysis before interpreting observed production as attributable to injection.

Technical background: [SPE mature-flood surveillance](https://jpt.spe.org/mature-flood-surveillance-using-streamlines), [SPE surveillance plots](https://jpt.spe.org/twa/role-surveillance-plots-diagnosing-waterfloods).
