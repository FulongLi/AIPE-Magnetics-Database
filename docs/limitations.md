# Current limitations

* `design_transformer` is analytical sizing with the `P / V` square-current approximation; the
  waveform-informed design adds exact waveform RMS / harmonics, 1-D leakage, per-layer Dowell and a
  network / lumped thermal estimate, still without finite-element, insulation or manufacturing analysis
  (the v0.3 FEA / thermal-FE workflow layer is present but not yet released). Results are design
  estimates with explicit assumptions.
* Core-loss models need a parameter set covering the operating temperature, frequency and flux
  density; the engine refuses to extrapolate. Steinmetz/iGSE ignore DC bias and minor loops (flagged).
* Datasheet parameters are transcribed for N87 only; other catalogue materials carry MagNet-derived
  loss parameters but no Bsat/Curie data yet (reported as missing, not guessed).
* Jiles-Atherton ships an illustrative parameter set, not a material fit. Data-driven inference is
  demonstrated on a synthetic reference; real challenge models need the optional `magnethub` package.
* Verification checks are L1/L2 (software, numerical and physical consistency). No L3 experimental
  validation is bundled; optional external tests exist for users with the source datasets.
* The bundled evidence file `src/aipe_magnetics/verification/evidence/v0.2.0.json` is written by the
  maintainer's release step (`verify --update-bundled`); until then every model-based result reports
  `L0 unverified` with a statement naming the missing file, and `verify --check-bundled` exits 1.
* The repository licence is `NOASSERTION` pending the owner's decision ([LICENSING.md](../LICENSING.md)).

Known limitations of the unreleased waveform-informed capability are listed in the
[CHANGELOG](../CHANGELOG.md) ("Unreleased").
