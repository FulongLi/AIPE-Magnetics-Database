/* AIPE Magnetics Data Explorer: read-only client over the JSON indexes written by tools/build_site.py.
   Hash routing (works under the GitHub Pages sub-path), data loaded on demand, no external requests. */
(function () {
  "use strict";
  var app = document.getElementById("app");
  var cache = {};
  var PAGE = 100;

  // -- data ------------------------------------------------------------------------------------------------
  function load(path) {
    if (!cache[path]) {
      cache[path] = fetch(path).then(function (r) {
        if (!r.ok) throw new Error("Could not load " + path + " (HTTP " + r.status + ")");
        return r.json();
      });
      cache[path].catch(function () { delete cache[path]; });
    }
    return cache[path];
  }
  var file = function (id) { return encodeURIComponent(id) + ".json"; };

  // -- vocabulary --------------------------------------------------------------------------------------------
  var CLASS_LABEL = { ferrite_mnzn: "MnZn ferrite", ferrite_nizn: "NiZn ferrite", powder_alloy: "Powder alloy", nanocrystalline: "Nanocrystalline",
    amorphous: "Amorphous", silicon_steel: "Silicon steel", other: "Other" };
  var DATA_CLASS = {
    experimental_measurement: ["Experimental measurement", "exp", "Measured on a test bench by the source laboratory."],
    manufacturer_datasheet: ["Manufacturer datasheet", "ds", "Typical values or curves published by the manufacturer, often digitised by a third party."],
    manufacturer_tool_export: ["Manufacturer tool export", "ds", "Exported from a manufacturer's design tool; not an independent measurement."],
    derived_from_measurement: ["Derived from measurement", "der", "Computed from another table of the same experiment; not counted as independent."],
    fitted_parameters: ["Fitted parameters", "fit", "Model coefficients fitted to data; not measurements."],
    aggregated_literature: ["Aggregated literature", "der", "Values collected from publications."],
    identity: ["Identity record", "", "Names the material in this source; carries no typed values."]
  };
  var QTY = { core_loss: "Core loss", mass_core_loss: "Core loss per mass", bh_curve: "B-H curve", complex_permeability: "Complex permeability",
    complex_permittivity: "Complex permittivity", initial_permeability: "Initial permeability", amplitude_permeability: "Amplitude permeability",
    loss_factor: "Loss factor", dielectric_properties: "Dielectric properties", energy_loss: "Energy loss per cycle", energy_loss_and_permeability: "Energy loss and permeability" };
  var SOURCE = { openmagnetics_mas: "OpenMagnetics MAS", princeton_magnet: "Princeton MagNet", paderborn_lea_materialdatabase: "Paderborn LEA",
    zenodo_8292861_hefmag: "HEFMAG (Zenodo 8292861)", zenodo_21626116_smc_ferrite_triangular: "SMC/ferrite triangular (Zenodo 21626116)",
    aipe_curated: "AIPE curated catalogue" };
  var PROP = [["initial_permeability", "Initial permeability μi"], ["saturation_flux_density", "Saturation flux density Bsat"],
    ["remanence", "Remanence Br"], ["coercivity", "Coercivity Hc"], ["curie_temperature", "Curie temperature Tc"], ["density", "Density"],
    ["resistivity", "Resistivity"], ["recommended_frequency_range", "Recommended frequency range"],
    ["permeability_vs_temperature", "Permeability vs temperature"]];
  var COND = [["temperature", "T"], ["frequency", "f"], ["field", "H"], ["flux_density_peak", "B̂"], ["dc_bias_field", "H_DC"]];
  var STATUS = { primary: "Primary", duplicate: "Duplicate", derived_view: "Derived view" };
  var PUBLISHED = { full: "All rows", sample: "Sampled preview", none: "Not published" };
  var NA = '<span class="na" title="Not reported by the source">—</span>';
  var COLORS = [1, 2, 3, 4, 5, 6, 7, 8].map(function (i) { return "var(--series-" + i + ")"; });
  var SEQ = [1, 2, 3, 4, 5].map(function (i) { return "var(--seq-" + i + ")"; }); // one hue, light to dark: ordered groups
  // display name, display unit and conversion of table columns (stored values stay SI; only the chart converts)
  var COLDEF = { frequency: ["Frequency f"], temperature: ["Temperature T"], flux_density_peak: ["Peak flux density B̂"],
    flux_density_peak_to_peak: ["Flux density swing ΔB"], flux_density: ["Flux density B"], field: ["Magnetic field H"],
    dc_bias_field: ["DC bias field H_DC"], loss_density: ["Core loss Pv"], mass_loss_density: ["Core loss per mass"],
    energy_density_per_cycle: ["Energy loss per cycle"], mu_real: ["μ′"], mu_imag: ["μ″"], eps_real: ["ε′"], eps_imag: ["ε″"],
    relative_permeability: ["Relative permeability μr"], amplitude_permeability: ["Amplitude permeability μa"],
    loss_factor: ["Loss factor"], duty: ["Duty cycle"] };
  function ident(v) { return v; }
  function colInfo(c) {
    var label = (COLDEF[c.name] || [c.name])[0], u = c.unit, conv = ident;
    if (u === "K") { u = "°C"; conv = function (v) { return v - 273.15; }; }
    else if (u === "T") { u = "mT"; conv = function (v) { return v * 1e3; }; }
    return { name: c.name, label: label, unit: u, conv: conv, axis: label + (unit(u) ? " [" + unit(u) + "]" : "") };
  }
  function colLabel(n) { return (COLDEF[n] || [n])[0]; }

  // -- formatting --------------------------------------------------------------------------------------------
  function esc(s) {
    return String(s === null || s === undefined ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function num(v, digits) {
    if (v === null || v === undefined) return null;
    if (typeof v !== "number") return String(v);
    if (v === 0) return "0";
    var a = Math.abs(v);
    if (a >= 1e6 || a < 1e-3) return v.toExponential((digits || 4) - 1).replace(/\.?0+e/, "e");
    return String(parseFloat(v.toPrecision(digits || 4)));
  }
  var UNIT = { "W/m^3": "W/m³", "kg/m^3": "kg/m³", "J/m^3": "J/m³", "Ohm*m": "Ω·m", "1": "", "degC": "°C", "m^2": "m²", "m^3": "m³" };
  function unit(u) { return u in UNIT ? UNIT[u] : (u || ""); }
  var PREFIXED = { Hz: 1, "W/m^3": 1, "W/kg": 1, "J/m^3": 1, "A/m": 1 };
  function qty(v, u) {
    if (v === null || v === undefined) return "—";
    if (typeof v !== "number") return String(v);
    if (u === "K") return num(v, 5) + " K (" + num(v - 273.15) + " °C)";
    if (PREFIXED[u]) {
      var a = Math.abs(v), p = [[1e9, "G"], [1e6, "M"], [1e3, "k"]];
      for (var i = 0; i < p.length; i++) if (a >= p[i][0]) return num(v / p[i][0]) + " " + p[i][1] + unit(u);
    }
    return (num(v) + " " + unit(u)).trim();
  }
  function qtyHtml(v, u, exact) {
    if (v === null || v === undefined) return NA;
    return '<span title="' + esc(exact === false ? "" : "stored value: " + v + " " + (u || "")) + '">' + esc(qty(v, u)) + "</span>";
  }
  function qobj(o) { // {value, unit, original}
    if (!o || o.value === undefined || o.value === null) return NA;
    var s = qtyHtml(o.value, o.unit);
    if (o.original && o.original.unit && o.original.unit !== o.unit && o.original.unit !== "degC")
      s += ' <span class="muted small">(published as ' + esc(num(o.original.value) + " " + unit(o.original.unit)) + ")</span>";
    return s;
  }
  function tick(u) { // axis ticks: number with SI prefix only (the unit is in the axis title); tooltips: full quantity
    return function (v, isTick) {
      if (!isTick) return qty(v, u);
      if (PREFIXED[u]) {
        var a = Math.abs(v), p = [[1e9, "G"], [1e6, "M"], [1e3, "k"]];
        for (var i = 0; i < p.length; i++) if (a >= p[i][0]) return num(v / p[i][0]) + p[i][1];
      }
      return num(v);
    };
  }
  function int(n) { return (n === null || n === undefined) ? "—" : Number(n).toLocaleString("en-GB"); }
  function dcBadge(dc) { var d = DATA_CLASS[dc] || [dc, "", ""]; return '<span class="badge ' + d[1] + '" title="' + esc(d[2]) + '">' + esc(d[0]) + "</span>"; }
  function flags(list) { return (list || []).map(function (f) { return '<span class="badge flag">' + esc(f) + "</span>"; }).join(" "); }
  function src(id) { return SOURCE[id] || id; }
  function shortTemp(k) { return num(k - 273.15) + " °C"; }
  function range(key, c) {
    if (!c) return "";
    var lo = c.min, hi = c.max, f;
    if (key === "temperature") f = function (v) { return num(v - 273.15); };
    else if (key === "flux_density_peak") f = function (v) { return num(v * 1e3); };
    else f = function (v) { return qty(v, c.unit).replace(/\s?\S+$/, ""); };
    var u = key === "temperature" ? " °C" : key === "flux_density_peak" ? " mT" : key === "frequency" ? " " + qty(hi, "Hz").split(" ")[1] : " " + unit(c.unit);
    if (key === "frequency") { var a = qty(lo, "Hz"), b = qty(hi, "Hz"); return lo === hi ? a : a + " – " + b; }
    return (lo === hi ? f(lo) : f(lo) + " – " + f(hi)) + u;
  }
  function conditions(c) {
    var out = [];
    [["frequency", "f"], ["temperature", "T"], ["flux_density_peak", "B̂"], ["dc_bias_field", "H_DC"], ["duty", "duty"]].forEach(function (k) {
      if (c && c[k[0]]) out.push(k[1] + " " + range(k[0], c[k[0]]));
    });
    if (c && c.waveforms) out.push(c.waveforms.join(", "));
    return out.length ? esc(out.join(" · ")) : '<span class="na">no condition columns</span>';
  }
  function setTitle(t) { document.title = (t ? t + " · " : "") + "AIPE Magnetics Data Explorer"; }

  // -- routing -----------------------------------------------------------------------------------------------
  function route() {
    var raw = location.hash.replace(/^#\/?/, ""), i = raw.indexOf("?");
    var path = i < 0 ? raw : raw.slice(0, i), qs = i < 0 ? "" : raw.slice(i + 1);
    return { parts: path.split("/").filter(Boolean).map(decodeURIComponent), q: new URLSearchParams(qs) };
  }
  function writeQuery(base, state) {
    var s = new URLSearchParams();
    Object.keys(state).forEach(function (k) { var v = state[k]; if (v !== "" && v !== null && v !== undefined && v !== false) s.set(k, v === true ? "1" : v); });
    var h = "#/" + base + (s.toString() ? "?" + s.toString() : "");
    if (location.hash !== h) history.replaceState(null, "", h);
  }
  function nav(section) {
    document.querySelectorAll("[data-nav]").forEach(function (a) {
      if (a.getAttribute("data-nav") === section) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
  }
  function fail(err) {
    app.innerHTML = '<div class="notice"><strong>Could not show this page.</strong> ' + esc(err && err.message || err) +
      '</div><p><a href="#/">Back to the overview</a></p>';
  }
  var renderToken = 0;
  function render() {
    var r = route(), p = r.parts, token = ++renderToken, view;
    var section = { compare: "compare", materials: "materials", material: "materials", measurements: "measurements", measurement: "measurements", cores: "cores", core: "cores" }[p[0]] || "overview";
    nav(section);
    if (!p.length) view = overview();
    else if (p[0] === "materials") view = materials(r.q);
    else if (p[0] === "material" && p[1]) view = material(p[1]);
    else if (p[0] === "measurements") view = measurements(r.q);
    else if (p[0] === "measurement" && p[1]) view = measurement(p[1]);
    else if (p[0] === "compare") view = compare(r.q);
    else if (p[0] === "cores") view = cores(r.q);
    else if (p[0] === "core" && p[1]) view = core(p[1]);
    else view = Promise.reject(new Error("Unknown page."));
    var slow = setTimeout(function () { if (token === renderToken) app.innerHTML = '<p class="loading" role="status">Loading data…</p>'; }, 150);
    Promise.resolve(view).finally(function () { clearTimeout(slow); })
      .then(function (fn) { if (token === renderToken && typeof fn === "function") { fn(); window.scrollTo(0, 0); } })
      .catch(function (e) { if (token === renderToken) fail(e); });
  }
  window.addEventListener("hashchange", render);

  // -- overview ----------------------------------------------------------------------------------------------
  function hubNotice(h) {
    if (!h) return "";
    if (h.published) return '<p class="small">Original files and complete tables: <a href="' + esc(h.url) + '" rel="noopener">Hugging Face dataset ' +
      esc(h.repo_id) + '</a> at commit <code>' + esc(String(h.commit).slice(0, 12)) + "</code> (" + int(h.files) + " files, SHA-256 listed in manifests/checksums.json).</p>";
    return '<p class="small muted">The <a href="' + esc(h.url) + '" rel="noopener">Hugging Face dataset</a> has no published release recorded yet; ' +
      "the data shown here come from the repository.</p>";
  }
  function hubCell(h) {
    if (!h) return NA;
    if (h.published) return '<a href="' + esc(h.raw) + '" rel="noopener">raw</a> · <a href="' + esc(h.normalized) + '" rel="noopener">normalised</a>';
    return '<span class="muted">' + esc(h.eligible ? "pending first release" : "not published") + "</span>";
  }
  function overview() {
    setTitle("");
    return Promise.all([load("data/summary.json"), load("data/sources.json")]).then(function (d) {
      var s = d[0], sources = d[1], c = s.counts;
      return function () {
        var integrated = sources.filter(function (x) { return x.status === "integrated"; });
        var other = sources.filter(function (x) { return x.status !== "integrated"; });
        app.innerHTML =
          '<section class="hero"><p class="eyebrow">AIPE Hub · Magnetics data</p>' +
          '<h1 class="display">Magnetics Data Explorer</h1>' +
          '<p class="lead">A read-only view of the AIPE Magnetics Database: soft-magnetic material properties, core geometries and ' +
          'measured core loss, permeability, permittivity and B-H data for power electronics. Every value keeps its source, ' +
          'licence and data class. Records are standardised in AIPE Magnetic Data Schema v' + esc(s.schema_version) + ' in SI units.</p>' +
          '<form class="search-pill" id="qs" role="search"><input type="search" name="q" placeholder="Search materials, e.g. N87, 3C90, MPP" aria-label="Search materials">' +
          '<button type="submit">Search</button></form>' +
          '<p class="try"><span>Try:</span>' + ["N87", "3C90", "3F4", "MPP", "Nanoperm", "Vitroperm"].map(function (q) {
            return '<a class="chip" href="#/materials?q=' + encodeURIComponent(q) + '">' + esc(q) + "</a>"; }).join("") + "</p></section>" +
          '<div class="stats">' +
          stat("#/materials", int(c.canonical_materials), "Materials", int(c.source_material_records) + " source records consolidated; " + int(c.canonical_materials_with_multiple_sources) + " combine several sources") +
          stat("#/cores", int(c.core_shapes + c.core_parts), "Core geometries", int(c.core_shapes) + " standard shapes, " + int(c.core_parts) + " manufacturer parts") +
          stat("#/measurements", int(c.measurement_records), "Measurement tables", int(c.independent_measurement_records) + " independent; " + int(c.table_rows_total) + " rows, of which " + int(c.experimental_rows_independent) + " experimental") +
          stat("#sources", int(integrated.length), "Data sources", "integrated, of " + int(c.sources_investigated) + " investigated; " + int(c.fitted_loss_models) + " fitted loss models kept separately") +
          '</div>' +
          '<h2>Material map</h2><p>Every material that reports an initial permeability and a saturation flux density, coloured by class. ' +
          '<a href="#/materials?view=map">Filter the map</a> or <a href="#/compare">compare materials</a>.</p><div id="ov-map"><p class="loading" role="status">Loading materials…</p></div>' +
          '<h2>Data classes</h2><p>Values are never mixed across classes. Each table and value is labelled with one of:</p>' +
          '<div class="table-wrap"><table><thead><tr><th>Class</th><th>Meaning</th><th class="num">Tables</th></tr></thead><tbody>' +
          ["experimental_measurement", "manufacturer_datasheet", "manufacturer_tool_export", "derived_from_measurement", "fitted_parameters"].map(function (k) {
            var n = k === "fitted_parameters" ? c.fitted_loss_models + " models" : k === "derived_from_measurement" ? int(c.derived_view_records) : int(s.data_classes[k] || 0);
            return "<tr><td>" + dcBadge(k) + "</td><td>" + esc(DATA_CLASS[k][2]) + '</td><td class="num">' + esc(n) + "</td></tr>";
          }).join("") + "</tbody></table></div>" +
          '<p class="small muted">Of ' + int(c.measurement_records) + " tables, " + int(c.tables_published_full) + " are published here in full, " +
          int(c.tables_published_sample) + " as labelled previews of up to " + int(s.preview_rows) + " rows (full table linked), and " +
          int(c.tables_not_published) + " large regenerated tables are listed without rows.</p>" +
          '<h2 id="sources">Data sources</h2>' + hubNotice(s.huggingface) +
          '<div class="table-wrap"><table><thead><tr><th>Source</th><th>Licence</th><th>Version</th>' +
          '<th class="num">Records</th><th class="num">Table rows</th><th>Full dataset</th><th>Notes</th></tr></thead><tbody>' +
          integrated.map(function (x) {
            var recs = x.records ? Object.keys(x.records).reduce(function (n, k) { return n + x.records[k]; }, 0) : null;
            return "<tr><td>" + (x.url ? '<a href="' + esc(x.url) + '" rel="noopener">' + esc(x.title) + "</a>" : esc(x.title)) +
              "</td><td>" + esc(x.license) + (x.licence_files && x.licence_files.length ? "<br>" + x.licence_files.map(function (u) {
                return '<a class="small" href="' + esc(u) + '">' + esc(u.split("/").pop()) + "</a>"; }).join(" · ") : "") +
              '</td><td class="mono small">' + esc(String(x.version || "").slice(0, 12)) + '</td><td class="num">' + (recs === null ? NA : int(recs)) +
              '</td><td class="num">' + (rowsTotal(x.table_rows) === null ? NA : int(rowsTotal(x.table_rows))) + '</td><td class="small">' + hubCell(x.huggingface) +
              '</td><td class="small">' + esc(x.notes || x.redistribution || "") + "</td></tr>";
          }).join("") + "</tbody></table></div>" +
          "<details><summary>" + other.length + " further sources investigated: reference entries only, no data available here or on Hugging Face</summary>" +
          '<div class="table-wrap"><table><thead><tr><th>Source</th><th>Status</th><th>Licence</th><th>Reason</th></tr></thead><tbody>' +
          other.map(function (x) {
            return "<tr><td>" + (x.url ? '<a href="' + esc(x.url) + '" rel="noopener">' + esc(x.title) + "</a>" : esc(x.title)) + "</td><td>" + esc(x.status) +
              "</td><td>" + esc(x.license || "") + '</td><td class="small">' + esc(x.redistribution || x.notes || "") + "</td></tr>";
          }).join("") + "</tbody></table></div></details>" +
          '<p class="small muted">Full catalogue and licensing: <a href="' + esc(s.repository) + '/blob/main/docs/source-catalogue.md">source catalogue</a>, ' +
          '<a href="' + esc(s.repository) + '/blob/main/LICENSING.md">LICENSING.md</a>, <a href="' + esc(s.repository) +
          '/blob/main/docs/provenance-and-limitations.md">provenance and limitations</a>.</p>' +
          '<h2>Programmatic access</h2><div class="card"><p>The same records are served to software and engineering agents by the Python API, ' +
          'the <code>aipe-magnetics-db</code> CLI and a read-only MCP server for Claude Code and Codex.</p>' +
          '<pre class="json">pip install "aipe-magnetics-database[mcp] @ git+https://github.com/FulongLi/AIPE-Magnetics-Database.git"\naipe-magnetics-db materials get "TDK N87"</pre>' +
          '<p class="small"><a href="' + esc(s.repository) + '#readme">README</a> · <a href="' + esc(s.repository) + '/blob/main/docs/api.md">API, CLI and MCP reference</a> · ' +
          '<a href="' + esc(s.repository) + '/blob/main/docs/data-format.md">data format</a></p></div>';
        var mapHost = document.getElementById("ov-map");
        load("data/materials.json").then(function (list) { if (mapHost.isConnected) materialMap(mapHost, list); })
          .catch(function () { if (mapHost.isConnected) mapHost.innerHTML = '<p class="muted">The material map could not be loaded.</p>'; });
        document.getElementById("qs").addEventListener("submit", function (e) {
          e.preventDefault();
          location.hash = "#/materials?q=" + encodeURIComponent(new FormData(e.target).get("q") || "");
        });
      };
    });
  }
  function rowsTotal(t) { // table_rows is a per-quantity map in the source index
    if (t === undefined || t === null) return null;
    if (typeof t === "number") return t;
    return Object.keys(t).reduce(function (n, k) { return n + (Number(t[k]) || 0); }, 0);
  }
  function stat(href, n, label, sub) {
    return '<a class="stat" href="' + href + '"><div class="label">' + esc(label) + '</div><div class="num">' + esc(n) + '</div><div class="sub">' + esc(sub) + "</div></a>";
  }

  // -- material map -------------------------------------------------------------------------------------------
  var CLASS_ORDER = ["ferrite_mnzn", "ferrite_nizn", "powder_alloy", "nanocrystalline", "amorphous", "silicon_steel", "other"]; // fixed colour per class
  function materialMap(host, rows) {
    var pts = rows.filter(function (m) { return m.mu_i && m.bsat && m.mu_i.v > 0 && m.bsat.v > 0; });
    if (!pts.length) { host.className = ""; host.innerHTML = '<div class="notice info">None of these materials reports both an initial permeability and a saturation flux density.</div>'; return; }
    var series = CLASS_ORDER.map(function (c, k) {
      return { label: CLASS_LABEL[c], color: COLORS[k], mode: "points", points: pts.filter(function (m) {
        return c === "other" ? CLASS_ORDER.indexOf(m["class"]) < 0 || m["class"] === "other" : m["class"] === c; }).map(function (m) { return [m.mu_i.v, m.bsat.v * 1e3, m]; }) };
    }).filter(function (s) { return s.points.length; });
    MiniPlot.render(host, {
      ariaLabel: "Saturation flux density versus initial permeability of " + pts.length + " materials",
      series: series,
      x: { label: "Initial permeability μi", log: true, format: tick("1") },
      y: { label: "Saturation flux density Bsat [mT]", log: true, format: tick("mT") },
      note: int(pts.length) + " of " + int(rows.length) + " materials report both values. Bsat is the value nearest 25 °C; its temperature and field differ between sources. Click a point to open the material.",
      title: function (m) { return m.name; },
      extra: function (m) {
        return (m.bsat.T !== undefined ? '<br><span class="muted">Bsat at ' + esc(shortTemp(m.bsat.T)) + "</span>" : "") +
          '<br><span class="muted">' + (m.measurements ? int(m.measurements) + (m.measurements === 1 ? " table" : " tables") + (m.experimental ? ", " + m.experimental + " measured" : "") : "no measurement table") + "</span>";
      },
      onPick: function (m) { location.hash = "#/material/" + encodeURIComponent(m.id); }
    });
  }

  // -- materials list ----------------------------------------------------------------------------------------
  function headCell(h, key) { return h ? (h.u === "K" ? '<span title="stored value: ' + h.v + ' K">' + esc(num(h.v - 273.15)) + " °C</span>" : qtyHtml(h.v, h.u)) + (h.T !== undefined ? ' <span class="muted small">@ ' + esc(shortTemp(h.T)) + "</span>" : "") +
    (h.note ? ' <span class="muted small" title="' + esc(h.note) + '">ⓘ</span>' : "") : NA; }

  function materials(q) {
    setTitle("Materials");
    return load("data/materials.json").then(function (list) {
      return function () {
        var state = { q: q.get("q") || "", cls: q.get("class") || "", mfr: q.get("manufacturer") || "", source: q.get("source") || "",
          exp: q.get("experimental") === "1", sort: q.get("sort") || "name", view: q.get("view") === "map" ? "map" : "" };
        var limit = PAGE;
        var mfrs = uniq(list.map(function (m) { return m.manufacturer; }));
        var classes = uniq(list.map(function (m) { return m.class; }));
        var sources = uniq([].concat.apply([], list.map(function (m) { return m.sources; })));
        app.innerHTML = '<h1>Materials</h1><p class="lead">Canonical materials consolidated across sources. Typed values come unchanged ' +
          'from one source record; every member record and its values are shown on the material page. ' + NA + ' means the value is not reported (never zero).</p>' +
          '<div class="filters" role="search">' +
          '<label class="wide">Search name, grade or manufacturer<input type="search" id="f-q" value="' + esc(state.q) + '" placeholder="e.g. N87, 3C95, Kool Mu"></label>' +
          select("f-class", "Material class", classes, state.cls, function (v) { return CLASS_LABEL[v] || v; }) +
          select("f-mfr", "Manufacturer", mfrs, state.mfr) +
          select("f-source", "Source", sources, state.source, src) +
          '<label class="check"><input type="checkbox" id="f-exp"' + (state.exp ? " checked" : "") + "> Has experimental data</label>" +
          '</div><div class="tabs" role="tablist" aria-label="View"><button type="button" role="tab" data-view="">Table</button>' +
          '<button type="button" role="tab" data-view="map">Map: Bsat vs μi</button></div>' +
          "<p id=\"count\" class=\"muted small\" aria-live=\"polite\"></p><div id=\"list\"></div><div class=\"more\" id=\"more\"></div>";
        document.querySelectorAll("#app [data-view]").forEach(function (b) {
          b.addEventListener("click", function () { state.view = b.getAttribute("data-view"); update(); });
        });
        var SORTS = { name: null, mu_i: "mu_i", bsat: "bsat", rho: "rho", density: "density", tc: "tc", measurements: "measurements" };
        function update() {
          writeQuery("materials", { q: state.q, "class": state.cls, manufacturer: state.mfr, source: state.source, experimental: state.exp,
            sort: state.sort === "name" ? "" : state.sort, view: state.view });
          document.querySelectorAll("#app [data-view]").forEach(function (b) { b.setAttribute("aria-selected", b.getAttribute("data-view") === state.view ? "true" : "false"); });
          var needle = state.q.trim().toLowerCase();
          var rows = list.filter(function (m) {
            if (state.cls && m.class !== state.cls) return false;
            if (state.mfr && m.manufacturer !== state.mfr) return false;
            if (state.source && m.sources.indexOf(state.source) < 0) return false;
            if (state.exp && !m.experimental) return false;
            if (needle) {
              var hay = [m.name, m.grade, m.manufacturer, m.id, CLASS_LABEL[m.class] || m.class].join(" ").toLowerCase();
              return needle.split(/\s+/).every(function (t) { return hay.indexOf(t) >= 0; });
            }
            return true;
          });
          var key = SORTS[state.sort];
          if (key) rows = rows.slice().sort(function (a, b) {
            var x = key === "measurements" ? a.measurements : a[key] && a[key].v, y = key === "measurements" ? b.measurements : b[key] && b[key].v;
            if (x === undefined || x === null) return 1; if (y === undefined || y === null) return -1; return y - x;
          });
          document.getElementById("count").textContent = int(rows.length) + " of " + int(list.length) + " materials" + (key ? ", sorted by " + state.sort + " (descending, missing last)" : "");
          if (state.view === "map") {
            document.getElementById("count").textContent = int(rows.length) + " of " + int(list.length) + " materials match the filters.";
            document.getElementById("more").innerHTML = "";
            var host = document.getElementById("list");
            host.innerHTML = '<div id="mat-map"></div>';
            materialMap(host.firstChild, rows);
            return;
          }
          var th = function (k, label, cls) { return '<th class="' + (cls || "") + '"><button class="sort" data-sort="' + k + '" aria-pressed="' + (state.sort === k) + '">' + label + (state.sort === k ? " ▾" : "") + "</button></th>"; };
          document.getElementById("list").innerHTML = rows.length ? '<div class="table-wrap"><table><thead><tr>' + th("name", "Material") +
            '<th class="hide-sm">Class</th>' + th("mu_i", "μi", "num") + th("bsat", "Bsat", "num") + th("rho", "ρ", "num hide-sm") + th("density", "Density", "num hide-sm") +
            th("tc", "Tc", "num hide-sm") + th("measurements", "Tables", "num") + '<th class="hide-sm">Sources</th></tr></thead><tbody>' +
            rows.slice(0, limit).map(function (m) {
              return '<tr><td><a href="#/material/' + encodeURIComponent(m.id) + '">' + esc(m.name) + "</a></td>" +
                '<td class="hide-sm">' + esc(CLASS_LABEL[m.class] || m.class || "") + "</td>" +
                '<td class="num">' + headCell(m.mu_i) + '</td><td class="num">' + headCell(m.bsat) + '</td><td class="num hide-sm">' + headCell(m.rho) +
                '</td><td class="num hide-sm">' + headCell(m.density) + '</td><td class="num hide-sm">' + headCell(m.tc) + '</td><td class="num">' +
                (m.measurements ? int(m.measurements) : NA) + (m.experimental ? ' <span class="badge exp" title="experimental tables">' + m.experimental + " exp</span>" : "") +
                '</td><td class="hide-sm small">' + m.sources.map(src).map(esc).join("<br>") + "</td></tr>";
            }).join("") + "</tbody></table></div>" : '<div class="notice info">No material matches these filters.</div>';
          document.getElementById("more").innerHTML = rows.length > limit ? '<button class="secondary" id="more-btn">Show ' + Math.min(PAGE, rows.length - limit) + " more</button>" : "";
          var mb = document.getElementById("more-btn");
          if (mb) mb.addEventListener("click", function () { limit += PAGE; update(); });
          document.querySelectorAll("button.sort").forEach(function (b) { b.addEventListener("click", function () { state.sort = b.getAttribute("data-sort"); update(); }); });
        }
        bind("f-q", "input", function (v) { state.q = v; limit = PAGE; update(); });
        bind("f-class", "change", function (v) { state.cls = v; limit = PAGE; update(); });
        bind("f-mfr", "change", function (v) { state.mfr = v; limit = PAGE; update(); });
        bind("f-source", "change", function (v) { state.source = v; limit = PAGE; update(); });
        document.getElementById("f-exp").addEventListener("change", function (e) { state.exp = e.target.checked; limit = PAGE; update(); });
        update();
      };
    });
  }
  function uniq(a) { return a.filter(function (v, i) { return v !== null && v !== undefined && v !== "" && a.indexOf(v) === i; }).sort(function (x, y) { return String(x).localeCompare(String(y)); }); }
  function select(id, label, values, current, fmt) {
    return '<label>' + esc(label) + '<select id="' + id + '"><option value="">All</option>' + values.map(function (v) {
      return '<option value="' + esc(v) + '"' + (String(v) === String(current) ? " selected" : "") + ">" + esc(fmt ? fmt(v) : v) + "</option>";
    }).join("") + "</select></label>";
  }
  function bind(id, ev, fn) { var n = document.getElementById(id); if (n) n.addEventListener(ev, function (e) { fn(e.target.value); }); }

  // -- material detail ---------------------------------------------------------------------------------------
  function evidenceLink(ref, evidence) {
    var e = (evidence || []).filter(function (x) { return x.id === ref; })[0];
    if (!e) return ref ? '<span class="mono small">' + esc(ref) + "</span>" : NA;
    var label = esc(e.reference || e.id) + (e.locator ? ' <span class="muted">' + esc(e.locator) + "</span>" : "");
    return '<span class="small">' + (e.url ? '<a href="' + esc(e.url) + '" rel="noopener">' + label + "</a>" : label) + "</span>";
  }
  function condText(e) {
    var parts = [];
    COND.forEach(function (c) { if (e[c[0]] && e[c[0]].value !== undefined) parts.push(c[1] + " = " + qty(e[c[0]].value, e[c[0]].unit)); });
    return parts.length ? esc(parts.join(", ")) : '<span class="na">not stated</span>';
  }
  function propertyTable(props, evidence, opts) {
    opts = opts || {};
    var rows = "";
    PROP.forEach(function (p) {
      var key = p[0], v = props[key];
      if (v === undefined || v === null) {
        if (opts.showMissing) rows += "<tr><th>" + esc(p[1]) + '</th><td colspan="4"><span class="na">— not reported</span></td></tr>';
        return;
      }
      if (key === "recommended_frequency_range") {
        rows += "<tr><th>" + esc(p[1]) + "</th><td>" + (v.min !== undefined ? qtyHtml(v.min, v.unit) : NA) + " – " + (v.max !== undefined ? qtyHtml(v.max, v.unit) : NA) +
          "</td><td></td><td></td><td></td></tr>";
        return;
      }
      if (key === "permeability_vs_temperature") {
        var ts = v.map(function (e) { return e.temperature && e.temperature.value; }).filter(function (t) { return t !== undefined; });
        rows += "<tr><th>" + esc(p[1]) + "</th><td>" + int(v.length) + " points</td><td>T " + esc(shortTemp(Math.min.apply(null, ts)) + " – " + shortTemp(Math.max.apply(null, ts))) +
          '</td><td class="small">plotted below</td><td>' + evidenceLink(v[0] && v[0].source_ref, evidence) + "</td></tr>";
        return;
      }
      (Array.isArray(v) ? v : [v]).forEach(function (e, i) {
        rows += "<tr>" + (i ? "<th></th>" : "<th>" + esc(p[1]) + "</th>") + "<td>" + qobj(e.value) + "</td><td>" + condText(e) + '</td><td class="small">' +
          esc(e.note || "") + "</td><td>" + evidenceLink(e.source_ref, evidence) + "</td></tr>";
      });
    });
    return rows ? '<div class="table-wrap"><table class="kv"><thead><tr><th>Property</th><th>Value</th><th>Conditions</th><th>Note</th><th>Evidence</th></tr></thead><tbody>' +
      rows + "</tbody></table></div>" : '<p class="muted">No typed property values in this record.</p>';
  }
  function steinmetzTable(sets) {
    return '<div class="table-wrap"><table><thead><tr><th>Set</th><th>Basis</th><th class="num">k</th><th class="num">α</th><th class="num">β</th><th>f range</th><th>B̂ range</th><th>T</th><th>Excitation</th><th>Fit</th></tr></thead><tbody>' +
      sets.map(function (s) {
        var fr = s.frequency_range || {}, br = s.flux_density_range || {}, fm = s.fit_metrics || {};
        return '<tr><td class="mono small">' + esc(s.id || "") + "</td><td>" + esc(s.basis || "") + '</td><td class="num">' + coef(s.k) + '</td><td class="num">' + esc(num(s.alpha)) +
          '</td><td class="num">' + esc(num(s.beta)) + "</td><td>" + esc(qty(fr.min, fr.unit) + " – " + qty(fr.max, fr.unit)) + "</td><td>" + esc(qty(br.min, br.unit) + " – " + qty(br.max, br.unit)) +
          "</td><td>" + (s.temperature ? esc(qty(s.temperature.value, s.temperature.unit)) : NA) + "</td><td>" + esc(s.excitation || "") + '</td><td class="small">' +
          (fm.n_points ? esc(fm.n_points + " points, mean |rel. error| " + num(fm.mean_abs_rel_error)) : "") + (s.note ? "<br>" + esc(s.note) : "") + "</td></tr>";
      }).join("") + "</tbody></table></div>";
  }
  function coef(k) { // Steinmetz k is stored as {value, unit}; older sets carry a bare number
    if (k && typeof k === "object") return k.value === undefined || k.value === null ? NA : esc(num(k.value)) + (unit(k.unit) ? ' <span class="muted small">' + esc(unit(k.unit)) + "</span>" : "");
    return k === undefined || k === null ? NA : esc(num(k));
  }
  function lossPointTable(points) {
    return '<div class="table-wrap"><table><thead><tr><th>Basis</th><th>f</th><th>B̂</th><th>T</th><th>Loss density</th></tr></thead><tbody>' +
      points.map(function (p) {
        return "<tr><td>" + esc(p.basis || "") + "</td><td>" + qobj(p.frequency) + "</td><td>" + qobj(p.flux_density_peak) + "</td><td>" + qobj(p.temperature) + "</td><td>" + qobj(p.loss_density) + "</td></tr>";
      }).join("") + "</tbody></table></div>";
  }
  function measurementTable(list) {
    if (!list.length) return '<p class="muted">No measurement tables.</p>';
    return '<div class="table-wrap"><table><thead><tr><th>Table</th><th>Quantity</th><th>Data class</th><th class="hide-sm">Conditions</th><th class="num">Rows</th><th class="hide-sm">On this site</th></tr></thead><tbody>' +
      list.map(function (m) {
        return '<tr><td><a href="#/measurement/' + encodeURIComponent(m.id) + '">' + esc(m.name) + '</a><div class="small muted">' + esc(src(m.source_id)) + (m.status === "primary" ? "" : ' <span class="badge warn">' + esc(STATUS[m.status] || m.status) + "</span>") + "</div></td><td>" +
          esc(QTY[m.quantity] || m.quantity) + "</td><td>" + dcBadge(m.data_class) + '</td><td class="small hide-sm cond">' + conditions(m.conditions) + '</td><td class="num">' + int(m.row_count) +
          '</td><td class="small hide-sm">' +
          esc(PUBLISHED[m.published] || "") + "</td></tr>";
      }).join("") + "</tbody></table></div>";
  }

  function material(id) {
    return load("data/materials/" + file(id)).then(function (d) {
      var r = d.record, s = d.summary, p = r.properties, canon = r.extensions["aipe.canonical"];
      setTitle(r.name);
      return function () {
        var order = { experimental_measurement: 0, derived_from_measurement: 1, manufacturer_tool_export: 2, manufacturer_datasheet: 3 };
        var meas = d.measurements.slice().sort(function (a, b) { return (order[a.data_class] - order[b.data_class]) || a.name.localeCompare(b.name); });
        app.innerHTML = '<div class="crumbs"><a href="#/materials">Materials</a> / ' + esc(r.name) + "</div>" +
          "<h1>" + esc(r.name) + "</h1>" +
          '<div class="head-meta"><span class="badge">' + esc(CLASS_LABEL[p.material_class] || p.material_class || "class not stated") + "</span>" +
          (s.manufacturer ? '<span class="badge">' + esc(s.manufacturer) + "</span>" : "") + '<span class="badge">grade ' + esc(p.grade || "—") + "</span>" +
          s.licenses.map(function (l) { return '<span class="badge" title="licence">' + esc(l) + "</span>"; }).join("") + "</div>" +
          '<p class="small muted">Canonical id <code>' + esc(r.id) + "</code>. Typed values below are copied unchanged from <code>" + esc(canon.properties_from) +
          "</code>; the other source records are listed under Sources with their own values.</p>" +
          '<div class="table-wrap" style="margin-bottom:8px"><table class="kv"><tbody>' +
          "<tr><th>Sources</th><td>" + canon.members.map(function (m) { return esc(src(m.source_id)) + " " + dcBadge(m.data_class); }).join("<br>") + "</td></tr>" +
          "<tr><th>Measurement tables</th><td>" + int(s.measurements) + " primary (" + int(s.experimental) + " experimental)" + "</td></tr>" +
          "<tr><th>Fitted loss models</th><td>" + int(s.loss_models) + "</td></tr>" +
          "<tr><th>Core parts</th><td>" + (s.parts ? '<a href="#/cores?view=parts&material=' + encodeURIComponent(r.id) + '">' + int(s.parts) + " manufacturer parts</a>" : NA) + "</td></tr>" +
          "<tr><th>Quality flags</th><td>" + flags(r.quality.flags) + "</td></tr></tbody></table></div>" +
          '<div class="more"><a class="btn-pill" href="#/compare?m=' + encodeURIComponent(r.id) + '">Compare with other materials</a></div>' +
          '<h2>Charts</h2><div id="mcharts"></div>' +
          "<h2>Properties</h2>" + propertyTable(p, r.evidence, { showMissing: true }) +
          (p.permeability_vs_temperature ? '<h3>Permeability vs temperature</h3><div id="pvt"></div>' : "") +
          (p.loss_reference_points ? "<h3>Datasheet loss reference points</h3>" + lossPointTable(p.loss_reference_points) : "") +
          (p.steinmetz_parameter_sets ? '<h3>Steinmetz parameter sets</h3><p>' + dcBadge("fitted_parameters") + ' Coefficients, not measurements.</p>' + steinmetzTable(p.steinmetz_parameter_sets) : "") +
          "<h2>Measurement tables</h2>" + measurementTable(meas) +
          (d.loss_models.length ? "<h2>Fitted loss models</h2><p>" + dcBadge("fitted_parameters") + " Published coefficients, kept apart from measurements. " +
            'Evaluation and fitting belong to <a href="https://github.com/FulongLi/AIPE-Magnetics-Design">AIPE-Magnetics-Design</a>.</p>' +
            d.loss_models.map(function (m) {
              var v = m.validity && m.validity.frequency_range;
              return '<div class="card"><strong>' + esc(m.name) + '</strong> <span class="mono small">' + esc(m.id) + "</span><br>" +
                '<span class="small">' + esc(src(m.source_id)) + " · " + esc(m.license) + " · " + esc(m.architecture || m.model_id || "") +
                (v ? " · valid " + esc(qty(v.min, v.unit) + " – " + qty(v.max, v.unit)) : "") + "</span>" +
                (m.description ? '<p class="small muted">' + esc(m.description) + "</p>" : "") +
                "<details><summary>Coefficients</summary><pre class=\"json\">" + esc(JSON.stringify(m.inference_config, null, 2)) + "</pre></details>" +
                (m.shard ? '<a class="small" href="' + esc(m.shard) + '">Record file on GitHub</a>' : "") + "</div>";
            }).join("") : "") +
          "<h2>Sources and provenance</h2>" + d.members.map(function (mem) {
            var rec = mem.record, dc = (rec.extensions && rec.extensions["aipe.data"] || {}).data_class ||
              (canon.members.filter(function (m) { return m.record === rec.id; })[0] || {}).data_class;
            return '<div class="card"><strong>' + esc(src(mem.source_id)) + "</strong> " + dcBadge(dc) + ' <span class="badge">' + esc(rec.license) + "</span>" +
              '<div class="small">Record <code>' + esc(rec.id) + "</code>" + (mem.shard ? ' · <a href="' + esc(mem.shard) + '">record file on GitHub</a>' : "") + "</div>" +
              '<ul class="small">' + (rec.evidence || []).map(function (e) {
                return "<li>" + (e.url ? '<a href="' + esc(e.url) + '" rel="noopener">' + esc(e.reference || e.id) + "</a>" : esc(e.reference || e.id)) +
                  (e.locator ? " · " + esc(e.locator) : "") + (e.version ? ' · <span class="mono">' + esc(String(e.version).slice(0, 12)) + "</span>" : "") +
                  " · " + esc(e.review_status || "") + (e.description ? '<br><span class="muted">' + esc(e.description) + "</span>" : "") + "</li>";
              }).join("") + "</ul>" + flags(rec.quality && rec.quality.flags) +
              "<details><summary>Values in this source record</summary>" + propertyTable(rec.properties, rec.evidence) +
              (rec.properties.steinmetz_parameter_sets ? steinmetzTable(rec.properties.steinmetz_parameter_sets) : "") + "</details></div>";
          }).join("") +
          '<details><summary>Canonical record (JSON)</summary><pre class="json">' + esc(JSON.stringify(r, null, 2)) + "</pre></details>";
        materialCharts(document.getElementById("mcharts"), d.measurements);
        if (p.permeability_vs_temperature) {
          MiniPlot.render(document.getElementById("pvt"), {
            ariaLabel: "Permeability versus temperature",
            series: [{ label: "μ", color: COLORS[0], mode: "lines", points: p.permeability_vs_temperature.map(function (e) { return [e.temperature.value - 273.15, e.value.value]; }) }],
            x: { label: "Temperature T [°C]", format: tick("°C") }, y: { label: "Relative permeability μr", format: tick("1") },
            note: "Values as published by " + src(canon.members.filter(function (m) { return m.record === canon.properties_from; }).map(function (m) { return m.source_id; })[0]) + "."
          });
        }
      };
    });
  }

  // -- measurements list -------------------------------------------------------------------------------------
  var COND_INPUTS = [["frequency", "Frequency", "kHz", 1e3, 0], ["temperature", "Temperature", "°C", 1, 273.15], ["flux_density_peak", "Flux density B̂", "mT", 1e-3, 0],
    ["dc_bias_field", "DC bias H", "A/m", 1, 0]];
  function measurements(q) {
    setTitle("Measurements");
    return load("data/measurements.json").then(function (list) {
      return function () {
        var state = { q: q.get("q") || "", quantity: q.get("quantity") || "", dc: q.get("data_class") || "", source: q.get("source") || "",
          waveform: q.get("waveform") || "", all: q.get("all") === "1" };
        COND_INPUTS.forEach(function (c) { state[c[0]] = q.get(c[0]) || ""; });
        var limit = PAGE;
        var waveforms = uniq([].concat.apply([], list.map(function (m) { return (m.conditions && m.conditions.waveforms) || []; })));
        app.innerHTML = '<h1>Measurements</h1><p class="lead">Core-loss tables, B-H curves, complex permeability and permittivity, and other ' +
          'tabulated data. Experimental measurements, manufacturer datasheet curves and derived tables are labelled separately; fitted model ' +
          'coefficients are listed on the material pages, not here.</p>' +
          '<div class="filters" role="search">' +
          '<label class="wide">Search table or material<input type="search" id="m-q" value="' + esc(state.q) + '" placeholder="e.g. N87, MagNet, 3C90"></label>' +
          select("m-quantity", "Quantity", uniq(list.map(function (m) { return m.quantity; })), state.quantity, function (v) { return QTY[v] || v; }) +
          select("m-dc", "Data class", uniq(list.map(function (m) { return m.data_class; })), state.dc, function (v) { return (DATA_CLASS[v] || [v])[0]; }) +
          select("m-source", "Source", uniq(list.map(function (m) { return m.source_id; })), state.source, src) +
          select("m-waveform", "Waveform", waveforms, state.waveform) +
          COND_INPUTS.map(function (c) {
            return "<label>" + esc(c[1]) + " [" + esc(c[2]) + ']<input type="number" step="any" id="m-' + c[0] + '" value="' + esc(state[c[0]]) + '" placeholder="covers value"></label>';
          }).join("") +
          '<label class="check"><input type="checkbox" id="m-all"' + (state.all ? " checked" : "") + "> Include duplicates and derived views</label>" +
          "</div><p id=\"count\" class=\"muted small\" aria-live=\"polite\"></p><div id=\"list\"></div><div class=\"more\" id=\"more\"></div>";
        function update() {
          var qs = { q: state.q, quantity: state.quantity, data_class: state.dc, source: state.source, waveform: state.waveform, all: state.all };
          COND_INPUTS.forEach(function (c) { qs[c[0]] = state[c[0]]; });
          writeQuery("measurements", qs);
          var needle = state.q.trim().toLowerCase();
          var rows = list.filter(function (m) {
            if (!state.all && m.status !== "primary") return false;
            if (state.quantity && m.quantity !== state.quantity) return false;
            if (state.dc && m.data_class !== state.dc) return false;
            if (state.source && m.source_id !== state.source) return false;
            if (state.waveform && (!m.conditions.waveforms || m.conditions.waveforms.indexOf(state.waveform) < 0)) return false;
            for (var i = 0; i < COND_INPUTS.length; i++) {
              var c = COND_INPUTS[i], raw = state[c[0]];
              if (raw === "" || isNaN(Number(raw))) continue;
              var v = Number(raw) * c[3] + c[4], r = m.conditions[c[0]];
              if (!r || r.max < v * (1 - 1e-9) - 1e-12 || r.min > v * (1 + 1e-9) + 1e-12) return false;
            }
            if (needle) {
              var hay = [m.name, m.material_name, m.id, src(m.source_id), QTY[m.quantity]].join(" ").toLowerCase();
              return needle.split(/\s+/).every(function (t) { return hay.indexOf(t) >= 0; });
            }
            return true;
          });
          var order = { experimental_measurement: 0, derived_from_measurement: 1, manufacturer_tool_export: 2, manufacturer_datasheet: 3 };
          rows.sort(function (a, b) { return (order[a.data_class] - order[b.data_class]) || a.name.localeCompare(b.name); });
          document.getElementById("count").textContent = int(rows.length) + " tables" + (state.all ? "" : " (primary tables only)") + "; experimental measurements first.";
          document.getElementById("list").innerHTML = rows.length ? measurementTable(rows.slice(0, limit)) : '<div class="notice info">No table matches these filters. Condition filters keep tables whose stated range covers the value.</div>';
          document.getElementById("more").innerHTML = rows.length > limit ? '<button class="secondary" id="more-btn">Show ' + Math.min(PAGE, rows.length - limit) + " more</button>" : "";
          var mb = document.getElementById("more-btn");
          if (mb) mb.addEventListener("click", function () { limit += PAGE; update(); });
        }
        bind("m-q", "input", function (v) { state.q = v; limit = PAGE; update(); });
        [["m-quantity", "quantity"], ["m-dc", "dc"], ["m-source", "source"], ["m-waveform", "waveform"]].forEach(function (b) {
          bind(b[0], "change", function (v) { state[b[1]] = v; limit = PAGE; update(); });
        });
        COND_INPUTS.forEach(function (c) { bind("m-" + c[0], "input", function (v) { state[c[0]] = v; limit = PAGE; update(); }); });
        document.getElementById("m-all").addEventListener("change", function (e) { state.all = e.target.checked; limit = PAGE; update(); });
        update();
      };
    });
  }

  // -- measurement detail ------------------------------------------------------------------------------------
  var TEXT_COLS = { waveform: 1, specimen: 1, branch: 1 };
  var DEFAULTS = { // x, y list, log x, log y, mode, group preference
    core_loss: ["flux_density_peak", ["loss_density"], true, true, "points", ["frequency", "temperature", "waveform"]],
    mass_core_loss: ["flux_density_peak", ["mass_loss_density"], true, true, "points", ["frequency", "temperature", "waveform"]],
    complex_permeability: ["frequency", ["mu_real", "mu_imag"], true, true, "lines", ["temperature", "flux_density_peak", "specimen", "dc_bias_field"]],
    complex_permittivity: ["frequency", ["eps_real", "eps_imag"], true, true, "lines", ["temperature", "specimen"]],
    bh_curve: ["field", ["flux_density"], false, false, "lines", ["temperature", "branch", "frequency"]],
    loss_factor: ["frequency", ["loss_factor"], true, true, "lines", ["temperature"]],
    initial_permeability: [null, ["relative_permeability"], false, false, "lines", ["frequency", "dc_bias_field", "flux_density_peak", "temperature"]],
    amplitude_permeability: ["flux_density_peak", ["amplitude_permeability"], false, false, "lines", ["temperature", "frequency", "dc_bias_field"]],
    energy_loss: ["flux_density_peak", ["energy_density_per_cycle"], true, true, "points", ["frequency", "duty"]],
    energy_loss_and_permeability: ["flux_density_peak", ["energy_density_per_cycle"], true, true, "points", ["frequency"]]
  };
  function groupLabel(col, v) {
    if (v === null || v === undefined) return col + ": not reported";
    if (col === "temperature") return shortTemp(v);
    if (col === "flux_density_peak") return num(v * 1e3) + " mT";
    if (col === "frequency") return qty(v, "Hz");
    if (col === "dc_bias_field") return "H_DC " + qty(v, "A/m");
    if (col === "duty") return "duty " + num(v);
    return String(v);
  }
  function distinct(rows, i) {
    var seen = {}, out = [];
    rows.forEach(function (r) { var k = String(r[i]); if (!(k in seen)) { seen[k] = 1; out.push(r[i]); } });
    return out.sort(function (a, b) { return (a === null) - (b === null) || (typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b))); });
  }

  // Groups of one column for colouring: exact values when there are 2-8 of them; otherwise whole-degree temperatures or
  // up to five value bands (log-spaced when the range spans more than a factor of four). Nothing is averaged: every row
  // is drawn at its own value and only takes the colour of its band.
  var NUMERIC_GROUPS = { frequency: 1, temperature: 1, flux_density_peak: 1, flux_density_peak_to_peak: 1, dc_bias_field: 1, duty: 1, field: 1 };
  function grouper(rows, i, name) {
    var vals = distinct(rows, i), map = {};
    if (vals.length < 2) return null;
    if (vals.length <= 8) {
      vals.forEach(function (v, k) { map[String(v)] = k; });
      return { labels: vals.map(function (v) { return groupLabel(name, v); }), of: function (r) { return map[String(r[i])]; },
        ordered: !!NUMERIC_GROUPS[name], binned: false };
    }
    var nums = vals.filter(function (v) { return typeof v === "number"; });
    if (!NUMERIC_GROUPS[name] || nums.length < 2) return null;
    if (name === "temperature") {
      var deg = distinct(nums.map(function (v) { return [Math.round(v - 273.15)]; }), 0);
      if (deg.length >= 2 && deg.length <= 8) {
        deg.forEach(function (v, k) { map[v] = k; });
        return { labels: deg.map(function (v) { return v + " °C"; }), ordered: true, binned: false,
          of: function (r) { return typeof r[i] === "number" ? map[Math.round(r[i] - 273.15)] : -1; } };
      }
    }
    var lo = nums[0], hi = nums[nums.length - 1], n = SEQ.length, log = lo > 0 && hi / lo > 4, edges = [];
    for (var k = 0; k <= n; k++) edges.push(parseFloat((log ? lo * Math.pow(hi / lo, k / n) : lo + (hi - lo) * k / n).toPrecision(2)));
    return { labels: edges.slice(0, n).map(function (a, k) { return groupLabel(name, a) + " – " + groupLabel(name, edges[k + 1]); }),
      ordered: true, binned: true,
      of: function (r) {
        var v = r[i]; if (typeof v !== "number") return -1;
        for (var k = n - 1; k > 0; k--) if (v >= edges[k]) return k;
        return 0;
      } };
  }

  // Default chart of one table: axes, y columns, colouring and marks.
  function chartState(d) {
    var e = d.entry, cols = d.columns, rows = d.rows, names = cols.map(function (c) { return c.name; });
    var idx = function (n) { return names.indexOf(n); };
    var numeric = cols.filter(function (c) { return !TEXT_COLS[c.name] && c.unit !== ""; }).map(function (c) { return c.name; });
    var def = DEFAULTS[e.quantity] || [null, [], false, false, "points", []];
    // x default: preferred column, else the condition column with the most distinct values
    var x = def[0] && idx(def[0]) >= 0 ? def[0] : null;
    if (!x) {
      var cand = ["temperature", "frequency", "flux_density_peak", "field", "dc_bias_field"].filter(function (n) { return idx(n) >= 0; });
      cand.sort(function (a, b) { return distinct(rows, idx(b)).length - distinct(rows, idx(a)).length; });
      x = cand[0] || numeric[0];
    }
    var ys = def[1].filter(function (n) { return idx(n) >= 0 && n !== x; });
    if (!ys.length) ys = [numeric.filter(function (n) { return n !== x; }).slice(-1)[0]].filter(Boolean);
    var groupers = {};
    names.forEach(function (n) { var g = grouper(rows, idx(n), n); if (g) groupers[n] = g; });
    var group = def[5].filter(function (n) { return groupers[n] && n !== x && ys.indexOf(n) < 0; })[0] || "";
    var st = { x: x, ys: ys, group: group, logx: def[2], logy: def[3], mode: def[4], f: {} };
    // lines only where x runs monotonically within each series (B-H loops are drawn in upstream traversal order)
    if (st.mode === "lines" && e.quantity !== "bh_curve") {
      var xi0 = idx(st.x), G0 = st.group ? groupers[st.group] : null, last = {}, dir = {};
      var monotonic = rows.every(function (r) {
        var g = G0 ? G0.of(r) : 0, v = r[xi0];
        if (v === null) return true;
        if (g in last && v !== last[g]) {
          var s = v > last[g] ? 1 : -1;
          if (dir[g] && dir[g] !== s) return false;
          dir[g] = s;
        }
        last[g] = v;
        return true;
      });
      if (!monotonic) st.mode = "points";
    }
    return { st: st, groupers: groupers, numeric: numeric, names: names };
  }

  // MiniPlot spec of the rows `shown` of one table under chart state `st` (G: grouper of st.group, or null).
  function plotSpec(d, st, shown, G) {
    var cols = d.columns, names = cols.map(function (c) { return c.name; }), idx = function (n) { return names.indexOf(n); };
    var xi = idx(st.x), X = colInfo(cols[xi]), k = G ? G.labels.length : 1;
    var Y = st.ys.map(function (y) { return colInfo(cols[idx(y)]); }), yi = st.ys.map(idx);
    var seq = G && G.ordered && st.ys.length === 1 && k <= SEQ.length, series = [];
    Y.forEach(function (y, yk) {
      for (var gk = 0; gk < k; gk++) {
        series.push({ label: (Y.length > 1 || !G ? y.label : "") + (G ? (Y.length > 1 ? ", " : "") + G.labels[gk] : ""), mode: st.mode, points: [],
          color: seq ? SEQ[k === 1 ? 2 : Math.round(gk * (SEQ.length - 1) / (k - 1))] : COLORS[(G ? gk + yk * k : yk) % 8] });
      }
    });
    shown.forEach(function (r, ri) {
      var gk = G ? G.of(r) : 0, xv = r[xi];
      if (gk === undefined || gk < 0) return;
      Y.forEach(function (y, yk) {
        var v = r[yi[yk]];
        series[yk * k + gk].points.push([typeof xv === "number" ? X.conv(xv) : null, typeof v === "number" ? y.conv(v) : null, ri]);
      });
    });
    var drawn = series.filter(function (s) { return s.points.length > 0; });
    if (!drawn.length) drawn = series.slice(0, 1);
    var tooMany = drawn.length > 8;
    if (tooMany) drawn = drawn.slice(0, 8);
    var yunits = uniq(Y.map(function (y) { return y.unit; }));
    return {
      ariaLabel: d.record.name + ": " + Y.map(function (y) { return y.label; }).join(", ") + " versus " + X.label,
      series: drawn,
      x: { label: X.axis, log: st.logx, format: tick(X.unit) },
      y: { label: Y.length === 1 ? Y[0].axis : Y.map(function (y) { return y.label; }).join(", ") + (yunits.length === 1 && unit(yunits[0]) ? " [" + unit(yunits[0]) + "]" : ""),
        log: st.logy, format: tick(yunits.length === 1 ? yunits[0] : "1") },
      note: (d.sampling.sampled ? "Sampled preview: " + int(d.sampling.published_rows) + " of " + int(d.sampling.total_rows) + " rows." : "") +
        (G && G.binned ? " Colour shows the " + colLabel(st.group).toLowerCase() + " band of each row; values are not averaged." : "") +
        (tooMany ? " Only the first 8 series are drawn; narrow the filters or change the colouring." : ""),
      extra: function (ri) {
        var r = shown[ri]; if (!r) return "";
        return names.filter(function (n) { return n !== st.x && st.ys.indexOf(n) < 0; }).map(function (n) {
          var c = cols[idx(n)]; return "<br><span class=\"muted\">" + esc(colLabel(n)) + ": " + esc(r[idx(n)] === null ? "not reported" : c.unit ? qty(r[idx(n)], c.unit) : r[idx(n)]) + "</span>";
        }).join("");
      }
    };
  }

  function measurement(id) {
    return load("data/measurements/" + file(id)).then(function (d) {
      var rec = d.record, e = d.entry, cols = d.columns, names = cols.map(function (c) { return c.name; }), props = rec.properties || {};
      setTitle(rec.name);
      return function () {
        var smp = d.sampling;
        var head = '<div class="crumbs"><a href="#/measurements">Measurements</a> / ' + esc(rec.name) + "</div><h1>" + esc(rec.name) + "</h1>" +
          '<div class="head-meta">' + dcBadge(e.data_class) + '<span class="badge">' + esc(QTY[e.quantity] || e.quantity) + '</span><span class="badge" title="licence">' + esc(rec.license) + "</span>" +
          (e.status !== "primary" ? '<span class="badge warn">' + esc(STATUS[e.status] || e.status) + "</span>" : "") + "</div>" +
          '<p class="small">' + esc((DATA_CLASS[e.data_class] || ["", "", ""])[2]) + " Source: " + esc(src(e.source_id)) + ". Material: " +
          (e.material ? '<a href="#/material/' + encodeURIComponent(e.material) + '">' + esc(e.material_name || e.material) + "</a>" : NA) + ".</p>" +
          (e.status === "duplicate" ? '<div class="notice">Duplicate of ' + (e.duplicate_of ? '<a href="#/measurement/' + encodeURIComponent(e.duplicate_of) + '">' + esc(e.duplicate_of) + "</a>" : "another table") +
            (e.duplicate_match ? " (" + esc(JSON.stringify(e.duplicate_match)) + ")" : "") + "; not counted as an independent measurement.</div>" : "") +
          (e.status === "derived_view" ? '<div class="notice">Derived view of ' + (e.derived_from || []).map(function (x) {
            return x.indexOf("measurement.") === 0 ? '<a href="#/measurement/' + encodeURIComponent(x) + '">' + esc(x) + "</a>" : esc(x); }).join(", ") +
            "; not an independent measurement.</div>" : "");
        var avail = d.unavailable ? '<div class="notice"><strong>Rows not published on this site.</strong> ' + esc(d.unavailable) +
            (d.links.huggingface ? ' <a href="' + esc(d.links.huggingface) + '" rel="noopener">Complete table on Hugging Face</a>.' : "") +
            " The record metadata and provenance are shown below.</div>" :
          smp.sampled ? '<div class="notice"><strong>Sampled preview: ' + int(smp.published_rows) + " of " + int(smp.total_rows) + " rows.</strong> " + esc(smp.method) + ". " +
            "Charts and downloads on this page use the sample, not the complete dataset. " +
            (d.links.rows_file ? 'Full table: <a href="' + esc(d.links.rows_file) + '">JSONL (' + int(smp.total_rows) + " rows)</a>, or " : "") +
            "<code>aipe-magnetics-db measurements table " + esc(rec.id) + " --csv table.csv</code>.</div>" :
          '<p class="small muted">All ' + int(smp.total_rows) + " rows of this table are published here.</p>";
        app.innerHTML = head + avail + (d.unavailable ? "" :
          '<h2>Data</h2><div class="filters" id="row-filters"></div><div class="chart-controls" id="controls"></div><div id="chart"></div>' +
          '<p id="shown" class="small muted" aria-live="polite"></p><div class="more"><button class="secondary" id="csv">Download shown rows (CSV)</button>' +
          '<span class="small muted" id="csv-note"></span></div><details><summary>Table view</summary><div id="rows"></div></details>') +
          "<h2>Columns</h2>" + '<div class="table-wrap"><table><thead><tr><th>Column</th><th>Unit</th><th>Original unit</th><th>Definition</th></tr></thead><tbody>' +
          cols.map(function (c) { return '<tr><td class="mono">' + esc(c.name) + "</td><td>" + (c.unit ? esc(unit(c.unit) || "1 (dimensionless)") : '<span class="muted">text</span>') +
            "</td><td>" + (c.original_unit ? esc(c.original_unit) : NA) + '</td><td class="small">' + esc(c.definition || "") + "</td></tr>"; }).join("") +
          "</tbody></table></div>" +
          ((rec.extensions["aipe.table"].notes || []).length ? '<ul class="small">' + rec.extensions["aipe.table"].notes.map(function (n) { return "<li>" + esc(n) + "</li>"; }).join("") + "</ul>" : "") +
          "<h2>Provenance</h2>" + provenance(d) +
          '<details><summary>Record metadata (JSON, rows omitted)</summary><pre class="json">' + esc(JSON.stringify(rec, null, 2)) + "</pre></details>";
        if (d.unavailable) return;
        dataView(d, names);
      };
    });
  }

  function provenance(d) {
    var rec = d.record, props = rec.properties || {}, ds = d.dataset, e = d.entry, data = rec.extensions["aipe.data"] || {};
    var kv = [["Record id", "<code>" + esc(rec.id) + "</code>"], ["Licence", esc(rec.license)], ["Data class", dcBadge(e.data_class)],
      ["Underlying experiment", e.underlying_source ? "<code>" + esc(e.underlying_source) + "</code>" : NA], ["Quality flags", flags(rec.quality && rec.quality.flags) || NA]];
    ["description", "specimen", "instrument", "method"].forEach(function (k) {
      if (props[k]) kv.push([k.charAt(0).toUpperCase() + k.slice(1), props[k].length > 240 ? "<details><summary>" + esc(props[k].slice(0, 160)) + "…</summary><p class=\"small\">" + esc(props[k]) + "</p></details>" : esc(props[k])]);
    });
    if (data.specimen_core) kv.push(["Specimen core (as published)", esc(Object.keys(data.specimen_core).filter(function (k) { return k !== "units"; }).map(function (k) {
      var u = (data.specimen_core.units || {})[k]; return k + " " + data.specimen_core[k] + (u ? " " + unit(u) : ""); }).join(", "))]);
    if (ds) kv.push(["Dataset", (ds.url ? '<a href="' + esc(ds.url) + '" rel="noopener">' + esc(ds.title || ds.id) + "</a>" : esc(ds.title || ds.id)) +
      (ds.version ? ' <span class="mono small">' + esc(String(ds.version).slice(0, 12)) + "</span>" : "") + (ds.publisher ? "<br><span class=\"small\">" + esc(ds.publisher) + "</span>" : "")]);
    if (ds && ds.citation) kv.push(["Cite", '<span class="small">' + esc(ds.citation) + "</span>"]);
    kv.push(["Evidence", '<ul class="small" style="margin:0;padding-left:18px">' + (rec.evidence || []).map(function (ev) {
      return "<li>" + (ev.url ? '<a href="' + esc(ev.url) + '" rel="noopener">' + esc(ev.reference || ev.id) + "</a>" : esc(ev.reference || ev.id)) +
        (ev.locator ? " · " + esc(ev.locator) : "") + " · " + esc(ev.review_status || "") + (ev.description ? '<br><span class="muted">' + esc(ev.description) + "</span>" : "") + "</li>";
    }).join("") + "</ul>"]);
    var links = [];
    if (d.links.record) links.push('<a href="' + esc(d.links.record) + '">record file on GitHub</a>');
    if (d.links.rows_file) links.push('<a href="' + esc(d.links.rows_file) + '">full rows (JSONL)</a>');
    if (d.links.huggingface) links.push('<a href="' + esc(d.links.huggingface) + '" rel="noopener">full rows on Hugging Face (pinned commit)</a>');
    kv.push(["Original files", links.join(" · ") || NA]);
    return '<div class="table-wrap"><table class="kv"><tbody>' + kv.map(function (r) { return "<tr><th>" + esc(r[0]) + "</th><td>" + r[1] + "</td></tr>"; }).join("") + "</tbody></table></div>";
  }

  function dataView(d, names) {
    var rec = d.record, e = d.entry, cols = d.columns, rows = d.rows;
    var idx = function (n) { return names.indexOf(n); };
    var cs = chartState(d), st = cs.st, groupers = cs.groupers, numeric = cs.numeric;

    // row filters
    var fbox = document.getElementById("row-filters"), fhtml = "";
    COND_INPUTS.forEach(function (c) {
      var i = idx(c[0]); if (i < 0) return;
      var vals = rows.map(function (r) { return r[i]; }).filter(function (v) { return typeof v === "number"; });
      if (!vals.length) return;
      var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals), toD = function (v) { return num((v - c[4]) / c[3]); };
      fhtml += "<label>" + esc(c[1]) + " [" + esc(c[2]) + ']<span class="range"><input type="number" step="any" data-f="' + c[0] + '" data-b="min" placeholder="' + esc(toD(lo)) +
        '" aria-label="minimum"><span>–</span><input type="number" step="any" data-f="' + c[0] + '" data-b="max" placeholder="' + esc(toD(hi)) + '" aria-label="maximum"></span>' +
        '<span class="hint">table: ' + esc(toD(lo) + " – " + toD(hi)) + "</span></label>";
    });
    Object.keys(TEXT_COLS).forEach(function (n) {
      var i = idx(n); if (i < 0) return;
      fhtml += select("rf-" + n, n.charAt(0).toUpperCase() + n.slice(1), distinct(rows, i).filter(function (v) { return v !== null; }), "");
    });
    if (fhtml) fbox.innerHTML = fhtml + '<p class="hint" style="grid-column:1/-1;margin:0">Row filters apply to the published rows. Empty = no limit.</p>'; else fbox.remove();
    fbox.querySelectorAll("input[data-f]").forEach(function (inp) {
      inp.addEventListener("input", function () {
        var c = COND_INPUTS.filter(function (k) { return k[0] === inp.getAttribute("data-f"); })[0];
        st.f[c[0]] = st.f[c[0]] || {};
        st.f[c[0]][inp.getAttribute("data-b")] = inp.value === "" || isNaN(Number(inp.value)) ? null : Number(inp.value) * c[3] + c[4];
        draw();
      });
    });
    Object.keys(TEXT_COLS).forEach(function (n) { bind("rf-" + n, "change", function (v) { st.f[n] = v; draw(); }); });

    // chart controls
    var ctl = document.getElementById("controls");
    function opt(list, cur) { return list.map(function (n) { return '<option value="' + esc(n) + '"' + (n === cur ? " selected" : "") + ">" + esc(colLabel(n)) + "</option>"; }).join(""); }
    ctl.innerHTML = '<label>x axis<select id="c-x">' + opt(numeric, st.x) + "</select></label>" +
      '<fieldset><legend>y axis</legend>' + numeric.map(function (n) {
        return '<label class="check"><input type="checkbox" data-y="' + esc(n) + '"' + (st.ys.indexOf(n) >= 0 ? " checked" : "") + "> " + esc(colLabel(n)) + "</label>"; }).join("") + "</fieldset>" +
      '<label>colour by<select id="c-g"><option value="">none</option>' + names.filter(function (n) { return n !== st.x; }).map(function (n) {
        var k = distinct(rows, idx(n)).length, g = groupers[n];
        return '<option value="' + esc(n) + '"' + (n === st.group ? " selected" : "") + (g ? "" : " disabled") + ">" + esc(colLabel(n)) +
          (g ? (g.binned ? " (" + g.labels.length + " bands)" : "") : " (" + k + " values)") + "</option>"; }).join("") + "</select></label>" +
      '<label>marks<select id="c-mode">' + ["points", "lines"].map(function (m) { return '<option value="' + m + '"' + (m === st.mode ? " selected" : "") + ">" + m + "</option>"; }).join("") + "</select></label>" +
      '<label class="check"><input type="checkbox" id="c-lx"' + (st.logx ? " checked" : "") + "> log x</label>" +
      '<label class="check"><input type="checkbox" id="c-ly"' + (st.logy ? " checked" : "") + "> log y</label>";
    bind("c-x", "change", function (v) { st.x = v; draw(); });
    bind("c-g", "change", function (v) { st.group = v; draw(); });
    bind("c-mode", "change", function (v) { st.mode = v; draw(); });
    document.getElementById("c-lx").addEventListener("change", function (ev) { st.logx = ev.target.checked; draw(); });
    document.getElementById("c-ly").addEventListener("change", function (ev) { st.logy = ev.target.checked; draw(); });
    ctl.querySelectorAll("input[data-y]").forEach(function (b) {
      b.addEventListener("change", function () {
        st.ys = Array.prototype.filter.call(ctl.querySelectorAll("input[data-y]"), function (c) { return c.checked; }).map(function (c) { return c.getAttribute("data-y"); });
        draw();
      });
    });

    var shown = rows;
    function filtered() {
      return rows.filter(function (r) {
        for (var k in st.f) {
          var f = st.f[k], i = idx(k);
          if (TEXT_COLS[k]) { if (f && String(r[i]) !== f) return false; continue; }
          if (f && f.min !== null && f.min !== undefined && !(r[i] !== null && r[i] >= f.min * (1 - 1e-9) - 1e-12)) return false;
          if (f && f.max !== null && f.max !== undefined && !(r[i] !== null && r[i] <= f.max * (1 + 1e-9) + 1e-12)) return false;
        }
        return true;
      });
    }
    function draw() {
      shown = filtered();
      var chart = document.getElementById("chart");
      if (!st.ys.length) { chart.innerHTML = '<div class="notice info">Select at least one y column.</div>'; }
      else MiniPlot.render(chart, plotSpec(d, st, shown, st.group ? groupers[st.group] : null));
      document.getElementById("shown").textContent = int(shown.length) + " of " + int(rows.length) + " published rows match the filters" +
        (d.sampling.sampled ? " (published rows are a sample of " + int(d.sampling.total_rows) + ")." : ".");
      document.getElementById("csv-note").textContent = d.sampling.sampled ? "The CSV contains sampled rows; the file name says so." : "";
      document.getElementById("rows").innerHTML = '<div class="table-wrap" style="max-height:480px"><table><thead><tr>' + cols.map(function (c) {
        return '<th class="' + (c.unit ? "num" : "") + '" title="' + esc(colLabel(c.name)) + '">' + esc(c.name) + (c.unit ? ' <span class="muted">[' + esc(unit(c.unit) || "1") + "]</span>" : "") + "</th>"; }).join("") +
        "</tr></thead><tbody>" + shown.slice(0, 200).map(function (r) {
          return "<tr>" + r.map(function (v, i) { return '<td class="' + (cols[i].unit ? "num" : "") + '"' + (typeof v === "number" ? ' title="' + v + '"' : "") + ">" +
            (v === null ? NA : esc(typeof v === "number" ? num(v) : v)) + "</td>"; }).join("") + "</tr>";
        }).join("") + "</tbody></table></div>" + (shown.length > 200 ? '<p class="small muted">First 200 of ' + int(shown.length) + " rows; download the CSV for all shown rows.</p>" : "");
    }
    document.getElementById("csv").addEventListener("click", function () {
      var lines = [cols.map(function (c) { return c.name + (c.unit ? " [" + c.unit + "]" : ""); }).join(",")];
      shown.forEach(function (r) { lines.push(r.map(function (v) { return v === null ? "" : typeof v === "number" ? String(v) : '"' + String(v).replace(/"/g, '""') + '"'; }).join(",")); });
      var name = rec.id + (d.sampling.sampled ? "-sample-" + d.sampling.published_rows + "-of-" + d.sampling.total_rows : "") +
        (shown.length !== rows.length ? "-filtered-" + shown.length + "-rows" : "") + ".csv";
      var a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], { type: "text/csv" }));
      a.download = name; document.body.appendChild(a); a.click();
      setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 0);
    });
    draw();
  }

  // -- charts shared by the material page and the comparison ---------------------------------------------------
  var CLASS_RANK = { experimental_measurement: 0, derived_from_measurement: 1, manufacturer_tool_export: 2, manufacturer_datasheet: 3 };
  var CHART_QTY = ["core_loss", "complex_permeability", "bh_curve", "amplitude_permeability", "initial_permeability", "complex_permittivity"];
  function rankTables(list) { // independent tables first, measured before datasheet, larger first
    return list.slice().sort(function (a, b) {
      return ((a.status !== "primary") - (b.status !== "primary")) || ((CLASS_RANK[a.data_class] || 0) - (CLASS_RANK[b.data_class] || 0)) ||
        (b.row_count - a.row_count) || a.name.localeCompare(b.name);
    });
  }
  function degrees(rows, i) { // distinct whole-degree temperatures of a table [°C]
    return distinct(rows.filter(function (r) { return typeof r[i] === "number"; }).map(function (r) { return [Math.round(r[i] - 273.15)]; }), 0);
  }
  function nearest(values, target) {
    return values.reduce(function (best, v) { return best === null || Math.abs(v - target) < Math.abs(best - target) ? v : best; }, null);
  }

  // One chart per quantity for a material: the best-ranked published table, drawn at one waveform and temperature.
  function materialCharts(host, meas) {
    var usable = meas.filter(function (m) { return m.published !== "none" && m.row_count > 1; });
    var qs = uniq(usable.map(function (m) { return m.quantity; })).sort(function (a, b) {
      var x = CHART_QTY.indexOf(a), y = CHART_QTY.indexOf(b); return (x < 0 ? 99 : x) - (y < 0 ? 99 : y) || a.localeCompare(b); });
    if (!qs.length) { host.innerHTML = '<p class="muted">No table of this material has rows published on this site, so there is nothing to plot.</p>'; return; }
    host.innerHTML = '<div class="tabs" role="tablist" aria-label="Quantity">' + qs.map(function (q) {
      return '<button type="button" role="tab" data-q="' + esc(q) + '">' + esc(QTY[q] || q) + "</button>"; }).join("") + "</div>" +
      '<div class="chart-controls" id="mc-ctl"></div><div id="mc-chart"></div><p class="small" id="mc-cap"></p>';
    var ctl = host.querySelector("#mc-ctl"), chart = host.querySelector("#mc-chart"), cap = host.querySelector("#mc-cap"), token = 0;
    function pick(q) {
      host.querySelectorAll("[data-q]").forEach(function (b) { b.setAttribute("aria-selected", b.getAttribute("data-q") === q ? "true" : "false"); });
      var cands = rankTables(usable.filter(function (m) { return m.quantity === q; }));
      show(cands, cands[0].id);
    }
    function show(cands, id) {
      var mine = ++token, entry = cands.filter(function (c) { return c.id === id; })[0];
      ctl.innerHTML = ""; cap.textContent = ""; chart.className = ""; chart.innerHTML = '<p class="loading" role="status">Loading table…</p>';
      load("data/measurements/" + file(id)).then(function (d) {
        if (mine !== token || !host.isConnected) return;
        var cs = chartState(d), st = cs.st, rows = d.rows, slices = [], yAll = st.ys.slice();
        if (yAll.length > 1 && st.group) st.ys = [yAll[0]]; // one quantity at a time, so colour can follow the grouping column
        ["waveform", "temperature"].forEach(function (n) {
          var i = cs.names.indexOf(n);
          if (i < 0 || n === st.x || n === st.group || st.ys.indexOf(n) >= 0) return;
          var opts = n === "temperature" ? degrees(rows, i) : distinct(rows, i).filter(function (v) { return v !== null; });
          if (opts.length < 2 || opts.length > 12) return;
          var value = n === "temperature" ? String(nearest(opts, d.entry.quantity === "core_loss" ? 100 : 25)) : opts.indexOf("sinusoidal") >= 0 ? "sinusoidal" : "";
          slices.push({ n: n, i: i, opts: opts, value: value });
        });
        ctl.innerHTML = (cands.length > 1 ? '<label>Table<select id="mc-table">' + cands.map(function (c) {
            return '<option value="' + esc(c.id) + '"' + (c.id === id ? " selected" : "") + ">" + esc(c.name + " · " + (DATA_CLASS[c.data_class] || [c.data_class])[0].toLowerCase()) + "</option>"; }).join("") + "</select></label>" : "") +
          slices.map(function (s, k) {
            return "<label>" + esc(s.n === "temperature" ? "Temperature" : "Waveform") + '<select data-slice="' + k + '"><option value="">All</option>' + s.opts.map(function (v) {
              return '<option value="' + esc(v) + '"' + (String(v) === s.value ? " selected" : "") + ">" + esc(s.n === "temperature" ? v + " °C" : v) + "</option>"; }).join("") + "</select></label>"; }).join("") +
          (yAll.length > st.ys.length ? '<label>Quantity<select id="mc-y">' + yAll.map(function (y) { return '<option value="' + esc(y) + '">' + esc(colLabel(y)) + "</option>"; }).join("") + "</select></label>" : "");
        function draw() {
          var shown = rows.filter(function (r) {
            return slices.every(function (s) {
              if (s.value === "") return true;
              return s.n === "temperature" ? typeof r[s.i] === "number" && Math.round(r[s.i] - 273.15) === Number(s.value) : String(r[s.i]) === s.value;
            });
          });
          MiniPlot.render(chart, plotSpec(d, st, shown, st.group ? cs.groupers[st.group] : null));
          cap.innerHTML = dcBadge(entry.data_class) + " " + esc(src(entry.source_id)) + " · " + int(shown.length) + " of " + int(rows.length) + " published rows drawn" +
            (d.sampling.sampled ? " (a sample of " + int(d.sampling.total_rows) + ")" : "") + (entry.status !== "primary" ? " · " + esc(STATUS[entry.status] || entry.status) : "") +
            ' · <a href="#/measurement/' + encodeURIComponent(id) + '">Open the table: filters, all columns, download</a>';
        }
        var sel = ctl.querySelector("#mc-table");
        if (sel) sel.addEventListener("change", function () { show(cands, sel.value); });
        var ysel = ctl.querySelector("#mc-y");
        if (ysel) ysel.addEventListener("change", function () { st.ys = [ysel.value]; draw(); });
        ctl.querySelectorAll("[data-slice]").forEach(function (s) {
          s.addEventListener("change", function () { slices[Number(s.getAttribute("data-slice"))].value = s.value; draw(); });
        });
        draw();
      }).catch(function (err) {
        if (mine === token) chart.innerHTML = '<div class="notice">Could not load this table. ' + esc(err && err.message || err) + "</div>";
      });
    }
    host.querySelectorAll("[data-q]").forEach(function (b) { b.addEventListener("click", function () { pick(b.getAttribute("data-q")); }); });
    pick(qs[0]);
  }

  // -- comparison ----------------------------------------------------------------------------------------------
  var CMP = { // x, y choices, log x, log y, marks, frequency condition
    core_loss: { x: "flux_density_peak", y: ["loss_density"], logx: true, logy: true, mode: "points", f: true },
    complex_permeability: { x: "frequency", y: ["mu_real", "mu_imag"], logx: true, logy: true, mode: "lines" },
    amplitude_permeability: { x: "flux_density_peak", y: ["amplitude_permeability"], mode: "lines" },
    bh_curve: { x: "field", y: ["flux_density"], mode: "lines" }
  };
  var CMP_F = [25e3, 50e3, 100e3, 200e3, 300e3, 500e3, 1e6, 2e6], CMP_T = [25, 60, 80, 100, 120], CMP_MAX = 6, F_TOL = 0.1, T_TOL = 10;
  var CMP_EXAMPLE = ["material.canonical-tdk-n87", "material.canonical-ferroxcube-3c94", "material.canonical-ferroxcube-3f4"];

  // Rows of one table at the requested conditions. Rows are selected, never interpolated: the frequency must lie within
  // ±10 % of the target, the temperature is the table's nearest whole degree within ±10 °C, sinusoidal excitation is used
  // where the table distinguishes waveforms, and a further condition column with a few discrete values (DC bias, frequency, flux density) is pinned to one of them.
  function compareRows(d, cfg, st) {
    var names = d.columns.map(function (c) { return c.name; }), idx = function (n) { return names.indexOf(n); };
    var xi = idx(cfg.x), yi = idx(st.y), rows = d.rows, used = [];
    if (xi < 0 || yi < 0) return { rows: [], used: used };
    var wi = idx("waveform");
    if (wi >= 0 && rows.some(function (r) { return r[wi] === "sinusoidal"; })) { rows = rows.filter(function (r) { return r[wi] === "sinusoidal"; }); used.push("sinusoidal"); }
    var fi = idx("frequency");
    if (cfg.f && fi >= 0) {
      rows = rows.filter(function (r) { return typeof r[fi] === "number" && Math.abs(r[fi] - st.f) <= F_TOL * st.f; });
      if (rows.length) used.push("f " + range("frequency", { min: Math.min.apply(null, rows.map(function (r) { return r[fi]; })), max: Math.max.apply(null, rows.map(function (r) { return r[fi]; })) }));
    }
    var ti = idx("temperature");
    if (ti >= 0 && rows.length) {
      var deg = degrees(rows, ti), t = nearest(deg, st.t);
      if (t === null || Math.abs(t - st.t) > T_TOL) return { rows: [], used: used };
      rows = rows.filter(function (r) { return typeof r[ti] === "number" && Math.round(r[ti] - 273.15) === t; });
      used.push("T " + t + " °C");
    } else if (ti < 0) used.push("temperature not stated");
    ["dc_bias_field", "frequency", "flux_density_peak"].forEach(function (n) { // remaining condition columns: one value
      var i = idx(n);
      if (i < 0 || n === cfg.x || n === st.y || (n === "frequency" && cfg.f) || !rows.length) return;
      var vals = distinct(rows, i);
      if (vals.length < 2 || vals.length > 8) return; // a swept column (many values) is kept whole and drawn as points
      var keep;
      if (n === "dc_bias_field") keep = vals.filter(function (v) { return typeof v === "number"; }).reduce(function (b, v) { return b === null || Math.abs(v) < Math.abs(b) ? v : b; }, null);
      else {
        var count = {}; rows.forEach(function (r) { count[String(r[i])] = (count[String(r[i])] || 0) + 1; });
        keep = vals.slice().sort(function (a, b) { return count[String(b)] - count[String(a)]; })[0];
      }
      rows = rows.filter(function (r) { return String(r[i]) === String(keep); });
      used.push(groupLabel(n, keep));
    });
    rows = rows.filter(function (r) { return typeof r[xi] === "number" && typeof r[yi] === "number"; });
    return { rows: rows, used: used, xi: xi, yi: yi, names: names };
  }

  function compare(q) {
    setTitle("Compare materials");
    return load("data/materials.json").then(function (list) {
      var byId = {}; list.forEach(function (m) { byId[m.id] = m; });
      return function () {
        var st = { m: (q.get("m") || "").split(",").filter(function (id) { return byId[id]; }).slice(0, CMP_MAX), q: CMP[q.get("q")] ? q.get("q") : "core_loss",
          f: Number(q.get("f")) || 1e5, t: q.get("t") !== null && q.get("t") !== "" && !isNaN(Number(q.get("t"))) ? Number(q.get("t")) : 100, y: q.get("y") || "", dc: q.get("class") || "" };
        var token = 0;
        app.innerHTML = '<h1>Compare materials</h1><p class="lead">Overlay up to ' + CMP_MAX + ' materials at the same operating point. Rows are selected from each ' +
          "material's best published table; nothing is interpolated or fitted, and the data class of every curve is shown.</p>" +
          '<div class="filters"><label class="wide">Add a material<input type="search" id="cmp-q" list="cmp-list" placeholder="e.g. TDK N87, 3C94, Kool Mu" autocomplete="off"></label>' +
          '<datalist id="cmp-list">' + list.map(function (m) { return '<option value="' + esc(m.name) + '">'; }).join("") + "</datalist>" +
          '<div><button type="button" id="cmp-add">Add</button></div><div id="cmp-chips" class="cmp-chips" style="grid-column:1/-1"></div></div>' +
          '<div class="tabs" role="tablist" aria-label="Quantity">' + Object.keys(CMP).map(function (k) {
            return '<button type="button" role="tab" data-q="' + k + '">' + esc(QTY[k]) + "</button>"; }).join("") + "</div>" +
          '<div class="chart-controls" id="cmp-ctl"></div><div id="cmp-chart"></div><div id="cmp-status"></div>';
        var chart = document.getElementById("cmp-chart"), status = document.getElementById("cmp-status");
        function add() {
          var inp = document.getElementById("cmp-q"), needle = inp.value.trim().toLowerCase();
          if (!needle) return;
          var hit = list.filter(function (m) { return m.name.toLowerCase() === needle; })[0] || list.filter(function (m) {
            var hay = [m.name, m.grade, m.manufacturer].join(" ").toLowerCase(); return needle.split(/\s+/).every(function (t) { return hay.indexOf(t) >= 0; }); })[0];
          if (hit && st.m.indexOf(hit.id) < 0 && st.m.length < CMP_MAX) { st.m.push(hit.id); inp.value = ""; update(); }
          else if (!hit) inp.setCustomValidity("No material matches."), inp.reportValidity(), inp.setCustomValidity("");
        }
        document.getElementById("cmp-add").addEventListener("click", add);
        document.getElementById("cmp-q").addEventListener("keydown", function (ev) { if (ev.key === "Enter") { ev.preventDefault(); add(); } });
        document.querySelectorAll("#app [data-q]").forEach(function (b) { b.addEventListener("click", function () { st.q = b.getAttribute("data-q"); st.y = ""; update(); }); });

        function update() {
          var cfg = CMP[st.q], mine = ++token;
          if (cfg.y.indexOf(st.y) < 0) st.y = cfg.y[0];
          writeQuery("compare", { m: st.m.join(","), q: st.q === "core_loss" ? "" : st.q, f: cfg.f && st.f !== 1e5 ? st.f : "", t: st.t === 100 ? "" : String(st.t),
            y: st.y === cfg.y[0] ? "" : st.y, "class": st.dc });
          document.querySelectorAll("#app [data-q]").forEach(function (b) { b.setAttribute("aria-selected", b.getAttribute("data-q") === st.q ? "true" : "false"); });
          document.getElementById("cmp-chips").innerHTML = st.m.map(function (id, k) {
            return '<span class="chip"><span class="swatch" style="background:' + COLORS[k] + '"></span><a href="#/material/' + encodeURIComponent(id) + '">' + esc(byId[id].name) +
              '</a> <button type="button" class="chip-x" data-rm="' + esc(id) + '" aria-label="Remove ' + esc(byId[id].name) + '">×</button></span>'; }).join("") ||
            '<span class="muted small">No material selected yet. <a href="#/compare?m=' + CMP_EXAMPLE.filter(function (id) { return byId[id]; }).join(",") + '">Try N87, 3C94 and 3F4</a>.</span>';
          document.querySelectorAll("[data-rm]").forEach(function (b) { b.addEventListener("click", function () { st.m = st.m.filter(function (id) { return id !== b.getAttribute("data-rm"); }); update(); }); });
          var ctl = document.getElementById("cmp-ctl");
          ctl.innerHTML = (cfg.f ? "<label>Frequency<select id=\"cmp-f\">" + CMP_F.map(function (f) { return '<option value="' + f + '"' + (f === st.f ? " selected" : "") + ">" + esc(qty(f, "Hz")) + "</option>"; }).join("") + "</select></label>" : "") +
            '<label>Temperature<select id="cmp-t">' + CMP_T.map(function (v) { return '<option value="' + v + '"' + (v === st.t ? " selected" : "") + ">" + v + " °C</option>"; }).join("") + "</select></label>" +
            (cfg.y.length > 1 ? '<label>Quantity<select id="cmp-y">' + cfg.y.map(function (y) { return '<option value="' + y + '"' + (y === st.y ? " selected" : "") + ">" + esc(colLabel(y)) + "</option>"; }).join("") + "</select></label>" : "") +
            '<label>Data class<select id="cmp-dc"><option value="">Best available</option>' + ["experimental_measurement", "manufacturer_datasheet"].map(function (k) {
              return '<option value="' + k + '"' + (k === st.dc ? " selected" : "") + ">" + esc(DATA_CLASS[k][0]) + " only</option>"; }).join("") + "</select></label>";
          bind("cmp-f", "change", function (v) { st.f = Number(v); update(); });
          bind("cmp-t", "change", function (v) { st.t = Number(v); update(); });
          bind("cmp-y", "change", function (v) { st.y = v; update(); });
          bind("cmp-dc", "change", function (v) { st.dc = v; update(); });
          status.innerHTML = "";
          if (!st.m.length) { chart.className = ""; chart.innerHTML = '<div class="notice info">Add two or more materials to compare them.</div>'; return; }
          chart.className = ""; chart.innerHTML = '<p class="loading" role="status">Loading tables…</p>';
          Promise.all(st.m.map(function (id) {
            return load("data/materials/" + file(id)).then(function (detail) {
              var cands = rankTables(detail.measurements.filter(function (e) {
                return e.quantity === st.q && e.published !== "none" && e.status === "primary" && (!st.dc || e.data_class === st.dc); })).slice(0, 6);
              var tryNext = function (k) {
                if (k >= cands.length) return { id: id, tried: cands.length };
                return load("data/measurements/" + file(cands[k].id)).then(function (d) {
                  var sel = compareRows(d, cfg, st);
                  return sel.rows.length ? { id: id, entry: cands[k], d: d, sel: sel } : tryNext(k + 1);
                });
              };
              return tryNext(0);
            });
          })).then(function (res) {
            if (mine !== token) return;
            var X = null, Y = null;
            var series = res.map(function (r, k) {
              if (!r.sel) return null;
              var cols = r.d.columns; X = X || colInfo(cols[r.sel.xi]); Y = Y || colInfo(cols[r.sel.yi]);
              var cx = colInfo(cols[r.sel.xi]), cy = colInfo(cols[r.sel.yi]);
              var pts = r.sel.rows.map(function (row) { return [cx.conv(row[r.sel.xi]), cy.conv(row[r.sel.yi]), { row: row, r: r }]; });
              var mode = cfg.mode, dir = 0;
              if (mode === "lines" && st.q !== "bh_curve") for (var i = 1; i < pts.length; i++) {
                var s = pts[i][0] > pts[i - 1][0] ? 1 : pts[i][0] < pts[i - 1][0] ? -1 : 0;
                if (s && dir && s !== dir) { mode = "points"; break; }
                dir = s || dir;
              }
              return { label: byId[r.id].name + " · " + (DATA_CLASS[r.entry.data_class] || [r.entry.data_class])[0].toLowerCase(), color: COLORS[k], mode: mode, points: pts };
            }).filter(Boolean);
            if (!series.length) { chart.innerHTML = '<div class="notice info">No selected material has published ' + esc((QTY[st.q] || st.q).toLowerCase()) + " rows at these conditions. Try another temperature" + (cfg.f ? " or frequency" : "") + ".</div>"; }
            else MiniPlot.render(chart, {
              ariaLabel: "Comparison of " + Y.label + " versus " + X.label, series: series,
              x: { label: X.axis, log: !!cfg.logx, format: tick(X.unit) }, y: { label: Y.axis, log: !!cfg.logy, format: tick(Y.unit) },
              note: "Each curve shows rows of one table; curves of different data classes or sources are not directly equivalent.",
              extra: function (o) {
                return o.r.sel.names.filter(function (n) { return n !== cfg.x && n !== st.y && o.row[o.r.sel.names.indexOf(n)] !== null; }).slice(0, 6).map(function (n) {
                  var c = o.r.d.columns[o.r.sel.names.indexOf(n)], v = o.row[o.r.sel.names.indexOf(n)];
                  return '<br><span class="muted">' + esc(colLabel(n)) + ": " + esc(c.unit ? qty(v, c.unit) : v) + "</span>"; }).join("");
              }
            });
            status.innerHTML = '<h2>Tables used</h2><div class="table-wrap"><table><thead><tr><th>Material</th><th>Table</th><th>Data class</th><th>Rows selected at</th><th class="num">Points</th></tr></thead><tbody>' +
              res.map(function (r, k) {
                var name = '<span class="swatch" style="background:' + COLORS[k] + '"></span><a href="#/material/' + encodeURIComponent(r.id) + '">' + esc(byId[r.id].name) + "</a>";
                if (!r.sel) return "<tr><td>" + name + '</td><td colspan="4" class="muted">' + (r.tried ? "None of its " + r.tried + " published " + esc((QTY[st.q] || st.q).toLowerCase()) + " tables has rows at these conditions." :
                  "No published " + esc((QTY[st.q] || st.q).toLowerCase()) + " table" + (st.dc ? " of this data class" : "") + ".") + "</td></tr>";
                return "<tr><td>" + name + '</td><td><a href="#/measurement/' + encodeURIComponent(r.entry.id) + '">' + esc(r.entry.name) + '</a><div class="small muted">' + esc(src(r.entry.source_id)) +
                  (r.d.sampling.sampled ? " · sampled preview, " + int(r.d.sampling.published_rows) + " of " + int(r.d.sampling.total_rows) + " rows" : "") + "</div></td><td>" + dcBadge(r.entry.data_class) +
                  '</td><td class="small">' + esc(r.sel.used.join(" · ") || "all rows") + '</td><td class="num">' + int(r.sel.rows.length) + "</td></tr>";
              }).join("") + "</tbody></table></div>" +
              '<p class="small muted">Selection rule: ' + (cfg.f ? "frequency within ±" + (F_TOL * 100) + " % of the target; " : "") + "the table's nearest temperature within ±" + T_TOL +
              " °C; sinusoidal excitation where the table distinguishes waveforms; measured tables are preferred over datasheet curves unless a data class is chosen.</p>";
          }).catch(function (err) { if (mine === token) chart.innerHTML = '<div class="notice">Could not load the tables. ' + esc(err && err.message || err) + "</div>"; });
        }
        update();
      };
    });
  }

  // -- core catalogue ----------------------------------------------------------------------------------------
  function mm(v) { return v === undefined || v === null ? NA : '<span title="' + v + ' m">' + esc(num(v * 1e3)) + "</span>"; }
  function partsTable(data) {
    var C = {}; data.columns.forEach(function (c, i) { C[c] = i; });
    return C;
  }
  function gapText(g) {
    if (!g || !g.length) return "ungapped";
    return g.map(function (x) { return (x[0] || "gap") + (x[1] !== null && x[1] !== undefined ? " " + num(x[1] * 1e3) + " mm" : ""); }).join(", ");
  }
  function cores(q) {
    setTitle("Core catalogue");
    var view = q.get("view") === "parts" ? "parts" : "shapes";
    var need = [load("data/cores/shapes.json")];
    if (view === "parts") need.push(load("data/cores/parts.json"), load("data/materials.json"));
    return Promise.all(need).then(function (d) {
      var shapes = d[0].shapes, parts = d[1], mats = d[2];
      return function () {
        var tabs = '<nav class="tabs" aria-label="Catalogue"><a href="#/cores"' + (view === "shapes" ? ' aria-current="page"' : "") + ">Standard shapes (" + int(shapes.length) + ")</a>" +
          '<a href="#/cores?view=parts"' + (view === "parts" ? ' aria-current="page"' : "") + ">Manufacturer parts</a></nav>";
        app.innerHTML = '<h1>Core catalogue</h1><p class="lead">Standard core shapes and manufacturer parts from OpenMagnetics MAS (Apache-2.0). ' +
          "Shapes carry dimension tolerance ranges as published. Effective parameters (A<sub>e</sub>, l<sub>e</sub>, V<sub>e</sub>) are not published by the source and are not computed here.</p>" + tabs +
          '<div class="filters" id="cf"></div><p id="count" class="muted small" aria-live="polite"></p><div id="list"></div><div class="more" id="more"></div>';
        var limit = PAGE, cf = document.getElementById("cf");
        if (view === "shapes") {
          var st = { q: q.get("q") || "", family: q.get("family") || "" };
          cf.innerHTML = '<label class="wide">Search shape name or alias<input type="search" id="c-q" value="' + esc(st.q) + '" placeholder="e.g. E 42/21/15, ETD 34, PQ"></label>' +
            select("c-family", "Family", uniq(shapes.map(function (s) { return s.family; })), st.family, function (v) { return v.toUpperCase(); });
          var upd = function () {
            writeQuery("cores", { q: st.q, family: st.family });
            var n = st.q.trim().toLowerCase();
            var rows = shapes.filter(function (s) {
              if (st.family && s.family !== st.family) return false;
              return !n || [s.name].concat(s.aliases || []).join(" ").toLowerCase().indexOf(n) >= 0;
            });
            document.getElementById("count").textContent = int(rows.length) + " of " + int(shapes.length) + " shapes";
            document.getElementById("list").innerHTML = rows.length ? '<div class="table-wrap"><table><thead><tr><th>Shape</th><th>Family</th><th class="hide-sm">Type</th>' +
              '<th class="hide-sm">Magnetic circuit</th><th class="num">Dimensions</th><th>Nominal values</th></tr></thead><tbody>' + rows.slice(0, limit).map(function (s) {
                return '<tr><td><a href="#/core/' + encodeURIComponent(s.id) + '">' + esc(s.name) + "</a>" + (s.aliases && s.aliases.length ? '<div class="small muted">' + esc(s.aliases.join(", ")) + "</div>" : "") +
                  "</td><td>" + esc((s.family || "").toUpperCase()) + '</td><td class="hide-sm">' + esc(s.type || "") + '</td><td class="hide-sm">' + (s.circuit ? esc(s.circuit) : NA) +
                  '</td><td class="num">' + Object.keys(s.ranges || {}).length + "</td><td>" + (s.nominal ? "published" : NA) + "</td></tr>";
              }).join("") + "</tbody></table></div>" : '<div class="notice info">No shape matches.</div>';
            document.getElementById("more").innerHTML = rows.length > limit ? '<button class="secondary" id="more-btn">Show ' + Math.min(PAGE, rows.length - limit) + " more</button>" : "";
            var mb = document.getElementById("more-btn"); if (mb) mb.addEventListener("click", function () { limit += PAGE; upd(); });
          };
          bind("c-q", "input", function (v) { st.q = v; limit = PAGE; upd(); });
          bind("c-family", "change", function (v) { st.family = v; limit = PAGE; upd(); });
          upd();
          return;
        }
        var C = partsTable(parts), rows0 = parts.rows, matName = {};
        mats.forEach(function (m) { matName[m.id] = m.name; });
        var shapeName = {}; shapes.forEach(function (s) { shapeName[s.id] = s.name; });
        var sp = { q: q.get("q") || "", mfr: q.get("manufacturer") || "", status: q.get("status") || "", material: q.get("material") || "", shape: q.get("shape") || "" };
        cf.innerHTML = '<label class="wide">Search part number, name or material<input type="search" id="p-q" value="' + esc(sp.q) + '" placeholder="e.g. ETD34, N87, 0077"></label>' +
          select("p-mfr", "Manufacturer", uniq(rows0.map(function (r) { return r[C.manufacturer]; })), sp.mfr) +
          select("p-status", "Status", uniq(rows0.map(function (r) { return r[C.status]; })), sp.status) +
          (sp.material ? '<p class="small" style="margin:0">Material: <strong>' + esc(matName[sp.material] || sp.material) + '</strong> <a href="#" id="clr-m">clear</a></p>' : "") +
          (sp.shape ? '<p class="small" style="margin:0">Shape: <strong>' + esc(shapeName[sp.shape] || sp.shape) + '</strong> <a href="#" id="clr-s">clear</a></p>' : "");
        var updp = function () {
          writeQuery("cores", { view: "parts", q: sp.q, manufacturer: sp.mfr, status: sp.status, material: sp.material, shape: sp.shape });
          var n = sp.q.trim().toLowerCase();
          var rows = rows0.filter(function (r) {
            if (sp.mfr && r[C.manufacturer] !== sp.mfr) return false;
            if (sp.status && r[C.status] !== sp.status) return false;
            if (sp.material && r[C.material] !== sp.material) return false;
            if (sp.shape && r[C.shape_ref] !== sp.shape) return false;
            if (n) {
              var hay = [r[C.part_number], r[C.name], r[C.material_name], r[C.shape_name]].join(" ").toLowerCase();
              return n.split(/\s+/).every(function (t) { return hay.indexOf(t) >= 0; });
            }
            return true;
          });
          document.getElementById("count").textContent = int(rows.length) + " of " + int(rows0.length) + " parts";
          document.getElementById("list").innerHTML = rows.length ? '<div class="table-wrap"><table><thead><tr><th>Part</th><th>Manufacturer</th><th>Material</th>' +
            '<th class="hide-sm">Shape</th><th class="hide-sm">Gapping</th><th class="hide-sm">Status</th><th>Datasheet</th></tr></thead><tbody>' + rows.slice(0, limit).map(function (r) {
              return '<tr><td><a href="#/core/' + encodeURIComponent(r[C.id]) + '">' + esc(r[C.part_number] || r[C.name]) + "</a></td><td>" + esc(r[C.manufacturer] || "") + "</td><td>" +
                (r[C.material] ? '<a href="#/material/' + encodeURIComponent(r[C.material]) + '">' + esc(r[C.material_name]) + "</a>" : esc(r[C.material_name] || "") || NA) +
                '</td><td class="hide-sm">' + (r[C.shape_ref] ? '<a href="#/core/' + encodeURIComponent(r[C.shape_ref]) + '">' + esc(r[C.shape_name]) + "</a>" : esc(r[C.shape_name] || "")) +
                '</td><td class="hide-sm small">' + esc(gapText(r[C.gapping])) + '</td><td class="hide-sm">' + esc(r[C.status] || "") + "</td><td>" +
                (r[C.datasheet_url] ? '<a href="' + esc(r[C.datasheet_url]) + '" rel="noopener nofollow">PDF</a>' : NA) + "</td></tr>";
            }).join("") + "</tbody></table></div>" : '<div class="notice info">No part matches.</div>';
          document.getElementById("more").innerHTML = rows.length > limit ? '<button class="secondary" id="more-btn">Show ' + Math.min(PAGE, rows.length - limit) + " more</button>" : "";
          var mb = document.getElementById("more-btn"); if (mb) mb.addEventListener("click", function () { limit += PAGE; updp(); });
        };
        bind("p-q", "input", function (v) { sp.q = v; limit = PAGE; updp(); });
        bind("p-mfr", "change", function (v) { sp.mfr = v; limit = PAGE; updp(); });
        bind("p-status", "change", function (v) { sp.status = v; limit = PAGE; updp(); });
        [["clr-m", "material"], ["clr-s", "shape"]].forEach(function (c) {
          var a = document.getElementById(c[0]); if (a) a.addEventListener("click", function (ev) { ev.preventDefault(); sp[c[1]] = ""; a.parentNode.remove(); limit = PAGE; updp(); });
        });
        updp();
      };
    });
  }

  function core(id) {
    var isPart = id.indexOf("core.mas-part-") === 0;
    return Promise.all([load("data/cores/shapes.json"), load("data/cores/parts.json"), isPart ? load("data/materials.json") : null]).then(function (d) {
      var shapes = d[0].shapes, parts = d[1], C = partsTable(parts);
      var lic = '<p class="small muted">Source: OpenMagnetics MAS, Apache-2.0. Third-party data, not independently verified.</p>';
      if (!isPart) {
        var s = shapes.filter(function (x) { return x.id === id; })[0];
        if (!s) throw new Error("No core shape '" + id + "'.");
        setTitle(s.name);
        var using = parts.rows.filter(function (r) { return r[C.shape_ref] === id; });
        return function () {
          var letters = uniq(Object.keys(s.ranges || {}).concat(Object.keys(s.nominal || {})));
          app.innerHTML = '<div class="crumbs"><a href="#/cores">Core catalogue</a> / ' + esc(s.name) + "</div><h1>" + esc(s.name) + "</h1>" +
            '<div class="head-meta"><span class="badge">' + esc((s.family || "").toUpperCase()) + '</span><span class="badge">' + esc(s.type || "") + "</span>" +
            (s.circuit ? '<span class="badge">' + esc(s.circuit) + " magnetic circuit</span>" : "") + '<span class="badge">' + esc(s.license) + "</span></div>" + lic +
            "<h2>Dimensions [mm]</h2>" + '<div class="table-wrap"><table><thead><tr><th>Dimension</th><th class="num">Minimum</th><th class="num">Maximum</th><th class="num">Nominal</th></tr></thead><tbody>' +
            letters.map(function (k) {
              var r = (s.ranges || {})[k] || {}, bad = s.inconsistent && s.inconsistent[k];
              return "<tr><td>" + esc(k) + (bad ? ' <span class="badge warn" title="upstream minimum exceeds maximum; kept as published">inverted upstream</span>' : "") +
                '</td><td class="num">' + mm(r.minimum_m) + '</td><td class="num">' + mm(r.maximum_m) + '</td><td class="num">' + mm(s.nominal && s.nominal[k]) + "</td></tr>";
            }).join("") + "</tbody></table></div>" +
            '<p class="small muted">Stored in metres; shown in millimetres. ' + NA + " = not published. " + esc(d[0].note) + "</p>" +
            "<p>Quality flags: " + flags(s.flags) + "</p>" +
            "<h2>Effective parameters</h2><p class=\"muted\">Not published by the source for this shape; this explorer does not calculate them.</p>" +
            "<h2>Manufacturer parts</h2>" + (using.length ? '<p><a href="#/cores?view=parts&shape=' + encodeURIComponent(id) + '">' + int(using.length) + " parts use this shape</a></p>" : '<p class="muted">No part in the catalogue references this shape.</p>') +
            "<h2>Provenance</h2><p class=\"small\">" + (s.evidence ? '<a href="' + esc(s.evidence) + '" rel="noopener">MAS core_shapes.ndjson record</a> · ' : "") + "<code>" + esc(s.id) + "</code></p>" +
            (s.description ? '<p class="small muted">' + esc(s.description) + "</p>" : "");
        };
      }
      var r = parts.rows.filter(function (x) { return x[C.id] === id; })[0];
      if (!r) throw new Error("No core part '" + id + "'.");
      var matName = {}; (d[2] || []).forEach(function (m) { matName[m.id] = m.name; });
      setTitle(r[C.part_number] || r[C.name]);
      return function () {
        var g = r[C.gapping] || [];
        var kv = [["Name", esc(r[C.name])], ["Part number", esc(r[C.part_number] || "") || NA], ["Manufacturer", esc(r[C.manufacturer] || "") || NA],
          ["Material", r[C.material] ? '<a href="#/material/' + encodeURIComponent(r[C.material]) + '">' + esc(matName[r[C.material]] || r[C.material_name]) + "</a>" : (esc(r[C.material_name] || "") || NA) + ' <span class="small muted">(not linked to a canonical material)</span>'],
          ["Shape", r[C.shape_ref] ? '<a href="#/core/' + encodeURIComponent(r[C.shape_ref]) + '">' + esc(r[C.shape_name]) + "</a>" : esc(r[C.shape_name] || "") || NA],
          ["Set", esc(r[C.set_description] || "") || NA], ["Stacks", r[C.number_stacks] === null || r[C.number_stacks] === undefined ? NA : esc(r[C.number_stacks])],
          ["Coating", esc(r[C.coating] || "") || NA], ["Status", esc(r[C.status] || "") || NA],
          ["Datasheet", r[C.datasheet_url] ? '<a href="' + esc(r[C.datasheet_url]) + '" rel="noopener nofollow">' + esc(r[C.datasheet_url]) + "</a>" : NA],
          ["Effective parameters", '<span class="muted">Not published by the source; not computed here.</span>'],
          ["Record", "<code>" + esc(r[C.id]) + '</code> · <a href="' + esc(parts.evidence_base + "#L" + r[C.source_line]) + '" rel="noopener">MAS cores.ndjson line ' + esc(r[C.source_line]) + "</a>"],
          ["Licence", esc(r[C.license])]];
        app.innerHTML = '<div class="crumbs"><a href="#/cores?view=parts">Manufacturer parts</a> / ' + esc(r[C.part_number] || r[C.name]) + "</div><h1>" + esc(r[C.part_number] || r[C.name]) + "</h1>" + lic +
          '<div class="table-wrap"><table class="kv"><tbody>' + kv.map(function (x) { return "<tr><th>" + esc(x[0]) + "</th><td>" + x[1] + "</td></tr>"; }).join("") + "</tbody></table></div>" +
          "<h2>Gapping</h2>" + (g.length ? '<div class="table-wrap"><table><thead><tr><th>Gap</th><th>Type</th><th class="num">Length [mm]</th></tr></thead><tbody>' + g.map(function (x, i) {
            return "<tr><td>" + (i + 1) + "</td><td>" + esc(x[0] || "") + '</td><td class="num">' + mm(x[1]) + "</td></tr>"; }).join("") + "</tbody></table></div>" : "<p>Ungapped (no gaps published).</p>");
      };
    });
  }

  // -- boot --------------------------------------------------------------------------------------------------
  load("data/summary.json").then(function (s) {
    var b = document.getElementById("build-info");
    if (b) b.innerHTML = "Data: package " + esc(s.version) + ", commit <a href=\"" + esc(s.repository) + "/tree/" + esc(s.commit) + "\"><code>" + esc(String(s.commit).slice(0, 7)) +
      "</code></a> (" + esc(String(s.data_date).slice(0, 10)) + ").";
  }).catch(function () {});
  render();
})();
