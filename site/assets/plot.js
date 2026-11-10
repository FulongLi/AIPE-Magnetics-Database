/* MiniPlot: a small, dependency-free SVG x-y plotter for the explorer (linear/log axes, points or lines,
   nearest-point tooltip, legend with series toggling, optional click on a point). It draws the values it is given; it never fits,
   smooths or interpolates. */
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";

  function el(name, attrs, parent) {
    var n = document.createElementNS(NS, name);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }

  function niceStep(span, count) {
    var raw = span / Math.max(1, count), mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * mag;
  }

  function linearTicks(lo, hi, count) {
    if (lo === hi) { var d = Math.abs(lo) * 0.1 || 1; lo -= d; hi += d; }
    var step = niceStep(hi - lo, count), a = Math.floor(lo / step) * step, b = Math.ceil(hi / step) * step, t = [];
    for (var v = a; v <= b + step * 1e-9; v += step) t.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return { lo: a, hi: b, ticks: t };
  }

  function logTicks(lo, hi) {
    var a = Math.floor(Math.log10(lo)), b = Math.ceil(Math.log10(hi));
    if (a === b) { a -= 1; b += 1; }
    var t = [], minor = [];
    for (var e = a; e <= b; e++) {
      t.push(Math.pow(10, e));
      if (e < b) for (var m = 2; m <= 9; m++) minor.push(m * Math.pow(10, e));
    }
    if (b - a <= 2) { // few decades: label 2 and 5 as well
      minor.forEach(function (v) { var m = Math.round(v / Math.pow(10, Math.floor(Math.log10(v) + 1e-9))); if (m === 2 || m === 5) t.push(v); });
      t.sort(function (x, y) { return x - y; });
    }
    return { lo: Math.pow(10, a), hi: Math.pow(10, b), ticks: t, minor: minor };
  }

  function render(container, spec) {
    container.innerHTML = "";
    container.classList.add("plot");
    var hidden = spec.hidden || {};
    var tip = document.createElement("div");
    tip.className = "tooltip";
    var svgHost = document.createElement("div");
    container.appendChild(svgHost);
    container.appendChild(tip);
    var note = document.createElement("p");
    note.className = "plot-note";
    container.appendChild(note);
    var legend = document.createElement("ul");
    legend.className = "legend-list";
    if (spec.series.length > 1) container.appendChild(legend);

    var fx = spec.x.format || String, fy = spec.y.format || String;
    var visiblePts = [];

    function draw() {
      svgHost.innerHTML = "";
      var W = Math.max(300, container.clientWidth - 18), H = Math.round(Math.min(440, Math.max(260, W * 0.56)));
      var m = { l: W < 480 ? 54 : 70, r: 26, t: 12, b: 46 }; // right margin holds half of the last tick label
      var dropped = 0, xs = [], ys = [];
      var series = spec.series.map(function (s, i) {
        var pts = [];
        s.points.forEach(function (p) {
          var x = p[0], y = p[1];
          if (x === null || y === null || !isFinite(x) || !isFinite(y)) return;
          if ((spec.x.log && x <= 0) || (spec.y.log && y <= 0)) { dropped++; return; }
          pts.push(p);
        });
        if (!hidden[i]) pts.forEach(function (p) { xs.push(p[0]); ys.push(p[1]); });
        return { s: s, i: i, pts: pts };
      });
      var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": spec.ariaLabel || "chart" }, svgHost);
      if (!xs.length) {
        el("text", { x: W / 2, y: H / 2, "text-anchor": "middle" }, svg).textContent = "No plottable rows for the current selection.";
        note.textContent = dropped ? dropped + " non-positive values are hidden on logarithmic axes." : "";
        visiblePts = [];
        return;
      }
      var xmin = Math.min.apply(null, xs), xmax = Math.max.apply(null, xs), ymin = Math.min.apply(null, ys), ymax = Math.max.apply(null, ys);
      if (!spec.x.log && spec.x.zero && xmin > 0) xmin = 0;
      if (!spec.y.log && spec.y.zero && ymin > 0) ymin = 0;
      var X = spec.x.log ? logTicks(xmin, xmax) : linearTicks(xmin, xmax, W < 480 ? 4 : 7);
      var Y = spec.y.log ? logTicks(ymin, ymax) : linearTicks(ymin, ymax, 6);
      function scale(ax, t, a, b) {
        if (ax.log) { var l0 = Math.log10(t.lo), l1 = Math.log10(t.hi); return function (v) { return a + (Math.log10(v) - l0) / (l1 - l0) * (b - a); }; }
        return function (v) { return a + (v - t.lo) / (t.hi - t.lo) * (b - a); };
      }
      var sx = scale(spec.x, X, m.l, W - m.r), sy = scale(spec.y, Y, H - m.b, m.t);
      var grid = el("g", { class: "grid" }, svg);
      (X.minor || []).forEach(function (v) { if (v > X.lo && v < X.hi) el("line", { x1: sx(v), x2: sx(v), y1: m.t, y2: H - m.b, "stroke-opacity": 0.45 }, grid); });
      (Y.minor || []).forEach(function (v) { if (v > Y.lo && v < Y.hi) el("line", { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), "stroke-opacity": 0.45 }, grid); });
      var lastX = -1e9;
      X.ticks.forEach(function (v) {
        var px = sx(v);
        el("line", { x1: px, x2: px, y1: m.t, y2: H - m.b }, grid);
        if (px - lastX > 46) { el("text", { x: px, y: H - m.b + 16, "text-anchor": "middle" }, svg).textContent = fx(v, true); lastX = px; }
      });
      var lastY = 1e9;
      Y.ticks.forEach(function (v) {
        var py = sy(v);
        el("line", { x1: m.l, x2: W - m.r, y1: py, y2: py }, grid);
        if (lastY - py > 16) { el("text", { x: m.l - 6, y: py + 4, "text-anchor": "end" }, svg).textContent = fy(v, true); lastY = py; }
      });
      el("rect", { class: "frame", x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b }, svg);
      el("text", { class: "axis-title", x: (m.l + W - m.r) / 2, y: H - 8, "text-anchor": "middle" }, svg).textContent = spec.x.label;
      var yt = el("text", { class: "axis-title", x: 0, y: 0, "text-anchor": "middle", transform: "translate(13," + ((m.t + H - m.b) / 2) + ") rotate(-90)" }, svg);
      yt.textContent = spec.y.label;
      var clip = "clip" + Math.random().toString(36).slice(2);
      el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b }, el("clipPath", { id: clip }, el("defs", {}, svg)));
      var layer = el("g", { "clip-path": "url(#" + clip + ")" }, svg);
      visiblePts = [];
      var total = series.reduce(function (n, s) { return n + (hidden[s.i] ? 0 : s.pts.length); }, 0);
      var dot = total > 1500 ? 4 : total > 300 ? 6 : 8;
      series.forEach(function (s) {
        if (hidden[s.i] || !s.pts.length) return;
        var color = s.s.color, d = "";
        if (s.s.mode === "lines") {
          s.pts.forEach(function (p, j) { d += (j ? "L" : "M") + sx(p[0]).toFixed(1) + " " + sy(p[1]).toFixed(1); });
          el("path", { d: d, fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, layer);
          d = "";
        }
        if (s.s.mode !== "lines" || s.pts.length <= 60) {
          s.pts.forEach(function (p) { d += "M" + sx(p[0]).toFixed(1) + " " + sy(p[1]).toFixed(1) + "h0"; });
          el("path", { d: d, fill: "none", stroke: color, "stroke-width": s.s.mode === "lines" ? 6 : dot, "stroke-linecap": "round",
            "stroke-opacity": total > 1500 ? 0.75 : 1 }, layer);
        }
        s.pts.forEach(function (p) { visiblePts.push({ px: sx(p[0]), py: sy(p[1]), p: p, s: s.s }); });
      });
      var hl = el("circle", { r: 6, fill: "none", stroke: "currentColor", "stroke-width": 2, visibility: "hidden" }, svg);
      hl.style.color = "var(--fg)";
      note.textContent = (spec.note ? spec.note + " " : "") + (dropped ? dropped + " non-positive values are hidden on logarithmic axes." : "");

      var picked = null;
      function move(ev) {
        var r = svg.getBoundingClientRect(), k = W / r.width;
        var mx = (ev.clientX - r.left) * k, my = (ev.clientY - r.top) * k, best = null, bd = 400 * k * k;
        for (var i = 0; i < visiblePts.length; i++) {
          var q = visiblePts[i], dd = (q.px - mx) * (q.px - mx) + (q.py - my) * (q.py - my);
          if (dd < bd) { bd = dd; best = q; }
        }
        if (!best) { picked = null; svg.style.cursor = ""; tip.style.display = "none"; hl.setAttribute("visibility", "hidden"); return; }
        hl.setAttribute("cx", best.px); hl.setAttribute("cy", best.py); hl.setAttribute("visibility", "visible");
        picked = best;
        svg.style.cursor = spec.onPick ? "pointer" : "";
        tip.innerHTML = (spec.title && best.p[2] !== undefined ? "<strong>" + escapeHtml(spec.title(best.p[2])) + "</strong><br>" : "") + (spec.series.length > 1 ? '<span class="sw" style="background:' + best.s.color + '"></span>' + escapeHtml(best.s.label) + "<br>" : "") +
          escapeHtml(spec.x.label) + ": " + escapeHtml(fx(best.p[0])) + "<br>" + escapeHtml(spec.y.label) + ": " + escapeHtml(fy(best.p[1])) +
          (spec.extra && best.p[2] !== undefined ? spec.extra(best.p[2]) : "");
        tip.style.display = "block";
        var cr = container.getBoundingClientRect(), left = best.px / k + (r.left - cr.left) + 12, top = best.py / k + (r.top - cr.top) - 10;
        if (left + tip.offsetWidth > cr.width - 4) left = left - tip.offsetWidth - 24;
        tip.style.left = Math.max(4, left) + "px"; tip.style.top = Math.max(4, top) + "px";
      }
      svg.addEventListener("pointermove", move);
      svg.addEventListener("pointerdown", move);
      svg.addEventListener("pointerleave", function () { picked = null; tip.style.display = "none"; hl.setAttribute("visibility", "hidden"); });
      if (spec.onPick) svg.addEventListener("click", function () { if (picked && picked.p[2] !== undefined) spec.onPick(picked.p[2]); });
    }

    function drawLegend() {
      legend.innerHTML = "";
      spec.series.forEach(function (s, i) {
        var li = document.createElement("li"), b = document.createElement("button");
        b.type = "button"; b.className = "secondary"; b.style.padding = "2px 8px"; b.style.minHeight = "28px";
        b.setAttribute("aria-pressed", hidden[i] ? "false" : "true");
        b.innerHTML = '<span class="swatch" style="background:' + (hidden[i] ? "transparent;border:1px solid " + s.color : s.color) + '"></span>' + escapeHtml(s.label);
        if (hidden[i]) b.style.opacity = "0.6";
        b.title = hidden[i] ? "Show series" : "Hide series";
        b.addEventListener("click", function () { hidden[i] = !hidden[i]; drawLegend(); draw(); });
        li.appendChild(b); legend.appendChild(li);
      });
    }

    function escapeHtml(s) { return String(s).replace(/[&<>"']/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; }); }

    drawLegend();
    draw();
    if (window.ResizeObserver) {
      var lastW = container.clientWidth, ro = new ResizeObserver(function () {
        if (!container.isConnected) { ro.disconnect(); return; }
        if (Math.abs(container.clientWidth - lastW) > 4) { lastW = container.clientWidth; draw(); }
      });
      ro.observe(container);
    }
  }

  window.MiniPlot = { render: render };
})();
