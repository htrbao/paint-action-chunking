/* ============================================================
   PAINT — interactive diagrams (vanilla JS)
   ============================================================ */
(function () {
  "use strict";
  const SVGNS = "http://www.w3.org/2000/svg";
  const prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function el(tag, attrs, kids) {
    const n = document.createElementNS(SVGNS, tag);
    if (attrs) for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (kids) (Array.isArray(kids) ? kids : [kids]).forEach((c) => n.appendChild(c));
    return n;
  }
  const lerp = (a, b, t) => a + (b - a) * t;
  const easeOut = (t) => 1 - Math.pow(1 - t, 3);

  /* ----------------------------------------------------------
     Shared trajectory data (normalized action value 0..1)
     ---------------------------------------------------------- */
  const H = 8, D = 2;
  const vB = 0.52;
  const NAIVE  = [0.80, 0.74, 0.66, 0.60, 0.55, 0.535, 0.55, 0.585];
  const TARGET = [0.52, 0.585, 0.66, 0.60, 0.55, 0.535, 0.55, 0.585];
  const PAINT  = [0.52, 0.585, 0.60, 0.585, 0.55, 0.535, 0.55, 0.585];
  const FLAT   = new Array(H).fill(vB);
  const FREE_NOISE = [0.42, 0.70, 0.52, 0.78, 0.46, 0.62, 0.56, 0.73];

  function grayFill(v) {
    const lo = 0.30, hi = 0.86; const g = lo + (hi - lo) * v;
    const c = Math.round(g * 255);
    return `rgb(${c},${c},${c})`;
  }

  /* ============================================================
     BIG STAGE — Algorithm 1 step-through
     ============================================================ */
  function buildAlgoStage() {
    const host = document.getElementById("algo-svg");
    if (!host) return;

    const W = 560, Hh = 392;
    const xL = 70, xR = 520, colW = (xR - xL) / H;
    const cx = Array.from({ length: H }, (_, i) => xL + colW * (i + 0.5));
    const sq = 36;
    const sqY = 70;
    const pTop = 234, pBot = 356;
    const mapY = (v) => pBot - v * (pBot - pTop);

    const svg = el("svg", { viewBox: `0 0 ${W} ${Hh}`, class: "stage__svg", role: "img" });
    svg.setAttribute("aria-label", "Animated step-through of the PAINT algorithm");

    const lbl = (x, y, t, anchor, cls) =>
      el("text", { x, y, "text-anchor": anchor || "start", class: "dg-lbl" + (cls ? " " + cls : "") }, document.createTextNode(t));
    svg.appendChild(lbl(xL, 44, "INITIAL NOISE  x₀", "start"));
    svg.appendChild(lbl(xL, 214, "ACTION CHUNK  x₁", "start"));

    const squares = [];
    for (let i = 0; i < H; i++) {
      const r = el("rect", {
        x: cx[i] - sq / 2, y: sqY, width: sq, height: sq, rx: 6,
        fill: grayFill(FREE_NOISE[i]), stroke: "rgba(0,0,0,0.10)", "stroke-width": 1,
        class: "dg-sq",
      });
      svg.appendChild(r);
      squares.push(r);
    }
    const noiseBrace = el("rect", {
      x: cx[0] - sq / 2 - 4, y: sqY - 6, width: (cx[D - 1] - cx[0]) + sq + 8, height: sq + 12,
      rx: 9, fill: "none", stroke: "var(--accent)", "stroke-width": 1.6, class: "dg-brace",
    });
    svg.appendChild(noiseBrace);

    const arrowG = el("g", { class: "dg-arrow" });
    const aX = W / 2;
    const aLine = el("line", { x1: aX, y1: 128, x2: aX, y2: 196, stroke: "var(--accent)", "stroke-width": 2.4, "stroke-linecap": "round" });
    const aHead = el("path", { d: "", fill: "var(--accent)" });
    const aText = el("text", { x: aX + 16, y: 166, "text-anchor": "start", class: "dg-flow" }, document.createTextNode(""));
    arrowG.appendChild(aLine); arrowG.appendChild(aHead); arrowG.appendChild(aText);
    svg.appendChild(arrowG);

    const baseY = mapY(0.5);
    svg.appendChild(el("line", { x1: 30, y1: pBot + 10, x2: xR, y2: pBot + 10, stroke: "var(--border)", "stroke-width": 1 }));
    const bx = cx[0] - colW / 2;
    svg.appendChild(el("line", { x1: bx, y1: pTop - 8, x2: bx, y2: pBot + 16, stroke: "var(--border-2)", "stroke-width": 1, "stroke-dasharray": "3 4" }));
    svg.appendChild(el("text", { x: bx - 6, y: pTop - 12, "text-anchor": "end", class: "dg-mini" }, document.createTextNode("executed")));
    svg.appendChild(el("text", { x: bx + 6, y: pTop - 12, "text-anchor": "start", class: "dg-mini" }, document.createTextNode("new chunk")));

    const band = el("rect", {
      x: bx, y: pTop - 4, width: cx[D - 1] + colW / 2 - bx, height: (pBot - pTop) + 22,
      fill: "var(--accent-tint)", class: "dg-band", rx: 4,
    });
    band.style.opacity = "0";
    svg.appendChild(band);

    const exX0 = 36;
    svg.appendChild(el("path", {
      d: `M ${exX0} ${mapY(0.34)} L ${bx} ${mapY(0.45)} L ${cx[0]} ${mapY(vB)}`,
      fill: "none", stroke: "var(--ink-faint)", "stroke-width": 2.4, "stroke-linecap": "round", "stroke-linejoin": "round",
      opacity: 0.55,
    }));
    const bDot = el("circle", { cx: cx[0], cy: mapY(vB), r: 4.5, fill: "var(--ink-soft)" });
    svg.appendChild(bDot);

    const chunkPath = el("path", { d: "", fill: "none", "stroke-width": 2.8, "stroke-linecap": "round", "stroke-linejoin": "round", class: "dg-chunk" });
    chunkPath.style.opacity = "0";
    svg.appendChild(chunkPath);
    const dots = [];
    for (let i = 0; i < H; i++) {
      const c = el("circle", { cx: cx[i], cy: mapY(FLAT[i]), r: 3.6, class: "dg-dot" });
      c.style.opacity = "0";
      svg.appendChild(c);
      dots.push(c);
    }

    const jumpG = el("g", { class: "dg-jump" });
    const jLine = el("line", { x1: cx[0], y1: mapY(vB), x2: cx[0], y2: mapY(NAIVE[0]), stroke: "var(--warn)", "stroke-width": 2, "stroke-dasharray": "2 3" });
    const jTxt = el("text", { x: cx[0] + 9, y: mapY((vB + NAIVE[0]) / 2) + 4, class: "dg-jumptxt" }, document.createTextNode("jump"));
    jumpG.appendChild(jLine); jumpG.appendChild(jTxt);
    jumpG.style.opacity = "0";
    svg.appendChild(jumpG);

    host.appendChild(svg);

    function pathFrom(vals) {
      let d = "";
      for (let i = 0; i < H; i++) d += (i ? " L " : "M ") + cx[i].toFixed(1) + " " + mapY(vals[i]).toFixed(1);
      return d;
    }

    const STEPS = [
      {
        num: "01 / 06", title: "Sample fresh noise",
        body: "Start from an ordinary Gaussian noise vector. Each cell maps roughly to one position of the action chunk — the locality of optimal-transport flow matching.",
        eq: "x₀ᶠʳᵉᵉ ~ 𝒩(0, I)",
        chunk: null, color: "ink", arrow: 0, band: 0, jump: 0, noisePrefix: false,
      },
      {
        num: "02 / 06", title: "Naive forward pass",
        body: "Running the policy πθ produces a valid chunk — but under async delay its prefix doesn’t line up with the actions already executed. The robot would lurch at the boundary.",
        eq: "x₁ⁿᵃᶠᵉ = πθ(x₀ᶠʳᵉᵉ, oₜ)",
        chunk: NAIVE, color: "warn", arrow: 1, band: 0, jump: 1, noisePrefix: false,
      },
      {
        num: "03 / 06", title: "Construct the target",
        body: "Pin the prefix to the actions just executed; keep the naive tail so the target stays on the data manifold. This is the endpoint we want the flow to reach.",
        eq: "x₁ᵗᵃʳᶠᵉᵗ = [ executed prefix | naive tail ]",
        chunk: TARGET, color: "ink", arrow: 0, band: 1, jump: 0, noisePrefix: false,
      },
      {
        num: "04 / 06", title: "Invert the flow ODE",
        body: "Run the ODE backwards with Euler steps to recover the noise the model associates with this target. The prefix cells are now “anchored” — no gradients required.",
        eq: "xₜ₋Δₜ = xₜ − Δₜ·vθ(xₜ, oₜ, τ)   →  x₀ⁱⁿᵛ",
        chunk: TARGET, color: "ink", arrow: -1, band: 1, jump: 0, noisePrefix: true,
      },
      {
        num: "05 / 06", title: "Re-paint the noise",
        body: "Keep the inverted noise for the prefix; restore the original free noise for the tail. Mixing in fresh noise would smear the mismatch across every position — this avoids it.",
        eq: "x₀* = [ x₀ⁱⁿᵛ[:d] | x₀ᶠʳᵉᵉ[d:] ]",
        chunk: TARGET, color: "ink", arrow: 0, band: 1, jump: 0, noisePrefix: true,
      },
      {
        num: "06 / 06", title: "Forward pass → consistent chunk",
        body: "The unmodified policy, run from x₀*, lands on a chunk whose prefix matches the executed motion and whose tail follows the policy’s own distribution. Start right, arrive right.",
        eq: "Aₜ = πθ(x₀*, oₜ)    A₀··ᵈ⁻¹ ≈ executed prefix",
        chunk: PAINT, color: "accent", arrow: 1, band: 1, jump: 0, noisePrefix: true,
      },
    ];

    const colorVar = { warn: "var(--warn)", accent: "var(--accent)", ink: "var(--ink-soft)" };

    const tNum = document.getElementById("algo-num");
    const tTitle = document.getElementById("algo-title");
    const tBody = document.getElementById("algo-body");
    const tEq = document.getElementById("algo-eq");

    let cur = 0;
    let displayed = FLAT.slice();
    let raf = null;

    function setArrow(dir) {
      arrowG.style.opacity = dir === 0 ? "0.18" : "1";
      const top = 128, bot = 196;
      if (dir >= 0) {
        aLine.setAttribute("y1", top); aLine.setAttribute("y2", bot);
        aHead.setAttribute("d", `M ${aX - 7} ${bot - 9} L ${aX + 7} ${bot - 9} L ${aX} ${bot + 2} Z`);
        aText.textContent = dir === 1 ? "πθ  generate" : "";
      } else {
        aLine.setAttribute("y1", bot); aLine.setAttribute("y2", top);
        aHead.setAttribute("d", `M ${aX - 7} ${top + 9} L ${aX + 7} ${top + 9} L ${aX} ${top - 2} Z`);
        aText.textContent = "invert";
      }
    }

    function tweenTo(target, color) {
      const from = displayed.slice();
      const dur = prefersReduced ? 0 : 520;
      const t0 = performance.now();
      chunkPath.style.stroke = colorVar[color];
      dots.forEach((d) => (d.style.fill = colorVar[color]));
      cancelAnimationFrame(raf);
      function frame(now) {
        const t = dur === 0 ? 1 : Math.min(1, (now - t0) / dur);
        const e = easeOut(t);
        for (let i = 0; i < H; i++) displayed[i] = lerp(from[i], target[i], e);
        chunkPath.setAttribute("d", pathFrom(displayed));
        for (let i = 0; i < H; i++) dots[i].setAttribute("cy", mapY(displayed[i]).toFixed(1));
        if (t < 1) raf = requestAnimationFrame(frame);
      }
      raf = requestAnimationFrame(frame);
    }

    function render(i) {
      const s = STEPS[i];
      cur = i;
      tNum.textContent = s.num; tTitle.textContent = s.title; tBody.textContent = s.body; tEq.textContent = s.eq;
      if (s.chunk) {
        chunkPath.style.opacity = "1"; dots.forEach((d) => (d.style.opacity = "1"));
        tweenTo(s.chunk, s.color);
      } else {
        chunkPath.style.opacity = "0"; dots.forEach((d) => (d.style.opacity = "0"));
        displayed = FLAT.slice();
      }
      setArrow(s.arrow);
      band.style.opacity = s.band ? "1" : "0";
      jumpG.style.opacity = s.jump ? "1" : "0";
      for (let k = 0; k < H; k++) {
        if (s.noisePrefix && k < D) {
          squares[k].setAttribute("fill", "var(--accent)");
          squares[k].setAttribute("stroke", "var(--accent-deep)");
        } else {
          squares[k].setAttribute("fill", grayFill(FREE_NOISE[k]));
          squares[k].setAttribute("stroke", "rgba(0,0,0,0.10)");
        }
      }
      noiseBrace.style.opacity = s.noisePrefix ? "1" : "0";
      document.querySelectorAll("#algo-dots button").forEach((b, bi) => b.classList.toggle("on", bi === i));
      prevBtn.disabled = i === 0;
      nextBtn.disabled = i === STEPS.length - 1;
    }

    const prevBtn = document.getElementById("algo-prev");
    const nextBtn = document.getElementById("algo-next");
    const playBtn = document.getElementById("algo-play");
    const dotsWrap = document.getElementById("algo-dots");
    STEPS.forEach((_, i) => {
      const b = document.createElement("button");
      b.setAttribute("aria-label", "Step " + (i + 1));
      b.addEventListener("click", () => { stop(); render(i); });
      dotsWrap.appendChild(b);
    });

    let playing = false, timer = null;
    function setPlayIcon() {
      playBtn.innerHTML = playing
        ? '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>'
        : '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';
    }
    function step() { render((cur + 1) % STEPS.length); }
    function play() { playing = true; setPlayIcon(); timer = setInterval(step, 2600); }
    function stop() { playing = false; setPlayIcon(); clearInterval(timer); }
    playBtn.addEventListener("click", () => (playing ? stop() : play()));
    prevBtn.addEventListener("click", () => { stop(); if (cur > 0) render(cur - 1); });
    nextBtn.addEventListener("click", () => { stop(); if (cur < STEPS.length - 1) render(cur + 1); });

    render(0);

    let kicked = false;
    if (!prefersReduced && "IntersectionObserver" in window) {
      const io = new IntersectionObserver((es) => {
        es.forEach((e) => { if (e.isIntersecting && !kicked) { kicked = true; play(); } });
      }, { threshold: 0.4 });
      io.observe(host);
    }
  }

  /* ============================================================
     SMALL PROBLEM PLOT — jump vs. smooth (toggle)
     ============================================================ */
  function buildProblemPlot() {
    const host = document.getElementById("prob-svg");
    if (!host) return;
    const W = 480, Hh = 248;
    const xL = 150, xR = 452, colW = (xR - xL) / H;
    const cx = Array.from({ length: H }, (_, i) => xL + colW * (i + 0.5));
    const pTop = 46, pBot = 196;
    const mapY = (v) => pBot - v * (pBot - pTop);
    const NAIVE2 = [0.82, 0.72, 0.62, 0.55, 0.5, 0.49, 0.52, 0.56];
    const PAINT2 = [0.52, 0.58, 0.6, 0.57, 0.52, 0.49, 0.52, 0.56];

    const svg = el("svg", { viewBox: `0 0 ${W} ${Hh}`, class: "stage__svg", role: "img" });
    svg.setAttribute("aria-label", "Prefix continuity: discontinuous jump versus a smooth match");

    svg.appendChild(el("text", { x: 20, y: 30, class: "dg-mini" }, document.createTextNode("action")));
    svg.appendChild(el("line", { x1: 30, y1: pBot + 6, x2: xR, y2: pBot + 6, stroke: "var(--border)", "stroke-width": 1 }));
    svg.appendChild(el("text", { x: xR, y: pBot + 26, "text-anchor": "end", class: "dg-mini" }, document.createTextNode("time →")));

    const bx = cx[0] - colW / 2;
    svg.appendChild(el("line", { x1: bx, y1: pTop - 6, x2: bx, y2: pBot + 10, stroke: "var(--border-2)", "stroke-width": 1, "stroke-dasharray": "3 4" }));
    svg.appendChild(el("text", { x: bx - 6, y: pTop - 10, "text-anchor": "end", class: "dg-mini" }, document.createTextNode("executed")));
    svg.appendChild(el("text", { x: bx + 6, y: pTop - 10, "text-anchor": "start", class: "dg-mini" }, document.createTextNode("predicted chunk")));

    const sufBx = bx;
    const sufBand = el("rect", {
      x: sufBx, y: pTop - 4, width: D * colW, height: (pBot - pTop) + 22,
      fill: "var(--accent-tint)", rx: 4,
    });
    sufBand.style.opacity = "0.45";
    svg.appendChild(sufBand);
    const sufVals = [0.52, 0.585];
    const sufRings = sufVals.map((v, i) =>
      el("circle", { cx: cx[i], cy: mapY(v), r: 6.5, fill: "none", stroke: "var(--ink-soft)", "stroke-width": 1.8 })
    );

    svg.appendChild(el("path", {
      d: `M 40 ${mapY(0.32)} L ${bx} ${mapY(0.45)} L ${cx[0]} ${mapY(sufVals[0])} L ${cx[1]} ${mapY(sufVals[1])}`,
      fill: "none", stroke: "var(--ink-faint)", "stroke-width": 2.6, "stroke-linecap": "round", "stroke-linejoin": "round", opacity: 0.6,
    }));
    sufVals.forEach((v, i) => svg.appendChild(el("circle", { cx: cx[i], cy: mapY(v), r: 4.5, fill: "var(--ink-soft)" })));

    function pathFrom(vals) {
      let d = "";
      for (let i = 0; i < H; i++) d += (i ? " L " : "M ") + cx[i].toFixed(1) + " " + mapY(vals[i]).toFixed(1);
      return d;
    }
    const chunk = el("path", { d: pathFrom(NAIVE2), fill: "none", "stroke-width": 3, "stroke-linecap": "round", "stroke-linejoin": "round", stroke: "var(--warn)" });
    svg.appendChild(chunk);
    const pts = NAIVE2.map((v, i) => { const c = el("circle", { cx: cx[i], cy: mapY(v), r: 3.6, fill: "var(--warn)" }); svg.appendChild(c); return c; });

    const jLine = el("line", { x1: cx[0], y1: mapY(sufVals[0]), x2: cx[0], y2: mapY(NAIVE2[0]), stroke: "var(--warn)", "stroke-width": 2, "stroke-dasharray": "2 3" });
    const jTxt = el("text", { x: cx[0] + 9, y: mapY((sufVals[0] + NAIVE2[0]) / 2) + 4, class: "dg-jumptxt" }, document.createTextNode("jump"));
    const jLine2 = el("line", { x1: cx[1], y1: mapY(sufVals[1]), x2: cx[1], y2: mapY(NAIVE2[1]), stroke: "var(--warn)", "stroke-width": 2, "stroke-dasharray": "2 3" });
    jLine2.style.opacity = "0";
    svg.appendChild(jLine); svg.appendChild(jTxt); svg.appendChild(jLine2);
    sufRings.forEach((r) => svg.appendChild(r));

    host.appendChild(svg);

    let displayed = NAIVE2.slice(), raf = null;
    function tween(target, color, showJump) {
      const from = displayed.slice(); const dur = prefersReduced ? 0 : 520; const t0 = performance.now();
      chunk.style.stroke = color; pts.forEach((p) => (p.style.fill = color));
      jLine.style.opacity = showJump ? "1" : "0"; jTxt.style.opacity = showJump ? "1" : "0"; jLine2.style.opacity = showJump ? "1" : "0";
      cancelAnimationFrame(raf);
      (function frame(now) {
        const t = dur === 0 ? 1 : Math.min(1, (now - t0) / dur); const e = easeOut(t);
        for (let i = 0; i < H; i++) displayed[i] = lerp(from[i], target[i], e);
        chunk.setAttribute("d", pathFrom(displayed));
        for (let i = 0; i < H; i++) pts[i].setAttribute("cy", mapY(displayed[i]).toFixed(1));
        if (t < 1) raf = requestAnimationFrame(frame);
      })(performance.now());
    }
    const segNaive = document.getElementById("prob-naive");
    const segPaint = document.getElementById("prob-paint");
    function set(m) {
      segNaive.classList.toggle("on", m === "naive");
      segPaint.classList.toggle("on", m === "paint");
      if (m === "naive") {
        tween(NAIVE2, "var(--warn)", true);
        sufBand.style.opacity = "0.45";
        sufRings.forEach((r) => { r.style.stroke = "var(--ink-soft)"; r.style.opacity = "0.7"; });
      } else {
        tween(PAINT2, "var(--accent)", false);
        sufBand.style.opacity = "1";
        sufRings.forEach((r) => { r.style.stroke = "var(--accent)"; r.style.opacity = "1"; });
      }
    }
    segNaive.addEventListener("click", () => set("naive"));
    segPaint.addEventListener("click", () => set("paint"));
    set("naive");
  }

  /* ============================================================
     SCROLL REVEAL
     ============================================================ */
  function initReveal() {
    const items = document.querySelectorAll(".reveal");
    if (prefersReduced || !("IntersectionObserver" in window)) return;
    document.documentElement.classList.add("reveal-on");
    const reveal = (n) => n.classList.add("in");
    const io = new IntersectionObserver((es) => {
      es.forEach((e) => { if (e.isIntersecting) { reveal(e.target); io.unobserve(e.target); } });
    }, { threshold: 0.08, rootMargin: "0px 0px -6% 0px" });
    items.forEach((i) => {
      const r = i.getBoundingClientRect();
      if (r.top < window.innerHeight && r.bottom > 0) reveal(i);
      else io.observe(i);
    });
    setTimeout(() => items.forEach(reveal), 2200);
  }

  /* ---- boot ---- */
  function boot() {
    buildAlgoStage();
    buildProblemPlot();
    initReveal();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
