/**
 * Reactive psychological experience — typography, atmosphere, analysis, charts.
 * Visible but restrained; cinematic, not gimmicky.
 */

(function () {
  "use strict";

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const root = document.documentElement;
  const body = document.body;

  /* --- Pointer state (smoothed) --- */
  let pointerX = 0.5;
  let pointerY = 0.5;
  let targetX = 0.5;
  let targetY = 0.5;
  let rafId = null;

  function setPointerVars(x, y) {
    root.style.setProperty("--pointer-x", String(x));
    root.style.setProperty("--pointer-y", String(y));
    root.style.setProperty("--pointer-dx", String((x - 0.5) * 2));
    root.style.setProperty("--pointer-dy", String((y - 0.5) * 2));
  }

  function smoothPointer() {
    pointerX += (targetX - pointerX) * 0.08;
    pointerY += (targetY - pointerY) * 0.08;
    setPointerVars(pointerX, pointerY);
    updateHeadlines();
    updateLangDepth();

    if (Math.abs(targetX - pointerX) > 0.0008 || Math.abs(targetY - pointerY) > 0.0008) {
      rafId = requestAnimationFrame(smoothPointer);
    } else {
      rafId = null;
    }
  }

  if (!reducedMotion) {
    document.addEventListener(
      "mousemove",
      (e) => {
        targetX = e.clientX / window.innerWidth;
        targetY = e.clientY / window.innerHeight;
        if (!rafId) rafId = requestAnimationFrame(smoothPointer);
      },
      { passive: true }
    );
    document.addEventListener(
      "mouseleave",
      () => {
        targetX = 0.5;
        targetY = 0.5;
        if (!rafId) rafId = requestAnimationFrame(smoothPointer);
      },
      { passive: true }
    );
  }

  /* --- Typography reaction --- */
  const headlines = document.querySelectorAll(".reactive-headline");

  function updateHeadlines() {
    const dx = (pointerX - 0.5) * 2;
    const dy = (pointerY - 0.5) * 2;

    headlines.forEach((el) => {
      const rect = el.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const distX = (pointerX * window.innerWidth - cx) / window.innerWidth;
      const distY = (pointerY * window.innerHeight - cy) / window.innerHeight;
      const influence = Math.max(0, 1 - Math.hypot(distX, distY) * 1.8);

      const shiftX = dx * 14 * influence;
      const shiftY = dy * 8 * influence;
      const tracking = -0.03 + Math.abs(dx) * 0.045 * influence;
      const glow = 0.22 + Math.abs(dx) * 0.28 * influence;
      const blur = Math.abs(dy) * 0.35 * influence;

      el.style.setProperty("--headline-x", shiftX.toFixed(2) + "px");
      el.style.setProperty("--headline-y", shiftY.toFixed(2) + "px");
      el.style.setProperty("--headline-tracking", tracking.toFixed(4) + "em");
      el.style.setProperty("--headline-glow", glow.toFixed(3));
      el.style.setProperty("--headline-blur", blur.toFixed(2) + "px");
    });
  }

  /* --- Language depth parallax --- */
  const langWords = document.querySelectorAll(".lang-word");

  function updateLangDepth() {
    const dx = (pointerX - 0.5) * 2;
    const dy = (pointerY - 0.5) * 2;

    langWords.forEach((word) => {
      const depth = parseFloat(word.dataset.depth) || 0.5;
      const mx = dx * 18 * depth;
      const my = dy * 12 * depth;
      word.style.setProperty("--lang-mx", mx.toFixed(1) + "px");
      word.style.setProperty("--lang-my", my.toFixed(1) + "px");
    });
  }

  /* --- Personality mode classes (stronger than data-theme alone) --- */
  const MODE_MAP = {
    introvert: "mode-introvert",
    intuition: "mode-intuition",
    sensing: "mode-sensing",
    thinking: "mode-thinking",
    feeling: "mode-feeling",
    perceiving: "mode-perceiving",
  };

  const themedSections = document.querySelectorAll("[data-theme]");
  const modeObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const theme = entry.target.dataset.theme;
        Object.values(MODE_MAP).forEach((cls) => body.classList.remove(cls));
        if (MODE_MAP[theme]) body.classList.add(MODE_MAP[theme]);
        body.dataset.activeMode = theme;
      });
    },
    { threshold: 0.38, rootMargin: "-12% 0px -12% 0px" }
  );
  themedSections.forEach((s) => modeObserver.observe(s));

  /* --- Analysis whisper system --- */
  const ANALYSIS_LINES = [
    "detecting patterns…",
    "analyzing language…",
    "extracting personality signals…",
    "mapping dimensions…",
    "semantic structures identified…",
    "correlating lexical features…",
    "processing linguistic traces…",
  ];

  const whisperSections = document.querySelectorAll("[data-analysis-zone]");

  whisperSections.forEach((section) => {
    const whisper = document.createElement("div");
    whisper.className = "analysis-whisper";
    whisper.setAttribute("aria-hidden", "true");
    section.appendChild(whisper);

    let lineIndex = 0;
    let cycleTimer = null;
    let visible = false;

    function showLine() {
      if (!visible) return;
      whisper.textContent = ANALYSIS_LINES[lineIndex];
      whisper.classList.remove("is-active");
      void whisper.offsetWidth;
      whisper.classList.add("is-active");
      lineIndex = (lineIndex + 1) % ANALYSIS_LINES.length;
    }

    const whisperObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          visible = entry.isIntersecting;
          if (visible) {
            showLine();
            cycleTimer = window.setInterval(showLine, 4200);
          } else {
            whisper.classList.remove("is-active");
            if (cycleTimer) {
              clearInterval(cycleTimer);
              cycleTimer = null;
            }
          }
        });
      },
      { threshold: 0.25 }
    );
    whisperObserver.observe(section);
  });

  /* --- Cinematic chart reveals --- */
  const chartCards = document.querySelectorAll(".chart-reveal");

  const chartObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const card = entry.target;
        card.classList.add("is-visible");

        const stage = card.querySelector(".chart-card__stage");
        if (stage) stage.classList.add("is-revealed");

        const featureWords = card.querySelectorAll(".feature-word");
        featureWords.forEach((word, i) => {
          window.setTimeout(() => word.classList.add("is-emerged"), 200 + i * 120);
        });

        chartObserver.unobserve(card);
      });
    },
    { threshold: 0.18, rootMargin: "0px 0px -8% 0px" }
  );

  chartCards.forEach((card) => chartObserver.observe(card));

  /* --- Pipeline step stagger on scroll --- */
  const pipelineSteps = document.querySelectorAll(".pipeline__step");
  if (pipelineSteps.length) {
    const pipeObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          pipelineSteps.forEach((step, i) => {
            window.setTimeout(() => step.classList.add("is-lit"), i * 180);
          });
          pipeObserver.disconnect();
        });
      },
      { threshold: 0.4 }
    );
    const pipeline = document.querySelector(".pipeline");
    if (pipeline) pipeObserver.observe(pipeline);
  }

  /* --- Stat cards pulse on reveal --- */
  document.querySelectorAll(".stat").forEach((stat, i) => {
    stat.style.setProperty("--stat-delay", i * 0.15 + "s");
  });

  const initialTheme = body.dataset.theme || "introvert";
  if (MODE_MAP[initialTheme]) body.classList.add(MODE_MAP[initialTheme]);
})();
