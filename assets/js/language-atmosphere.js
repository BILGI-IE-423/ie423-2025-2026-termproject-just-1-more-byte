/**
 * Language atmosphere — ambient typography fragments by section.
 * Boosted visibility (~40%) with depth layers for mouse parallax.
 */

(function () {
  "use strict";

  const VISIBILITY = 1.4;

  const PHRASES = {
    hero: [
      { text: "identity", size: 7.5, x: 72, y: 18, opacity: 0.07, blur: 1.8, depth: 0.4 },
      { text: "language", size: 9.5, x: 58, y: 62, opacity: 0.065, blur: 2.2, depth: 0.7 },
      { text: "patterns", size: 6, x: 12, y: 28, opacity: 0.068, blur: 1.5, depth: 0.5 },
      { text: "emotion", size: 5.5, x: 88, y: 38, opacity: 0.055, blur: 2, depth: 0.6 },
      { text: "words", size: 8.5, x: 22, y: 78, opacity: 0.062, blur: 2, depth: 0.8 },
      { text: "who we are", size: 3.4, x: 68, y: 42, opacity: 0.078, blur: 0.8, phrase: true, depth: 0.3 },
      { text: "structure", size: 4.5, x: 42, y: 12, opacity: 0.05, blur: 1.8, depth: 0.55 },
    ],
    question: [
      { text: "who are we", size: 2.8, x: 24, y: 40, opacity: 0.082, blur: 0.6, phrase: true, depth: 0.35 },
      { text: "patterns emerge", size: 2.4, x: 32, y: 58, opacity: 0.075, blur: 0.5, phrase: true, depth: 0.45 },
      { text: "language leaves traces", size: 2, x: 16, y: 70, opacity: 0.07, blur: 0.4, phrase: true, depth: 0.5 },
      { text: "meaning", size: 5, x: 36, y: 24, opacity: 0.06, blur: 1.8, depth: 0.6 },
      { text: "possibility", size: 3.8, x: 10, y: 50, opacity: 0.055, blur: 1.5, depth: 0.7 },
    ],
    data: [
      { text: "experience", size: 4.5, x: 85, y: 20, opacity: 0.055, blur: 1.5, depth: 0.5 },
      { text: "details", size: 4, x: 10, y: 55, opacity: 0.052, blur: 1.8, depth: 0.6 },
      { text: "present", size: 3.5, x: 75, y: 80, opacity: 0.048, blur: 1.5, depth: 0.45 },
      { text: "reality", size: 3.2, x: 55, y: 15, opacity: 0.05, blur: 1.6, depth: 0.55 },
    ],
    pipeline: [
      { text: "logic", size: 5.5, x: 88, y: 30, opacity: 0.072, blur: 1.2, depth: 0.65 },
      { text: "structure", size: 4, x: 6, y: 40, opacity: 0.065, blur: 1.5, depth: 0.5 },
      { text: "analysis", size: 4.5, x: 15, y: 75, opacity: 0.058, blur: 1.8, depth: 0.7 },
      { text: "systems", size: 3.6, x: 80, y: 68, opacity: 0.062, blur: 1.2, depth: 0.55 },
      { text: "objective", size: 3, x: 50, y: 12, opacity: 0.05, blur: 1.4, depth: 0.4 },
    ],
    results: [
      { text: "emotion", size: 3.5, x: 92, y: 15, opacity: 0.038, blur: 1.5, depth: 0.35 },
      { text: "care", size: 3.2, x: 5, y: 88, opacity: 0.035, blur: 1.2, depth: 0.4 },
      { text: "connection", size: 2.8, x: 78, y: 92, opacity: 0.032, blur: 1, phrase: true, depth: 0.3 },
    ],
    interpret: [
      { text: "logic", size: 6.5, x: 8, y: 12, opacity: 0.085, blur: 1.5, depth: 0.75 },
      { text: "empathy", size: 5.5, x: 85, y: 22, opacity: 0.08, blur: 1.2, depth: 0.65 },
      { text: "systems", size: 5, x: 72, y: 55, opacity: 0.072, blur: 1.8, depth: 0.8 },
      { text: "connection", size: 3.5, x: 12, y: 48, opacity: 0.078, blur: 0.8, phrase: true, depth: 0.5 },
      { text: "understanding", size: 4.2, x: 55, y: 8, opacity: 0.065, blur: 1.2, depth: 0.6 },
      { text: "objective", size: 4, x: 28, y: 82, opacity: 0.07, blur: 1.5, depth: 0.55 },
      { text: "feeling", size: 6, x: 90, y: 78, opacity: 0.075, blur: 1.8, depth: 0.7 },
      { text: "analysis", size: 4.5, x: 42, y: 70, opacity: 0.062, blur: 1.5, depth: 0.45 },
      { text: "future", size: 4, x: 62, y: 35, opacity: 0.058, blur: 2, depth: 0.85 },
    ],
    reflection: [
      { text: "openness", size: 4.5, x: 20, y: 30, opacity: 0.062, blur: 1.5, depth: 0.5 },
      { text: "exploration", size: 3, x: 70, y: 60, opacity: 0.068, blur: 0.8, phrase: true, depth: 0.6 },
      { text: "identity", size: 5.5, x: 50, y: 85, opacity: 0.055, blur: 2, depth: 0.7 },
    ],
    rq3: [
      { text: "topic", size: 5, x: 82, y: 22, opacity: 0.055, blur: 1.6, depth: 0.45 },
      { text: "style", size: 6, x: 14, y: 38, opacity: 0.062, blur: 1.4, depth: 0.55 },
      { text: "how we say it", size: 2.6, x: 58, y: 72, opacity: 0.07, blur: 0.6, phrase: true, depth: 0.35 },
      { text: "uncertainty", size: 3.8, x: 28, y: 18, opacity: 0.05, blur: 1.8, depth: 0.65 },
      { text: "signal", size: 4.2, x: 72, y: 48, opacity: 0.058, blur: 1.5, depth: 0.5 },
      { text: "what we discuss", size: 2.2, x: 8, y: 68, opacity: 0.065, blur: 0.5, phrase: true, depth: 0.4 },
    ],
  };

  function spawnWords(container, items) {
    const intensity = container.dataset.langIntensity || "medium";
    const opacityScale =
      intensity === "strong" ? 1.35
        : intensity === "light" ? 0.95
          : intensity === "minimal" ? 0.75
            : 1.15;

    items.forEach((item, index) => {
      const el = document.createElement("span");
      el.className = "lang-word" + (item.phrase ? " lang-word--phrase" : "");
      el.textContent = item.text;
      el.dataset.depth = String(item.depth ?? 0.5 + (index % 3) * 0.15);
      el.style.setProperty("--lang-x", item.x + "%");
      el.style.setProperty("--lang-y", item.y + "%");
      el.style.setProperty("--lang-size", item.size + "rem");
      el.style.setProperty("--lang-opacity", String(item.opacity * opacityScale * VISIBILITY));
      el.style.setProperty("--lang-blur", item.blur + "px");
      el.style.setProperty("--lang-delay", (index * 1.4) + "s");
      el.style.setProperty("--lang-duration", (18 + index * 2.5) + "s");
      el.style.setProperty("--lang-mx", "0px");
      el.style.setProperty("--lang-my", "0px");
      container.appendChild(el);
    });
  }

  document.querySelectorAll(".lang-atmosphere").forEach((container) => {
    const key = container.dataset.langSet;
    if (key && PHRASES[key]) {
      spawnWords(container, PHRASES[key]);
    }
  });
})();
