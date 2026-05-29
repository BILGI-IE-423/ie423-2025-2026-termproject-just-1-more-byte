/**
 * Personality in Words — IE423 Research Website
 * Cinematic scroll experience: reveal, ambient themes, subtle parallax
 */

(function () {
  "use strict";

  /* --- Scroll reveal --- */
  const revealEls = document.querySelectorAll(".reveal:not(.chart-reveal)");

  const revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
        }
      });
    },
    { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
  );

  revealEls.forEach((el) => revealObserver.observe(el));

  /* --- Ambient theme transitions per section --- */
  const themedSections = document.querySelectorAll("[data-theme]");
  const body = document.body;

  const themeObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const theme = entry.target.dataset.theme;
          if (theme) body.setAttribute("data-theme", theme);
        }
      });
    },
    { threshold: 0.35, rootMargin: "-10% 0px -10% 0px" }
  );

  themedSections.forEach((section) => themeObserver.observe(section));

  /* --- Animated stat counter --- */
  const statNumbers = document.querySelectorAll("[data-count]");

  const counterObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;

        const el = entry.target;
        const target = parseInt(el.dataset.count, 10);
        if (isNaN(target)) return;

        const duration = 1800;
        const start = performance.now();

        function tick(now) {
          const progress = Math.min((now - start) / duration, 1);
          const eased = 1 - Math.pow(1 - progress, 3);
          el.textContent = Math.floor(eased * target).toLocaleString();
          if (progress < 1) requestAnimationFrame(tick);
        }

        requestAnimationFrame(tick);
        counterObserver.unobserve(el);
      });
    },
    { threshold: 0.5 }
  );

  statNumbers.forEach((el) => counterObserver.observe(el));

  /* --- Subtle language drift on scroll (depth, not distraction) --- */
  const langLayers = document.querySelectorAll(".lang-atmosphere");
  let langTicking = false;

  function updateLangParallax() {
    const scrollY = window.scrollY;
    langLayers.forEach((layer) => {
      const rect = layer.getBoundingClientRect();
      const center = rect.top + rect.height / 2 + scrollY;
      const offset = (scrollY - center) * 0.028;
      layer.style.setProperty("--lang-scroll-y", offset + "px");
    });
    langTicking = false;
  }

  if (langLayers.length && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    window.addEventListener(
      "scroll",
      () => {
        if (!langTicking) {
          langTicking = true;
          requestAnimationFrame(updateLangParallax);
        }
      },
      { passive: true }
    );
  }

  /* --- Smooth CTA scroll --- */
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", (e) => {
      const id = anchor.getAttribute("href");
      if (id === "#") return;

      const target = document.querySelector(id);
      if (!target) return;

      e.preventDefault();
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });

  /* --- Nav background on scroll --- */
  const nav = document.querySelector(".nav");

  window.addEventListener(
    "scroll",
    () => {
      if (!nav) return;
      nav.style.background =
        window.scrollY > 60
          ? "rgba(244, 242, 239, 0.95)"
          : "linear-gradient(to bottom, rgba(244, 242, 239, 0.92) 0%, transparent 100%)";
    },
    { passive: true }
  );
})();
