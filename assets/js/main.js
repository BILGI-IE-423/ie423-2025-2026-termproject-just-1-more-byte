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

  /* --- Metric showcase count-up (0.809) --- */
  const metricEl = document.querySelector("[data-metric]");

  if (metricEl) {
    const target = parseFloat(metricEl.dataset.metric);
    const metricObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          const duration = 2200;
          const start = performance.now();

          function tick(now) {
            const progress = Math.min((now - start) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 4);
            metricEl.textContent = (eased * target).toFixed(3);
            if (progress < 1) requestAnimationFrame(tick);
          }

          requestAnimationFrame(tick);
          metricObserver.unobserve(entry.target);
        });
      },
      { threshold: 0.45 }
    );
    metricObserver.observe(metricEl.closest(".metric-showcase") || metricEl);
  }

  /* --- Chapter navigation & progress --- */
  const chapterNav = document.querySelector(".chapter-nav");
  const chapterProgress = document.querySelector(".chapter-nav__progress");
  const chapterLinks = document.querySelectorAll(".chapter-nav__list a");
  const chapterSections = document.querySelectorAll("[data-chapter]");
  const heroSection = document.getElementById("hero");
  let chapterTicking = false;

  function updateChapters() {
    if (!chapterSections.length) return;

    if (heroSection) {
      const pastHero = window.scrollY > heroSection.offsetHeight * 0.35;
      body.classList.toggle("chapters-visible", pastHero);
    }

    const scrollY = window.scrollY + window.innerHeight * 0.38;
    let activeChapter = chapterSections[0].dataset.chapter;

    chapterSections.forEach((section) => {
      if (section.offsetTop <= scrollY) {
        activeChapter = section.dataset.chapter;
      }
    });

    chapterLinks.forEach((link) => {
      link.classList.toggle("is-active", link.dataset.chapter === activeChapter);
    });

    document.querySelectorAll(".nav__links a").forEach((link) => {
      const href = link.getAttribute("href");
      if (href && href.startsWith("#")) {
        const id = href.slice(1);
        link.classList.toggle("is-active", id === activeChapter);
      }
    });

    if (chapterProgress && chapterSections.length > 1) {
      const first = chapterSections[0];
      const last = chapterSections[chapterSections.length - 1];
      const start = first.offsetTop;
      const end = last.offsetTop + last.offsetHeight;
      const range = end - start;
      const pct = range > 0 ? Math.min(100, Math.max(0, ((window.scrollY - start) / range) * 100)) : 0;
      chapterProgress.style.height = pct + "%";
    }

    chapterTicking = false;
  }

  if (chapterNav && chapterSections.length) {
    window.addEventListener(
      "scroll",
      () => {
        if (!chapterTicking) {
          chapterTicking = true;
          requestAnimationFrame(updateChapters);
        }
      },
      { passive: true }
    );
    updateChapters();
  }

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
