(function () {
  'use strict';

  var STORAGE_KEY = 'bg-mode-pref';
  var canvas = document.getElementById('homepage-particles');
  var blowfish = document.getElementById('homepage-blowfish');
  var body = document.body;
  if (!canvas || !blowfish) return;

  var ctx = canvas.getContext('2d');
  var dpr = window.devicePixelRatio || 1;
  var w = 0, h = 0;
  var particles = [];
  var rafId = null;
  var running = false;

  var reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function readPref() {
    try {
      var v = localStorage.getItem(STORAGE_KEY);
      if (v === 'dynamic' || v === 'static') return v;
    } catch (e) {}
    return 'dynamic';
  }

  function writePref(v) {
    try { localStorage.setItem(STORAGE_KEY, v); } catch (e) {}
  }

  function resize() {
    w = window.innerWidth;
    h = window.innerHeight;
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function isDark() {
    return document.documentElement.classList.contains('dark');
  }

  function particleColor(alpha) {
    return isDark()
      ? 'rgba(56, 189, 248, ' + alpha + ')'
      : 'rgba(14, 165, 233, ' + alpha + ')';
  }

  function makeParticle() {
    var isLine = Math.random() < 0.18;
    return {
      x: Math.random() * w,
      y: h + Math.random() * h * 0.4,
      r: isLine ? 0 : 0.8 + Math.random() * 1.6,
      len: isLine ? 6 + Math.random() * 14 : 0,
      vy: -(0.18 + Math.random() * 0.42),
      sway: 0.4 + Math.random() * 0.9,
      phase: Math.random() * Math.PI * 2,
      alpha: 0.25 + Math.random() * 0.45,
      isLine: isLine
    };
  }

  function initParticles() {
    var count = Math.min(60, Math.floor(w * h / 22000));
    if (count < 24) count = 24;
    particles = [];
    for (var i = 0; i < count; i++) {
      var p = makeParticle();
      p.y = Math.random() * h;
      particles.push(p);
    }
  }

  function draw() {
    ctx.clearRect(0, 0, w, h);
    var t = performance.now() / 1000;
    for (var i = 0; i < particles.length; i++) {
      var p = particles[i];
      p.phase += 0.008;
      p.y += p.vy;
      var dx = Math.sin(t * 0.6 + p.phase) * p.sway;
      var px = p.x + dx;
      if (p.isLine) {
        ctx.strokeStyle = particleColor(p.alpha * 0.7);
        ctx.lineWidth = 0.6;
        ctx.beginPath();
        ctx.moveTo(px, p.y);
        ctx.lineTo(px + dx * 0.4, p.y + p.len);
        ctx.stroke();
      } else {
        ctx.fillStyle = particleColor(p.alpha);
        ctx.beginPath();
        ctx.arc(px, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
      }
      if (p.y < -20) {
        var np = makeParticle();
        np.y = h + 10;
        particles[i] = np;
      }
    }
    rafId = requestAnimationFrame(draw);
  }

  function start() {
    if (running) return;
    running = true;
    resize();
    initParticles();
    rafId = requestAnimationFrame(draw);
  }

  function stop() {
    running = false;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;
    ctx.clearRect(0, 0, w, h);
  }

  function applyMode(mode) {
    body.setAttribute('data-bg-mode', mode);
    if (mode === 'dynamic' && !reducedMotion) {
      start();
    } else {
      stop();
    }
    updateToggleIcon(mode);
  }

  // ---- Toggle button (inject only on homepage; nav lives in desktop-menu) ----
  var btn = null;
  var iconDynamic =
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">' +
      '<path d="M5 12 C 5 7, 13 6, 17 8 C 21 10, 21 15, 17 17 C 13 19, 5 17, 5 12 Z" />' +
      '<path d="M19 9 L 22 7 L 21 12 L 22 17 L 19 15" />' +
      '<circle cx="9" cy="11" r="0.9" fill="currentColor" stroke="none" />' +
      '<path d="M6 13 Q 4.5 14, 5.5 15.5" stroke-opacity="0.7" />' +
    '</svg>';
  var iconStatic =
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">' +
      '<path d="M3 9 Q 7.5 5, 12 9 T 21 9" />' +
      '<path d="M3 14 Q 7.5 10, 12 14 T 21 14" stroke-opacity="0.6" />' +
      '<path d="M3 19 Q 7.5 15, 12 19 T 21 19" stroke-opacity="0.35" />' +
    '</svg>';

  function updateToggleIcon(mode) {
    if (!btn) return;
    btn.innerHTML = mode === 'dynamic' ? iconDynamic : iconStatic;
    btn.setAttribute('title', mode === 'dynamic' ? '切换为静态背景' : '切换为动态背景');
    btn.setAttribute('aria-label', mode === 'dynamic' ? '切换为静态背景' : '切换为动态背景');
    btn.setAttribute('aria-pressed', mode === 'dynamic' ? 'true' : 'false');
  }

  function injectToggle() {
    var appearance = document.getElementById('appearance-switcher');
    if (!appearance) return;
    var wrapper = appearance.parentElement;     // <div class="flex items-center">
    var nav = wrapper && wrapper.parentElement; // <nav class="flex items-center gap-x-5 h-12">
    if (!nav) return;
    btn = document.createElement('button');
    btn.id = 'bg-mode-toggle';
    btn.type = 'button';
    btn.className = 'text-base bf-icon-color-hover';
    btn.addEventListener('click', function () {
      var current = body.getAttribute('data-bg-mode') === 'static' ? 'static' : 'dynamic';
      var next = current === 'dynamic' ? 'static' : 'dynamic';
      writePref(next);
      applyMode(next);
    });
    if (reducedMotion) {
      btn.disabled = true;
      btn.style.opacity = '0.4';
      btn.style.cursor = 'not-allowed';
    }
    nav.insertBefore(btn, wrapper);
  }

  // ---- Bootstrap ----
  function boot() {
    injectToggle();
    var pref = readPref();
    if (reducedMotion) pref = 'static';
    applyMode(pref);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }

  // resize
  var resizeTimer;
  window.addEventListener('resize', function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      if (running) {
        resize();
        initParticles();
      }
    }, 150);
  });

  // pause when tab hidden
  document.addEventListener('visibilitychange', function () {
    if (document.hidden) {
      if (rafId) cancelAnimationFrame(rafId);
      rafId = null;
    } else if (running && !rafId) {
      rafId = requestAnimationFrame(draw);
    }
  });

  // re-color particles on theme switch (just keeps drawing — color fn checks each frame)
})();
