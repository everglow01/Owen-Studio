(function () {
  'use strict';

  var STORAGE_KEY = 'bg-mode-pref';
  var body = document.body;

  var blowfishCanvas = document.getElementById('homepage-particles');
  var blowfishSvg    = document.getElementById('homepage-blowfish');
  var starCanvas     = document.getElementById('starfield-canvas');

  var hasBlowfish = !!(blowfishCanvas && blowfishSvg);
  var hasStarfield = !!starCanvas;

  var reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var dpr = window.devicePixelRatio || 1;

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
  function isDark() { return document.documentElement.classList.contains('dark'); }

  // =========================================================================
  // Blowfish scene (homepage): particles drift up + sway
  // =========================================================================
  var bfCtx, bfW = 0, bfH = 0, bfParticles = [], bfRaf = null, bfRunning = false;

  function bfColor(a) {
    return isDark() ? 'rgba(56,189,248,' + a + ')' : 'rgba(14,165,233,' + a + ')';
  }
  function bfResize() {
    bfW = window.innerWidth; bfH = window.innerHeight;
    blowfishCanvas.width = Math.floor(bfW * dpr);
    blowfishCanvas.height = Math.floor(bfH * dpr);
    blowfishCanvas.style.width = bfW + 'px';
    blowfishCanvas.style.height = bfH + 'px';
    bfCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  function bfMake() {
    var isLine = Math.random() < 0.18;
    return {
      x: Math.random() * bfW, y: bfH + Math.random() * bfH * 0.4,
      r: isLine ? 0 : 0.8 + Math.random() * 1.6,
      len: isLine ? 6 + Math.random() * 14 : 0,
      vy: -(0.18 + Math.random() * 0.42),
      sway: 0.4 + Math.random() * 0.9,
      phase: Math.random() * Math.PI * 2,
      alpha: 0.25 + Math.random() * 0.45,
      isLine: isLine
    };
  }
  function bfInit() {
    var n = Math.min(60, Math.floor(bfW * bfH / 22000));
    if (n < 24) n = 24;
    bfParticles = [];
    for (var i = 0; i < n; i++) { var p = bfMake(); p.y = Math.random() * bfH; bfParticles.push(p); }
  }
  function bfDraw() {
    bfCtx.clearRect(0, 0, bfW, bfH);
    var t = performance.now() / 1000;
    for (var i = 0; i < bfParticles.length; i++) {
      var p = bfParticles[i];
      p.phase += 0.008; p.y += p.vy;
      var dx = Math.sin(t * 0.6 + p.phase) * p.sway;
      var px = p.x + dx;
      if (p.isLine) {
        bfCtx.strokeStyle = bfColor(p.alpha * 0.7); bfCtx.lineWidth = 0.6;
        bfCtx.beginPath(); bfCtx.moveTo(px, p.y); bfCtx.lineTo(px + dx * 0.4, p.y + p.len); bfCtx.stroke();
      } else {
        bfCtx.fillStyle = bfColor(p.alpha);
        bfCtx.beginPath(); bfCtx.arc(px, p.y, p.r, 0, Math.PI * 2); bfCtx.fill();
      }
      if (p.y < -20) { var np = bfMake(); np.y = bfH + 10; bfParticles[i] = np; }
    }
    bfRaf = requestAnimationFrame(bfDraw);
  }
  function bfStart() {
    if (bfRunning) return;
    bfRunning = true; bfCtx = blowfishCanvas.getContext('2d');
    bfResize(); bfInit(); bfRaf = requestAnimationFrame(bfDraw);
  }
  function bfStop() {
    bfRunning = false;
    if (bfRaf) cancelAnimationFrame(bfRaf);
    bfRaf = null;
    if (bfCtx) bfCtx.clearRect(0, 0, bfW, bfH);
  }

  // =========================================================================
  // Starfield scene (sub-pages): twinkling stars + occasional meteors
  // =========================================================================
  var sfCtx, sfW = 0, sfH = 0, sfStars = [], sfMeteors = [], sfRaf = null, sfRunning = false;
  var sfNextMeteor = 0;
  var sfNextFlash = 0;

  // Star color by type (dot/glow use cool blue, cross/burst use warm gold to echo --he-spark)
  function sfStarColor(type, a) {
    if (type === 'cross' || type === 'burst') {
      return isDark()
        ? 'rgba(252,211,77,'  + a + ')'   // amber-300
        : 'rgba(245,158,11,'  + a + ')';  // amber-500
    }
    return isDark()
      ? 'rgba(224,242,254,' + a + ')'     // sky-100
      : 'rgba(15,76,129,'   + a + ')';    // deep navy (high contrast on light bg)
  }
  function sfResize() {
    sfW = window.innerWidth; sfH = window.innerHeight;
    starCanvas.width = Math.floor(sfW * dpr);
    starCanvas.height = Math.floor(sfH * dpr);
    starCanvas.style.width = sfW + 'px';
    starCanvas.style.height = sfH + 'px';
    sfCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  function sfPickStarType() {
    var r = Math.random();
    if (r < 0.05) return 'burst';   // 5-pt star,  5%
    if (r < 0.18) return 'cross';   // 4-pt cross, 13%
    if (r < 0.40) return 'glow';    // halo dot,   22%
    return 'dot';                   // plain dot,  60%
  }
  function sfInitStars() {
    var n = Math.min(150, Math.floor(sfW * sfH / 9000));
    if (n < 70) n = 70;
    sfStars = [];
    for (var i = 0; i < n; i++) {
      var type = sfPickStarType();
      var r =
        type === 'burst' ? 2.5 + Math.random() * 1.5 :
        type === 'cross' ? 2.0 + Math.random() * 1.2 :
        type === 'glow'  ? 1.0 + Math.random() * 1.3 :
                           0.6 + Math.random() * 1.0;
      // ~15% of plain dots are "stable" (no twinkle, distant background stars)
      var isStill = type === 'dot' && Math.random() < 0.15;
      sfStars.push({
        x: Math.random() * sfW,
        y: Math.random() * sfH,
        type: type,
        r: r,
        base:  isStill ? 0.7  + Math.random() * 0.2  : 0.55 + Math.random() * 0.30,
        amp:   isStill ? 0    : 0.18 + Math.random() * 0.30,
        speed: 0.3 + Math.random() * 1.4,
        phase: Math.random() * Math.PI * 2,
        flashUntil: 0
      });
    }
    sfNextFlash = performance.now() + 1500 + Math.random() * 2000;
  }
  function sfScheduleFlash(now) {
    if (!sfStars.length) return;
    // weighted pick: cross/burst get 3x weight
    var weights = [], total = 0;
    for (var i = 0; i < sfStars.length; i++) {
      var w = (sfStars[i].type === 'cross' || sfStars[i].type === 'burst') ? 3 : 1;
      total += w; weights.push(total);
    }
    var pick = Math.random() * total;
    for (var j = 0; j < weights.length; j++) {
      if (pick <= weights[j]) { sfStars[j].flashUntil = now + 600; break; }
    }
  }
  function sfDrawDot(x, y, r, color) {
    sfCtx.fillStyle = color;
    sfCtx.beginPath();
    sfCtx.arc(x, y, r, 0, Math.PI * 2);
    sfCtx.fill();
  }
  function sfDrawGlow(x, y, r, type, a) {
    var center = sfStarColor(type, a);
    var edge   = sfStarColor(type, 0);
    var R = r * 2.6;
    var grad = sfCtx.createRadialGradient(x, y, 0, x, y, R);
    grad.addColorStop(0, center);
    grad.addColorStop(0.5, sfStarColor(type, a * 0.35));
    grad.addColorStop(1, edge);
    sfCtx.fillStyle = grad;
    sfCtx.beginPath();
    sfCtx.arc(x, y, R, 0, Math.PI * 2);
    sfCtx.fill();
    // bright pinpoint center
    sfDrawDot(x, y, r * 0.55, sfStarColor(type, Math.min(1, a * 1.2)));
  }
  function sfDrawCross(x, y, size, color) {
    sfCtx.strokeStyle = color;
    sfCtx.lineWidth = 0.9;
    sfCtx.lineCap = 'round';
    sfCtx.beginPath();
    sfCtx.moveTo(x - size, y); sfCtx.lineTo(x + size, y);
    sfCtx.moveTo(x, y - size); sfCtx.lineTo(x, y + size);
    sfCtx.stroke();
    // tiny bright center
    sfDrawDot(x, y, 0.7, color);
  }
  function sfDrawBurst(x, y, size, color) {
    sfCtx.fillStyle = color;
    sfCtx.beginPath();
    for (var i = 0; i < 5; i++) {
      var outer = -Math.PI / 2 + i * (Math.PI * 2 / 5);
      var inner = outer + Math.PI / 5;
      if (i === 0) sfCtx.moveTo(x + Math.cos(outer) * size, y + Math.sin(outer) * size);
      else         sfCtx.lineTo(x + Math.cos(outer) * size, y + Math.sin(outer) * size);
      sfCtx.lineTo(x + Math.cos(inner) * size * 0.42, y + Math.sin(inner) * size * 0.42);
    }
    sfCtx.closePath();
    sfCtx.fill();
  }
  function sfSpawnMeteor() {
    var startX = sfW * (0.55 + Math.random() * 0.45);
    var startY = sfH * (0.02 + Math.random() * 0.35);
    var angleDeg = 155 + Math.random() * 20;            // 155-175deg
    var angle = angleDeg * Math.PI / 180;
    var speed = 620 + Math.random() * 220;              // px/sec
    var trailLen = 90 + Math.random() * 60;
    sfMeteors.push({
      x: startX, y: startY,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      life: 0,
      max: 1.4 + Math.random() * 0.4,
      trail: trailLen
    });
    if (sfMeteors.length > 2) sfMeteors.shift();
  }
  function sfDrawMeteor(m) {
    var headX = m.x, headY = m.y;
    var mag = Math.hypot(m.vx, m.vy) || 1;
    var tailX = headX - (m.vx / mag) * m.trail;
    var tailY = headY - (m.vy / mag) * m.trail;
    var t = m.life / m.max;
    var alpha = t < 0.2 ? t / 0.2 : (1 - (t - 0.2) / 0.8);
    if (alpha < 0) alpha = 0;
    var grad = sfCtx.createLinearGradient(tailX, tailY, headX, headY);
    grad.addColorStop(0,    'rgba(255,220,180,0)');
    grad.addColorStop(0.45, 'rgba(120,180,230,' + (alpha * 0.45) + ')');
    grad.addColorStop(1,    'rgba(255,220,180,' + (alpha * 0.95) + ')');
    sfCtx.strokeStyle = grad;
    sfCtx.lineWidth = 1.4;
    sfCtx.lineCap = 'round';
    sfCtx.beginPath();
    sfCtx.moveTo(tailX, tailY);
    sfCtx.lineTo(headX, headY);
    sfCtx.stroke();
    var glow = sfCtx.createRadialGradient(headX, headY, 0, headX, headY, 10);
    glow.addColorStop(0,   'rgba(255,255,255,' + (alpha * 0.95) + ')');
    glow.addColorStop(0.4, 'rgba(255,220,180,' + (alpha * 0.7) + ')');
    glow.addColorStop(1,   'rgba(255,220,180,0)');
    sfCtx.fillStyle = glow;
    sfCtx.beginPath();
    sfCtx.arc(headX, headY, 10, 0, Math.PI * 2);
    sfCtx.fill();
  }
  function sfDraw() {
    var now = performance.now();
    if (!sfDraw._last) sfDraw._last = now;
    var dt = Math.min(0.05, (now - sfDraw._last) / 1000);
    sfDraw._last = now;
    var t = now / 1000;

    sfCtx.clearRect(0, 0, sfW, sfH);

    // sparkle scheduler — pick a star to flash
    if (now >= sfNextFlash) {
      sfScheduleFlash(now);
      sfNextFlash = now + 1500 + Math.random() * 2000; // every 1.5-3.5s
    }

    for (var i = 0; i < sfStars.length; i++) {
      var s = sfStars[i];
      var a = s.base + Math.sin(t * s.speed + s.phase) * s.amp;
      if (a < 0.05) a = 0.05;
      // flash boost: cosine-ease bump over 600ms window
      if (now < s.flashUntil) {
        var p = 1 - (s.flashUntil - now) / 600; // 0 -> 1
        var ease = 0.5 - 0.5 * Math.cos(p * Math.PI * 2); // 0 -> 1 -> 0
        a = Math.min(1, a + ease * 1.0);
      }
      if (s.type === 'cross')      sfDrawCross(s.x, s.y, s.r, sfStarColor(s.type, a));
      else if (s.type === 'burst') sfDrawBurst(s.x, s.y, s.r, sfStarColor(s.type, a));
      else if (s.type === 'glow')  sfDrawGlow(s.x, s.y, s.r, s.type, a);
      else                         sfDrawDot(s.x, s.y, s.r, sfStarColor(s.type, a));
    }

    if (now >= sfNextMeteor) {
      sfSpawnMeteor();
      sfNextMeteor = now + (8000 + Math.random() * 4000); // 8-12s
    }

    for (var j = sfMeteors.length - 1; j >= 0; j--) {
      var m = sfMeteors[j];
      m.x += m.vx * dt;
      m.y += m.vy * dt;
      m.life += dt;
      sfDrawMeteor(m);
      if (m.life >= m.max || m.x < -m.trail || m.y > sfH + m.trail || m.x > sfW + m.trail || m.y < -m.trail) {
        sfMeteors.splice(j, 1);
      }
    }

    sfRaf = requestAnimationFrame(sfDraw);
  }
  function sfStart() {
    if (sfRunning) return;
    sfRunning = true; sfCtx = starCanvas.getContext('2d');
    sfResize(); sfInitStars();
    sfNextMeteor = performance.now() + 1500;
    sfDraw._last = 0;
    sfRaf = requestAnimationFrame(sfDraw);
  }
  function sfStop() {
    sfRunning = false;
    if (sfRaf) cancelAnimationFrame(sfRaf);
    sfRaf = null;
    if (sfCtx) sfCtx.clearRect(0, 0, sfW, sfH);
    sfMeteors = [];
  }

  // =========================================================================
  // Mode dispatch
  // =========================================================================
  function applyMode(mode) {
    body.setAttribute('data-bg-mode', mode);
    if (mode === 'dynamic' && !reducedMotion) {
      if (hasBlowfish) bfStart();
      if (hasStarfield) sfStart();
    } else {
      if (hasBlowfish) bfStop();
      if (hasStarfield) sfStop();
    }
    updateToggleIcon(mode);
  }

  // =========================================================================
  // Toggle button
  // =========================================================================
  var btn = null;
  var iconDynamic =
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">' +
      '<path d="M3 14 C 6 8, 12 6, 15 9" stroke-opacity="0.85"/>' +
      '<circle cx="15" cy="9" r="1.4" fill="currentColor" stroke="none"/>' +
      '<circle cx="6"  cy="6"  r="0.9" fill="currentColor" stroke="none" opacity="0.7"/>' +
      '<circle cx="18" cy="5"  r="0.9" fill="currentColor" stroke="none" opacity="0.6"/>' +
      '<circle cx="20" cy="14" r="0.9" fill="currentColor" stroke="none" opacity="0.7"/>' +
      '<circle cx="9"  cy="18" r="0.9" fill="currentColor" stroke="none" opacity="0.5"/>' +
      '<circle cx="4"  cy="20" r="0.9" fill="currentColor" stroke="none" opacity="0.5"/>' +
    '</svg>';
  var iconStatic =
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">' +
      '<circle cx="6"  cy="6"  r="0.9" fill="currentColor" stroke="none"/>' +
      '<circle cx="12" cy="9"  r="0.9" fill="currentColor" stroke="none"/>' +
      '<circle cx="18" cy="6"  r="0.9" fill="currentColor" stroke="none"/>' +
      '<circle cx="9"  cy="15" r="0.9" fill="currentColor" stroke="none"/>' +
      '<circle cx="16" cy="17" r="0.9" fill="currentColor" stroke="none"/>' +
      '<circle cx="5"  cy="19" r="0.9" fill="currentColor" stroke="none"/>' +
    '</svg>';

  function updateToggleIcon(mode) {
    if (!btn) return;
    btn.innerHTML = mode === 'dynamic' ? iconDynamic : iconStatic;
    btn.setAttribute('title', mode === 'dynamic' ? '切换为静态背景' : '切换为动态背景');
    btn.setAttribute('aria-label', mode === 'dynamic' ? '切换为静态背景' : '切换为动态背景');
    btn.setAttribute('aria-pressed', mode === 'dynamic' ? 'true' : 'false');
  }
  function injectToggle() {
    if (!hasBlowfish && !hasStarfield) return;  // no toggle on plain article pages
    var appearance = document.getElementById('appearance-switcher');
    if (!appearance) return;
    var wrapper = appearance.parentElement;
    var nav = wrapper && wrapper.parentElement;
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

  // =========================================================================
  // Bootstrap
  // =========================================================================
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

  var resizeTimer;
  window.addEventListener('resize', function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      if (bfRunning) { bfResize(); bfInit(); }
      if (sfRunning) { sfResize(); sfInitStars(); }
    }, 150);
  });

  document.addEventListener('visibilitychange', function () {
    if (document.hidden) {
      if (bfRaf) { cancelAnimationFrame(bfRaf); bfRaf = null; }
      if (sfRaf) { cancelAnimationFrame(sfRaf); sfRaf = null; }
    } else {
      if (bfRunning && !bfRaf) bfRaf = requestAnimationFrame(bfDraw);
      if (sfRunning && !sfRaf) { sfDraw._last = 0; sfRaf = requestAnimationFrame(sfDraw); }
    }
  });
})();
