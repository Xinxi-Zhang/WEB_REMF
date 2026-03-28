(function () {
  'use strict';

  const FREQ = 2;
  const SURF_N = 50;
  const ARROW_COUNT = 12;
  const Z_SCALE = 0.65;
  const VIEW = { xMin: -0.15, xMax: 1.15, yMin: -0.3, yMax: 1.3 };

  // 3D projection: matching matplotlib view_init(elev=25, azim=135)
  const azRad = -45 * Math.PI / 180;
  const cosAz = Math.cos(azRad), sinAz = Math.sin(azRad);
  const elRad = 25 * Math.PI / 180;
  const cosEl = Math.cos(elRad), sinEl = Math.sin(elRad);

  const BLUES = [
    [0.00, 247, 251, 255],
    [0.12, 222, 235, 247],
    [0.25, 198, 219, 239],
    [0.38, 158, 202, 225],
    [0.50, 107, 174, 214],
    [0.62,  66, 146, 198],
    [0.75,  33, 113, 181],
    [0.87,   8,  81, 156],
    [1.00,   8,  48, 107],
  ];

  let fixedMaxAngle = 90;
  let trajCtx, surfCtx;
  const TRAJ_W = 620, TRAJ_H = 560;
  const SURF_W = 680, SURF_H = 580;

  // ── Math ──────────────────────────────────────────────

  function curvedPos(t, amp) {
    const w = FREQ * 2 * Math.PI;
    return [t, amp * Math.sin(w * t) * (1 - t) + t];
  }

  function curvedVel(t, amp) {
    const w = FREQ * 2 * Math.PI;
    return [1.0, amp * (w * Math.cos(w * t) * (1 - t) - Math.sin(w * t)) + 1];
  }

  function angleDiff(vel, dir) {
    const nv = Math.hypot(vel[0], vel[1]);
    const nd = Math.hypot(dir[0], dir[1]);
    if (nv < 1e-12 || nd < 1e-12) return 0;
    const c = Math.max(-1, Math.min(1,
      (vel[0] * dir[0] + vel[1] * dir[1]) / (nv * nd)));
    return Math.acos(c) * (180 / Math.PI);
  }

  // ── Colormap ──────────────────────────────────────────

  function bluesRGB(t) {
    t = Math.max(0, Math.min(1, t));
    let i = 0;
    while (i < BLUES.length - 2 && BLUES[i + 1][0] < t) i++;
    const [t0, r0, g0, b0] = BLUES[i];
    const [t1, r1, g1, b1] = BLUES[i + 1];
    const f = (t - t0) / (t1 - t0);
    return [
      Math.round(r0 + (r1 - r0) * f),
      Math.round(g0 + (g1 - g0) * f),
      Math.round(b0 + (b1 - b0) * f),
    ];
  }

  // ── 3D projection ────────────────────────────────────

  function proj(r, t, z) {
    const x = r - 0.5, y = t - 0.5;
    const xr = cosAz * x - sinAz * y;
    const yr = sinAz * x + cosAz * y;
    const zs = z * Z_SCALE;
    return {
      sx: xr,
      sy: -sinEl * yr + cosEl * zs,
      depth: cosEl * yr + sinEl * zs,
    };
  }

  // ── Canvas setup ──────────────────────────────────────

  function setupCanvas(canvas, w, h) {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    return ctx;
  }

  // ── Trajectory panel (2D) ─────────────────────────────

  function drawTrajectory(amp) {
    const ctx = trajCtx;
    const w = TRAJ_W, h = TRAJ_H;
    const pad = 25;
    const pw = w - 2 * pad, ph = h - 2 * pad;

    function toC(x, y) {
      return [
        pad + ((x - VIEW.xMin) / (VIEW.xMax - VIEW.xMin)) * pw,
        pad + ((VIEW.yMax - y) / (VIEW.yMax - VIEW.yMin)) * ph,
      ];
    }

    ctx.clearRect(0, 0, w, h);

    ctx.setLineDash([6, 5]);
    ctx.strokeStyle = '#ccc';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(...toC(0, 0));
    ctx.lineTo(...toC(1, 1));
    ctx.stroke();
    ctx.setLineDash([]);

    const nPts = 400;
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (let i = 0; i <= nPts; i++) {
      const t = i / nPts;
      const [cx, cy] = toC(...curvedPos(t, amp));
      i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    ctx.lineWidth = 1.5;
    for (let k = 0; k < ARROW_COUNT; k++) {
      const t = 0.04 + (k / (ARROW_COUNT - 1)) * 0.92;
      const [px, py] = curvedPos(t, amp);
      const [vx, vy] = curvedVel(t, amp);
      const norm = Math.hypot(vx, vy);
      const sc = 0.065;
      const dx = (vx / norm) * sc, dy = (vy / norm) * sc;
      const [ax, ay] = toC(px, py);
      const [bx, by] = toC(px + dx, py + dy);
      const ang = Math.atan2(by - ay, bx - ax);
      const hl = 6;

      ctx.strokeStyle = '#888';
      ctx.fillStyle = '#888';
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(bx, by);
      ctx.lineTo(bx - hl * Math.cos(ang - 0.45), by - hl * Math.sin(ang - 0.45));
      ctx.lineTo(bx - hl * Math.cos(ang + 0.45), by - hl * Math.sin(ang + 0.45));
      ctx.closePath(); ctx.fill();
    }

    const [dx, dy] = toC(0, 0);
    const [nx, ny] = toC(1, 1);
    ctx.fillStyle = '#c0392b';
    ctx.beginPath(); ctx.arc(dx, dy, 7, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = '#27ae60';
    ctx.beginPath(); ctx.arc(nx, ny, 7, 0, 2 * Math.PI); ctx.fill();

    ctx.font = 'italic 15px "adobe-garamond-pro", Georgia, serif';
    ctx.fillStyle = '#c0392b';
    ctx.textAlign = 'left';
    ctx.fillText('x  (data)', dx + 14, dy + 5);
    ctx.fillStyle = '#27ae60';
    ctx.textAlign = 'right';
    ctx.fillText('z  (noise)', nx - 14, ny - 8);
  }

  // ── 3D Surface panel ──────────────────────────────────

  function drawSurface(amp) {
    const ctx = surfCtx;
    ctx.clearRect(0, 0, SURF_W, SURF_H);

    const scale = 370;
    const cx = SURF_W * 0.48;
    const cy = SURF_H * 0.58;

    function toS(p) {
      return [cx + p.sx * scale, cy - p.sy * scale];
    }

    // Compute angle grid
    const N = SURF_N;
    const vals = new Float64Array(N * N);
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        const r = i / (N - 1), t = j / (N - 1);
        if (t > r + 0.02) {
          const posT = curvedPos(t, amp);
          const posR = curvedPos(r, amp);
          const vel = curvedVel(t, amp);
          const dir = [posT[0] - posR[0], posT[1] - posR[1]];
          vals[j * N + i] = angleDiff(vel, dir) / fixedMaxAngle;
        } else {
          vals[j * N + i] = -1;
        }
      }
    }

    // Build quads
    const quads = [];
    for (let j = 0; j < N - 1; j++) {
      for (let i = 0; i < N - 1; i++) {
        const z00 = vals[j * N + i];
        const z10 = vals[j * N + i + 1];
        const z01 = vals[(j + 1) * N + i];
        const z11 = vals[(j + 1) * N + i + 1];
        if (z00 < 0 || z10 < 0 || z01 < 0 || z11 < 0) continue;

        const r0 = i / (N - 1), r1 = (i + 1) / (N - 1);
        const t0 = j / (N - 1), t1 = (j + 1) / (N - 1);
        const p00 = proj(r0, t0, z00);
        const p10 = proj(r1, t0, z10);
        const p01 = proj(r0, t1, z01);
        const p11 = proj(r1, t1, z11);

        quads.push({
          pts: [p00, p10, p11, p01],
          z: (z00 + z10 + z01 + z11) / 4,
          d: (p00.depth + p10.depth + p01.depth + p11.depth) / 4,
        });
      }
    }

    quads.sort((a, b) => a.d - b.d);

    // ─ Draw floor grid (behind everything) ─
    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 0.7;
    for (let v = 0; v <= 1; v += 0.25) {
      ctx.beginPath();
      ctx.moveTo(...toS(proj(v, 0, 0)));
      ctx.lineTo(...toS(proj(v, 1, 0)));
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(...toS(proj(0, v, 0)));
      ctx.lineTo(...toS(proj(1, v, 0)));
      ctx.stroke();
    }

    // Back edges of bounding box (behind the surface)
    ctx.strokeStyle = '#bbb';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(...toS(proj(0, 0, 0))); ctx.lineTo(...toS(proj(1, 0, 0))); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(...toS(proj(1, 0, 0))); ctx.lineTo(...toS(proj(1, 1, 0))); ctx.stroke();

    // Vertical guides at back corners
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = '#ccc';
    ctx.lineWidth = 0.7;
    ctx.beginPath(); ctx.moveTo(...toS(proj(1, 0, 0))); ctx.lineTo(...toS(proj(1, 0, 1))); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(...toS(proj(1, 1, 0))); ctx.lineTo(...toS(proj(1, 1, 1))); ctx.stroke();
    ctx.setLineDash([]);

    // ─ Draw surface quads ─
    for (const q of quads) {
      const [cr, cg, cb] = bluesRGB(q.z);
      ctx.fillStyle = 'rgb(' + cr + ',' + cg + ',' + cb + ')';
      ctx.beginPath();
      ctx.moveTo(...toS(q.pts[0]));
      for (let k = 1; k < 4; k++) ctx.lineTo(...toS(q.pts[k]));
      ctx.closePath();
      ctx.fill();

      ctx.strokeStyle = 'rgba(' + Math.max(0, cr - 30) + ',' + Math.max(0, cg - 30) + ',' + Math.max(0, cb - 30) + ',0.2)';
      ctx.lineWidth = 0.4;
      ctx.stroke();
    }

    // ─ Front axes (on top) ─
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1.2;

    // r-axis: front-right edge (0,1,0)→(1,1,0)
    ctx.beginPath(); ctx.moveTo(...toS(proj(0, 1, 0))); ctx.lineTo(...toS(proj(1, 1, 0))); ctx.stroke();
    // t-axis: front-left edge (0,0,0)→(0,1,0)
    ctx.beginPath(); ctx.moveTo(...toS(proj(0, 0, 0))); ctx.lineTo(...toS(proj(0, 1, 0))); ctx.stroke();
    // z-axis: vertical at left corner (0,0,0)→(0,0,1)
    ctx.beginPath(); ctx.moveTo(...toS(proj(0, 0, 0))); ctx.lineTo(...toS(proj(0, 0, 1))); ctx.stroke();

    // ─ Tick marks ─
    ctx.fillStyle = '#000';
    ctx.font = '11px "adobe-garamond-pro", Georgia, serif';

    // r-axis ticks along (v, 1, 0), labels offset outward
    for (const v of [0, 0.5, 1]) {
      const tp = proj(v, 1.05, 0);
      const [tx, ty] = toS(tp);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(v === 0 ? '0' : v === 1 ? '1' : '0.5', tx, ty + 2);
      ctx.beginPath();
      ctx.moveTo(...toS(proj(v, 1, 0)));
      ctx.lineTo(...toS(proj(v, 1.03, 0)));
      ctx.strokeStyle = '#000'; ctx.lineWidth = 1;
      ctx.stroke();
    }

    // t-axis ticks along (0, v, 0), labels offset outward
    for (const v of [0, 0.5, 1]) {
      const tp = proj(-0.05, v, 0);
      const [tx, ty] = toS(tp);
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(v === 0 ? '0' : v === 1 ? '1' : '0.5', tx - 2, ty);
      ctx.beginPath();
      ctx.moveTo(...toS(proj(0, v, 0)));
      ctx.lineTo(...toS(proj(-0.02, v, 0)));
      ctx.strokeStyle = '#000'; ctx.lineWidth = 1;
      ctx.stroke();
    }

    // z-axis ticks along (0, 0, v), labels offset left
    var zTicks = [0, 0.25, 0.5, 0.75, 1.0];
    for (const v of zTicks) {
      const angle = Math.round(v * fixedMaxAngle);
      const tp = proj(-0.05, 0, v);
      const [tx, ty] = toS(tp);
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(angle + '°', tx - 2, ty);
      ctx.beginPath();
      ctx.moveTo(...toS(proj(0, 0, v)));
      ctx.lineTo(...toS(proj(-0.02, 0, v)));
      ctx.strokeStyle = '#000'; ctx.lineWidth = 1;
      ctx.stroke();
    }

    // ─ Axis labels ─
    ctx.font = 'italic 14px "adobe-garamond-pro", Georgia, serif';
    ctx.fillStyle = '#000';

    // r label (along front-right edge)
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    var rp = toS(proj(0.5, 1.15, 0));
    ctx.fillText('r', rp[0], rp[1]);

    // t label (along front-left edge)
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    var tlp = toS(proj(-0.14, 0.5, 0));
    ctx.fillText('t', tlp[0], tlp[1]);

    // z label (along left vertical)
    ctx.save();
    var zp = toS(proj(-0.14, 0, 0.5));
    ctx.translate(zp[0], zp[1]);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText('Angle diff (°)', 0, -4);
    ctx.restore();
  }

  // ── Compute fixed max angle ───────────────────────────

  function computeMaxAngle(amp) {
    let mx = 0;
    const N = SURF_N;
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        const r = i / (N - 1), t = j / (N - 1);
        if (t <= r + 0.02) continue;
        const posT = curvedPos(t, amp);
        const posR = curvedPos(r, amp);
        const vel = curvedVel(t, amp);
        const dir = [posT[0] - posR[0], posT[1] - posR[1]];
        mx = Math.max(mx, angleDiff(vel, dir));
      }
    }
    return mx;
  }

  // ── Main ──────────────────────────────────────────────

  function update(amp) {
    drawTrajectory(amp);
    drawSurface(amp);
  }

  function init() {
    const trajCanvas = document.getElementById('trajCanvas');
    const surfCanvas = document.getElementById('heatCanvas');
    const slider = document.getElementById('curvatureSlider');

    trajCtx = setupCanvas(trajCanvas, TRAJ_W, TRAJ_H);
    surfCtx = setupCanvas(surfCanvas, SURF_W, SURF_H);

    fixedMaxAngle = computeMaxAngle(0.8) * 1.05;

    slider.addEventListener('input', function () {
      update(parseFloat(this.value) / 100);
    });

    update(parseFloat(slider.value) / 100);
  }

  document.addEventListener('DOMContentLoaded', init);
})();
