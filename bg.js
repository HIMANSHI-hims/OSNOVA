// OSNOVA animated background: drifting nodes joined by shining web-like lines
(() => {
  const c = document.getElementById('bgCanvas'), x = c.getContext('2d');
  let W, H, pts = [], mouse = { x: -999, y: -999 };
  const resize = () => {
    W = c.width = innerWidth; H = c.height = innerHeight;
    const n = Math.min(110, Math.floor(W * H / 14000));
    pts = Array.from({ length: n }, () => ({
      x: Math.random() * W, y: Math.random() * H,
      vx: (Math.random() - .5) * .35, vy: (Math.random() - .5) * .35
    }));
  };
  addEventListener('resize', resize);
  addEventListener('mousemove', e => { mouse.x = e.clientX; mouse.y = e.clientY; });
  resize();
  const D = 150;
  (function frame(t) {
    const light = document.body.getAttribute('data-theme') === 'light';
    const rgb = light ? '30,34,40' : '255,255,255';
    x.clearRect(0, 0, W, H);
    for (const p of pts) {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > W) p.vx *= -1;
      if (p.y < 0 || p.y > H) p.vy *= -1;
    }
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i];
      for (let j = i + 1; j < pts.length; j++) {
        const b = pts[j], d = Math.hypot(a.x - b.x, a.y - b.y);
        if (d < D) {
          // shine pulse travelling across the web
          const shine = Math.max(0, Math.sin(t / 900 + (a.x + b.y) / 140)) ** 6;
          x.strokeStyle = `rgba(${rgb},${(1 - d / D) * (.16 + shine * .6)})`;
          x.lineWidth = .6 + shine * .9;
          x.beginPath(); x.moveTo(a.x, a.y); x.lineTo(b.x, b.y); x.stroke();
        }
      }
      const m = Math.hypot(a.x - mouse.x, a.y - mouse.y);
      if (m < 190) {
        x.strokeStyle = `rgba(${rgb},${(1 - m / 190) * .45})`;
        x.beginPath(); x.moveTo(a.x, a.y); x.lineTo(mouse.x, mouse.y); x.stroke();
      }
      x.fillStyle = `rgba(${rgb},.55)`;
      x.beginPath(); x.arc(a.x, a.y, 1.3, 0, 6.283); x.fill();
    }
    requestAnimationFrame(frame);
  })(0);
})();