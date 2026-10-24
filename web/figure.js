// Spotter's coach: a flat 2D girl drawn on a canvas from joint positions.
// Each move is a function of time returning joints in a 400x300 box (ground at y = 285).
"use strict";

const COL = {
  skin: "#C98A5B", skinFar: "#A96F44",
  legs: "#111111", legsFar: "#3A3A3A",
  top: "#FF0000", topFar: "#B30000",
  hair: "#2B1A12", shoe: "#FFFFFF", sole: "#111111",
};

const ease = (k) => k * k * (3 - 2 * k);
const lerp = (a, b, k) => a + (b - a) * k;
function mix(p, q, k) {
  const out = {};
  for (const key in p) {
    out[key] = Array.isArray(p[key]) && typeof p[key][0] === "number"
      ? [lerp(p[key][0], q[key][0], k), lerp(p[key][1], q[key][1], k)]
      : p[key];
  }
  return out;
}
// play keyframes in a loop: frames = [pose, pose, ...], period in seconds
function loop(frames, period, t) {
  const n = frames.length;
  const x = ((t % period) / period) * n;
  const i = Math.floor(x) % n;
  return mix(frames[i], frames[(i + 1) % n], ease(x - Math.floor(x)));
}
const add = (p, dx, dy) => [p[0] + dx, p[1] + dy];

// ---------- poses (side view: "f" = far limb, "n" = near limb; front view: "l"/"r" = viewer's left/right)
const STAND_FRONT = {
  view: "front", head: [200, 95], face: [0, 1],
  shl: [172, 127], shr: [228, 127], ell: [166, 160], elr: [234, 160], wrl: [163, 190], wrr: [237, 190],
  hipl: [186, 192], hipr: [214, 192], knl: [182, 237], knr: [218, 237], anl: [180, 280], anr: [220, 280],
};
const SQUAT_FRONT = {
  ...STAND_FRONT, head: [200, 132],
  shl: [172, 162], shr: [228, 162], ell: [176, 182], elr: [224, 182], wrl: [190, 176], wrr: [210, 176],
  hipl: [182, 226], hipr: [218, 226], knl: [160, 242], knr: [240, 242], anl: [176, 280], anr: [224, 280],
};
const STAND_SIDE = {
  view: "side", head: [198, 95], face: [-1, 0],
  sh: [200, 127], elf: [203, 160], wrf: [204, 190], eln: [197, 160], wrn: [196, 190],
  hip: [200, 192], knf: [202, 237], anf: [203, 280], tof: [190, 283],
  knn: [198, 237], ann: [198, 280], ton: [185, 283],
};
const LUNGE_N = {
  ...STAND_SIDE, head: [198, 125], sh: [200, 157], elf: [204, 190], wrf: [205, 218], eln: [196, 190], wrn: [195, 218],
  hip: [202, 222], knn: [160, 228], ann: [160, 280], ton: [146, 283], knf: [236, 268], anf: [270, 266], tof: [280, 283],
};
const LUNGE_F = {
  ...LUNGE_N, knf: [160, 228], anf: [160, 280], tof: [146, 283], knn: [236, 268], ann: [270, 266], ton: [280, 283],
};
const PLANK = {
  view: "side", head: [86, 206], face: [-1, 0.25],
  sh: [116, 216], elf: [112, 272], wrf: [72, 276], eln: [118, 272], wrn: [78, 276],
  hip: [240, 238], knf: [305, 250], anf: [370, 262], tof: [378, 283],
  knn: [305, 250], ann: [370, 262], ton: [378, 283],
};
const KNEE_PLANK = {
  ...PLANK, hip: [232, 250], knf: [305, 276], anf: [350, 252], tof: [362, 244],
  knn: [305, 276], ann: [350, 252], ton: [362, 244],
};
const BRIDGE_DOWN = {
  view: "side", head: [92, 268], face: [0, -1],
  sh: [122, 272], elf: [165, 276], wrf: [205, 278], eln: [168, 277], wrn: [208, 279],
  hip: [214, 270], knf: [268, 222], anf: [300, 278], tof: [314, 284],
  knn: [268, 222], ann: [300, 278], ton: [314, 284],
};
const BRIDGE_UP = { ...BRIDGE_DOWN, hip: [214, 228], knf: [272, 214], knn: [272, 214] };
const HANDS_ON_HIPS = { ell: [152, 162], elr: [248, 162], wrl: [182, 188], wrr: [218, 188] };
const LEG_STAND = { ...STAND_FRONT, ...HANDS_ON_HIPS };
const LEG_R = { ...LEG_STAND, head: [194, 95], knr: [246, 230], anr: [274, 266] };
const LEG_L = { ...LEG_STAND, head: [206, 95], knl: [154, 230], anl: [126, 266] };
const PRESS_DOWN = { ...STAND_FRONT, ell: [148, 142], elr: [252, 142], wrl: [150, 106], wrr: [250, 106] };
const PRESS_UP = { ...STAND_FRONT, ell: [160, 88], elr: [240, 88], wrl: [170, 52], wrr: [230, 52] };
const WALL_OUT = {
  view: "side", head: [118, 138], face: [-1, 0], wall: 78,
  sh: [136, 152], elf: [108, 152], wrf: [82, 150], eln: [110, 154], wrn: [82, 152],
  hip: [193, 220], knf: [217, 251], anf: [242, 281], tof: [232, 284], knn: [217, 251], ann: [242, 281], ton: [232, 284],
};
const WALL_IN = {
  ...WALL_OUT, head: [100, 150], sh: [118, 164], elf: [110, 190], wrf: [82, 150], eln: [112, 192], wrn: [82, 152],
  hip: [182, 226],
};

const MOVES = {
  idle: (t) => mix(STAND_FRONT, { ...STAND_FRONT, head: [200, 93], shl: [172, 125], shr: [228, 125] }, (Math.sin(t * 2) + 1) / 2),
  squat: (t) => loop([STAND_FRONT, SQUAT_FRONT], 3.2, t),
  lunge: (t) => loop([STAND_SIDE, LUNGE_N, STAND_SIDE, LUNGE_F], 6, t),
  bridge: (t) => loop([BRIDGE_DOWN, BRIDGE_UP], 3, t),
  legraise: (t) => loop([LEG_STAND, LEG_R, LEG_STAND, LEG_L], 4, t),
  armcircles: (t) => {
    const a = t * 5, r = 12;
    return { ...STAND_FRONT, ell: [136, 128], elr: [264, 128],
      wrl: [102 + Math.cos(a) * r, 126 + Math.sin(a) * r], wrr: [298 - Math.cos(a) * r, 126 + Math.sin(a) * r] };
  },
  press: (t) => loop([PRESS_DOWN, PRESS_UP], 2.6, t),
  wallpush: (t) => loop([WALL_OUT, WALL_IN], 2.6, t),
  plank: (t) => mix(PLANK, { ...PLANK, hip: [240, 236] }, (Math.sin(t * 1.6) + 1) / 2),
  kneeplank: (t) => mix(KNEE_PLANK, { ...KNEE_PLANK, hip: [232, 248] }, (Math.sin(t * 1.6) + 1) / 2),
};

// ---------- drawing
function capsule(ctx, pts, width, color) {
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = "round"; ctx.lineJoin = "round";
  ctx.beginPath(); ctx.moveTo(...pts[0]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(...pts[i]);
  ctx.stroke();
}
function foot(ctx, ankle, toe) {
  capsule(ctx, [ankle, toe], 13, COL.sole);
  capsule(ctx, [ankle, toe], 9, COL.shoe);
}
const along = (a, b, k) => [lerp(a[0], b[0], k), lerp(a[1], b[1], k)];

function drawHead(ctx, j, t) {
  const [hx, hy] = j.head, [fx, fy] = j.face;
  const sway = Math.sin(t * 3) * 3;
  ctx.fillStyle = COL.hair;
  // ponytail behind the head
  ctx.strokeStyle = COL.hair; ctx.lineWidth = 9; ctx.lineCap = "round";
  ctx.beginPath();
  if (j.view === "front") {
    ctx.moveTo(hx + 12, hy - 12);
    ctx.quadraticCurveTo(hx + 26 + sway, hy - 4, hx + 22 + sway, hy + 22);
  } else {
    const bx = hx - fx * 15, by = hy - fy * 15 - (fy === 0 ? 6 : 0);
    ctx.moveTo(bx, by);
    ctx.quadraticCurveTo(bx - fx * 12, by + 4, bx - fx * 10 + sway * 0.5, by + 24 * (fy < 0 ? 0.2 : 1));
  }
  ctx.stroke();
  // hair cap, then face slightly forward so the hair shows on top/back
  ctx.beginPath(); ctx.arc(hx - fx * 2.5, hy - 2.5 - fy * 1.5, 18.5, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = COL.skin;
  ctx.beginPath(); ctx.arc(hx + fx * 2, hy + 2 + fy * 1.5, 15.5, 0, Math.PI * 2); ctx.fill();
}

function drawSide(ctx, j, t) {
  if (j.wall) {
    ctx.fillStyle = "#000"; ctx.fillRect(j.wall - 6, 20, 6, 265);
  }
  // far limbs first (darker)
  capsule(ctx, [j.sh, j.elf, j.wrf], 11, COL.skinFar);
  capsule(ctx, [j.hip, j.knf, j.anf], 19, COL.legsFar);
  foot(ctx, j.anf, j.tof);
  // neck + torso
  const neckBase = along(j.sh, j.head, 0.25);
  capsule(ctx, [j.head, neckBase], 11, COL.skin);
  capsule(ctx, [j.sh, j.hip], 34, COL.skin);
  capsule(ctx, [j.sh, along(j.sh, j.hip, 0.38)], 36, COL.top);
  capsule(ctx, [along(j.sh, j.hip, 0.82), j.hip], 34, COL.legs);
  // near limbs
  capsule(ctx, [j.hip, j.knn, j.ann], 21, COL.legs);
  foot(ctx, j.ann, j.ton);
  capsule(ctx, [j.sh, j.eln, j.wrn], 11, COL.skin);
  drawHead(ctx, j, t);
}

function drawFront(ctx, j, t) {
  const neck = [(j.shl[0] + j.shr[0]) / 2, (j.shl[1] + j.shr[1]) / 2];
  // legs
  capsule(ctx, [j.hipl, j.knl, j.anl], 20, COL.legs);
  capsule(ctx, [j.hipr, j.knr, j.anr], 20, COL.legs);
  foot(ctx, j.anl, add(j.anl, -6, 3));
  foot(ctx, j.anr, add(j.anr, 6, 3));
  // torso: rounded quad
  const quad = (a, b, c, d, color) => {
    ctx.fillStyle = color; ctx.strokeStyle = color; ctx.lineWidth = 14; ctx.lineJoin = "round";
    ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.lineTo(...c); ctx.lineTo(...d); ctx.closePath();
    ctx.fill(); ctx.stroke();
  };
  capsule(ctx, [j.head, neck], 12, COL.skin);
  quad(j.shl, j.shr, j.hipr, j.hipl, COL.skin);
  quad(j.shl, j.shr, along(j.shr, j.hipr, 0.42), along(j.shl, j.hipl, 0.42), COL.top);
  quad(along(j.shl, j.hipl, 0.84), along(j.shr, j.hipr, 0.84), j.hipr, j.hipl, COL.legs);
  // arms
  capsule(ctx, [j.shl, j.ell, j.wrl], 11, COL.skin);
  capsule(ctx, [j.shr, j.elr, j.wrr], 11, COL.skin);
  drawHead(ctx, j, t);
}

class Coach {
  constructor(canvas) {
    this.canvas = canvas; this.ctx = canvas.getContext("2d");
    this.move = "idle"; this.t0 = performance.now();
    const tick = () => { this.draw(); requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  }
  setMove(name) { if (MOVES[name] && name !== this.move) { this.move = name; } }
  draw() {
    const c = this.canvas, ctx = this.ctx, dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, c.width, c.height);
    // keep clear of the title/timer (top) and the subtitle bar (bottom)
    const top = 62 * dpr, bottom = 66 * dpr, availH = Math.max(50, c.height - top - bottom);
    const s = Math.min(c.width / 400, availH / 290);
    ctx.setTransform(s, 0, 0, s, (c.width - 400 * s) / 2, top + (availH - 290 * s) / 2 - 2 * s);
    // floor line
    ctx.fillStyle = "#DDDDDD"; ctx.fillRect(0, 285, 400, 3);
    const t = (performance.now() - this.t0) / 1000;
    const j = MOVES[this.move](t);
    (j.view === "side" ? drawSide : drawFront)(ctx, j, t);
  }
}
window.Coach = Coach;
