import { useEffect, useRef } from "react";
import { GRID, SPECIAL_CELLS, VICore } from "./vicore";

// Special-cell border colors (kept consistent with the header legend)
export const REWARD_BORDER = "#f4b400"; // gold: reward cells
export const PENALTY_BORDER = "#2f6bff"; // blue: penalty cells

interface Props {
  core: VICore;
  /** bumped on every step/reset to trigger a redraw */
  version: number;
  sqsize: number;
  brightness: number;
  fontSize: number;
}

/** Format a value, mirroring the original DecimalFormat("0.##") */
function fmt(v: number): string {
  return Number(v.toFixed(2)).toString();
}

// Color basis: value 0 is light gray, fading toward green/red for positive/negative
const BASE = [228, 228, 230]; // light gray
const POS = [34, 160, 34]; // green
const NEG = [200, 48, 48]; // red

function lerp(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * t);
}

function cellColor(v: number, brightness: number): string {
  const target = v >= 0 ? POS : NEG;
  const t = Math.min(Math.pow(Math.abs(v) / 10, brightness), 1);
  const r = lerp(BASE[0], target[0], t);
  const g = lerp(BASE[1], target[1], t);
  const b = lerp(BASE[2], target[2], t);
  return `rgb(${r},${g},${b})`;
}

/** Draw a thin arrow from the cell center pointing in a given direction */
function drawArrow(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  dir: number,
  len: number,
) {
  // dir: 0=up 1=right 2=down 3=left
  const dx = dir === 1 ? 1 : dir === 3 ? -1 : 0;
  const dy = dir === 2 ? 1 : dir === 0 ? -1 : 0;
  const tipX = cx + dx * len;
  const tipY = cy + dy * len;
  const head = Math.max(3, len * 0.45); // arrowhead wing length

  ctx.beginPath();
  // shaft
  ctx.moveTo(cx, cy);
  ctx.lineTo(tipX, tipY);
  // two arrowhead wings (~30° from the shaft)
  // unit vector perpendicular to the direction is (-dy, dx)
  ctx.moveTo(tipX, tipY);
  ctx.lineTo(tipX - dx * head + -dy * head * 0.55, tipY - dy * head + dx * head * 0.55);
  ctx.moveTo(tipX, tipY);
  ctx.lineTo(tipX - dx * head - -dy * head * 0.55, tipY - dy * head - dx * head * 0.55);
  ctx.stroke();
}

export function GridCanvas({ core, version, sqsize, brightness, fontSize }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const size = sqsize * GRID;

    // Handle high-DPI screens
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.clearRect(0, 0, size, size);
    ctx.font = `${fontSize}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineJoin = "round";
    ctx.lineCap = "round";

    const arrowLen = sqsize * 0.3;

    for (let x = 0; x < GRID; x++) {
      for (let y = 0; y < GRID; y++) {
        const v = core.values[x][y];
        const cx = x * sqsize + sqsize / 2;
        const cy = y * sqsize + sqsize / 2;

        // 1) Cell background
        ctx.fillStyle = cellColor(v, brightness);
        ctx.fillRect(x * sqsize, y * sqsize, sqsize, sqsize);

        // 2) Value (drawn first, as the bottom layer): white text centered,
        //    with a dark outline for readability.
        ctx.lineWidth = 3;
        ctx.strokeStyle = "rgba(0,0,0,0.65)";
        ctx.strokeText(fmt(v), cx, cy);
        ctx.fillStyle = "white";
        ctx.fillText(fmt(v), cx, cy);

        // 3) Thin black arrows for the optimal action(s), drawn on top of the
        //    value so they aren't hidden. If all four actions tie (e.g. the
        //    all-zero initial state) there is no clear policy, so skip them.
        const q = core.qvalues[x][y];
        const optimal: number[] = [];
        for (let a = 0; a < 4; a++) if (v === q[a]) optimal.push(a);
        if (optimal.length < 4) {
          ctx.strokeStyle = "rgba(0,0,0,0.85)";
          ctx.lineWidth = Math.max(1, sqsize / 40);
          for (const a of optimal) drawArrow(ctx, cx, cy, a, arrowLen);
        }
      }
    }

    // 4) Grid lines (mid gray, clearly visible)
    ctx.strokeStyle = "#9a9a9a";
    ctx.lineWidth = 1;
    for (let i = 0; i <= GRID; i++) {
      ctx.beginPath();
      ctx.moveTo(i * sqsize + 0.5, 0);
      ctx.lineTo(i * sqsize + 0.5, size);
      ctx.moveTo(0, i * sqsize + 0.5);
      ctx.lineTo(size, i * sqsize + 0.5);
      ctx.stroke();
    }

    // 5) Colored borders for special cells (topmost layer): gold = reward, blue = penalty
    const bw = Math.max(2, sqsize / 14);
    ctx.lineWidth = bw;
    for (const c of SPECIAL_CELLS) {
      ctx.strokeStyle = c.kind === "reward" ? REWARD_BORDER : PENALTY_BORDER;
      const inset = bw / 2 + 0.5;
      ctx.strokeRect(
        c.x * sqsize + inset,
        c.y * sqsize + inset,
        sqsize - 2 * inset,
        sqsize - 2 * inset,
      );
    }
  }, [core, version, sqsize, brightness, fontSize]);

  return <canvas ref={canvasRef} />;
}
