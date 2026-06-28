/**
 * Value iteration core — a TypeScript port of David Poole's grid-world demo.
 * Faithfully reproduced from the original Java program VIcore.java
 * (GPL, © 2006-2007 David Poole).
 *
 * Structure:
 *   - VICore (abstract): owns the value function `values` and the generic
 *     value-iteration loop `dostep()`. It does not know *how* a state's value
 *     is backed up, nor what the greedy policy is — those are abstract hooks.
 *   - QValueVICore: the default solver. Backs up each state as max_a Q(s,a)
 *     and exposes the greedy policy as argmax_a Q(s,a). Swap in another
 *     subclass (e.g. a plain value-function backup) without touching the loop.
 *
 * Coordinate convention (matches the original):
 *   x = column (horizontal, increases rightward), y = row (vertical, increases
 *   downward), both in the range 0..9.
 * Action convention:
 *   0 = up (y-1), 1 = right (x+1), 2 = down (y+1), 3 = left (x-1).
 * Transition noise: the intended direction is taken with probability
 *   `transitionProb` (default 0.7, adjustable 0..1), and the remaining
 *   (1 - transitionProb) is split uniformly among the other three directions
 *   ((1 - p)/3 each). transitionProb = 1.0 makes the world deterministic.
 * Hitting a wall: the agent stays in place and receives a -1 reward.
 */

export const GRID = 10;
export const NUM_ACTIONS = 4;

export interface SpecialCell {
  x: number;
  y: number;
  reward: number;
  kind: "reward" | "penalty";
}

/** The four special cells (single source of truth, matching the positions/rewards hardcoded in q()) */
export const SPECIAL_CELLS: SpecialCell[] = [
  { x: 8, y: 7, reward: 10, kind: "reward" },
  { x: 7, y: 2, reward: 3, kind: "reward" },
  { x: 3, y: 4, reward: -5, kind: "penalty" },
  { x: 3, y: 7, reward: -10, kind: "penalty" },
];

function zeros2d(): number[][] {
  return Array.from({ length: GRID }, () => new Array<number>(GRID).fill(0));
}

function zeros3d(): number[][][] {
  return Array.from({ length: GRID }, () =>
    Array.from({ length: GRID }, () => new Array<number>(NUM_ACTIONS).fill(0)),
  );
}

/**
 * Abstract value-iteration solver. Owns the value function and the generic
 * sweep; subclasses decide how a single state is backed up and what the greedy
 * policy is.
 */
export abstract class VICore {
  /** values[x][y] — the value function V(s) for state (x,y) */
  values: number[][] = zeros2d();
  discount = 0.9;
  /**
   * Probability the agent actually moves in its intended direction. The
   * remaining (1 - transitionProb) is split uniformly among the other three
   * directions. 1.0 = deterministic; 0.7 is the classic noisy setting.
   * Part of the MDP model, so it lives on the base shared by all solvers.
   */
  transitionProb = 0.7;
  /** whether to treat +10 / +3 as absorbing states (stop on arrival) */
  absorbing = false;

  /**
   * One sweep of value iteration: V_{k+1}(s) = backupValue(s) for every state,
   * each computed from the current V_k (synchronous update).
   */
  dostep(newdiscount: number): void {
    this.discount = newdiscount;
    const newvalues = zeros2d();
    for (let x = 0; x < GRID; x++) {
      for (let y = 0; y < GRID; y++) {
        newvalues[x][y] = this.backupValue(x, y);
      }
    }
    this.values = newvalues;
  }

  /**
   * One Bellman backup for state (x,y): compute the new V(s) from the current
   * `values`. Subclasses implement the actual rule.
   */
  protected abstract backupValue(x: number, y: number): number;

  /**
   * The greedy policy at (x,y): one boolean per action marking the optimal
   * direction(s). Drives the policy arrows. Subclasses implement it.
   */
  abstract optimalActions(x: number, y: number): boolean[];

  /**
   * A string signature of the current greedy policy across all cells. Two
   * steps with the same signature mean the policy has stopped changing.
   */
  policyKey(): string {
    let s = "";
    for (let x = 0; x < GRID; x++) {
      for (let y = 0; y < GRID; y++) {
        const opt = this.optimalActions(x, y);
        for (let a = 0; a < NUM_ACTIONS; a++) s += opt[a] ? "1" : "0";
      }
    }
    return s;
  }

  /** Reset the value function to initVal (plus any subclass state). */
  doreset(initVal: number): void {
    for (let x = 0; x < GRID; x++) {
      for (let y = 0; y < GRID; y++) this.values[x][y] = initVal;
    }
    this.onReset(initVal);
  }

  /** Hook for subclasses to reset extra state (e.g. Q-values). */
  protected onReset(_initVal: number): void { }
}


/**
 * Value iteration via action values:
 *   V(s) = max_a Q(s,a),  Q(s,a) = R_s^a + γ Σ_s' P_ss'^a V(s').
 * The per-action Q-values are retained so the greedy policy (arrows) can be
 * read off as argmax_a Q(s,a).
 */
export class QValueVICore extends VICore {
  /** qvalues[x][y][a] — Q(s,a) from the most recent backup */
  qvalues: number[][][] = zeros3d();

  /** V(s) ← max_a Q(s,a); stores each Q(s,a) for policy extraction. */
  protected backupValue(x: number, y: number): number {
    let best = (this.qvalues[x][y][0] = this.q(x, y, 0));
    for (let a = 1; a < NUM_ACTIONS; a++) {
      const qa = (this.qvalues[x][y][a] = this.q(x, y, a));
      if (qa > best) best = qa;
    }
    return best;
  }

  /** Greedy policy: the action(s) whose Q-value ties the state value. */
  optimalActions(x: number, y: number): boolean[] {
    const v = this.values[x][y];
    return this.qvalues[x][y].map((qa) => v === qa);
  }

  /** Compute a single Q-value from the previous value function */
  q(x: number, y: number, action: number): number {
    // Two reward states: any action yields the reward; when not absorbing, the
    // agent is "teleported" to one of the four corners.
    if (x === 8 && y === 7) {
      return this.absorbing
        ? 10.0
        : 10.0 +
        this.discount *
        0.25 *
        (this.values[0][0] +
          this.values[0][9] +
          this.values[9][0] +
          this.values[9][9]);
    }
    if (x === 7 && y === 2) {
      return this.absorbing
        ? 3.0
        : 3.0 +
        this.discount *
        0.25 *
        (this.values[0][0] +
          this.values[0][9] +
          this.values[9][0] +
          this.values[9][9]);
    }

    let newqval = 0.0;
    // The part that depends on the actual direction traveled (affected by
    // transition noise): P(actual = dir | intended = action) is `transitionProb`
    // for the intended direction and (1 - transitionProb)/3 for each of the others.
    const p = this.transitionProb;
    const other = (1 - p) / 3;
    for (let dir = 0; dir < NUM_ACTIONS; dir++) {
      const contrib = this.contribution(x, y, dir);
      newqval += (action === dir ? p : other) * contrib;
    }

    // Action-independent penalty states
    if (x === 3 && y === 4) newqval += -5.0;
    else if (x === 3 && y === 7) newqval += -10.0;

    return newqval;
  }

  /** Contribution to the Q-value if the agent actually moves in direction dir */
  contribution(x: number, y: number, dir: number): number {
    switch (dir) {
      case 0: // up
        return y === 0
          ? -1.0 + this.discount * this.values[x][y] // hit wall
          : this.discount * this.values[x][y - 1];
      case 1: // right
        return x === 9
          ? -1.0 + this.discount * this.values[x][y]
          : this.discount * this.values[x + 1][y];
      case 2: // down
        return y === 9
          ? -1.0 + this.discount * this.values[x][y]
          : this.discount * this.values[x][y + 1];
      case 3: // left
        return x === 0
          ? -1.0 + this.discount * this.values[x][y]
          : this.discount * this.values[x - 1][y];
      default:
        return 0.0;
    }
  }

  /** Also reset the Q-values to initVal. */
  protected onReset(initVal: number): void {
    for (let x = 0; x < GRID; x++) {
      for (let y = 0; y < GRID; y++) {
        for (let a = 0; a < NUM_ACTIONS; a++) this.qvalues[x][y][a] = initVal;
      }
    }
  }
}
