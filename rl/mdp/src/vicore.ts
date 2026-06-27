/**
 * Value iteration core — a TypeScript port of David Poole's grid-world demo.
 * Faithfully reproduced from the original Java program VIcore.java
 * (GPL, © 2006-2007 David Poole).
 *
 * Coordinate convention (matches the original):
 *   x = column (horizontal, increases rightward), y = row (vertical, increases
 *   downward), both in the range 0..9.
 * Action convention:
 *   0 = up (y-1), 1 = right (x+1), 2 = down (y+1), 3 = left (x-1).
 * Transition noise: the intended direction is taken with probability 0.7, each
 *   of the other three directions with probability 0.1.
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

export class VICore {
  /** values[x][y] — the value function for state (x,y) */
  values: number[][] = zeros2d();
  /** qvalues[x][y][a] — the Q-value of taking action a in state (x,y) */
  qvalues: number[][][] = zeros3d();
  discount = 0.9;
  /** whether to treat +10 / +3 as absorbing states (stop on arrival) */
  absorbing = false;

  /** Perform one step of value iteration */
  dostep(newdiscount: number): void {
    this.discount = newdiscount;
    const newvalues = zeros2d();
    for (let x = 0; x < GRID; x++) {
      for (let y = 0; y < GRID; y++) {
        this.qvalues[x][y][0] = this.q(x, y, 0);
        newvalues[x][y] = this.qvalues[x][y][0];
        for (let a = 1; a < NUM_ACTIONS; a++) {
          this.qvalues[x][y][a] = this.q(x, y, a);
          if (this.qvalues[x][y][a] > newvalues[x][y]) {
            newvalues[x][y] = this.qvalues[x][y][a];
          }
        }
      }
    }
    this.values = newvalues;
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
    // The part that depends on the actual direction traveled (affected by transition noise)
    for (let dir = 0; dir < NUM_ACTIONS; dir++) {
      const contrib = this.contribution(x, y, dir);
      newqval += (action === dir ? 0.7 : 0.1) * contrib;
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

  /**
   * A string signature of the current greedy policy: for each cell, which
   * actions are optimal (i.e. tie the state value). Two steps with the same
   * signature mean the policy has stopped changing.
   */
  policyKey(): string {
    let s = "";
    for (let x = 0; x < GRID; x++) {
      for (let y = 0; y < GRID; y++) {
        const v = this.values[x][y];
        const q = this.qvalues[x][y];
        for (let a = 0; a < NUM_ACTIONS; a++) s += v === q[a] ? "1" : "0";
      }
    }
    return s;
  }

  /** Reset all values and Q-values to initVal */
  doreset(initVal: number): void {
    for (let x = 0; x < GRID; x++) {
      for (let y = 0; y < GRID; y++) {
        this.values[x][y] = initVal;
        for (let a = 0; a < NUM_ACTIONS; a++) this.qvalues[x][y][a] = initVal;
      }
    }
  }
}
