# Value Iteration Visualization

A value-iteration visualization of a 10×10 grid-world MDP. Pure front-end (React + TypeScript + Vite), ported from David Poole's Java applet.

## Run

```bash
cd rl/mdp
npm install
npm run dev      # local dev server
npm run build    # build static files into dist/
npm run preview  # preview the build output
```

## Usage

- **Step**: perform one step of value iteration. Click repeatedly to watch the value function converge and the optimal policy emerge.
- **Discount**: the discount factor γ (default 0.9), adjustable with −/+.
- **Reset / Initial Value**: reset all state values to the given initial value.
- **Brightness / Font Size / Grid Size**: display controls.
- **Absorbing States**: make the +10 / +3 reward cells absorbing (terminal) states.

Colors: green = positive value, red = negative, with brightness scaled by magnitude. Thin black arrows mark the optimal action(s) of each cell (there may be several tied). Special cells are outlined: gold = reward, blue = penalty.

## Code structure

| File | Responsibility |
|------|----------------|
| `src/vicore.ts` | Value-iteration algorithm (UI-independent, reusable on its own). Coordinates: x = column, y = row; actions 0/1/2/3 = up/right/down/left. |
| `src/GridCanvas.tsx` | Canvas rendering (cell coloring, arrows, values, grid lines, special-cell borders). |
| `src/Controls.tsx` | Control panel. |
| `src/App.tsx` | State management and composition. |

## Environment model

- Transition noise: the intended direction is taken with probability 0.7, each of the other three directions with probability 0.1; hitting a wall keeps the agent in place with a −1 reward.
- Reward cells: (8,7) = +10, (7,2) = +3 (when not absorbing, the agent is teleported to one of the four corners).
- Penalty cells: (3,4) = −5, (3,7) = −10.
- Update rule: V(s) ← max_a Q(s,a), where Q is computed from the previous V.

## Credits

Ported from the original Java applet by David Poole (GPL, © 2006-2007):
https://gist.github.com/benbotto/c3fb2e55117c5ef02917bfc1f48de4fe
