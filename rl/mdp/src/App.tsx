import { useRef, useState } from "react";
import { Controls } from "./Controls";
import { GridCanvas } from "./GridCanvas";
import { QValueVICore } from "./vicore";

export default function App() {
  // core is a mutable object kept as the same instance across renders.
  // Lazily created once so we can set initial model options (e.g. absorbing).
  const coreRef = useRef<QValueVICore | null>(null);
  if (coreRef.current === null) {
    coreRef.current = new QValueVICore();
    coreRef.current.absorbing = true; // default: Absorbing States on
  }
  const core = coreRef.current;

  const [version, setVersion] = useState(0); // bump to trigger canvas redraw
  const [discount, setDiscount] = useState("0.9");
  const [transitionProb, setTransitionProb] = useState("0.7");
  const [initialValue, setInitialValue] = useState("0.0");
  const [absorbing, setAbsorbingState] = useState(true);
  const [brightness, setBrightness] = useState(1.0);
  const [fontSize, setFontSize] = useState(11);
  const [sqsize, setSqsize] = useState(50);
  // true when the greedy policy is identical before and after the last step
  // (note: the value function may still be changing, so this is not a proof of optimality)
  const [policyUnchanged, setPolicyUnchanged] = useState(false);
  // number of value-iteration steps performed since the last reset
  const [stepCount, setStepCount] = useState(0);

  const redraw = () => setVersion((v) => v + 1);

  const onStep = () => {
    const d = parseFloat(discount);
    if (!Number.isNaN(d)) {
      const p = parseFloat(transitionProb);
      if (!Number.isNaN(p)) core.transitionProb = Math.min(Math.max(p, 0), 1);
      const before = core.policyKey();
      core.dostep(d);
      // The policy (arrow pattern) is identical before and after this step.
      setPolicyUnchanged(core.policyKey() === before);
      setStepCount((n) => n + 1);
    }
    redraw();
  };

  const onReset = () => {
    const v = parseFloat(initialValue);
    core.doreset(Number.isNaN(v) ? 0 : v);
    setPolicyUnchanged(false);
    setStepCount(0);
    redraw();
  };

  const setAbsorbing = (v: boolean) => {
    core.absorbing = v;
    setAbsorbingState(v);
    // The model changed, so the previous comparison no longer holds.
    setPolicyUnchanged(false);
  };

  // Original behavior: "-" dims (×1.1), "+" brightens (÷1.1)
  const onBrightness = (delta: "up" | "down" | "reset") => {
    if (delta === "reset") setBrightness(1.0);
    else if (delta === "down") setBrightness((b) => b * 1.1);
    else setBrightness((b) => b / 1.1);
  };

  return (
    <div className="app">
      <header>
        <h1>Value Iteration Visualization</h1>
        <p>
          A 10×10 grid-world MDP. Cell color: green = positive value, red = negative;
          thin black arrows show the current optimal action. Click <strong>Step </strong>
          repeatedly to watch the value function converge and the optimal policy emerge.
        </p>
        <ul className="legend">
          <li>
            <span className="arrow-mark">→</span>
            <strong>Arrows = policy</strong>: the optimal move direction(s) for each cell
            (argmax over actions). Multiple arrows mean tied-optimal directions.
          </li>
          <li>
            <span className="swatch" style={{ borderColor: "#f4b400" }} />
            <strong>Reward cells (gold border)</strong>: (7,2)=+3, (8,7)=+10. Reward granted on arrival.
          </li>
          <li>
            <span className="swatch" style={{ borderColor: "#2f6bff" }} />
            <strong>Penalty cells (blue border)</strong>: (3,4)=−5, (3,7)=−10. Only add a negative
            reward; movement is unaffected and they never terminate.
          </li>
          <li>
            <strong>Transition P</strong>: probability the agent moves in its intended
            direction (default 0.7); the remaining 1−P is split evenly among the other
            three directions. Set it to 1.0 for a deterministic world.
          </li>
          <li>
            <strong>Absorbing States</strong>: when checked, the two reward cells become{" "}
            <strong>terminal / absorbing</strong> — the agent stops on arrival (value fixed at
            +10 / +3). When unchecked (default), the agent collects the reward and is then
            teleported to a random corner, and the episode continues.
          </li>
        </ul>
        <p className="note">
          Original Java: <a href="https://gist.github.com/benbotto/c3fb2e55117c5ef02917bfc1f48de4fe">https://gist.github.com/benbotto/c3fb2e55117c5ef02917bfc1f48de4fe</a>
        </p>
      </header>
      <div className="layout">
        <div className="grid-col">
          <div className="grid-wrap">
            <GridCanvas
              core={core}
              version={version}
              sqsize={sqsize}
              brightness={brightness}
              fontSize={fontSize}
            />
          </div>
          <p className="step-count">Step: {stepCount}</p>
          {policyUnchanged && (
            <p className="optimal">Policy unchanged from previous step</p>
          )}
        </div>
        <Controls
          discount={discount}
          setDiscount={setDiscount}
          transitionProb={transitionProb}
          setTransitionProb={setTransitionProb}
          initialValue={initialValue}
          setInitialValue={setInitialValue}
          absorbing={absorbing}
          setAbsorbing={setAbsorbing}
          onStep={onStep}
          onReset={onReset}
          onBrightness={onBrightness}
          fontSize={fontSize}
          setFontSize={setFontSize}
          sqsize={sqsize}
          setSqsize={setSqsize}
        />
      </div>
    </div>
  );
}
