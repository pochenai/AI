interface Props {
  discount: string;
  setDiscount: (v: string) => void;
  initialValue: string;
  setInitialValue: (v: string) => void;
  absorbing: boolean;
  setAbsorbing: (v: boolean) => void;
  onStep: () => void;
  onReset: () => void;
  onBrightness: (delta: "up" | "down" | "reset") => void;
  fontSize: number;
  setFontSize: (f: number) => void;
  sqsize: number;
  setSqsize: (s: number) => void;
}

const round1 = (n: number) => Math.round(n * 10) / 10;

export function Controls(p: Props) {
  return (
    <div className="controls">
      <button className="big" onClick={p.onStep}>
        Step
      </button>

      <div className="row">
        <label>Discount</label>
        <button onClick={() => p.setDiscount(String(round1(parseFloat(p.discount) - 0.1)))}>
          −
        </button>
        <input
          value={p.discount}
          size={3}
          onChange={(e) => p.setDiscount(e.target.value)}
        />
        <button onClick={() => p.setDiscount(String(round1(parseFloat(p.discount) + 0.1)))}>
          +
        </button>
      </div>

      <button className="big" onClick={p.onReset}>
        Reset
      </button>
      <div className="row">
        <label>Initial Value</label>
        <input
          value={p.initialValue}
          size={3}
          onChange={(e) => p.setInitialValue(e.target.value)}
        />
      </div>

      <div className="row">
        <label>Brightness</label>
        <button onClick={() => p.onBrightness("down")}>−</button>
        <button onClick={() => p.onBrightness("reset")}>0</button>
        <button onClick={() => p.onBrightness("up")}>+</button>
      </div>

      <div className="row">
        <label>Font Size</label>
        <button onClick={() => p.setFontSize(Math.max(4, p.fontSize - 1))}>−</button>
        <button onClick={() => p.setFontSize(p.fontSize + 1)}>+</button>
      </div>

      <div className="row">
        <label>Grid Size</label>
        <button onClick={() => p.setSqsize(Math.max(15, p.sqsize - 5))}>−</button>
        <button onClick={() => p.setSqsize(p.sqsize + 5)}>+</button>
      </div>

      <div className="row">
        <label>
          <input
            type="checkbox"
            checked={p.absorbing}
            onChange={(e) => p.setAbsorbing(e.target.checked)}
          />
          Absorbing States
        </label>
      </div>
    </div>
  );
}
