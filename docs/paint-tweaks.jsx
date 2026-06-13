/* paint-tweaks.jsx — Tweaks panel for the PAINT page.
   Loaded after tweaks-panel.jsx (which exports the controls to window). */

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "accent": "#1846c8",
  "headFont": "Newsreader",
  "dark": false
}/*EDITMODE-END*/;

// chosen (light-mode) accent -> brighter variant for dark mode
const ACCENT_DARK = {
  "#1846c8": "#6f97ff",
  "#047857": "#34c98a",
  "#6d28d9": "#a98bf5",
  "#191b20": "#cfd2da",
};

function applyPaintTweaks(t) {
  const root = document.documentElement;
  root.setAttribute("data-theme", t.dark ? "dark" : "light");
  const accent = t.dark ? (ACCENT_DARK[t.accent] || t.accent) : t.accent;
  root.style.setProperty("--accent", accent);
  root.style.setProperty("--font-head", `"${t.headFont}", Georgia, "Times New Roman", serif`);
}

function PaintTweaks() {
  const { useTweaks, TweaksPanel, TweakSection, TweakColor, TweakRadio, TweakToggle } = window;
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);

  React.useEffect(() => { applyPaintTweaks(t); }, [t.accent, t.headFont, t.dark]);

  return (
    <TweaksPanel title="Tweaks">
      <TweakSection label="Color" />
      <TweakColor
        label="Accent"
        value={t.accent}
        options={["#1846c8", "#047857", "#6d28d9", "#191b20"]}
        onChange={(v) => setTweak("accent", v)}
      />
      <TweakToggle
        label="Dark mode"
        value={t.dark}
        onChange={(v) => setTweak("dark", v)}
      />
      <TweakSection label="Typography" />
      <TweakRadio
        label="Headings"
        value={t.headFont}
        options={["Newsreader", "Spectral", "Space Grotesk"]}
        onChange={(v) => setTweak("headFont", v)}
      />
    </TweaksPanel>
  );
}

// apply defaults immediately (before React mounts) to avoid a flash
applyPaintTweaks(TWEAK_DEFAULTS);

(function mount() {
  const root = document.getElementById("tweaks-root");
  if (root && window.ReactDOM && window.useTweaks) {
    ReactDOM.createRoot(root).render(<PaintTweaks />);
  } else {
    setTimeout(mount, 60);
  }
})();
