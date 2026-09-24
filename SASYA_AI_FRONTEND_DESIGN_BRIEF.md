# Sasya AI — Frontend Design Brief

**Purpose of this document**: this is the design spec Antigravity should build and refine the UI against. It defines the visual identity, every feature the UI must surface, and the implementation approach given our current stack (Streamlit). Treat this as a brief from a client who has already rejected generic, templated-looking AI output — the palette and layout choices below are deliberate and grounded in what this product actually is, not defaults.

---

## 0. Two implementation paths — pick one before building

**Path A — Supercharge the existing Streamlit app (recommended given our timeline).**
Streamlit supports full custom CSS injection and a themeable config, so a genuinely striking UI is achievable without touching our working backend integration. This is additive to code that already works — lowest risk.

**Path B — Separate custom HTML/CSS/JS frontend calling a small FastAPI wrapper around our existing pipeline.**
Gives full visual freedom (real animations, custom components, true futuristic HUD-style interactions) but requires standing up a new API layer around code that currently only runs inside Streamlit. This is a genuinely bigger lift this close to the deadline, with the same kind of integration risk already flagged for the WhatsApp bot idea — do not start this unless Path A is fully done and there's real time left.

**This brief is written primarily for Path A.** Section 8 covers what changes if you pursue Path B instead.

---

## 1. Design Vision

Sasya AI is a diagnostic tool — a farmer photographs a leaf and the system scans it, reads its condition, and reports back. The visual identity should feel like **a living diagnostic scanner**: something between a plant biology lab instrument and a sci-fi HUD, grounded in the actual subject matter (plant health, disease, growth) rather than generic "AI app" tropes.

Avoid these specific clichés (common tells of ungrounded AI-generated design — do not default into them):
- Warm cream background with a terracotta accent
- Near-black background with a single generic neon accent and no other color relationships
- Identical rounded "SaaS cards" with the same soft grey shadow on everything
- All-caps tracked-out eyebrow labels above every heading, middle-dot-joined meta text, or an arrow (→) tacked onto every button

Instead: lean into a **bio-diagnostic** identity — dark, focused, lab-instrument feel, with color that means something (green = health signal, violet = active AI processing, amber = caution), not decoration for its own sake.

---

## 2. Design Tokens

### Color — 6 core colors, each with a job, not just decoration
| Name | Hex | Role |
|---|---|---|
| Void Canopy | `#0E1512` | Base background — deep, near-black, faint green undertone (not generic pure black) |
| Canopy Surface | `#16211C` | Card/panel surface, one step up from base |
| Signal Green | `#3DFFA2` | Primary accent — healthy states, primary buttons, confidence bars for high-confidence results |
| Electric Violet | `#7C4DFF` | AI-activity accent — used ONLY during active processing/scanning and for anything AI-generated (voice Q&A responses) — this color visually signals "the model is doing something," a deliberate semantic use, not just a second pretty color |
| Amber Pulse | `#FFB020` | Caution accent — severity warnings (moderate/severe), weather caveats, low-confidence flags |
| Mist White | `#F1F5F0` | Primary text/foreground |

Derived/muted: `Fern Grey` `#8FA396` for secondary/meta text (timestamps, helper copy) — don't introduce more named colors beyond this set.

**Semantic color rule** (important — don't let this drift during implementation): green = confirmed healthy/high-confidence, amber = caution/moderate severity or low confidence, a soft red-orange (`#FF6B4A`, used sparingly, not a core token) = severe/high-severity only. Violet is reserved exclusively for "AI is actively processing" moments — a progress spinner, the scan animation, the voice-assistant response bubble. This gives the palette actual meaning instead of being arbitrary.

### Typography — two fonts, each doing a distinct job
- **Display/headline: `Chakra Petch`** (Google Fonts) — a technical, slightly angular geometric face that reads as instrumentation/HUD without being a cliché "sci-fi" font. Use for the app name, section headings, and the diagnosis result headline (e.g. the disease name).
- **Body: `Manrope`** (Google Fonts) — clean, modern grotesk for all body copy, labels, and advisory text. Deliberately not Inter/Poppins (the generic defaults) — Manrope has more character in its letterforms while staying highly readable.
- **Kannada text: `Noto Sans Kannada`** (Google Fonts) — required for correctly rendering the Kannada translation output. Pair it visually with Manrope's weight/size scale so Kannada and English text feel like one consistent system, not a mismatched afterthought.

Type scale: headline 32–40px, section headers 20–24px, body 16px, meta/labels 13px. Line length for advisory body text should stay under 80 characters — use column max-widths to enforce this even on wide screens.

### Layout concept
Single scrolling flow, top to bottom, center-aligned content within a max-width container (don't let text/cards stretch full-width on large screens — cap around 900–1000px for the main content column, wider only for the image/heatmap comparison).

```
┌─────────────────────────────────────────┐
│           SASYA AI  (ಸಸ್ಯ AI)            │  ← hero, headline treatment
│   one-line mission statement              │
├─────────────────────────────────────────┤
│  [ district selector ]  [ upload photo ] │  ← input row, side by side on
│                                            │    desktop, stacked on mobile
├─────────────────────────────────────────┤
│                                            │
│         [ scan animation state ]          │  ← appears only while processing
│                                            │
├─────────────────────────────────────────┤
│  ┌───────────────┐  ┌──────────────────┐ │
│  │ original image│  │ grad-cam heatmap │ │  ← side-by-side comparison
│  └───────────────┘  └──────────────────┘ │
│  Disease name · confidence · severity     │
├─────────────────────────────────────────┤
│  ADVISORY CARD                            │
│  cause / treatment / prevention           │
│  [ weather caveat banner, if present ]    │
├─────────────────────────────────────────┤
│  ಕನ್ನಡ  Kannada translation text          │
│  [ ▶ audio player ]                       │
├─────────────────────────────────────────┤
│  Ask a follow-up question (voice)         │
│  [ mic input ]  →  [ response + audio ]   │
├─────────────────────────────────────────┤
│  footer: how this works / trust note      │
└─────────────────────────────────────────┘
```

---

## 3. Motion — one deliberate moment, not scattered effects

Per design discipline: spend animation budget in exactly one place, make it count, and leave everything else static and calm. The one moment here is **the scan animation** during processing: a horizontal scan-line sweeping down over the uploaded leaf image (in Electric Violet, semi-transparent, glowing edge), echoing an actual diagnostic scanner. This single animation does double duty — it's the loading state AND it visually foreshadows the Grad-CAM heatmap reveal that follows it.

Do not add: hover-lift effects on every card, fade-and-slide-up entrances on every section as you scroll, or pulsing effects on multiple elements simultaneously. Pick this one moment and let the rest of the interface be still and confident.

---

## 4. Full Feature Inventory — every UI element the design must cover

This maps directly to what the backend already does — nothing here should be invented UI without a backend feature behind it, and nothing the backend does should be missing from the UI.

1. **App identity/hero** — name (English + Kannada script), one-line mission statement grounded in the real problem (not generic "AI-powered" copy — say what it actually does: photograph a leaf, get a diagnosis and spoken advice in Kannada).
2. **District selector** — dropdown for the weather-aware layer (Karnataka districts + "Other").
3. **Image upload** — drag-and-drop or tap-to-upload, with a clear preview of the uploaded photo before analysis runs.
4. **Scan/processing state** — the one animated moment described in Section 3. Copy during this state should be specific to what's happening (e.g. "Reading leaf pattern…" then "Checking against known diseases…"), not a generic spinner with "Loading...".
5. **Diagnosis result** — disease name (styled with the display font), confidence score (a bar or radial indicator using the semantic color rule — green/amber/red by confidence band), and severity label + percentage (from the Grad-CAM-based severity layer).
6. **Image comparison** — original photo vs. Grad-CAM heatmap overlay, side by side (stacked on mobile), with a brief one-line caption explaining what the heatmap shows ("Highlighted areas show what the model focused on").
7. **Advisory card** — cause, treatment, and prevention, clearly separated (not one wall of text). If a severity-scaled treatment variant applies, show which tier is being displayed.
8. **Weather caveat banner** — only appears when the weather layer returns a note; styled in Amber Pulse, positioned right above or inside the advisory card near the treatment section since that's what it modifies.
9. **Kannada output** — the translated advisory text rendered in Noto Sans Kannada, plus an audio player for the TTS output. This should feel like a primary feature, not an afterthought caption — give it real visual weight.
10. **Voice follow-up Q&A** — a clearly optional, collapsible section below the main result: a microphone/audio input control, a transcript display of what was heard (so the farmer can confirm it understood correctly), the response text, and a response audio player. Style AI-generated responses here in Electric Violet framing (a subtly bordered/tinted response bubble) to visually distinguish "the model said this" from the fixed, source-verified advisory content above it — this is also an honest design choice: it shows the user which content is generative vs. verified.
11. **Trust/how-it-works footer** — a short, plain-language note that the treatment advice comes from a verified agricultural source database, not AI-generated guesswork, and that the voice assistant is limited to the diagnosis context. This is a genuine trust-building feature, not just filler — make it visible, not buried in tiny print.
12. **Empty and error states** — before any image is uploaded, the empty state should invite action ("Upload a leaf photo to begin" with a simple leaf icon), not just show a blank box. If the model fails, if weather data is unavailable, or if the voice assistant can't answer, each should have a plain-language, specific message in the interface's own voice — never a raw exception, never "Something went wrong."

---

## 5. Copy Guidance

Write from the farmer/end-user's perspective, in plain language — never system-internal terms ("inference," "API call," "confidence score" is borderline acceptable but "model output" is not). Buttons say exactly what they do ("Analyze photo," not "Submit"). Keep tone calm and direct — this is a diagnostic tool someone may be relying on for a real crop decision, not a flashy consumer app; the futuristic visual identity should not tip into a jokey or hyperbolic voice. No exclamation marks in system copy, no "Oops!" in error states.

---

## 6. Streamlit Implementation Notes (Path A)

- Use `.streamlit/config.toml` to set the base theme colors (background, secondary background, text, primary color) to the token values in Section 2 — this handles native widget theming (buttons, sliders, selectboxes) so they don't clash with custom CSS elsewhere.
- Inject the two Google Fonts and any custom component styling via `st.markdown("<style>...</style>", unsafe_allow_html=True)` near the top of `app.py`. Import fonts via a `<link>` to Google Fonts in that same style block.
- Structure the layout using `st.columns()` for the side-by-side image comparison and the district/upload input row, and `st.container()` / `st.expander()` for the collapsible voice Q&A section.
- The scan animation (Section 3) can be a small CSS `@keyframes` sweep applied to a positioned overlay div shown only while `st.spinner()` or a custom processing flag is active — don't rely on Streamlit's default spinner alone, style over/replace it.
- Test the final look on both a wide desktop window (for the stage presentation laptop) and a narrow mobile width (in case judges scan a QR code to try it on their own phones) — confirm `st.columns()` sections stack sensibly on narrow viewports rather than squeezing.
- Respect reduced-motion: wrap the scan animation in a `prefers-reduced-motion` media query fallback (a static highlight instead of the sweep) — small effort, real accessibility floor.

---

## 7. Process Instructions for Antigravity

Before writing any code:
1. Restate this brief's color/type/layout tokens back in your own words to confirm understanding.
2. Sketch (in a short written plan, not code yet) how the hero section, the scan animation, and the advisory card will look concretely, using this project's real copy and content — not lorem ipsum or generic placeholder text.
3. Check your plan against Section 1's "avoid these clichés" list — if any part of your plan matches one of them, revise it and note what you changed and why.
4. Only then write the actual CSS/Streamlit code, building the full feature inventory in Section 4 — do not skip any item without flagging it back to me.
5. Take a screenshot/render of the result if your environment supports it and self-critique against this brief before calling it done — specifically check: does every color use follow the semantic rule in Section 2, is there exactly one animated moment (not several), and does every backend feature from our existing pipeline have a corresponding, clearly visible UI element?

Report back with what you built, a screenshot if possible, and anything from Section 4's feature list you weren't able to fully implement.

---

## 8. If pursuing Path B (separate frontend + API) instead

Only relevant if Path A is complete and there's real time left. In that case: stand up a minimal FastAPI wrapper exposing the existing pipeline (`/analyze` endpoint accepting an image + district, returning prediction/severity/advisory/audio URLs; `/ask` endpoint for the voice Q&A flow) and build the frontend in plain HTML/CSS/JS (or React) against Section 1–5 of this brief, with full freedom to implement true HUD-style animated components the Streamlit CSS approach can't achieve. This is a bigger, separate task — do not attempt to do both paths at once, and do not start this until Path A is confirmed working end-to-end as a fallback.
