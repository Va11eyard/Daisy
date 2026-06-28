import { readFileSync } from "node:fs";

const B9 = ["coding","recipe","cooking","video games","walkthrough","sports trivia","homework","finance","legal advice"];
const BANNED = ["I'm here for you!","Take care!","That makes so much sense!","Absolutely!","I hear you!","That's so valid.","I completely understand.","sort of","kind of","Does that make sense?","It sounds like you're going through a lot.","That must be really tough.","generic silver linings","unsolicited reframes during disclosure","That sounds really hard","That must be difficult","Of course!","I hear you","You're not alone"];
const TIER1 = ["suicide","kill myself","self-harm","суицид","убить себя"];
const ALL_EMOTION_LABELS = ["joy","trust","fear","surprise","sadness","disgust","anger","anticipation","anxiety","depression","dissociation","hopelessness","shame","guilt","grief","suicidal_ideation","panic","rumination","exhaustion","numbness","hypervigilance","flooded","restlessness","derealization","relief","groundedness","acceptance","flourishing","none"];
const TONE_EMOTION = {
  crisis: ["hopelessness","panic","flooded","grief"],
  suicidal_ideation: ["suicidal_ideation","hopelessness"],
  ambivalent: ["rumination","guilt","shame","anticipation"],
  guarded: ["hypervigilance","numbness","fear","dissociation"],
};

// Crisis-resource references: language-specific. Referenced exactly once per
// assistant turn in template_tone_only dialogs (crisis / suicidal_ideation),
// near the end before the closing question. RU dialogs MUST use the Russian
// helpline and MUST NOT carry the US-centric 988 reference.
const EN_CRISIS_REF = "988";
const RU_CRISIS_REF = "8-800-2000-122";
const RU_CRISIS_PHRASE = "Телефон доверия в России: 8-800-2000-122 (бесплатно), или местный аналог.";
const countOccurrences = (haystack, needle) => haystack.split(needle).length - 1;

const path = process.argv[2];
const lines = readFileSync(path, "utf8").split(/\r?\n/).filter(l => l.trim().length);
let errors = 0;
const tok = (s) => Math.round(s.trim().split(/\s+/).length * 1.3);

lines.forEach((line, i) => {
  const n = i + 1;
  let o;
  try { o = JSON.parse(line); } catch (e) { console.log(`L${n} JSON parse error: ${e.message}`); errors++; return; }
  const user = o.messages?.find(m => m.role === "user")?.content ?? "";
  const asst = o.messages?.find(m => m.role === "assistant")?.content ?? "";
  const uLow = user.toLowerCase(), aLow = asst.toLowerCase();
  for (const b of B9) if (uLow.includes(b)) { console.log(`L${n} B9 in USER: "${b}"`); errors++; }
  for (const t of TIER1) if (uLow.includes(t)) { console.log(`L${n} tier-1 verbatim in USER: "${t}"`); errors++; }
  for (const b of BANNED) if (aLow.includes(b.toLowerCase())) { console.log(`L${n} banned phrase in ASSISTANT: "${b}"`); errors++; }
  const tk = tok(asst);
  const isOff = o.generation_mode === "off_register";
  if (tk > 200) { console.log(`L${n} too long: ~${tk} tokens`); errors++; }
  if (!isOff && tk < 40) { console.log(`L${n} too short: ~${tk} tokens`); errors++; }
  if (!ALL_EMOTION_LABELS.includes(o.emotion_label)) { console.log(`L${n} bad emotion_label: ${o.emotion_label}`); errors++; }
  const allowed = TONE_EMOTION[o.tone];
  if (allowed && o.emotion_label !== "none" && !allowed.includes(o.emotion_label)) { console.log(`L${n} emotion_label "${o.emotion_label}" not allowed for tone ${o.tone}`); errors++; }
  if (o.source !== "claude-opus-4-8") { console.log(`L${n} bad source: ${o.source}`); errors++; }

  // Language-specific crisis-resource enforcement.
  if (o.lang === "ru" && asst.includes(EN_CRISIS_REF)) {
    console.log(`L${n} US-centric crisis ref "${EN_CRISIS_REF}" present in RU dialog (use ${RU_CRISIS_REF})`); errors++;
  }
  if (o.generation_mode === "template_tone_only") {
    const ref = o.lang === "ru" ? RU_CRISIS_PHRASE : EN_CRISIS_REF;
    const c = countOccurrences(asst, ref);
    if (c !== 1) { console.log(`L${n} expected exactly one ${o.lang} crisis ref "${ref}", found ${c}`); errors++; }
  }
});

console.log(`\n${path}`);
console.log(`lines=${lines.length} errors=${errors}`);
process.exit(errors ? 1 : 0);
