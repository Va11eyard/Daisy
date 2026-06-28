/**
 * B9 off-topic substring guard for *user turns*.
 *
 * Mirrors the matching semantics of `rawInputMatchesOffTopicDomain` in
 * src/layers/rules-engine/index.ts: case-insensitive substring search with a
 * word-boundary guard (a hit only counts when neither neighbouring character
 * is a word character), so "coding" matches "coding bootcamp" but not
 * "decoding". The forbidden list is supplied from generation_plan.json — never
 * hardcoded here.
 */

const WORD_CHAR = /[A-Za-z0-9_]/;

function isWordChar(ch: string | undefined): boolean {
  return ch !== undefined && WORD_CHAR.test(ch);
}

/** Returns the first forbidden phrase found in `userTurn`, or null. */
export function findForbiddenSubstring(
  userTurn: string,
  forbidden: readonly string[],
): string | null {
  const lower = userTurn.toLowerCase();
  for (const phrase of forbidden) {
    const needle = phrase.toLowerCase();
    if (needle.length === 0) continue;
    let from = 0;
    while (from <= lower.length) {
      const i = lower.indexOf(needle, from);
      if (i === -1) break;
      const before = i === 0 ? undefined : userTurn[i - 1];
      const afterIdx = i + needle.length;
      const after = afterIdx >= userTurn.length ? undefined : userTurn[afterIdx];
      if (!isWordChar(before) && !isWordChar(after)) {
        return phrase;
      }
      from = i + 1;
    }
  }
  return null;
}

export function hasForbiddenSubstring(userTurn: string, forbidden: readonly string[]): boolean {
  return findForbiddenSubstring(userTurn, forbidden) !== null;
}
