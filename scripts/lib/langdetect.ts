/**
 * Language detection for assistant turns. Primary: franc-min (ISO 639-3).
 * Fallback for short/undetermined text: Cyrillic-character ratio heuristic.
 * Returns a 2-letter code constrained to the dataset's languages.
 */

import { franc } from "franc-min";
import type { Lang } from "./types.js";

const ISO3_TO_2: Record<string, Lang> = { eng: "en", rus: "ru" };

function cyrillicRatio(text: string): number {
  const letters = text.replace(/[^a-zа-яё]/giu, "");
  if (letters.length === 0) return 0;
  const cyr = letters.replace(/[^а-яё]/giu, "").length;
  return cyr / letters.length;
}

export function detectLang(text: string): Lang {
  const iso3 = franc(text, { minLength: 1 });
  const mapped = ISO3_TO_2[iso3];
  if (mapped) return mapped;
  // Undetermined or unsupported language → fall back to script heuristic.
  return cyrillicRatio(text) >= 0.3 ? "ru" : "en";
}
