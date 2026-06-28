/** Token counting via tiktoken cl100k_base (closest public BPE to Qwen2 BBPE). */

import { get_encoding, type Tiktoken } from "tiktoken";

let enc: Tiktoken | null = null;

function encoder(): Tiktoken {
  if (enc === null) enc = get_encoding("cl100k_base");
  return enc;
}

export function countTokens(text: string): number {
  return encoder().encode(text).length;
}

/** Release native tiktoken resources (call once at process end if desired). */
export function freeTokenizer(): void {
  if (enc !== null) {
    enc.free();
    enc = null;
  }
}
