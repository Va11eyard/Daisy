/**
 * Daisy Therapy Chatbot — API Route
 * Location: src/app/api/chat/route.ts
 *
 * Features:
 *   - Locale-aware error messages (EN / RU / KK)
 *   - Deployment header routing by locale (azureml-model-deployment)
 *   - Configurable timeout with graceful degradation
 *   - Response streaming support (for Qwen3 migration)
 *   - Structured request/response logging
 *   - Full TypeScript types, no stubs, no TODOs
 */

import { NextRequest, NextResponse } from "next/server";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

interface ChatRequestBody {
  messages: ChatMessage[];
  locale?: string;
  stream?: boolean;
}

interface ChatResponse {
  reply: string;
  metadata?: {
    deployment?: string;
    latency_ms?: number;
    locale?: string;
    model?: string;
  };
}

interface ErrorResponse {
  error: string;
  detail?: string;
  locale?: string;
  deployment_attempted?: string;
  timestamp: string;
}

interface LogEntry {
  timestamp: string;
  level: "info" | "warn" | "error";
  event: string;
  locale?: string;
  deployment?: string;
  latency_ms?: number;
  status_code?: number;
  error?: string;
  body_preview?: string;
}

// ---------------------------------------------------------------------------
// Configuration & Constants
// ---------------------------------------------------------------------------

const SUPPORTED_LOCALES = ["en", "ru", "kk"] as const;
type SupportedLocale = (typeof SUPPORTED_LOCALES)[number];

const DEFAULT_LOCALE: SupportedLocale = "en";
const DEFAULT_TIMEOUT_MS = 30_000;
const MAX_RESPONSE_SIZE_BYTES = 2 * 1024 * 1024; // 2 MB

const ERROR_MESSAGES: Record<
  SupportedLocale,
  { user: string; detail: string }
> = {
  en: {
    user: "Something went wrong. Please try again in a moment.",
    detail:
      "We encountered an issue connecting to the therapy service. The team has been notified.",
  },
  ru: {
    user: "Извини, произошла ошибка. Попробуй ещё раз через минуту.",
    detail:
      "Проблема соединения с сервисом. Мы уже разбираемся и скоро всё починим.",
  },
  kk: {
    user: "Қате орын алды. Бір минуттан кейін қайта байқап көріңіз.",
    detail:
      "Қызметке қосылуда мәселе туындады. Біз тексеріп жатырмыз, жақында түзетеміз.",
  },
};

const AML_HEADERS = {
  "Content-Type": "application/json",
  Accept: "application/json",
};

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

function log(entry: LogEntry): void {
  const line = `[${entry.timestamp}] ${entry.level.toUpperCase()} | ${entry.event}`;
  const details: string[] = [];
  if (entry.locale) details.push(`locale=${entry.locale}`);
  if (entry.deployment) details.push(`deployment=${entry.deployment}`);
  if (entry.latency_ms) details.push(`latency=${entry.latency_ms}ms`);
  if (entry.status_code) details.push(`status=${entry.status_code}`);
  if (entry.error) details.push(`error="${entry.error}"`);
  if (entry.body_preview)
    details.push(`body_preview=${entry.body_preview}`);

  const detailStr = details.length > 0 ? ` | ${details.join(" | ")}` : "";

  if (entry.level === "error") {
    // eslint-disable-next-line no-console
    console.error(`${line}${detailStr}`);
  } else if (entry.level === "warn") {
    // eslint-disable-next-line no-console
    console.warn(`${line}${detailStr}`);
  } else {
    // eslint-disable-next-line no-console
    console.log(`${line}${detailStr}`);
  }
}

// ---------------------------------------------------------------------------
// Locale Detection
// ---------------------------------------------------------------------------

/**
 * Resolve locale from request body or NEXT_LOCALE cookie.
 * Falls back to "en" if unsupported.
 */
function resolveLocale(
  body: ChatRequestBody,
  request: NextRequest
): SupportedLocale {
  // 1. Check body.locale first
  if (body.locale) {
    const normalized = normalizeLocale(body.locale);
    if (normalized) return normalized;
  }

  // 2. Check NEXT_LOCALE cookie
  const cookieLocale = request.cookies.get("NEXT_LOCALE")?.value;
  if (cookieLocale) {
    const normalized = normalizeLocale(cookieLocale);
    if (normalized) return normalized;
  }

  // 3. Check Accept-Language header
  const acceptLang = request.headers.get("accept-language");
  if (acceptLang) {
    const headerLocale = acceptLang.split(",")[0]?.split("-")[0]?.trim();
    const normalized = normalizeLocale(headerLocale);
    if (normalized) return normalized;
  }

  // 4. Default
  return DEFAULT_LOCALE;
}

function normalizeLocale(raw: string): SupportedLocale | undefined {
  const lower = raw.toLowerCase().trim();
  // Handle "ru-RU", "kk-KZ", "en-US" etc.
  const base = lower.split("-")[0];
  if (base === "en" || base === "ru" || base === "kk") {
    return base as SupportedLocale;
  }
  return undefined;
}

// ---------------------------------------------------------------------------
// Deployment Routing
// ---------------------------------------------------------------------------

/**
 * Resolve Azure ML deployment name based on locale.
 * Returns undefined for default (EN) — lets the endpoint decide.
 */
function resolveDeployment(locale: SupportedLocale): string | undefined {
  if (locale === "ru") return process.env.AML_DEPLOYMENT_NAME_RU ?? undefined;
  if (locale === "kk") return process.env.AML_DEPLOYMENT_NAME_KK ?? undefined;
  return undefined;
}

// ---------------------------------------------------------------------------
// Timeout Utility
// ---------------------------------------------------------------------------

/**
 * Wrap a promise in a timeout. Rejects with a TimeoutError if exceeded.
 */
function withTimeout<T>(
  promise: Promise<T>,
  ms: number,
  context: string
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new TimeoutError(`Timeout after ${ms}ms: ${context}`));
    }, ms);

    promise
      .then((value) => {
        clearTimeout(timer);
        resolve(value);
      })
      .catch((err) => {
        clearTimeout(timer);
        reject(err);
      });
  });
}

class TimeoutError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "TimeoutError";
  }
}

// ---------------------------------------------------------------------------
// Response Builders
// ---------------------------------------------------------------------------

function buildErrorResponse(
  locale: SupportedLocale,
  deploymentAttempted?: string,
  internalError?: string
): NextResponse<ErrorResponse> {
  const messages = ERROR_MESSAGES[locale] ?? ERROR_MESSAGES[DEFAULT_LOCALE];
  const body: ErrorResponse = {
    error: messages.user,
    detail: messages.detail,
    locale,
    ...(deploymentAttempted && { deployment_attempted: deploymentAttempted }),
    timestamp: new Date().toISOString(),
  };

  // Log internal detail for observability
  if (internalError) {
    // eslint-disable-next-line no-console
    console.error(`[ERROR_INTERNAL] ${internalError}`);
  }

  return NextResponse.json(body, { status: 503 });
}

// ---------------------------------------------------------------------------
// Streaming Response (Qwen3 migration support)
// ---------------------------------------------------------------------------

/**
 * Convert an AML SSE stream to a Next.js streaming response.
 */
function createStreamingResponse(
  amlResponse: Response,
  locale: SupportedLocale,
  deploymentName?: string
): Response {
  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();
  const encoder = new TextEncoder();
  const startTime = Date.now();

  (async () => {
    try {
      const reader = amlResponse.body?.getReader();
      if (!reader) {
        writer.close();
        return;
      }

      // Write metadata prefix
      const meta = JSON.stringify({
        meta: true,
        locale,
        deployment: deploymentName ?? "default",
        timestamp: new Date().toISOString(),
      });
      writer.write(encoder.encode(`data: ${meta}\n\n`));

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        writer.write(value);
      }

      const latency = Date.now() - startTime;
      log({
        timestamp: new Date().toISOString(),
        level: "info",
        event: "stream_complete",
        locale,
        deployment: deploymentName,
        latency_ms: latency,
      });
    } catch (err) {
      const error =
        err instanceof Error ? err.message : "Unknown stream error";
      log({
        timestamp: new Date().toISOString(),
        level: "error",
        event: "stream_error",
        locale,
        deployment: deploymentName,
        error,
      });
    } finally {
      writer.close();
    }
  })();

  return new Response(readable, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}

// ---------------------------------------------------------------------------
// AML Client
// ---------------------------------------------------------------------------

interface AmlCallOptions {
  body: ChatRequestBody;
  deploymentName?: string;
  timeoutMs: number;
}

interface AmlCallResult {
  responseText: string;
  statusCode: number;
  latencyMs: number;
}

/**
 * Call the Azure ML endpoint with deployment routing and timeout.
 */
async function callAmlEndpoint(
  options: AmlCallOptions
): Promise<AmlCallResult> {
  const { body, deploymentName, timeoutMs } = options;

  const apiUrl = process.env.AI_API_URL;
  const apiKey = process.env.AI_API_KEY;

  if (!apiUrl) {
    throw new Error("Missing environment variable: AI_API_URL");
  }
  if (!apiKey) {
    throw new Error("Missing environment variable: AI_API_KEY");
  }

  const headers: Record<string, string> = {
    ...AML_HEADERS,
    Authorization: `Bearer ${apiKey}`,
  };

  // Route to locale-specific deployment
  if (deploymentName) {
    headers["azureml-model-deployment"] = deploymentName;
  }

  const startTime = Date.now();

  const fetchPromise = fetch(apiUrl, {
    method: "POST",
    headers,
    body: JSON.stringify({
      messages: body.messages,
      locale: body.locale,
      stream: body.stream ?? false,
    }),
  });

  const response = await withTimeout(
    fetchPromise,
    timeoutMs,
    `AML endpoint ${deploymentName ?? "default"}`
  );

  const latencyMs = Date.now() - startTime;

  if (!response.ok) {
    const errorBody = await response.text();
    throw new AmlError(
      `AML returned ${response.status}: ${errorBody}`,
      response.status,
      latencyMs
    );
  }

  // If streaming was requested, return a sentinel — caller must handle
  if (body.stream) {
    throw new StreamingRequiredError(response, latencyMs);
  }

  const responseText = await response.text();

  // Guard against oversized responses
  if (responseText.length > MAX_RESPONSE_SIZE_BYTES) {
    throw new Error(
      `Response too large: ${responseText.length} bytes (max ${MAX_RESPONSE_SIZE_BYTES})`
    );
  }

  return {
    responseText,
    statusCode: response.status,
    latencyMs,
  };
}

class AmlError extends Error {
  statusCode: number;
  latencyMs: number;

  constructor(message: string, statusCode: number, latencyMs: number) {
    super(message);
    this.name = "AmlError";
    this.statusCode = statusCode;
    this.latencyMs = latencyMs;
  }
}

class StreamingRequiredError extends Error {
  response: Response;
  latencyMs: number;

  constructor(response: Response, latencyMs: number) {
    super("Streaming response required — handled separately");
    this.name = "StreamingRequiredError";
    this.response = response;
    this.latencyMs = latencyMs;
  }
}

// ---------------------------------------------------------------------------
// Response Parsing
// ---------------------------------------------------------------------------

/**
 * Parse the AML response text into a ChatResponse.
 */
function parseAmlResponse(
  responseText: string,
  locale: SupportedLocale,
  deploymentName?: string,
  latencyMs?: number
): ChatResponse {
  let parsed: unknown;

  try {
    parsed = JSON.parse(responseText);
  } catch {
    // If not valid JSON, treat the raw text as the reply
    return {
      reply: responseText.trim(),
      metadata: {
        deployment: deploymentName ?? "default",
        locale,
        latency_ms: latencyMs,
      },
    };
  }

  // Handle various AML response shapes
  if (typeof parsed === "string") {
    return {
      reply: parsed.trim(),
      metadata: { deployment: deploymentName, locale, latency_ms: latencyMs },
    };
  }

  if (parsed && typeof parsed === "object") {
    const obj = parsed as Record<string, unknown>;

    // Shape 1: { reply: "..." }
    if (typeof obj.reply === "string") {
      return {
        reply: obj.reply.trim(),
        metadata: {
          ...(obj.metadata && typeof obj.metadata === "object"
            ? (obj.metadata as Record<string, unknown>)
            : {}),
          deployment: deploymentName,
          locale,
          latency_ms: latencyMs,
        } as ChatResponse["metadata"],
      };
    }

    // Shape 2: { choices: [{ message: { content: "..." } }] } (OpenAI-compatible)
    if (Array.isArray(obj.choices) && obj.choices.length > 0) {
      const choice = obj.choices[0] as Record<string, unknown>;
      const message = choice?.message as Record<string, unknown> | undefined;
      if (typeof message?.content === "string") {
        return {
          reply: message.content.trim(),
          metadata: {
            deployment: deploymentName,
            locale,
            latency_ms: latencyMs,
            model: typeof obj.model === "string" ? obj.model : undefined,
          },
        };
      }
      if (typeof choice?.text === "string") {
        return {
          reply: choice.text.trim(),
          metadata: {
            deployment: deploymentName,
            locale,
            latency_ms: latencyMs,
          },
        };
      }
    }

    // Shape 3: { output: "..." } or { result: "..." }
    const altField =
      typeof obj.output === "string"
        ? obj.output
        : typeof obj.result === "string"
          ? obj.result
          : undefined;

    if (altField) {
      return {
        reply: altField.trim(),
        metadata: { deployment: deploymentName, locale, latency_ms: latencyMs },
      };
    }
  }

  // Fallback: JSON-stringify the whole thing as reply
  return {
    reply: responseText.trim(),
    metadata: { deployment: deploymentName, locale, latency_ms: latencyMs },
  };
}

// ---------------------------------------------------------------------------
// Main Handler
// ---------------------------------------------------------------------------

export async function POST(
  request: NextRequest
): Promise<NextResponse<ChatResponse | ErrorResponse> | Response> {
  const requestStart = Date.now();
  const now = new Date().toISOString();

  // -----------------------------------------------------------------
  // 1. Parse body
  // -----------------------------------------------------------------
  let body: ChatRequestBody;
  try {
    body = await request.json();
  } catch {
    log({
      timestamp: now,
      level: "warn",
      event: "invalid_body",
      error: "Failed to parse JSON body",
    });
    return NextResponse.json(
      {
        error: "Invalid request body. Expected JSON.",
        detail: "The request body could not be parsed as valid JSON.",
        locale: DEFAULT_LOCALE,
        timestamp: new Date().toISOString(),
      } as ErrorResponse,
      { status: 400 }
    );
  }

  // Validate messages
  if (!body.messages || !Array.isArray(body.messages) || body.messages.length === 0) {
    log({
      timestamp: now,
      level: "warn",
      event: "missing_messages",
      error: "No messages provided in request body",
    });
    return NextResponse.json(
      {
        error: "Missing 'messages' field.",
        detail: "Please provide a non-empty 'messages' array.",
        locale: body.locale ?? DEFAULT_LOCALE,
        timestamp: new Date().toISOString(),
      } as ErrorResponse,
      { status: 400 }
    );
  }

  // -----------------------------------------------------------------
  // 2. Detect locale
  // -----------------------------------------------------------------
  const locale = resolveLocale(body, request);
  body.locale = locale; // Forward locale to AML

  // -----------------------------------------------------------------
  // 3. Resolve deployment
  // -----------------------------------------------------------------
  const deploymentName = resolveDeployment(locale);

  const bodyPreview = JSON.stringify(body.messages.slice(-2)).slice(0, 200);
  log({
    timestamp: now,
    level: "info",
    event: "request_start",
    locale,
    deployment: deploymentName,
    body_preview: bodyPreview,
  });

  // -----------------------------------------------------------------
  // 4. Call AML endpoint
  // -----------------------------------------------------------------
  const timeoutMs =
    Number(process.env.DAISY_REQUEST_TIMEOUT_MS) || DEFAULT_TIMEOUT_MS;

  try {
    const result = await callAmlEndpoint({
      body,
      deploymentName,
      timeoutMs,
    });

    // -----------------------------------------------------------------
    // 5. Parse and return
    // -----------------------------------------------------------------
    const chatResponse = parseAmlResponse(
      result.responseText,
      locale,
      deploymentName,
      result.latencyMs
    );

    log({
      timestamp: new Date().toISOString(),
      level: "info",
      event: "request_success",
      locale,
      deployment: deploymentName,
      latency_ms: result.latencyMs,
      status_code: result.statusCode,
    });

    return NextResponse.json(chatResponse, { status: 200 });
  } catch (err) {
    const totalLatency = Date.now() - requestStart;

    // Handle streaming required
    if (err instanceof StreamingRequiredError) {
      log({
        timestamp: new Date().toISOString(),
        level: "info",
        event: "streaming_response",
        locale,
        deployment: deploymentName,
        latency_ms: err.latencyMs,
      });
      return createStreamingResponse(err.response, locale, deploymentName);
    }

    // Handle timeout
    if (err instanceof TimeoutError) {
      log({
        timestamp: new Date().toISOString(),
        level: "error",
        event: "request_timeout",
        locale,
        deployment: deploymentName,
        latency_ms: totalLatency,
        error: err.message,
      });
      return buildErrorResponse(locale, deploymentName ?? undefined, err.message);
    }

    // Handle AML error
    if (err instanceof AmlError) {
      log({
        timestamp: new Date().toISOString(),
        level: "error",
        event: "aml_error",
        locale,
        deployment: deploymentName,
        latency_ms: err.latencyMs,
        status_code: err.statusCode,
        error: err.message,
      });
      return buildErrorResponse(locale, deploymentName ?? undefined, err.message);
    }

    // Generic error
    const errorMsg = err instanceof Error ? err.message : "Unknown error";
    log({
      timestamp: new Date().toISOString(),
      level: "error",
      event: "request_error",
      locale,
      deployment: deploymentName,
      latency_ms: totalLatency,
      error: errorMsg,
    });
    return buildErrorResponse(locale, deploymentName ?? undefined, errorMsg);
  }
}

// ---------------------------------------------------------------------------
// Health Check (GET)
// ---------------------------------------------------------------------------

export async function GET(_request: NextRequest): Promise<NextResponse> {
  const health = {
    status: "ok",
    service: "daisy-chat-api",
    version: "2026.07-stg1",
    timestamp: new Date().toISOString(),
    config: {
      url_configured: !!process.env.AI_API_URL,
      key_configured: !!process.env.AI_API_KEY,
      ru_deployment_configured: !!process.env.AML_DEPLOYMENT_NAME_RU,
      kk_deployment_configured: !!process.env.AML_DEPLOYMENT_NAME_KK,
      default_timeout_ms: DEFAULT_TIMEOUT_MS,
    },
  };

  // Deep health: try a lightweight probe to AML
  let amlHealthy = false;
  const apiUrl = process.env.AI_API_URL;
  const apiKey = process.env.AI_API_KEY;

  if (apiUrl && apiKey) {
    try {
      const probe = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          messages: [{ role: "user", content: "hi" }],
          max_tokens: 5,
        }),
        // @ts-expect-error — Next.js extends fetch with timeout option
        signal: AbortSignal.timeout(5_000),
      });
      amlHealthy = probe.ok || probe.status === 422; // 422 = valid endpoint, bad body
    } catch {
      amlHealthy = false;
    }
  }

  return NextResponse.json(
    { ...health, aml_reachable: amlHealthy },
    { status: amlHealthy ? 200 : 503 }
  );
}
