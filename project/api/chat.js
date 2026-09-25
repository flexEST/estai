import { GoogleGenAI } from "@google/genai";
import { createClient } from "@supabase/supabase-js";

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY
});

// Server-side Supabase client using the SERVICE ROLE key. This is what lets the
// daily-message count/insert be trusted: the old client-side check queried
// Supabase directly with the public anon key and disabled the UI in JS, which a
// user could simply skip (open devtools, call fetch() themselves, edit
// localStorage, etc). Now the browser can never see a count that lets more than
// MAX_DAILY_MESSAGES turns through, because the gate lives here.
//
// Add these two in the Vercel dashboard -> Project -> Settings -> Environment
// Variables (redeploy after adding them):
//   SUPABASE_URL
//   SUPABASE_SERVICE_ROLE_KEY   (Project Settings -> API -> service_role key in Supabase — NOT the anon key)
//
// Also run `npm install @supabase/supabase-js` in the function's project if it
// isn't already a dependency.
const supabase = (process.env.SUPABASE_URL && process.env.SUPABASE_SERVICE_ROLE_KEY)
  ? createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY)
  : null;

// ===== LIMITS =====
// Vercel Serverless Functions enforce a hard 4.5 MB request-body limit that cannot
// be raised — it's a fixed platform limit, not something bodyParser config can change
// (confirmed on Vercel's own "how to bypass the 4.5MB body size limit" article).
// Base64 inflates a file by ~33%, and the JSON envelope (chat history + system prompt)
// adds a bit more on top — so the raw file has to stay well under 4.5 MB for the
// encoded request to actually arrive. Keep this formula identical to the one in
// ai.html (client side), or a file that passes there could still get rejected here.
const VERCEL_BODY_LIMIT_BYTES = 4.5 * 1024 * 1024;
const JSON_OVERHEAD_BYTES = 200 * 1024;
const MAX_ATTACHMENT_BYTES = Math.floor(((VERCEL_BODY_LIMIT_BYTES - JSON_OVERHEAD_BYTES) * 3) / 4);

const MAX_TEXT_LENGTH = 200;                    // matches the client's per-message char limit
const MAX_MESSAGE_STORE_LEN = MAX_TEXT_LENGTH * 4;  // small buffer for older turns
const MAX_HISTORY_MESSAGES = 30;
const MAX_SYSTEM_PROMPT_LEN = 20000;

// Daily per-user message cap. This used to live in ai.html (MAX_DAILY_MESSAGES = 5,
// checked and enforced client-side). It's enforced here now instead, and raised to 10.
const MAX_DAILY_MESSAGES = 10;

// Fixed timezone for the "daily" boundary, in minutes to ADD to UTC (Azerbaijan
// is UTC+4 year-round — AZT has not observed DST since 2016, so this never needs
// to change with the seasons). This used to come from the client as `tzOffset`,
// which meant a user could just send a different offset on every request to keep
// shifting where "today" starts and blow through the cap. The reset time is now
// fixed and computed purely from the server's own clock — nothing in the request
// body or query string can move it.
const APP_TIMEZONE_OFFSET_MINUTES = 240;

// Actual message text is NEVER stored in the database — only this short default
// text is saved (rows are still used for the daily limit count). This used to be
// written by the client's recordMessageToServer(); that logic now lives here too.
const DB_DEFAULT_MESSAGE = "message_sent";

// This is the only set of mimeTypes the client will ever actually send: images go
// through as-is, and every audio attachment (recorded or picked) is re-encoded to
// audio/wav in the browser before upload, because Gemini's officially supported
// audio types are wav/mp3/aiff/aac/ogg/flac — not the webm/mp4 a MediaRecorder
// produces. Keeping the allow-list this tight also shrinks the attack surface.
const ALLOWED_MIME_TYPES = new Set([
  "image/jpeg", "image/png", "image/webp",
  "application/pdf",
  "text/plain",
  "audio/wav"
]);

// Magic-byte signatures so a file can't be smuggled through by relabeling its
// declared mimeType. text/plain has no reliable signature; size + allow-list still apply to it.
const MAGIC_BYTES = {
  "image/png": [0x89, 0x50, 0x4e, 0x47],
  "image/jpeg": [0xff, 0xd8, 0xff],
  "application/pdf": [0x25, 0x50, 0x44, 0x46]
};

function checkMagicBytes(mimeType, buffer) {
  if (mimeType === "audio/wav" || mimeType === "image/webp") {
    if (buffer.length < 12) return false;
    const riff = buffer.toString("ascii", 0, 4);
    const kind = buffer.toString("ascii", 8, 12);
    if (riff !== "RIFF") return false;
    return mimeType === "audio/wav" ? kind === "WAVE" : kind === "WEBP";
  }
  const sig = MAGIC_BYTES[mimeType];
  if (!sig) return true;
  if (buffer.length < sig.length) return false;
  for (let i = 0; i < sig.length; i++) {
    if (buffer[i] !== sig[i]) return false;
  }
  return true;
}

function validateAttachment(attachment) {
  if (!attachment || typeof attachment !== "object") {
    return { ok: false, status: 400, error: "invalid attachment" };
  }
  const { data, mimeType, name } = attachment;

  if (typeof mimeType !== "string" || !ALLOWED_MIME_TYPES.has(mimeType)) {
    return { ok: false, status: 415, error: "unsupported file type" };
  }
  if (typeof data !== "string" || !data || !/^[A-Za-z0-9+/]+={0,2}$/.test(data)) {
    return { ok: false, status: 400, error: "malformed file data" };
  }

  // Cheap length-based check before touching the buffer, then confirm on the real bytes.
  const approxBytes = Math.floor((data.length * 3) / 4);
  if (approxBytes > MAX_ATTACHMENT_BYTES) {
    return { ok: false, status: 413, error: "file too large" };
  }

  const buffer = Buffer.from(data, "base64");
  if (buffer.length > MAX_ATTACHMENT_BYTES) {
    return { ok: false, status: 413, error: "file too large" };
  }
  if (!checkMagicBytes(mimeType, buffer)) {
    return { ok: false, status: 415, error: "file content does not match its declared type" };
  }

  const safeName = (typeof name === "string" ? name : "file").replace(/[^\w.\- ]/g, "_").slice(0, 120);
  return { ok: true, buffer, mimeType, name: safeName };
}

// ===== DAILY LIMIT HELPERS =====

// Always computed from APP_TIMEZONE_OFFSET_MINUTES and the server's own clock —
// no request input feeds into this at all, so there's nothing for a client to
// spoof to move the reset point.
function computeLocalMidnightISO() {
  const nowUtcMs = Date.now();
  const localMs = nowUtcMs + APP_TIMEZONE_OFFSET_MINUTES * 60000;
  const local = new Date(localMs);
  const y = local.getUTCFullYear();
  const m = local.getUTCMonth();
  const d = local.getUTCDate();
  const localMidnightMs = Date.UTC(y, m, d, 0, 0, 0) - APP_TIMEZONE_OFFSET_MINUTES * 60000;
  return new Date(localMidnightMs).toISOString();
}

async function countMessagesToday(userId) {
  if (!supabase) throw new Error("Supabase not configured");
  const sinceISO = computeLocalMidnightISO();
  const { count, error } = await supabase
    .from("chat_messages")
    .select("*", { count: "exact", head: true })
    .eq("user_id", userId)
    .gte("created_at", sinceISO);
  if (error) throw error;
  return count || 0;
}

async function recordUsage(userId) {
  if (!supabase) throw new Error("Supabase not configured");
  const messageId = "msg_" + Date.now() + "_" + Math.random().toString(36).substr(2, 5);
  const { error } = await supabase
    .from("chat_messages")
    .insert([{ id: messageId, user_id: userId, message_text: DB_DEFAULT_MESSAGE }]);
  if (error) throw error;
}

function isValidUserId(userId) {
  return typeof userId === "string" && userId.length > 0 && userId.length <= 128;
}

// ===== CORS =====
// Optional origin allow-list. Set ALLOWED_ORIGINS in the Vercel dashboard
// (Project -> Settings -> Environment Variables) as a comma-separated list,
// e.g. https://flexest.github.io — no code changes or CLI needed.
// Leaving it unset keeps today's open behaviour so nothing breaks.
function isAllowedOrigin(origin) {
  if (!origin) return true; // non-browser callers (curl, server-to-server) send no Origin header
  const allowed = (process.env.ALLOWED_ORIGINS || "")
    .split(",")
    .map(s => s.trim())
    .filter(Boolean);
  if (allowed.length === 0) return true;
  return allowed.indexOf(origin) !== -1;
}

export default async function handler(req, res) {
  const origin = req.headers.origin;
  const originOk = isAllowedOrigin(origin);

  if (originOk) {
    res.setHeader("Access-Control-Allow-Origin", origin || "*");
  }
  res.setHeader("Vary", "Origin");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  res.setHeader("X-Content-Type-Options", "nosniff");

  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  if (!originOk) {
    return res.status(403).json({ error: "Origin not allowed" });
  }

  // ----- GET: display-only "how many messages do I have left today" check -----
  // ai.html calls this on load (and after each send) just to show a number to the
  // user — it is NOT what enforces the limit. The real gate is in the POST branch
  // below, so a user calling this directly can't get themselves more messages.
  if (req.method === "GET") {
    const userId = req.query && req.query.userId;

    if (!isValidUserId(userId)) {
      return res.status(400).json({ error: "invalid userId" });
    }
    if (!supabase) {
      return res.status(500).json({ error: "Missing Supabase configuration" });
    }
    try {
      const count = await countMessagesToday(userId);
      const remaining = Math.max(0, MAX_DAILY_MESSAGES - count);
      res.setHeader("X-Daily-Remaining", String(remaining));
      res.setHeader("X-Daily-Max", String(MAX_DAILY_MESSAGES));
      return res.status(200).json({ remaining, max: MAX_DAILY_MESSAGES });
    } catch (err) {
      console.error("Daily limit check error:", err && err.message ? err.message : err);
      return res.status(500).json({ error: "limit check failed" });
    }
  }

  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  if (!process.env.GEMINI_API_KEY) {
    return res.status(500).json({ error: "Missing Gemini API key" });
  }
  if (!supabase) {
    return res.status(500).json({ error: "Missing Supabase configuration" });
  }

  const { messages, system, stream, attachment, userId } = req.body || {};

  if (!isValidUserId(userId)) {
    return res.status(400).json({ error: "invalid userId" });
  }
  if (!messages || !Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: "messages array required" });
  }
  if (messages.length > MAX_HISTORY_MESSAGES) {
    return res.status(400).json({ error: "too many messages" });
  }
  for (const m of messages) {
    if (!m || typeof m.content !== "string" || typeof m.role !== "string") {
      return res.status(400).json({ error: "invalid message format" });
    }
    if (m.content.length > MAX_MESSAGE_STORE_LEN) {
      return res.status(400).json({ error: "message too long" });
    }
  }
  if (typeof system !== "string" || system.length > MAX_SYSTEM_PROMPT_LEN) {
    return res.status(400).json({ error: "invalid system prompt" });
  }

  // ----- Server-side daily limit — the real gate. Checked before we spend a
  // Gemini call or touch the attachment, so a limited-out user costs us nothing. -----
  let usedToday;
  try {
    usedToday = await countMessagesToday(userId);
  } catch (err) {
    console.error("Daily limit check error:", err && err.message ? err.message : err);
    return res.status(500).json({ error: "limit check failed" });
  }

  if (usedToday >= MAX_DAILY_MESSAGES) {
    res.setHeader("X-Daily-Remaining", "0");
    res.setHeader("X-Daily-Max", String(MAX_DAILY_MESSAGES));
    return res.status(429).json({
      error: "Günlük mesaj limitinə çatmısınız",
      remaining: 0,
      max: MAX_DAILY_MESSAGES
    });
  }

  let validatedAttachment = null;
  if (attachment) {
    const result = validateAttachment(attachment);
    if (!result.ok) {
      return res.status(result.status).json({ error: result.error });
    }
    validatedAttachment = result;
  }

  // Record usage before calling the model — same as the old client-side
  // recordMessageToServer() — so an attempted turn counts even if the Gemini
  // call itself fails afterward.
  try {
    await recordUsage(userId);
  } catch (err) {
    console.error("Usage record error:", err && err.message ? err.message : err);
    return res.status(500).json({ error: "could not record usage" });
  }

  const remaining = Math.max(0, MAX_DAILY_MESSAGES - (usedToday + 1));
  res.setHeader("X-Daily-Remaining", String(remaining));
  res.setHeader("X-Daily-Max", String(MAX_DAILY_MESSAGES));

  try {
    const formattedContents = messages.map(msg => ({
      role: msg.role === "assistant" ? "model" : "user",
      parts: [{ text: msg.content }]
    }));

    // Attach the (already-validated) file to the last user turn only — that's the
    // one the person just sent; older turns in history never carried a real file.
    if (validatedAttachment) {
      const lastPart = formattedContents[formattedContents.length - 1];
      if (lastPart && lastPart.role === "user") {
        lastPart.parts.push({
          inlineData: {
            mimeType: validatedAttachment.mimeType,
            data: validatedAttachment.buffer.toString("base64")
          }
        });
      }
    }

    const config = {
      maxOutputTokens: 1000,
      temperature: 0.7,
      systemInstruction: system || "You are a helpful assistant."
    };

    // If client requested a stream, use generateContentStream
    if (stream) {
      res.setHeader("Content-Type", "text/plain; charset=utf-8");
      res.setHeader("Transfer-Encoding", "chunked");

      const responseStream = await ai.models.generateContentStream({
        model: "gemini-3.5-flash-lite", // Updated model
        contents: formattedContents,
        config: config
      });

      for await (const chunk of responseStream) {
        if (chunk.text) {
          res.write(chunk.text);
        }
      }
      return res.end();
    }

    // Fallback standard JSON response if stream is not requested
    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash-lite", // Updated model
      contents: formattedContents,
      config: config
    });

    return res.status(200).json({ content: response.text });

  } catch (err) {
    // Never log request bodies or attachment bytes — only a short message.
    console.error("Gemini API error:", err && err.message ? err.message : err);
    if (!res.headersSent) {
      res.status(500).json({ error: "⚠️ AI error, try again later" });
    } else {
      res.end();
    }
  }
}
