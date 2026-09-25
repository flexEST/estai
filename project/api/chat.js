// code.js
import { GoogleGenAI } from "@google/genai";
import { createClient } from "@supabase/supabase-js";

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY
});

// ===== SUPABASE (server-side) =====
const SUPABASE_URL = process.env.SUPABASE_URL || "https://gtxysgqfuepywqwyciii.supabase.co";
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY; // ← service role key (RLS bypass)
const supabaseAdmin = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY, {
  auth: { persistSession: false, autoRefreshToken: false }
});

// ===== LIMITS =====
const MAX_DAILY_MESSAGES = 5;

const VERCEL_BODY_LIMIT_BYTES = 4.5 * 1024 * 1024;
const JSON_OVERHEAD_BYTES = 200 * 1024;
const MAX_ATTACHMENT_BYTES = Math.floor(((VERCEL_BODY_LIMIT_BYTES - JSON_OVERHEAD_BYTES) * 3) / 4);

const MAX_TEXT_LENGTH = 200;
const MAX_MESSAGE_STORE_LEN = MAX_TEXT_LENGTH * 4;
const MAX_HISTORY_MESSAGES = 30;
const MAX_SYSTEM_PROMPT_LEN = 20000;

const ALLOWED_MIME_TYPES = new Set([
  "image/jpeg", "image/png", "image/webp",
  "application/pdf",
  "text/plain",
  "audio/wav"
]);

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

// ===== CORS =====
function isAllowedOrigin(origin) {
  if (!origin) return true;
  const allowed = (process.env.ALLOWED_ORIGINS || "")
    .split(",")
    .map(s => s.trim())
    .filter(Boolean);
  if (allowed.length === 0) return true;
  return allowed.indexOf(origin) !== -1;
}

// ===== AUTH + LIMIT HELPERS =====
async function getUserIdFromRequest(req) {
  const authHeader = req.headers.authorization || "";
  const token = authHeader.replace(/^Bearer\s+/i, "").trim();
  if (!token) return null;
  try {
    const { data, error } = await supabaseAdmin.auth.getUser(token);
    if (error || !data || !data.user) return null;
    return data.user.id;
  } catch (e) {
    return null;
  }
}

function getClientIp(req) {
  const fwd = req.headers["x-forwarded-for"];
  if (fwd) return String(fwd).split(",")[0].trim();
  return req.socket?.remoteAddress || "unknown";
}

// Returns local midnight in ISO with the client's timezone offset.
// Client sends its offset (minutes) so the daily counter resets at the user's local midnight.
function getLocalMidnightISO(tzOffsetMinutes) {
  const now = new Date();
  const offset = Number.isFinite(tzOffsetMinutes) ? tzOffsetMinutes : 0;
  const local = new Date(now.getTime() + offset * 60 * 1000);
  const y = local.getUTCFullYear();
  const m = String(local.getUTCMonth() + 1).padStart(2, "0");
  const d = String(local.getUTCDate()).padStart(2, "0");
  const sign = offset >= 0 ? "+" : "-";
  const abs = Math.abs(offset);
  const oh = String(Math.floor(abs / 60)).padStart(2, "0");
  const om = String(abs % 60).padStart(2, "0");
  return `${y}-${m}-${d}T00:00:00${sign}${oh}:${om}`;
}

// Count today's messages for a user, then insert a new row (only if under limit).
async function checkAndRecordMessage(userId, tzOffsetMinutes) {
  const since = getLocalMidnightISO(tzOffsetMinutes);

  // 1. count today's rows
  const { count, error: countErr } = await supabaseAdmin
    .from("chat_messages")
    .select("*", { count: "exact", head: true })
    .eq("user_id", userId)
    .gte("created_at", since);

  if (countErr) throw new Error("limit check failed");

  const used = count || 0;
  if (used >= MAX_DAILY_MESSAGES) {
    return { allowed: false, used, remaining: 0 };
  }

  // 2. insert a new row (message_text stays as the short placeholder)
  const messageId = "msg_" + Date.now() + "_" + Math.random().toString(36).slice(2, 7);
  const { error: insertErr } = await supabaseAdmin
    .from("chat_messages")
    .insert([{ id: messageId, user_id: userId, message_text: "message_sent" }]);

  if (insertErr) throw new Error("record failed");

  return { allowed: true, used: used + 1, remaining: MAX_DAILY_MESSAGES - (used + 1) };
}

export default async function handler(req, res) {
  const origin = req.headers.origin;
  const originOk = isAllowedOrigin(origin);

  if (originOk) {
    res.setHeader("Access-Control-Allow-Origin", origin || "*");
  }
  res.setHeader("Vary", "Origin");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  res.setHeader("X-Content-Type-Options", "nosniff");

  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }
  if (!originOk) {
    return res.status(403).json({ error: "Origin not allowed" });
  }
  if (!process.env.GEMINI_API_KEY) {
    return res.status(500).json({ error: "Missing Gemini API key" });
  }

  const { messages, system, stream, attachment, tzOffsetMinutes } = req.body || {};

  // ---- 1. AUTH: user_id from Supabase JWT, else IP-based key ----
  let userId = await getUserIdFromRequest(req);
  if (!userId) {
    userId = "ip_" + getClientIp(req); // fallback: IP-based bucket
  }

  // ---- 2. VALIDATE BODY ----
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

  let validatedAttachment = null;
  if (attachment) {
    const result = validateAttachment(attachment);
    if (!result.ok) {
      return res.status(result.status).json({ error: result.error });
    }
    validatedAttachment = result;
  }

  // ---- 3. DAILY LIMIT (server-side) ----
  let limitInfo;
  try {
    limitInfo = await checkAndRecordMessage(userId, tzOffsetMinutes);
  } catch (e) {
    console.error("limit error:", e.message);
    return res.status(500).json({ error: "limit check failed" });
  }

  if (!limitInfo.allowed) {
    return res.status(429).json({
      error: "daily_limit_reached",
      used: limitInfo.used,
      remaining: 0,
      max: MAX_DAILY_MESSAGES
    });
  }

  // Tell client how many are left
  res.setHeader("X-Daily-Limit-Remaining", String(limitInfo.remaining));
  res.setHeader("X-Daily-Limit-Max", String(MAX_DAILY_MESSAGES));

  // ---- 4. GEMINI CALL ----
  try {
    const formattedContents = messages.map(msg => ({
      role: msg.role === "assistant" ? "model" : "user",
      parts: [{ text: msg.content }]
    }));

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

    if (stream) {
      res.setHeader("Content-Type", "text/plain; charset=utf-8");
      res.setHeader("Transfer-Encoding", "chunked");

      const responseStream = await ai.models.generateContentStream({
        model: "gemini-2.5-flash-lite",
        contents: formattedContents,
        config: config
      });

      for await (const chunk of responseStream) {
        if (chunk.text) res.write(chunk.text);
      }
      return res.end();
    }

    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash-lite",
      contents: formattedContents,
      config: config
    });

    return res.status(200).json({ content: response.text });

  } catch (err) {
    console.error("Gemini API error:", err && err.message ? err.message : err);
    if (!res.headersSent) {
      res.status(500).json({ error: "⚠️ AI error, try again later" });
    } else {
      res.end();
    }
  }
}
