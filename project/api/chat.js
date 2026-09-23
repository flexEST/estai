// code.js
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY
});

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

const MAX_TEXT_LENGTH = 200;                        // matches the client's per-message char limit
const MAX_MESSAGE_STORE_LEN = MAX_TEXT_LENGTH * 4;  // small buffer for older turns
const MAX_HISTORY_MESSAGES = 30;
const MAX_SYSTEM_PROMPT_LEN = 20000;

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
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
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

  const { messages, system, stream, attachment } = req.body || {};

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
        model: "gemini-2.5-flash-lite",
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
      model: "gemini-2.5-flash-lite",
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
