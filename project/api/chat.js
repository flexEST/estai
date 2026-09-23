// code.js
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY
});

export default async function handler(req, res) {
  // Allow CORS for testing from frontend
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  if (!process.env.GEMINI_API_KEY) {
    return res.status(500).json({ error: "Missing Gemini API key" });
  }

  const { messages, system, stream } = req.body;

  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: "messages array required" });
  }

  try {
    const formattedContents = messages.map(msg => ({
      role: msg.role === "assistant" ? "model" : "user",
      parts: [{ text: msg.content }]
    }));

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
    console.error("Gemini API error:", err);
    if (!res.headersSent) {
      res.status(500).json({ error: "⚠️ AI error, try again later" });
    } else {
      res.end();
    }
  }
}
