// code.js
import { GoogleGenAI } from "@google/genai";

// Initialize Gemini Client. 
// It will automatically pick up process.env.GEMINI_API_KEY if passed without an object,
// but explicitly providing it ensures compatibility with your existing environment style.
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY
});

export default async function handler(req, res) {
  // Allow CORS for testing from frontend
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    // Preflight request for CORS
    return res.status(200).end();
  }

  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  if (!process.env.GEMINI_API_KEY) {
    return res.status(500).json({ error: "Missing Gemini API key" });
  }

  const { messages, system } = req.body;

  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: "messages array required" });
  }

  try {
    // Format incoming OpenAI-style array [{role: "user", content: "..."}] 
    // to Gemini-style array [{role: "user", parts: [{text: "..."}]}]
    const formattedContents = messages.map(msg => ({
      role: msg.role === "assistant" ? "model" : "user",
      parts: [{ text: msg.content }]
    }));

    // Generate content using Google's new GenAI client syntax
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash", // Fast, low-cost flagship model
      contents: formattedContents,
      config: {
        maxOutputTokens: 1000,   // Limits response size/cost
        temperature: 0.7,        // Creativity controls
        systemInstruction: system || "You are a helpful assistant." // Clean separation for system prompt
      }
    });

    // Extract the text content from the Gemini response structure
    const reply = response.text;

    res.status(200).json({ content: reply });

  } catch (err) {
    console.error("Gemini API error:", err);
    res.status(500).json({ error: "⚠️ AI error, try again later" });
  }
}
