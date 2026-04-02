import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST");

  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const { messages, system } = req.body;

  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: "messages array required" });
  }

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",   // ✅ cheap model
      max_tokens: 300,        // ✅ limit cost
      messages: [
        { role: "system", content: system || "You are a helpful assistant." },
        ...messages,
      ],
    });

    const reply = completion.choices[0].message.content;

    res.status(200).json({ content: reply });

  } catch (err) {
    console.error(err);

    res.status(500).json({
      error: "⚠️ AI error, try again later"
    });
  }
}
