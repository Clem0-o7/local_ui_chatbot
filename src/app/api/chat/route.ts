import { createOllama } from 'ollama-ai-provider';
import { streamText, convertToCoreMessages, UserContent } from 'ai';

export const runtime = 'edge';
export const dynamic = 'force-dynamic';

export async function POST(req: Request) {
  const { messages, selectedModel, data } = await req.json();
  const ollamaUrl = process.env.OLLAMA_URL!;
  const ragUrl = process.env.RAG_SERVER_URL || 'http://localhost:8000';

  // RAG helpers
  async function fetchContextRAG(text: string): Promise<string[]> { 
    const res = await fetch(`${ragUrl}/rag/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const json = await res.json();
    return Array.isArray(json.context) ? json.context : [];
  
  }
  async function storeToRAG(text: string) { 
    await fetch(`${ragUrl}/rag/store`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    }); 
  }

  const initialMessages = messages.slice(0, -1);
  const currentMessage = messages.at(-1)!;
  const currentText = currentMessage.content;

  // Retrieve up to 3 past snippets
  const contextChunks = await fetchContextRAG(currentText);
  const contextBlock = contextChunks.length
    ? `Here’s some past context:\n${contextChunks.join('\n')}\n\n`
    : '';

  // Prepend and build the user prompt
  const promptWithContext = contextBlock + currentText;
  const messageContent: UserContent = [{ type: 'text', text: promptWithContext }];
  data?.images?.forEach((url: string) => {
    messageContent.push({ type: 'image', image: new URL(url) });
  });

  // 1) Stream to Ollama, 2) capture full reply via onFinish
  const ollama = createOllama({ baseURL: ollamaUrl + '/api' });
  const result = streamText({
    model: ollama(selectedModel),
    messages: [
      ...convertToCoreMessages(initialMessages), // now optional in v4 :contentReference[oaicite:7]{index=7}
      { role: 'user', content: messageContent },
    ],
    onFinish: async ({ text: replyText }) => {
      await storeToRAG(currentText + ' ' + replyText);
    },
  });

  // Stream back to the client
  return result.toDataStreamResponse();
}
