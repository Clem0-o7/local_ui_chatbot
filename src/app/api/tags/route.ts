export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET(req: Request) {
    const OLLAMA_URL = process.env.OLLAMA_URL;
    console.log('OLLAMA_URL:', OLLAMA_URL); // Debugging

    if (!OLLAMA_URL) {
        return new Response(JSON.stringify({ error: "OLLAMA_URL is missing" }), { status: 500 });
    }

    try {
        const res = await fetch(`${OLLAMA_URL}/api/tags`);
        if (!res.ok) throw new Error(`Fetch failed: ${res.statusText}`);

        const data = await res.json();
        return new Response(JSON.stringify(data), { status: 200, headers: { "Content-Type": "application/json" } });
    } catch (error) {
        console.error("Fetch error:", error);
        return new Response(JSON.stringify({ error: "Failed to fetch tags" }), { status: 500 });
    }
}
