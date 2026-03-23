import asyncio
import os
import time
from openai import AsyncOpenAI
from supervisor import Supervisor, Task

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


async def summarize_agent(task: Task) -> str:
    text = task.payload.get("text", "")
    if not text:
        raise ValueError("No text provided in payload")

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize the given text in 2-3 sentences."},
            {"role": "user",   "content": text},
        ],
        max_tokens=200,
    )
    return response.choices[0].message.content


async def sentiment_agent(task: Task) -> dict:
    text = task.payload.get("text", "")
    if not text:
        raise ValueError("No text provided in payload")

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Classify sentiment as positive, negative or neutral. Reply with JSON: {\"sentiment\": \"...\", \"score\": 0.0-1.0}"},
            {"role": "user",   "content": text},
        ],
        max_tokens=50,
    )

    import json
    raw   = response.choices[0].message.content or "{}"
    clean = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)


async def keyword_agent(task: Task) -> list[str]:
    text = task.payload.get("text", "")
    if not text:
        raise ValueError("No text provided in payload")

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract the 5 most important keywords. Reply with JSON array: [\"word1\", \"word2\", ...]"},
            {"role": "user",   "content": text},
        ],
        max_tokens=100,
    )

    import json
    raw   = response.choices[0].message.content or "[]"
    clean = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)


async def main():
    supervisor = Supervisor(max_restarts=3, task_timeout_s=30.0)

    summarize_id  = supervisor.register("summarizer",  summarize_agent)
    sentiment_id  = supervisor.register("sentiment",   sentiment_agent)
    keyword_id    = supervisor.register("keywords",    keyword_agent)

    text = (
        "Artificial intelligence is transforming the software development industry. "
        "Developers now use AI tools to write code faster, catch bugs earlier and "
        "design better architectures. Companies that adopt AI early gain significant "
        "competitive advantages in both speed and quality of delivery."
    )

    t1 = supervisor.submit("summarize", {"text": text})
    t2 = supervisor.submit("sentiment", {"text": text})
    t3 = supervisor.submit("keywords",  {"text": text})

    await supervisor.run_until_empty([summarize_id, sentiment_id, keyword_id])

    print("\n=== SUPERVISOR RESULTS ===\n")
    for task in [t1, t2, t3]:
        print(f"Task: {task.name}")
        print(f"Status: {task.status}")
        if task.result:
            print(f"Result: {task.result}")
        if task.error:
            print(f"Error: {task.error}")
        if task.duration():
            print(f"Duration: {task.duration():.2f}s")
        print()

    print("=== STATS ===")
    stats = supervisor.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
