from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from openai import OpenAI
import logging
import os

logger = logging.getLogger("uvicorn")
app = FastAPI()
load_dotenv()

client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

@app.post("/analyze_trip")
async def analyze_trip(body: dict):
    trip_data = body.get("trip_data", None)

    if not trip_data:
        raise HTTPException(status_code=400, detail="Invalid trip data.")

    prompt = f"""
        You are a driving analytics expert.

        Analyze the following trip data and produce a concise, human-readable insight and summary.
        Note: The waypoints provided are a representative sample of the full route.
        They include evenly distributed points across the trip and do not represent the complete path.
        The original total waypoint count is provided separately.

        IMPORTANT SPECIAL CASE:
        If the data strongly indicates that the movement was walking rather than a vehicle
        (e.g. consistently very low speeds, short distances, frequent direction changes),
        DO NOT perform a driving analysis.
        Instead, return a short response explaining that the trip appears to have been made on foot.
        Example tone:
        "We detected that this trip was likely done by walking rather than driving, so driving insights are not applicable."

        Focus on:
            - Overall trip characteristics (distance, duration, speed profile)
            - Driving behavior patterns (smoothness, stops, speed variation)
            - Route characteristics inferred from waypoints (urban vs open road, turns, consistency)
            - Any notable observations or anomalies
            - Practical insights (not raw stats repetition)

        Rules:
            - Do NOT repeat raw numbers unless they support an insight
            - Do NOT invent data that is not present
            - Keep the tone professional and neutral
            - Avoid bullet points — write as a flowing analysis
            - Maximum length: 85 words
            
        Trip data:
        <data>
            {{trip_data}}
        </data>
    """

    response = client.responses.create(model="gpt-4o-mini", input=prompt, temperature=0.2)
    return {"agent_message": response.output_text}