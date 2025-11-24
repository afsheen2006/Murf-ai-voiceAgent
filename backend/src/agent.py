import logging
import json
import os
from datetime import datetime
from typing import Annotated
from dataclasses import dataclass, field, asdict

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    MetricsCollectedEvent,
    RunContext,
    function_tool,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ======================================================
# 🧠 STATE MODELS
# ======================================================

@dataclass
class CheckInState:
    mood: str | None = None
    energy: str | None = None
    objectives: list[str] = field(default_factory=list)
    advice_given: str | None = None

    def is_complete(self):
        return (
            self.mood is not None
            and self.energy is not None
            and len(self.objectives) > 0
        )

    def to_dict(self):
        return asdict(self)

@dataclass
class Userdata:
    current_checkin: CheckInState
    history_summary: str
    session_start: datetime = field(default_factory=datetime.now)

# ======================================================
# 💾 JSON LOGGING SYSTEM
# ======================================================

LOG_FILE = "wellness_log.json"

def get_log_path():
    base = os.path.dirname(__file__)
    backend = os.path.abspath(os.path.join(base, ".."))
    return os.path.join(backend, LOG_FILE)

def load_history():
    path = get_log_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def save_checkin(entry: CheckInState):
    path = get_log_path()
    history = load_history()

    record = {
        "timestamp": datetime.now().isoformat(),
        "mood": entry.mood,
        "energy": entry.energy,
        "objectives": entry.objectives,
        "summary": entry.advice_given,
    }

    history.append(record)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    print(f"✔ Saved check-in → {path}")

# ======================================================
# 🛠️ FUNCTION TOOLS
# ======================================================

@function_tool
async def record_mood_energy(
    ctx: RunContext[Userdata],
    mood: Annotated[str, Field(description="How the user feels today")],
    energy: Annotated[str, Field(description="Energy level for today")],
) -> str:

    ctx.userdata.current_checkin.mood = mood
    ctx.userdata.current_checkin.energy = energy
    
    return f"Got it — you're feeling {mood} with {energy} energy."

@function_tool
async def record_goals(
    ctx: RunContext[Userdata],
    goals: Annotated[list[str], Field(description="1–3 goals for today")],
) -> str:

    ctx.userdata.current_checkin.objectives = goals
    return "Great, I’ve noted your goals for today."

@function_tool
async def finalize_checkin(
    ctx: RunContext[Userdata],
    summary: Annotated[str, Field(description="One-sentence supportive summary")],
) -> str:

    state = ctx.userdata.current_checkin
    state.advice_given = summary

    if not state.is_complete():
        return "I still need your mood, energy, and at least one goal before we finish."

    save_checkin(state)

    recap = f"""
Here’s your check-in summary for today:

• Mood: {state.mood}
• Energy: {state.energy}
• Goals: {', '.join(state.objectives)}

Remember: {summary}

I've saved this in your wellness log. Wishing you a grounded, peaceful day.
"""
    return recap

# ======================================================
# 🤖 AGENT BEHAVIOR
# ======================================================

class WellnessAgent(Agent):
    def __init__(self, history_context: str):
        super().__init__(
            instructions=f"""
You are a calm, supportive **Daily Wellness Voice Companion**.
Your role is to guide a short, grounding check-in.

🔹 ALWAYS follow this structure:
1. Warm greeting.
2. Mention **previous session** using this context:
   "{history_context}"
3. Ask:
   - “How are you feeling today?”
   - “What’s your energy level like?”
4. When the user answers → call **record_mood_energy**.
5. Next ask:
   - “What are 1–3 things you’d like to get done today?”
6. When the user answers → call **record_goals**.
7. Offer simple, non-medical, practical advice.
8. Then call **finalize_checkin** with a short encouraging summary.

🔹 SAFETY:
- NEVER diagnose health conditions.
- NEVER give medical or clinical advice.
- If user expresses crisis → reply gently:
  “I may not be able to help enough. Please reach out to someone you trust or a professional immediately.”

🔹 IMPORTANT:
- ALWAYS use the tools to log user mood, energy, goals.
- Keep responses short, warm, encouraging, and grounded.
""",
            tools=[record_mood_energy, record_goals, finalize_checkin],
        )

# ======================================================
# 🚀 SESSION ENTRYPOINT
# ======================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    history = load_history()

    if len(history) > 0:
        last = history[-1]
        history_ctx = (
            f"Last check-in on {last['timestamp']}: "
            f"You felt {last['mood']} with {last['energy']} energy. "
            f"Your goals were {', '.join(last['objectives'])}."
        )
    else:
        history_ctx = "This is your first wellness check-in."

    userdata = Userdata(
        current_checkin=CheckInState(),
        history_summary=history_ctx,
    )

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",   # ✔ Correct voice
            style="Conversation",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )

    await session.start(
        agent=WellnessAgent(history_context=history_ctx),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()

# ======================================================
# ▶️ RUN AGENT
# ======================================================

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
