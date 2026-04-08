"""
Inference Script — Context-Aware Content Moderation Environment
===============================================================
MANDATORY:
- API_BASE_URL   : The API endpoint for the LLM
- MODEL_NAME     : The model identifier to use for inference
- HF_TOKEN       : Your Hugging Face / API key

- This script must be named `inference.py` and placed in the root directory
- Uses OpenAI Client for all LLM calls
- Emits structured [START] [STEP] [END] logs for automated scoring
"""

import os
import re
import json
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Config — read from environment variables (mandatory for submission)
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

MAX_STEPS   = 8
TEMPERATURE = 0.2
MAX_TOKENS  = 300
FALLBACK_ACTION = json.dumps({
    "decision": "escalate",
    "harm_category": "safe",
    "confidence": 0.3,
    "justification": "Unable to determine — escalating for human review",
})

TASKS = ["task_1", "task_2", "task_3"]

# ---------------------------------------------------------------------------
# System prompt — tells the agent how to behave as a content moderator
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert content moderation agent for a video platform.

    Your job is to decide whether a comment should be approved, removed, or escalated
    for human review. CRITICALLY: you must consider the VIDEO CONTEXT — the same words
    can be friendly banter on a gaming video and a genuine threat on a news video.

    You will be given:
    - The comment text (may include Hindi, Hinglish, English, emojis)
    - The video title, category, tone, transcript snippet, thumbnail description
    - The commenter's history (for hard tasks)
    - The comment thread (for thread tasks)

    Respond with ONLY a valid JSON object — no explanation outside the JSON:
    {
        "decision": "<approve|remove|escalate|label_only>",
        "harm_category": "<safe|hate_speech|harassment|misinformation|spam|self_harm|violence|sexual_content>",
        "confidence": <0.0 to 1.0>,
        "justification": "<one sentence explaining your reasoning>"
    }

    Decision guide:
    - approve     : comment is safe given the video context
    - remove      : comment is clearly harmful given the video context
    - escalate    : comment is ambiguous — flag for human review
    - label_only  : comment needs a warning label but not removal

    Key rules:
    - Gaming slang and Hinglish banter on entertainment videos is usually SAFE
    - The same language on serious news/documentary videos may be HARMFUL
    - A new account (< 30 days) with prior violations deserves extra scrutiny
    - Coordinated similar comments = escalate even if each one seems borderline
    - When unsure, escalate rather than approve
    """
).strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_user_prompt(observation: Dict[str, Any], history: List[str]) -> str:
    """Build the per-step user prompt from observation fields."""
    comment    = observation.get("comment_text", "")
    video_title = observation.get("video_title", "")
    category   = observation.get("video_category", "")
    tone       = observation.get("video_tone", "")
    transcript = observation.get("transcript_snippet", "")
    thumbnail  = observation.get("thumbnail_description", "")
    lang_mix   = observation.get("language_mix", "english")
    commenter  = observation.get("commenter_history", {})
    thread     = observation.get("comment_thread", [])
    task       = observation.get("task_name", "task_1")
    step       = observation.get("step_number", 1)
    feedback   = observation.get("feedback", "")

    history_text = "\n".join(history[-3:]) if history else "None"

    thread_text = "N/A"
    if thread:
        thread_text = "\n".join(
            f"  [{i+1}] {c.get('commenter','?')}: {c.get('comment_text','')}"
            for i, c in enumerate(thread)
        )

    commenter_text = (
        f"Account age: {commenter.get('account_age_days','?')} days | "
        f"Prior violations: {commenter.get('prior_violations', 0)} | "
        f"Verified: {commenter.get('verified', False)}"
        if commenter else "Unknown"
    )

    return textwrap.dedent(f"""
        Task: {task} | Step: {step}
        Previous feedback: {feedback or 'None'}
        Prior decisions: {history_text}

        === COMMENT TO MODERATE ===
        "{comment}"
        Language: {lang_mix}
        Commenter history: {commenter_text}

        === VIDEO CONTEXT ===
        Title    : {video_title}
        Category : {category}
        Tone     : {tone}
        Transcript snippet: "{transcript}"
        Thumbnail: {thumbnail}

        === THREAD CONTEXT ===
        {thread_text}

        Reply with ONLY a JSON object matching the required format.
    """).strip()


def parse_agent_action(response_text: str) -> Dict[str, Any]:
    """Extract and parse JSON action from model response."""
    if not response_text:
        return json.loads(FALLBACK_ACTION)

    # Try to find JSON block anywhere in the response
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            # Validate required fields
            if "decision" in parsed:
                parsed.setdefault("harm_category", "safe")
                parsed.setdefault("confidence", 0.5)
                parsed.setdefault("justification", "")
                return parsed
        except json.JSONDecodeError:
            pass

    return json.loads(FALLBACK_ACTION)


def call_llm(client: OpenAI, user_prompt: str) -> str:
    """Call the LLM and return raw response text."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user",   "content": [{"type": "text", "text": user_prompt}]},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return completion.choices[0].message.content or ""
    except Exception as exc:
        print(f"Model request failed ({exc}). Using fallback action.")
        return FALLBACK_ACTION


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_episode(
    client: OpenAI,
    env,
    task: str,
    scenario_index: int,
) -> Dict[str, Any]:
    """Run a single episode for a given task and return episode summary."""

    # [START] log — mandatory format
    start_log = {
        "task": task,
        "scenario_index": scenario_index,
        "model": MODEL_NAME,
    }
    print(f"[START] {json.dumps(start_log)}")

    result = env.reset(task=task, scenario_index=scenario_index)
    observation = result.observation if hasattr(result, "observation") else result

    history: List[str] = []
    episode_reward = 0.0
    steps_taken = 0

    for step in range(1, MAX_STEPS + 1):
        obs_dict = observation.model_dump() if hasattr(observation, "model_dump") else vars(observation)

        if obs_dict.get("done", False) and step > 1:
            print("Environment signalled done. Stopping early.")
            break

        user_prompt = build_user_prompt(obs_dict, history)
        response_text = call_llm(client, user_prompt)
        action_dict = parse_agent_action(response_text)

        print(f"Step {step}: model suggested → {action_dict.get('decision')} "
              f"(confidence: {action_dict.get('confidence', 0.5):.2f})")

        # Build action object and call step()
        try:
            from moderation_env import ModerationAction
        except ImportError:
            from models import ModerationAction

        action = ModerationAction(**action_dict)
        result = env.step(action)
        observation = result.observation if hasattr(result, "observation") else result
        obs_dict = observation.model_dump() if hasattr(observation, "model_dump") else vars(observation)

        reward = obs_dict.get("reward") or 0.0
        done   = obs_dict.get("done", False)
        feedback = obs_dict.get("feedback", "")
        episode_reward += reward
        steps_taken = step

        history_line = (
            f"Step {step}: {action_dict.get('decision')} → "
            f"reward {reward:+.2f} | {feedback[:60]}"
        )
        history.append(history_line)

        # [STEP] log — mandatory format
        step_log = {
            "step": step,
            "action": action_dict.get("decision"),
            "harm_category": action_dict.get("harm_category"),
            "confidence": action_dict.get("confidence"),
            "reward": round(reward, 3),
            "done": done,
            "feedback": feedback[:80],
        }
        print(f"[STEP] {json.dumps(step_log)}")

        print(
            f"  Reward: {reward:+.2f} | Done: {done} | "
            f"Feedback: {feedback[:60]}"
        )

        if done:
            print("Episode complete.")
            break
    else:
        print(f"Reached max steps ({MAX_STEPS}).")

    # [END] log — mandatory format
    end_log = {
        "task": task,
        "scenario_index": scenario_index,
        "steps_taken": steps_taken,
        "total_reward": round(episode_reward, 3),
        "score": round(min(max(episode_reward, 0.0), 1.0), 3),
    }
    print(f"[END] {json.dumps(end_log)}")

    return end_log


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Import environment — works whether running locally or from Docker
    try:
        from moderation_env.client import ModerationEnv
        env_cls = ModerationEnv
        use_client = True
    except ImportError:
        # Fall back to direct environment import for local testing
        from server.moderation_environment import ModerationEnvironment
        env_cls = ModerationEnvironment
        use_client = False

    all_results = []

    # Run one episode per task (3 tasks × 1 scenario each = 3 episodes)
    TASK_SCENARIOS = [
        ("task_1", 0),
        ("task_2", 0),
        ("task_3", 1),
    ]

    if use_client:
        env_url = os.getenv("ENV_URL", "http://localhost:8000")
        with env_cls(base_url=env_url).sync() as env:
            for task, scenario_idx in TASK_SCENARIOS:
                result = run_episode(client, env, task, scenario_idx)
                all_results.append(result)
    else:
        # Direct mode — for local testing without Docker
        env = env_cls()
        for task, scenario_idx in TASK_SCENARIOS:
            result = run_episode(client, env, task, scenario_idx)
            all_results.append(result)

    # Final summary
    total_score = sum(r["score"] for r in all_results) / len(all_results)
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for r in all_results:
        print(f"  {r['task']:10s} | score: {r['score']:.3f} | steps: {r['steps_taken']}")
    print(f"\n  Overall score: {total_score:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
