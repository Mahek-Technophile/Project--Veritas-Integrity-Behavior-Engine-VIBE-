"""
Typed Pydantic models for the Context-Aware Content Moderation Environment.
Follows OpenEnv spec: Action, Observation, State all inherit from openenv base types.
"""

from typing import Any, Dict, List, Optional
from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from pydantic import BaseModel as Action
    from pydantic import BaseModel as Observation
    from pydantic import BaseModel as State


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class ModerationAction(Action):
    """
    One moderation decision by the agent.

    decision      : primary action taken on the comment
    harm_category : type of harm detected (or 'safe')
    confidence    : agent's self-reported confidence 0.0-1.0
    justification : free-text reasoning — fed directly to the LLM grader
    """
    decision: str = Field(
        ...,
        description="One of: approve | remove | escalate | label_only"
    )
    harm_category: str = Field(
        default="safe",
        description=(
            "One of: safe | hate_speech | harassment | misinformation "
            "| spam | self_harm | violence | sexual_content"
        )
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Agent confidence in this decision (0.0-1.0)"
    )
    justification: str = Field(
        default="",
        description="Agent's reasoning — used by LLM grader"
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class ModerationObservation(Observation):
    """
    Full context the agent sees at each step.
    Includes the comment AND the video context that gives it meaning.
    """

    # --- Comment being judged ---
    comment_id: str = Field(..., description="Unique comment ID")
    comment_text: str = Field(..., description="Raw comment text (may include Hinglish/emojis)")
    commenter_history: Dict[str, Any] = Field(
        default_factory=dict,
        description="Prior violations, account age, verified status"
    )

    # --- VIDEO CONTEXT LAYER (the core differentiator) ---
    video_title: str = Field(..., description="Title of the video the comment is on")
    video_category: str = Field(
        ...,
        description="Category: Gaming | News | Comedy | Education | Music | Sports | Politics"
    )
    video_tone: str = Field(
        ...,
        description="Tone: satirical | serious | educational | entertainment | documentary"
    )
    transcript_snippet: str = Field(
        default="",
        description="~200 words of transcript around the timestamp of the comment"
    )
    thumbnail_description: str = Field(
        default="",
        description="Vision-LLM description of the video thumbnail"
    )
    language_mix: str = Field(
        default="english",
        description="Language profile: english | hinglish | hindi | mixed"
    )

    # --- Thread context ---
    comment_thread: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Parent comment + sibling replies for thread tasks"
    )

    # --- Task metadata ---
    task_name: str = Field(..., description="task_1 | task_2 | task_3")
    ground_truth_label: str = Field(
        default="",
        description="Hidden ground truth — used by grader, not shown to agent in prompt"
    )
    step_number: int = Field(default=1)

    # --- OpenEnv required fields ---
    reward: Optional[float] = Field(default=None)
    done: bool = Field(default=False)
    feedback: str = Field(
        default="",
        description="Grader feedback from the previous step"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ModerationState(State):
    """
    Full episode state tracked server-side.
    """
    episode_id: str = Field(default="")
    step_count: int = Field(default=0)
    current_task: str = Field(default="task_1")
    total_reward: float = Field(default=0.0)
    decisions_made: List[Dict[str, Any]] = Field(default_factory=list)
    correct_decisions: int = Field(default=0)
    false_positives: int = Field(default=0)   # removed safe content
    false_negatives: int = Field(default=0)   # approved harmful content
    escalations: int = Field(default=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
