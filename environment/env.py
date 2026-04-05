from pydantic import BaseModel
from typing import Optional

class Observation(BaseModel):
    video_context: str
    chat_history: list[str]
    comment: str
    difficulty: str

class Action(BaseModel):
    decision: str
    reason: str

class Reward(BaseModel):
    score: float
    feedback: str

class AISafetyEnv:
    def __init__(self):
        self.current_scenario = None

    def reset(self):
        pass  # Person 1 fills dataset first, then you complete this

    def step(self, action: Action):
        pass  # you complete this in Task 4

    def state(self):
        pass  # you complete this in Task 4
