import random
from pydantic import BaseModel
from environment.dataset import SCENARIOS
from environment.graders import grade

class Observation(BaseModel):
    video_context: str
    chat_history: list
    comment: str
    difficulty: str

class Action(BaseModel):
    decision: str
    reason: str

class AISafetyEnv:
    def __init__(self, difficulty: str = "easy"):
        self.difficulty = difficulty
        self.current_scenario = None

    def reset(self):
        pool = [s for s in SCENARIOS if s["difficulty"] == self.difficulty]
        self.current_scenario = random.choice(pool)
        return Observation(
            video_context=self.current_scenario["video_context"],
            chat_history=self.current_scenario["chat_history"],
            comment=self.current_scenario["comment"],
            difficulty=self.current_scenario["difficulty"]
        )

    def step(self, action: Action):
        score = grade(
            prediction=action.decision,
            ground_truth=self.current_scenario["label"],
            threat_type=self.current_scenario["threat_type"]
        )
        done = True
        info = {"correct_label": self.current_scenario["label"]}
        return score, done, info

    def state(self):
        return self.current_scenario