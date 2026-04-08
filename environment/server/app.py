from fastapi import FastAPI
from environment.env import AISafetyEnv, Action

app = FastAPI()
env = AISafetyEnv()

@app.post("/reset")
def reset(difficulty: str = "easy"):
    env.difficulty = difficulty
    obs = env.reset()
    return obs

@app.post("/step")
def step(action: Action):
    score, done, info = env.step(action)
    return {"score": score, "done": done, "info": info}

@app.get("/state")
def state():
    return env.state()

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()