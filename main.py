from fastapi import FastAPI
from pipeline import get_state_data

app = FastAPI()

@app.get("/api/states")
def state_data():
    return get_state_data()