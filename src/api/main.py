import json
from datetime import datetime
import os
import time
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from api.config import config
from api.models import Schedule, ScheduleRequest
from api.postprocess import create_debug_visualizations
from api.preprocess import preprocess_checks
from engine.scheduler import SchedulingEngine
from engine.models import LegalHoursConstraint
from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Initialize the scheduling engine
scheduling_engine = SchedulingEngine(debug=config.debug)


@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.post("/auto-schedule", response_model=Schedule)
def auto_schedule(request: ScheduleRequest, background_tasks: BackgroundTasks) -> Schedule:
    start_time = time.time()
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = f"debug_logs/{now}"
    os.makedirs(debug_dir, exist_ok=True)

    # Time file writing
    write_start = time.time()
    with open(f"{debug_dir}/input.json", "w") as f:
        json.dump(request.model_dump(), f, indent=2)
    print(f"Input file writing took: {time.time() - write_start:.2f} seconds")

    # Time preprocessing
    preprocess_start = time.time()
    try:
        preprocess_checks(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    print(f"Preprocessing took: {time.time() - preprocess_start:.2f} seconds")

    # Time scheduling
    schedule_start = time.time()
    result = scheduling_engine.schedule(activities=request.activities, workers=request.workers, constraints=request.options.constraints)
    print(f"Scheduling computation took: {time.time() - schedule_start:.2f} seconds")

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error_message)

    schedule = Schedule(activities=result.filled_activities, score=result.score)

    # Move debug operations to background tasks
    if config.debug:
        background_tasks.add_task(create_debug_visualizations, now, schedule.model_dump(), request.model_dump())

        def write_debug_output():
            with open(f"{debug_dir}/output.json", "w") as f:
                json.dump(schedule.model_dump(), f, indent=2)

        background_tasks.add_task(write_debug_output)

    print(f"Total request processing time: {time.time() - start_time:.2f} seconds")
    return schedule
