#!/usr/bin/env python3
"""Monitor training progress from TensorBoard event files."""
import time
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOGDIR = Path("outputs/run_001/logs/version_0")

while True:
    try:
        ea = EventAccumulator(str(LOGDIR))
        ea.Reload()
        tags = ea.Tags()["scalars"]
        
        print(f"\n{'='*55}")
        for tag in sorted(tags):
            events = ea.Scalars(tag)
            if events:
                print(f"  {tag:35s} {events[-1].value:10.4f}")
        print(f"{'='*55}")
        time.sleep(30)
    except KeyboardInterrupt:
        break
    except Exception:
        time.sleep(10)
