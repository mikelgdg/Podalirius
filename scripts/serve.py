#!/usr/bin/env python3
"""Launch the Triage-MRI FastAPI server."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from triagemri.serving.model_loader import load_model

def main():
    parser = argparse.ArgumentParser(description="Triage-MRI API server")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config_dir", default="configs")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("triagemri.serve")

    logger.info("Loading model from %s", args.checkpoint)
    load_model(args.checkpoint, args.config_dir, args.device)

    import uvicorn
    logger.info("Starting server on %s:%d", args.host, args.port)
    uvicorn.run(
        "triagemri.serving.api:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )

if __name__ == "__main__":
    main()
