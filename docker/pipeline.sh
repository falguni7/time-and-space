#!/bin/bash
# Rebuild everything from scratch, then serve the app. Run by docker/Dockerfile.pipeline.
set -euo pipefail

if [ ! -f data/engine/metrics.duckdb ]; then
  python -m presslens.sim generate          # 56-match season -> data/matches
  python -m presslens.engine build          # xT, xPass, decisions, metrics cube, moments, validation
fi

if [ "${REBUILD_DEMO:-0}" = "1" ]; then
  # Live: every agent call goes to your Azure AI Foundry deployment (needs .env, costs ~$0.25).
  python -m presslens.agents.demo build --backend azure
else
  # Offline: replay the recorded model answers; fails loudly if any answer would be new.
  python -m presslens.agents.demo check || echo "WARNING: the replay differs from the recording (see above)"
fi

python -m presslens.web.clips               # tracking clips + page data -> demo/
exec python -m presslens.web.server
