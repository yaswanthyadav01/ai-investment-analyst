# 📈 Multi-Agent Investment & Market Analyst

An automated analyst built from cooperating agents, with a Streamlit dashboard.

> **Disclaimer:** Educational project. Output is rule-based and automated, can be wrong,
> and is **not financial advice**.

## How it works

```
 tickers ──► Data layer (yfinance / demo data)
                 │
     ┌───────────┼────────────┬─────────────┐   (run in parallel)
     ▼           ▼            ▼             ▼
 Technical   Fundamental   Sentiment      Risk
  agent        agent         agent        agent
     └───────────┴─────┬──────┴─────────────┘
                       ▼
                Strategist agent  ── weights × confidence, risk override
                       │              optional Claude-written note
                       ▼
        Rating · score · confidence · position size
```

| Agent | Looks at | Output |
|---|---|---|
| Technical | 50/200-day trend, RSI, MACD, Bollinger | score -1…+1 |
| Fundamental | P/E, PEG, margins, ROE, debt/equity, revenue growth | score + data-coverage confidence |
| Sentiment | Recent headlines, transparent keyword lexicon | score |
| Risk | Volatility, max drawdown, VaR, beta | risk level + position cap |
| Strategist | All of the above | final rating, position size, written note |

Design choices worth knowing:
- An agent with no data gets **zero confidence → zero vote** (never a fake neutral vote).
- A crashing agent is contained; the report still completes.
- The backtest shifts signals one bar to avoid look-ahead bias.
- The LLM is **optional**; every failure falls back to a rule-based note.

## Project layout

```
app.py                  Streamlit UI
analyst/
  data.py               yfinance + deterministic demo data
  indicators.py         RSI, MACD, Bollinger, ATR, VaR, beta...
  agents/               technical, fundamental, sentiment, risk, strategist
  orchestrator.py       parallel execution + AnalysisReport
  backtest.py           SMA-crossover vs buy & hold
  llm.py                optional Claude narrative
tests/                  offline pytest suite
.github/workflows/ci.yml
Dockerfile
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q                                              # optional sanity check
streamlit run app.py
```

Tick **Demo mode** in the sidebar to try everything without network data.

Optional Claude-written notes: copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml` and add `ANTHROPIC_API_KEY`, or paste a key in the sidebar.

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: multi-agent investment analyst"
git branch -M main
git remote add origin https://github.com/<your-username>/ai-investment-analyst.git
git push -u origin main
```

The included GitHub Actions workflow runs the tests on every push/PR.
`.streamlit/secrets.toml` and `.env` are git-ignored, so keys are never committed.

## Deploy on Streamlit Community Cloud (free)

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **Create app** → pick your repo, branch `main`, main file `app.py`.
3. *(Optional)* **Advanced settings → Secrets**: add `ANTHROPIC_API_KEY = "sk-ant-..."`.
4. Deploy. Every `git push` redeploys automatically.

## Deploy with Docker (alternative)

```bash
docker build -t ai-investment-analyst .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=sk-ant-... ai-investment-analyst
```

## Extending

Add an agent by subclassing `BaseAgent` (return an `AgentResult`), appending it in
`Orchestrator.__init__`, and giving it a weight in `DEFAULT_WEIGHTS`. Ideas: macro agent
(rates/inflation), options-flow agent, earnings-call transcript agent, portfolio optimiser.

## Known limitations

- Yahoo Finance data is free but unofficial: it can be delayed, incomplete or rate-limited.
- Sentiment is keyword-based (no negation handling); fundamentals are missing for ETFs/indices.
- Indian tickers work with Yahoo suffixes (`RELIANCE.NS`, `TCS.NS`; index `^NSEI`).
