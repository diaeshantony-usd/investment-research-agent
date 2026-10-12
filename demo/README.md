# Multi-Agent Equity Research System — Agent Demo Notebooks

This directory (`demo/`) contains 9 dedicated, self-contained Jupyter Notebooks demonstrating each agent and the complete pipeline in the institutional equity research system. Every notebook connects directly to the real LLM backend (`gpt-oss:120b` via Ollama Cloud) using `from research_agent.interfaces import get_llm` and showcases real agent activations, autonomous tool executions, and multi-agent coordination.

---

## Notebook Directory Index

| Notebook | Agent | Role & Architecture Summary | Input Sent | Output Produced |
| :--- | :--- | :--- | :--- | :--- |
| [`01_planner_agent.ipynb`](file:///demo/01_planner_agent.ipynb) | **Planner** (`ResearchPlanner`) | Lead Strategist & Query Dispatcher. Directs live research, in-memory resolution, and memory tool recall. | 1. NVDA equity query<br>2. Financial concept query<br>3. Session timeline query<br>4. Memory tools inspection (`list_state_ids`, `get_history_states`, `get_state`, `list_threads`)<br>5. Autonomous memory recall query | 1. Structured JSON: `{"research_required": true, "ticker": "NVDA"}`<br>2. Operating margin concept explanation (plain text)<br>3. Session history recall (plain text)<br>4. Programmatic memory inspection results<br>5. Autonomous memory recall trace from memory tools |
| [`02_orchestrator_agent.ipynb`](file:///demo/02_orchestrator_agent.ipynb) | **Router / Orchestrator** (`ResearchRouter`) | ReAct Dynamic Supervisor. Wraps specialist subagents (`EarningsAnalyst`, `MarketAnalyst`, `NewsAnalyst`) as callable tools (`delegate_to_<agent>`). | 1. Targeted earnings query<br>2. Multi-domain technical & news query<br>3. Dynamic subagent addition (`MacroAnalyst`) | 1. Dispatches `delegate_to_earningsanalyst` and synthesizes fundamentals<br>2. Dispatches `MarketAnalyst` and `NewsAnalyst` and synthesizes both<br>3. Dynamically adds new subagent to delegation tools |
| [`03_earnings_analyst_agent.ipynb`](file:///demo/03_earnings_analyst_agent.ipynb) | **Earnings Analyst** (`EarningsAnalyst`) | Quantitative fundamental analysis & SEC EDGAR 10-K/10-Q filing retrieval. | `"Analyze fundamental financial performance and SEC 10-Q filing disclosures for NVDA"` | Calls `get_earnings_data` and `search_sec_filings`; outputs revenue, gross margin, operating income, and 10-Q disclosures. |
| [`04_market_analyst_agent.ipynb`](file:///demo/04_market_analyst_agent.ipynb) | **Market Analyst** (`MarketAnalyst`) | Technical momentum, moving averages (SMA 50/200), RSI, volatility, and macro indicators. | `"Evaluate market price momentum, SMAs, RSI, and technical indicators for NVDA"` | Calls `get_market_technicals`; outputs price, moving averages, 14-day RSI, 52-week range, and macro context. |
| [`05_news_analyst_agent.ipynb`](file:///demo/05_news_analyst_agent.ipynb) | **News Analyst** (`NewsAnalyst`) | Live media catalyst categorization, corporate headlines, and net sentiment scoring. | `"Synthesize recent news catalysts, media headlines, and corporate sentiment for NVDA"` | Calls `get_news_data`; outputs recent news catalysts, corporate announcements, and net market sentiment. |
| [`06_draft_writer_agent.ipynb`](file:///demo/06_draft_writer_agent.ipynb) | **Draft Writer** (`DraftWriter`) | Institutional report synthesis. Unifies blackboard specialist findings into standard brief sections. | Blackboard dictionary of findings from Earnings, Market, and News analysts. | Complete 5-section institutional research brief: Executive Summary, Fundamentals, Technicals, Catalysts & Risks, Stance. |
| [`07_critic_agent.ipynb`](file:///demo/07_critic_agent.ipynb) | **Critic** (`Critic`) | Dual-stage Evaluator. Code-based number grounding check + 5-dimension weighted LLM judge. | Draft research brief + raw tool observation data from `state.tools`. | Grounding check results, 5-dimension score breakdown (grounding, coverage, balance, clarity, actionability), and verdict. |
| [`08_refiner_agent.ipynb`](file:///demo/08_refiner_agent.ipynb) | **Refiner** (`BriefRefiner`) | Targeted Optimizer in evaluator-optimizer loop. Rewrites flagged sections without altering solid text. | Draft brief + flagged issue descriptions from Critic report. | Refined research brief addressing issues + structured `### Change Summary` modification log. |
| [`09_end_to_end_pipeline.ipynb`](file:///demo/09_end_to_end_pipeline.ipynb) | **End-to-End Pipeline** (`StateGraph`) | Complete institutional research workflow compiled with `build_research_workflow`. | Equity research query (`"Provide an equity research briefing on NVDA covering fundamentals, technical momentum, and news catalysts."`) | Multi-node trajectory through `planner -> orchestrator -> draft_writer -> critic -> release` and final published report. |

---

## How to Run

All notebooks are pre-populated with rendered cell execution outputs from the real LLM backend. You can view them directly in VS Code or Jupyter Lab.

To re-run any notebook live:
1. Open the `.ipynb` file in your editor or Jupyter.
2. Select the Python environment at `.venv` (Python 3.13 / 3.11).
3. Click **Run All** or execute cells sequentially.

Every notebook directly instantiates the real LLM via `from research_agent.interfaces import get_llm`.
