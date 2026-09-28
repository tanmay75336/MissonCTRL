import { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const suggestions = [
  "Research the current AI startup landscape in India.",
  "Compare student-focused business opportunities in Mumbai.",
  "Find the latest developments in India's EV charging market."
];
const stages = ["Planning", "Researching", "Checking evidence", "Finalizing"];
const progressByEvent = { MISSION_STARTED: 5, STRATEGY_SELECTED: 15, RESEARCH_QUESTION: 20, SEARCH_STARTED: 25, SEARCH_COMPLETED: 40, EVIDENCE_EVALUATED: 60, DECISION: 70, STRATEGY_SWITCHED: 70, SYNTHESIS_STARTED: 85, MISSION_COMPLETED: 100, MISSION_STOPPED: 100 };

function getProgress(events, phase) {
  if (phase === "running" && !events.length) return { percent: 5, stage: 0, action: "Starting your research" };
  const event = events.at(-1);
  if (!event) return { percent: 0, stage: -1, action: "Ready when you are" };
  const type = event.type;
  const decisionAction = event.data?.next_action;
  const action = decisionAction === "SEARCH" ? "Looking for the missing information" : decisionAction === "SYNTHESIZE" ? "Enough evidence was found" : decisionAction === "STOP" ? "Research stopped with remaining uncertainty" : ({ MISSION_STARTED: "Starting your research", STRATEGY_SELECTED: "Planning the research approach", RESEARCH_QUESTION: "Choosing the next research question", SEARCH_STARTED: "Searching for relevant information", SEARCH_COMPLETED: "Reviewing search results", EVIDENCE_COLLECTED: "Reviewing useful sources", EVIDENCE_REJECTED: "Filtering out weak sources", EVIDENCE_EVALUATED: "Checking whether the sources answer your question", STRATEGY_SWITCHED: "Changing the research approach", SYNTHESIS_STARTED: "Preparing your answer", MISSION_COMPLETED: "Research complete", MISSION_STOPPED: "Research stopped with remaining uncertainty" }[type] || "Reviewing the research");
  const stage = ["MISSION_STARTED", "STRATEGY_SELECTED"].includes(type) ? 0 : ["RESEARCH_QUESTION", "SEARCH_STARTED", "SEARCH_COMPLETED", "STRATEGY_SWITCHED"].includes(type) ? 1 : ["EVIDENCE_COLLECTED", "EVIDENCE_REJECTED", "EVIDENCE_EVALUATED", "DECISION"].includes(type) ? 2 : 3;
  return { percent: progressByEvent[type] ?? 5, stage, action };
}

function humanActivity(event) {
  const results = event.data?.results;
  if (event.type === "MISSION_STARTED") return "Started the research";
  if (event.type === "STRATEGY_SELECTED") return "Planned the research approach";
  if (event.type === "SEARCH_COMPLETED") return `Reviewed ${results ?? "available"} sources`;
  if (event.type === "EVIDENCE_EVALUATED") return "Checked whether the evidence answered the question";
  if (event.type === "STRATEGY_SWITCHED") return "Changed the research approach";
  if (event.type === "DECISION") return event.data?.next_action === "SEARCH" ? "More evidence was needed" : event.data?.next_action === "SYNTHESIZE" ? "Enough evidence was found" : "Research stopped with uncertainty";
  if (event.type === "SYNTHESIS_STARTED") return "Prepared the final answer";
  if (event.type === "MISSION_COMPLETED") return "Completed the mission";
  if (event.type === "MISSION_STOPPED") return "Stopped with remaining uncertainty";
  return null;
}

function DecisionCard({ decision }) {
  const [open, setOpen] = useState(false);
  const tone = decision.next_action === "SYNTHESIZE" ? "sufficient" : decision.next_action === "STOP" ? "stopped" : "insufficient";
  const title = decision.next_action === "SYNTHESIZE" ? "EVIDENCE CHECK COMPLETE" : decision.next_action === "STOP" ? "RESEARCH STOPPED" : "MORE RESEARCH NEEDED";
  const summary = decision.next_action === "SYNTHESIZE" ? "Enough relevant evidence was found. Preparing the final answer." : decision.next_action === "STOP" ? "Research stopped with uncertainty preserved below." : "The first sources did not yet answer an important part of the mission.";
  return <article className={`decision-card ${tone}`}><div className="decision-topline"><span className="status-dot" /><strong>{title}</strong></div><p>{summary}</p><button className="why-button" type="button" onClick={() => setOpen(!open)}>{open ? "Hide why" : "Why?"}</button>{open && <div className="why-detail"><strong>{decision.reason}</strong>{decision.missing_information?.length > 0 && <ul>{decision.missing_information.map((item, index) => <li key={index}>{item}</li>)}</ul>}{decision.next_question && <p>Next, MISSIONCTRL looked for: {decision.next_question}</p>}</div>}</article>;
}

function SourceCard({ source }) {
  return <article className="source-card"><span className="source-kind">RESEARCH SOURCE</span><h3>{source.title || "Untitled source"}</h3>{source.claim_supported && <p>{source.claim_supported}</p>}{source.url && <a href={source.url} target="_blank" rel="noreferrer">Open source &rarr;</a>}</article>;
}

function App() {
  const [mission, setMission] = useState("");
  const [budget, setBudget] = useState(4);
  const [phase, setPhase] = useState("idle");
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [visibleEvents, setVisibleEvents] = useState([]);
  const [showActivity, setShowActivity] = useState(false);
  const [showSources, setShowSources] = useState(false);
  const running = phase === "running" || phase === "replaying";
  const events = visibleEvents.length ? visibleEvents : result?.events || [];
  const progress = getProgress(events, phase);
  const activity = useMemo(() => events.map(humanActivity).filter(Boolean).filter((item, index, all) => item !== all[index - 1]), [events]);
  const used = result?.searches_used ?? 0;
  const total = result ? result.searches_used + result.remaining_searches : budget;
  const finalDecision = result?.decisions?.at(-1);

  async function executeMission(event) {
    event.preventDefault();
    if (!mission.trim()) return setError("Enter a mission before starting research.");
    setPhase("running"); setError(""); setResult(null); setVisibleEvents([]); setShowActivity(false); setShowSources(false);
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 90000);
    try {
      const response = await fetch("/api/v1/missions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ mission: mission.trim(), plan: "", search_budget: budget }), signal: controller.signal });
      const payload = await response.json().catch(() => null);
      if (!response.ok || !payload || typeof payload.answer !== "string" || !Array.isArray(payload.events)) throw new Error();
      setPhase("replaying");
      for (const item of payload.events) { setVisibleEvents((current) => [...current, item]); await new Promise((resolve) => window.setTimeout(resolve, 220)); }
      setResult(payload); setPhase("complete");
    } catch (requestError) {
      setPhase("error");
      setError(requestError.name === "AbortError" ? "MISSIONCTRL couldn't complete the research in time. Check that the research server is running and try again." : "MISSIONCTRL couldn't complete the research. Check that the research server is running and try again.");
    } finally { window.clearTimeout(timeout); }
  }
  function startNewMission() { setMission(""); setResult(null); setVisibleEvents([]); setError(""); setPhase("idle"); }

  return <main className="shell">
    <header className="masthead"><div><p className="eyebrow"><span /> AUTONOMOUS RESEARCH</p><h1>MISSIONCTRL</h1><p className="subhead">Give me a mission. I’ll research it.</p></div><div className={`system-state ${running ? "active" : ""}`}><i /> {running ? "MISSION RUNNING" : phase === "complete" ? "MISSION COMPLETE" : "SYSTEM READY"}</div></header>
    <section className="mission-panel" aria-labelledby="mission-heading"><div className="section-heading"><span>01</span><h2 id="mission-heading">MISSION INPUT</h2></div><form onSubmit={executeMission}><textarea value={mission} onChange={(event) => setMission(event.target.value)} disabled={running} placeholder="Research a market, investigate a company, compare options, find current information, or answer a complex question..." /><div className="controls"><label>SEARCH BUDGET<select value={budget} onChange={(event) => setBudget(Number(event.target.value))} disabled={running}>{[2, 4, 6, 8].map((value) => <option key={value} value={value}>{value} searches</option>)}</select></label><button type="submit" disabled={running}>{running ? "MISSION RUNNING" : "EXECUTE MISSION"}<span>&rarr;</span></button></div></form><div className="suggestions">{suggestions.map((item) => <button key={item} type="button" onClick={() => setMission(item)} disabled={running}>{item}</button>)}</div>{error && <p className="error-message" role="alert">{error}</p>}</section>
    <section className="progress-panel" aria-live="polite"><div className="progress-topline"><div><p className="eyebrow">{phase === "complete" ? "MISSION COMPLETE" : running ? "RESEARCHING" : "MISSION PROGRESS"}</p><strong>{progress.percent}%</strong></div><div className="search-summary"><span>SEARCHES</span><b>{used} / {total}</b><small>{result?.remaining_searches ?? budget} remaining</small></div></div><div className="progress-track"><i style={{ width: `${progress.percent}%` }} /></div><p className="current-action">{progress.action}</p><div className="stage-list">{stages.map((stage, index) => <span key={stage} className={index < progress.stage ? "done" : index === progress.stage ? "current" : "future"}>{index < progress.stage ? "✓" : index === progress.stage ? "●" : "○"} {stage}</span>)}</div></section>
    <section className="operations-grid"><section className="activity-panel" aria-labelledby="activity-heading"><div className="section-heading"><span>02</span><h2 id="activity-heading">RESEARCH ACTIVITY</h2></div>{activity.length ? <><ol className="simple-timeline">{activity.slice(showActivity ? 0 : -5).map((item, index) => <li key={`${item}-${index}`}>{index === activity.slice(showActivity ? 0 : -5).length - 1 && running ? "●" : "✓"}<span>{item}</span></li>)}</ol>{activity.length > 5 && <button className="text-button" type="button" onClick={() => setShowActivity(!showActivity)}>{showActivity ? "Show less activity" : "View full activity"}</button>}</> : <div className="empty-state"><span className="pulse" /> Your research activity will appear here</div>}</section>{finalDecision && <aside className="autonomy-callout"><p className="eyebrow">AUTONOMOUS DECISION</p><strong>{finalDecision.next_action === "SEARCH" ? "MORE RESEARCH NEEDED" : finalDecision.next_action === "STOP" ? "RESEARCH STOPPED" : "EVIDENCE CHECK COMPLETE"}</strong><p>{finalDecision.next_action === "SEARCH" ? "The available sources did not yet provide enough direct evidence, so MISSIONCTRL searched for the missing information." : finalDecision.next_action === "STOP" ? "Research stopped with remaining uncertainty, shown below." : "Enough relevant evidence was found to prepare the final answer."}</p></aside>}</section>
    {(result || running) && <section className="content-grid"><section className="evidence-panel"><div className="section-heading"><span>03</span><h2>RESEARCH SOURCES</h2></div>{result?.decisions?.length > 0 && <div className="decision-list">{result.decisions.map((decision, index) => <DecisionCard key={index} decision={decision} />)}</div>}{result?.sources?.length > 0 ? <><div className="source-grid">{result.sources.slice(0, showSources ? result.sources.length : 3).map((source, index) => <SourceCard key={`${source.url}-${index}`} source={source} />)}</div>{result.sources.length > 3 && <button className="text-button source-toggle" type="button" onClick={() => setShowSources(!showSources)}>{showSources ? "Show fewer sources" : `View all ${result.sources.length} sources`}</button>}</> : <p className="muted">Sources will appear after the mission completes.</p>}</section><section className="result-panel"><div className="section-heading"><span>04</span><h2>MISSION RESULT</h2></div>{result ? <><article className="answer"><p className="eyebrow">FINAL FINDINGS</p><p>{result.answer}</p></article><div className="finding-columns"><div><h3>KEY FACTS</h3><ul>{result.facts?.map((item, index) => <li key={index}>{item}</li>)}</ul></div><div><h3>WHAT IS INFERRED</h3><ul>{result.inferences?.map((item, index) => <li key={index}>{item}</li>)}</ul></div></div><div className="uncertainties"><h3>STILL UNCERTAIN</h3><ul>{[...(result.unknowns || []), ...(result.unresolved_questions || [])].map((item, index) => <li key={index}>{item}</li>)}</ul></div><button className="new-mission" type="button" onClick={startNewMission}>START NEW MISSION &rarr;</button></> : <p className="muted">Final evidence-grounded findings will appear here.</p>}</section></section>}
  </main>;
}

createRoot(document.getElementById("root")).render(<App />);
