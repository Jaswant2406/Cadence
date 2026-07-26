import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  CircleGauge,
  Database,
  Download,
  FileCheck2,
  FileSearch,
  FileUp,
  Fingerprint,
  Gauge,
  Info,
  LayoutDashboard,
  Menu,
  Network,
  Play,
  Radar,
  RefreshCw,
  Search,
  ServerCog,
  Shield,
  ShieldAlert,
  Sparkles,
  UploadCloud,
  UserRoundSearch,
  X,
  XCircle,
} from "lucide-react";
import {
  Navigate,
  NavLink,
  Outlet,
  Route,
  Routes,
  useNavigate,
  useParams,
} from "react-router-dom";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "./api";
import type {
  Alert,
  ImportResult,
  Paginated,
  RunStatus,
  TelemetryEvent,
} from "./types";

const severityColors: Record<string, string> = {
  Critical: "#dc2626",
  High: "#ea580c",
  Medium: "#d97706",
  Low: "#2563eb",
};
const chartColors = ["#dc2626", "#ea580c", "#d97706", "#2563eb", "#7c3aed"];

function fmtNumber(value: number | undefined) {
  return new Intl.NumberFormat("en-US").format(value || 0);
}

function fmtDate(value?: string | null, includeTime = true) {
  if (!value) return "—";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    ...(includeTime
      ? { hour: "2-digit", minute: "2-digit", hour12: false }
      : {}),
  }).format(parsed);
}

function humanize(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function SeverityBadge({ severity }: { severity: string }) {
  return (
    <span className={`severity severity-${severity.toLowerCase()}`}>
      <span />
      {severity}
    </span>
  );
}

function Score({ value, compact = false }: { value: number; compact?: boolean }) {
  const tone = value >= 85 ? "critical" : value >= 65 ? "high" : value >= 40 ? "medium" : "low";
  return (
    <span className={`score score-${tone} ${compact ? "score-compact" : ""}`}>
      {Math.round(value)}
      {!compact && <small>/100</small>}
    </span>
  );
}

function ErrorState({
  title = "Unable to load this view",
  error,
  onRetry,
}: {
  title?: string;
  error: unknown;
  onRetry?: () => void;
}) {
  return (
    <div className="state-card error-state">
      <XCircle size={28} />
      <h2>{title}</h2>
      <p>{error instanceof Error ? error.message : "An unexpected error occurred."}</p>
      {onRetry && (
        <button className="button button-secondary" onClick={onRetry}>
          <RefreshCw size={16} /> Retry
        </button>
      )}
    </div>
  );
}

function LoadingState({ label = "Loading analysis…" }: { label?: string }) {
  return (
    <div className="loading-state" role="status">
      <span className="spinner" />
      <p>{label}</p>
    </div>
  );
}

function StartPage() {
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [imported, setImported] = useState<ImportResult | null>(null);
  const latest = useQuery({ queryKey: ["latest-run"], queryFn: api.latestRun });
  const upload = useMutation({
    mutationFn: api.uploadCsv,
    onSuccess: setImported,
  });
  const start = useMutation({
    mutationFn: ({ source, importId }: { source: "generated" | "csv"; importId?: string }) =>
      api.startRun(source, importId),
    onSuccess: (run) => navigate(`/runs/${run.run_id}`),
  });

  function useFile(file?: File) {
    if (file) {
      setImported(null);
      upload.mutate(file);
    }
  }

  return (
    <main className="launch-page">
      <header className="launch-header">
        <Brand />
        <div className="system-status">
          <span className="status-dot" />
          Analysis engine ready
        </div>
      </header>
      <section className="hero">
        <div className="eyebrow">
          <Sparkles size={15} /> Behavioral threat intelligence
        </div>
        <h1>See the behavior behind every security signal.</h1>
        <p>
          CADENCE combines sequence-aware detection, entity baselines, explainable
          risk, and MITRE ATT&amp;CK context in one analyst-ready workspace.
        </p>
      </section>

      <section className="source-grid">
        <article className="source-card featured">
          <div className="source-icon"><Database /></div>
          <div>
            <span className="card-kicker">Fastest path</span>
            <h2>Explore synthetic telemetry</h2>
            <p>
              Generate 2,500 reproducible enterprise events with controlled attack
              patterns and run the full model pipeline.
            </p>
          </div>
          <div className="source-meta">
            <span><CheckCircle2 size={15} /> 7 attack categories</span>
            <span><CheckCircle2 size={15} /> Evaluation labels included</span>
          </div>
          <button
            className="button button-primary button-wide"
            disabled={start.isPending}
            onClick={() => start.mutate({ source: "generated" })}
          >
            {start.isPending ? <span className="spinner small" /> : <Play size={17} />}
            Generate and analyze
          </button>
        </article>

        <article className="source-card">
          <div className="source-icon source-icon-indigo"><FileUp /></div>
          <div>
            <span className="card-kicker">Your data</span>
            <h2>Upload behavioral telemetry</h2>
            <p>
              Run inference without labels, or include labels for evaluation metrics only.
            </p>
          </div>
          <div
            className={`drop-zone ${dragging ? "dragging" : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              useFile(event.dataTransfer.files[0]);
            }}
            onClick={() => inputRef.current?.click()}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".csv,text/csv"
              hidden
              onChange={(event) => useFile(event.target.files?.[0])}
            />
            {upload.isPending ? (
              <>
                <span className="spinner" />
                <strong>Validating dataset…</strong>
              </>
            ) : (
              <>
                <UploadCloud size={26} />
                <strong>Drop a CSV here or browse</strong>
                <small>UTF-8 · up to 10 MB · 25,000 rows</small>
              </>
            )}
          </div>
          <a className="template-link" href="/api/imports/template">
            <Download size={15} /> Download CSV template
          </a>
        </article>
      </section>

      {(upload.error || start.error) && (
        <div className="inline-error">
          <XCircle size={18} />
          {(upload.error || start.error)?.message}
        </div>
      )}
      {imported && (
        <ImportReview
          result={imported}
          onClose={() => setImported(null)}
          onAnalyze={() =>
            start.mutate({ source: "csv", importId: imported.import_id })
          }
          pending={start.isPending}
        />
      )}
      {latest.data && (
        <section className="recent-run">
          <div>
            <span className="card-kicker">Recent workspace</span>
            <strong>
              {latest.data.filename || "Synthetic telemetry"} ·{" "}
              {humanize(latest.data.status)}
            </strong>
            <small>Started {fmtDate(latest.data.created_at)}</small>
          </div>
          <button
            className="button button-secondary"
            onClick={() => navigate(`/runs/${latest.data!.run_id}`)}
          >
            Open workspace <ChevronRight size={16} />
          </button>
        </section>
      )}
      <footer className="launch-footer">
        <span>CADENCE Security Analytics</span>
        <span>Local analysis · Data stays on this machine</span>
      </footer>
    </main>
  );
}

function ImportReview({
  result,
  onClose,
  onAnalyze,
  pending,
}: {
  result: ImportResult;
  onClose: () => void;
  onAnalyze: () => void;
  pending: boolean;
}) {
  const summary = result.summary;
  return (
    <div className="modal-backdrop">
      <section className="import-modal" role="dialog" aria-modal="true">
        <header>
          <div className={`validation-icon ${result.valid ? "valid" : "invalid"}`}>
            {result.valid ? <FileCheck2 /> : <AlertTriangle />}
          </div>
          <div>
            <span className="card-kicker">Dataset review</span>
            <h2>{result.filename}</h2>
            <p>
              {result.valid
                ? "Schema and sequence feasibility checks passed."
                : "Resolve the blocking validation issues before analysis."}
            </p>
          </div>
          <button className="icon-button" onClick={onClose} aria-label="Close">
            <X />
          </button>
        </header>

        {result.valid && (
          <div className="import-stats">
            <MetricMini label="Rows" value={fmtNumber(summary.rows)} />
            <MetricMini label="Entities" value={fmtNumber(summary.entities)} />
            <MetricMini label="Labels" value={summary.labels_present ? "Evaluation enabled" : "Inference only"} />
            <MetricMini label="Anomaly rate" value={summary.anomaly_rate == null ? "—" : `${summary.anomaly_rate}%`} />
          </div>
        )}

        {(result.errors.length > 0 || result.warnings.length > 0) && (
          <div className="issue-list">
            {result.errors.map((issue, index) => (
              <div className="issue error" key={`error-${index}`}>
                <XCircle size={16} />
                <span>
                  {issue.row && <strong>Row {issue.row} · </strong>}
                  {issue.column && <code>{issue.column}</code>} {issue.reason}
                </span>
              </div>
            ))}
            {result.warnings.map((issue, index) => (
              <div className="issue warning" key={`warning-${index}`}>
                <AlertTriangle size={16} />
                <span>{issue.reason}</span>
              </div>
            ))}
          </div>
        )}

        {result.preview.length > 0 && (
          <div className="preview-wrap">
            <div className="section-heading compact">
              <h3>Row preview</h3>
              <span>First {result.preview.length} records</span>
            </div>
            <div className="table-scroll">
              <table className="data-table compact-table">
                <thead>
                  <tr>
                    {Object.keys(result.preview[0]).slice(0, 7).map((key) => (
                      <th key={key}>{humanize(key)}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.preview.map((row, index) => (
                    <tr key={index}>
                      {Object.keys(result.preview[0]).slice(0, 7).map((key) => (
                        <td key={key}>{String(row[key] ?? "—")}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        <footer>
          <button className="button button-secondary" onClick={onClose}>
            Choose another file
          </button>
          <button
            className="button button-primary"
            disabled={!result.valid || pending}
            onClick={onAnalyze}
          >
            {pending ? <span className="spinner small" /> : <BrainCircuit size={17} />}
            Analyze dataset
          </button>
        </footer>
      </section>
    </div>
  );
}

function MetricMini({ label, value }: { label: string; value: string }) {
  return <div><span>{label}</span><strong>{value}</strong></div>;
}

function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`brand ${compact ? "brand-compact" : ""}`}>
      <span className="brand-mark"><Radar /></span>
      <div>
        <strong>CADENCE</strong>
        {!compact && <small>Security Analytics</small>}
      </div>
    </div>
  );
}

function RunWorkspace() {
  const { runId = "" } = useParams();
  const run = useQuery({
    queryKey: ["run", runId],
    queryFn: () => api.getRun(runId),
    refetchInterval: (query) =>
      query.state.data?.status === "completed" || query.state.data?.status === "failed"
        ? false
        : 900,
  });
  if (run.isLoading) return <LoadingState label="Opening analysis workspace…" />;
  if (run.error) return <ErrorState error={run.error} onRetry={() => run.refetch()} />;
  if (!run.data) return <Navigate to="/" replace />;
  if (run.data.status !== "completed") return <RunProgress run={run.data} />;
  return <WorkspaceLayout run={run.data} />;
}

function RunProgress({ run }: { run: RunStatus }) {
  const navigate = useNavigate();
  const stages = [
    ["Preparing telemetry", 12],
    ["Building entity baselines", 30],
    ["Training sequence model", 55],
    ["Scoring behavioral risk", 72],
    ["Mapping incidents", 87],
    ["Generating explanations", 96],
  ];
  if (run.status === "failed") {
    return (
      <main className="progress-page">
        <Brand />
        <ErrorState title="Analysis run failed" error={new Error(run.error || "Pipeline failed.")} />
        <button className="button button-secondary" onClick={() => navigate("/")}>
          <ArrowLeft size={16} /> Back to data sources
        </button>
      </main>
    );
  }
  return (
    <main className="progress-page">
      <Brand />
      <section className="progress-card">
        <div className="analysis-orbit">
          <div className="orbit orbit-one" />
          <div className="orbit orbit-two" />
          <BrainCircuit />
        </div>
        <span className="eyebrow">Analysis in progress</span>
        <h1>Building your behavioral risk picture</h1>
        <p>
          {run.filename || "Synthetic telemetry"} · {humanize(run.stage)}
        </p>
        <div className="progress-track">
          <span style={{ width: `${run.progress}%` }} />
        </div>
        <div className="progress-value">
          <span>{run.status === "queued" ? "Queued" : "Processing locally"}</span>
          <strong>{run.progress}%</strong>
        </div>
        <div className="stage-list">
          {stages.map(([label, threshold]) => {
            const done = run.progress >= Number(threshold);
            const active =
              run.progress < Number(threshold) &&
              run.progress >= Number(threshold) - 20;
            return (
              <div className={`${done ? "done" : ""} ${active ? "active" : ""}`} key={label}>
                {done ? <CheckCircle2 /> : active ? <span className="spinner small" /> : <span />}
                {label}
              </div>
            );
          })}
        </div>
      </section>
      <p className="progress-note">
        <Shield size={15} /> You can leave this page open. The analysis continues in the API process.
      </p>
    </main>
  );
}

const navItems = [
  { to: "", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "alerts", label: "Alert queue", icon: ShieldAlert },
  { to: "telemetry", label: "Telemetry", icon: Activity },
  { to: "models", label: "Model insights", icon: BarChart3 },
  { to: "methodology", label: "Methodology", icon: FileSearch },
];

function WorkspaceLayout({ run }: { run: RunStatus }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  useEffect(() => {
    if (!mobileOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMobileOpen(false);
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [mobileOpen]);

  const toggleSidebar = () => {
    if (window.matchMedia("(max-width: 820px)").matches) {
      setMobileOpen((open) => !open);
      return;
    }
    setSidebarCollapsed((collapsed) => !collapsed);
  };

  return (
    <div className="workspace">
      <aside className={`sidebar ${sidebarCollapsed ? "collapsed" : ""} ${mobileOpen ? "open" : ""}`}>
        <div className="sidebar-top">
          <Brand />
          <button className="icon-button mobile-close" onClick={() => setMobileOpen(false)} aria-label="Close sidebar">
            <X />
          </button>
        </div>
        <nav>
          <span className="nav-label">Workspace</span>
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              end={item.end}
              to={item.to}
              onClick={() => setMobileOpen(false)}
              className={({ isActive }) => (isActive ? "active" : "")}
              data-tooltip={item.label}
              aria-label={item.label}
            >
              <item.icon />
              <span className="nav-text">{item.label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-dataset">
          <span className="nav-label">Active dataset</span>
          <div className="dataset-icon"><Database /></div>
          <strong>{run.filename || "Synthetic telemetry"}</strong>
          <small>
            {run.source === "csv" ? "Uploaded CSV" : "Generated dataset"} · Complete
          </small>
          <span className="run-id">RUN {run.run_id.toUpperCase()}</span>
        </div>
        <NavLink to="/" className="new-analysis" data-tooltip="New analysis" aria-label="New analysis">
          <RefreshCw /> <span className="nav-text">New analysis</span>
        </NavLink>
      </aside>
      <button
        className={`sidebar-backdrop ${mobileOpen ? "open" : ""}`}
        onClick={() => setMobileOpen(false)}
        aria-label="Close sidebar"
      />
      <div className={`content-shell ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
        <header className="topbar">
          <div className="topbar-leading">
            <button className="icon-button menu-button" onClick={toggleSidebar} aria-label="Toggle sidebar">
              <Menu />
            </button>
            <div className="breadcrumbs">
              <span>CADENCE</span><ChevronRight /> <strong>Analysis workspace</strong>
            </div>
          </div>
          <div className="topbar-right">
            <span className="system-status"><span className="status-dot" /> Models online</span>
            <span className="avatar">SO</span>
          </div>
        </header>
        <div className="page-content"><Outlet /></div>
      </div>
    </div>
  );
}

function PageHeader({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow: string;
  title: string;
  description: string;
  action?: React.ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        <span>{eyebrow}</span>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {action}
    </header>
  );
}

function OverviewPage() {
  const { runId = "" } = useParams();
  const query = useQuery({ queryKey: ["summary", runId], queryFn: () => api.summary(runId) });
  if (query.isLoading) return <LoadingState />;
  if (query.error || !query.data) return <ErrorState error={query.error} onRetry={() => query.refetch()} />;
  const data = query.data;
  const timelineFloor = data.risk_timeline.length
    ? Math.max(
        0,
        Math.floor(
          Math.min(...data.risk_timeline.map((point) => point.average_risk)) / 10,
        ) *
          10 -
          10,
      )
    : 0;
  return (
    <>
      <PageHeader
        eyebrow="Threat overview"
        title="Behavioral risk at a glance"
        description="Prioritized signals across entities, sessions, and detection models."
        action={<div className="time-pill"><Activity /> Analysis complete</div>}
      />
      <section className="metric-grid">
        <MetricCard icon={Activity} label="Telemetry events" value={fmtNumber(data.total_events)} detail={`${data.total_entities} unique entities`} tone="blue" />
        <MetricCard icon={ShieldAlert} label="Detected alerts" value={fmtNumber(data.total_alerts)} detail={`${data.high_risk_alerts} high or critical`} tone="red" />
        <MetricCard icon={CircleGauge} label="Ground-truth anomaly rate" value={data.anomaly_rate == null ? "—" : `${data.anomaly_rate}%`} detail={data.labels_available ? "Evaluation context only" : "No labels supplied"} tone="amber" />
        <MetricCard icon={Gauge} label="Adaptive threshold" value={data.adaptive_threshold.toFixed(3)} detail="Conformal GRU calibration" tone="indigo" />
      </section>
      <section className="metric-grid">
        <MetricCard icon={BrainCircuit} label="Average confidence" value={`${data.average_confidence}%`} detail="Across detected alerts" tone="blue" />
        <MetricCard icon={BarChart3} label="Ensemble precision" value={data.detection_precision == null ? "—" : `${data.detection_precision}%`} detail={data.labels_available ? "Against evaluation labels" : "No labels supplied"} tone="indigo" />
        <MetricCard icon={AlertTriangle} label="Observed false-positive rate" value={data.false_positive_rate == null ? "—" : `${data.false_positive_rate}%`} detail={data.labels_available ? "Across labeled normal windows" : "No labels supplied"} tone="amber" />
        <MetricCard icon={CheckCircle2} label="Models agreeing" value={`${data.model_agreement_rate}%`} detail="Share of detected alerts" tone="red" />
      </section>

      <section className="dashboard-grid">
        <article className="panel panel-wide">
          <PanelHeader title="Risk activity" description="Detected incidents and risk intensity over time" />
          <div className="chart-large">
            {data.risk_timeline.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data.risk_timeline} margin={{ top: 10, right: 16, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="riskGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3157d5" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#3157d5" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e8ebf2" />
                  <XAxis dataKey="date" tickFormatter={(v) => v.slice(5)} axisLine={false} tickLine={false} />
                  <YAxis domain={[timelineFloor, 100]} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ borderRadius: 10, borderColor: "#e2e6ef" }} />
                  <Area type="monotone" dataKey="max_risk" name="Peak risk" stroke="#3157d5" strokeWidth={2.5} fill="url(#riskGradient)" />
                  <Area type="monotone" dataKey="average_risk" name="Average risk" stroke="#8da0dd" strokeWidth={2} fill="transparent" />
                </AreaChart>
              </ResponsiveContainer>
            ) : <EmptyChart />}
          </div>
        </article>
        <article className="panel">
          <PanelHeader title="Alert severity" description="Current detection distribution" />
          <div className="donut-layout">
            <div className="chart-donut">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={data.severity_distribution} dataKey="value" innerRadius={58} outerRadius={78} paddingAngle={3}>
                    {data.severity_distribution.map((entry) => <Cell key={entry.name} fill={severityColors[entry.name]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
              <div className="donut-total"><strong>{data.total_alerts}</strong><span>Total</span></div>
            </div>
            <div className="legend-list">
              {data.severity_distribution.map((entry) => (
                <div key={entry.name}><span style={{ background: severityColors[entry.name] }} /><label>{entry.name}</label><strong>{entry.value}</strong></div>
              ))}
            </div>
          </div>
        </article>
        <article className="panel">
          <PanelHeader title="MITRE ATT&CK coverage" description="Techniques represented in alerts" />
          <div className="technique-list">
            {data.technique_distribution.map((entry, index) => (
              <div key={entry.name}>
                <span className="technique-rank">{String(index + 1).padStart(2, "0")}</span>
                <div><strong>{entry.name}</strong><small>{entry.value} detected alerts</small></div>
                <div className="technique-bar"><span style={{ width: `${Math.max(8, entry.value / Math.max(1, data.total_alerts) * 100)}%` }} /></div>
              </div>
            ))}
          </div>
        </article>
        <article className="panel panel-wide">
          <PanelHeader
            title="Highest-risk alerts"
            description="Ranked by the strongest model signal"
            action={<NavLink className="text-link" to="alerts">View all alerts <ChevronRight /></NavLink>}
          />
          <AlertsTable items={data.top_alerts} compact />
        </article>
      </section>
    </>
  );
}

function MetricCard({ icon: Icon, label, value, detail, tone }: any) {
  return (
    <article className="metric-card">
      <div className={`metric-icon ${tone}`}><Icon /></div>
      <div><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>
    </article>
  );
}

function PanelHeader({ title, description, action }: { title: string; description: string; action?: React.ReactNode }) {
  return <header className="panel-header"><div><h2>{title}</h2><p>{description}</p></div>{action}</header>;
}

function EmptyChart() {
  return <div className="empty-chart"><BarChart3 /><span>No alert data for this run</span></div>;
}

function AlertsPage() {
  const { runId = "" } = useParams();
  const [pageNumber, setPageNumber] = useState(1);
  const [search, setSearch] = useState("");
  const [severity, setSeverity] = useState("");
  const [minRisk, setMinRisk] = useState(0);
  const query = useQuery({
    queryKey: ["alerts", runId, pageNumber, search, severity, minRisk],
    queryFn: () => api.alerts(runId, { page: pageNumber, search, severity, minRisk }),
  });
  return (
    <>
      <PageHeader
        eyebrow="Investigation"
        title="Alert queue"
        description="Risk-ranked behavioral anomalies with model and MITRE context."
        action={query.data && <span className="result-count">{fmtNumber(query.data.total)} alerts</span>}
      />
      <div className="filter-bar">
        <label className="search-field"><Search /><input value={search} onChange={(e) => { setSearch(e.target.value); setPageNumber(1); }} placeholder="Search entity, IP, resource…" /></label>
        <select value={severity} onChange={(e) => { setSeverity(e.target.value); setPageNumber(1); }}>
          <option value="">All severities</option><option>Critical</option><option>High</option><option>Medium</option><option>Low</option>
        </select>
        <select value={minRisk} onChange={(e) => { setMinRisk(Number(e.target.value)); setPageNumber(1); }}>
          <option value="0">Any risk score</option><option value="40">Risk 40+</option><option value="65">Risk 65+</option><option value="85">Risk 85+</option>
        </select>
      </div>
      <article className="panel table-panel">
        {query.isLoading ? <LoadingState label="Loading alert queue…" /> : query.error ? <ErrorState error={query.error} onRetry={() => query.refetch()} /> : query.data && (
          <>
            <AlertsTable items={query.data.items} />
            <Pagination page={query.data.page} pages={query.data.pages} total={query.data.total} onChange={setPageNumber} />
          </>
        )}
      </article>
    </>
  );
}

function AlertsTable({ items, compact = false }: { items: Alert[]; compact?: boolean }) {
  return (
    <div className="table-scroll">
      <table className="data-table alert-table">
        <thead><tr><th>Risk</th><th>Entity</th><th>Predicted category</th><th>Context</th><th>Models</th><th>Detected</th><th /></tr></thead>
        <tbody>
          {items.map((alert) => (
            <tr key={alert.alert_id}>
              <td><Score value={alert.combined_risk_score} compact /></td>
              <td><NavLink className="entity-link" to={`../entities/${alert.entity_id}`}>{alert.entity_id}</NavLink><small>{alert.source_ip || "Unknown IP"}</small></td>
              <td><SeverityBadge severity={alert.severity} /><strong className="table-primary">{alert.predicted_attack_category}</strong><small>{alert.prediction_confidence.toFixed(1)}% confidence · {alert.technique_id}</small></td>
              <td><span className="resource-cell" title={alert.resource_accessed}>{alert.resource_accessed}</span><small>{alert.auth_method}</small></td>
              <td><div className="model-pills"><span className={alert.gru_prediction ? "active" : ""}>GRU</span><span className={alert.if_prediction ? "active" : ""}>IF</span></div></td>
              <td>{fmtDate(alert.window_end)}<small>{alert.model_agreement ? "Models agree" : "Single-model signal"}</small></td>
              <td><NavLink className="row-action" to={`../alerts/${alert.alert_id}`} aria-label="Open alert"><ChevronRight /></NavLink></td>
            </tr>
          ))}
          {!items.length && <tr><td colSpan={7}><div className="empty-table"><Shield /><strong>No alerts match these filters</strong><span>Adjust the search or risk criteria.</span></div></td></tr>}
        </tbody>
      </table>
    </div>
  );
}

function Pagination({ page, pages, total, onChange }: { page: number; pages: number; total: number; onChange: (value: number) => void }) {
  return (
    <div className="pagination"><span>Page {page} of {pages} · {fmtNumber(total)} results</span><div><button disabled={page <= 1} onClick={() => onChange(page - 1)}><ChevronLeft /></button><button disabled={page >= pages} onClick={() => onChange(page + 1)}><ChevronRight /></button></div></div>
  );
}

function AlertDetailPage() {
  const { runId = "", alertId = "" } = useParams();
  const navigate = useNavigate();
  const query = useQuery({ queryKey: ["alert", runId, alertId], queryFn: () => api.alert(runId, alertId) });
  if (query.isLoading) return <LoadingState label="Loading incident context…" />;
  if (query.error || !query.data) return <ErrorState error={query.error} onRetry={() => query.refetch()} />;
  const alert = query.data;
  return (
    <>
      <button className="back-link" onClick={() => navigate(-1)}><ArrowLeft /> Back to alert queue</button>
      <header className="incident-header">
        <div>
          <div className="incident-label"><SeverityBadge severity={alert.severity} /><span>{alert.technique_id}</span><span>{alert.alert_id}</span></div>
          <h1>{alert.predicted_attack_category} behavior detected</h1>
          <p>Sequence deviation associated with <NavLink to={`../../entities/${alert.entity_id}`}>{alert.entity_id}</NavLink> between {fmtDate(alert.window_start)} and {fmtDate(alert.window_end)}.</p>
        </div>
        <RiskGauge value={alert.combined_risk_score} />
      </header>
      <section className="detail-grid">
        <article className="panel detail-main">
          <PanelHeader title="Detection narrative" description="Why this activity was surfaced for review" />
          <div className="narrative-callout"><ShieldAlert /><p>{alert.soc_recommendation}</p></div>
          <div className="context-grid">
            <ContextItem icon={UserRoundSearch} label="Entity" value={alert.entity_id} />
            <ContextItem icon={Network} label="Source IP" value={alert.source_ip || "Unknown"} />
            <ContextItem icon={Database} label="Resource" value={alert.resource_accessed} />
            <ContextItem icon={Fingerprint} label="Device" value={alert.device_fingerprint || "Unknown"} />
            <ContextItem icon={Gauge} label="Prediction confidence" value={`${alert.prediction_confidence.toFixed(1)}%`} />
            <ContextItem icon={Shield} label="MITRE ATT&CK" value={`${alert.technique_id} · ${alert.technique_name}`} />
          </div>
          <h3 className="subheading">Contributing factors</h3>
          {alert.explanation?.contributors?.length ? (
            <div className="factor-list">
              {alert.explanation.contributors.map((factor) => (
                <div key={factor.feature}><div><strong>{humanize(factor.feature)}</strong><small>{factor.value >= 0 ? "Increased" : "Reduced"} anomaly score</small></div><div className="factor-track"><span className={factor.value < 0 ? "negative" : ""} style={{ width: `${Math.min(100, Math.abs(factor.value) * 60 + 8)}%` }} /></div><code>{factor.value.toFixed(3)}</code></div>
              ))}
            </div>
          ) : (
            <div className="factor-chips">
              <span className={alert.new_device_indicator ? "active" : ""}><Fingerprint /> New device {alert.new_device_indicator ? "observed" : "not observed"}</span>
              <span className={alert.ip_subnet_change ? "active" : ""}><Network /> Subnet {alert.ip_subnet_change ? "changed" : "consistent"}</span>
              <span><Activity /> Login frequency {alert.login_frequency.toFixed(1)}</span>
            </div>
          )}
        </article>
        <aside className="detail-side">
          <article className="panel">
            <PanelHeader title="Model signals" description={alert.model_agreement ? "Both detectors agree" : "Single-detector signal"} />
            <ModelSignal name="Conformal GRU" value={alert.gru_risk_score} active={alert.gru_prediction} />
            <ModelSignal name="Isolation Forest" value={alert.if_risk_score} active={alert.if_prediction} />
          </article>
          <article className="panel network-card">
            <PanelHeader title="Network intelligence" description="Offline source-network context" />
            <div className="trust-score"><Network /><div><strong>{alert.network?.network_trust_score == null ? "—" : `${Math.round(alert.network.network_trust_score)}%`}</strong><span>Network trust</span></div></div>
            <dl><div><dt>Type</dt><dd>{alert.network?.network_type || "Unknown"}</dd></div><div><dt>ISP</dt><dd>{alert.network?.isp || "Unknown"}</dd></div><div><dt>ASN</dt><dd>{alert.network?.asn || "Unknown"}</dd></div></dl>
            <div className="reason-tags">{alert.network?.reasons?.map((reason: string) => <span key={reason}>{reason}</span>)}</div>
          </article>
        </aside>
        <article className="panel full-span">
          <PanelHeader title="Entity activity timeline" description="Events immediately surrounding the detection window" action={<NavLink className="text-link" to={`../../entities/${alert.entity_id}`}>View full entity history <ChevronRight /></NavLink>} />
          <TelemetryTable items={alert.surrounding_events || []} compact />
        </article>
      </section>
    </>
  );
}

function RiskGauge({ value }: { value: number }) {
  const color = value >= 85 ? "#dc2626" : value >= 65 ? "#ea580c" : "#d97706";
  return <div className="risk-gauge" style={{ background: `conic-gradient(${color} ${value * 3.6}deg, #e9edf4 0deg)` }}><div><strong>{Math.round(value)}</strong><span>Risk score</span></div></div>;
}

function ContextItem({ icon: Icon, label, value }: any) {
  return <div className="context-item"><Icon /><div><span>{label}</span><strong title={value}>{value}</strong></div></div>;
}

function ModelSignal({ name, value, active }: { name: string; value: number; active: boolean }) {
  return <div className="model-signal"><header><div><strong>{name}</strong><span className={active ? "signal-active" : "signal-clear"}>{active ? "Flagged" : "Clear"}</span></div><b>{Math.round(value)}</b></header><div><span style={{ width: `${value}%` }} /></div></div>;
}

function EntityPage() {
  const { runId = "", entityId = "" } = useParams();
  const query = useQuery({ queryKey: ["entity", runId, entityId], queryFn: () => api.entity(runId, entityId) });
  if (query.isLoading) return <LoadingState label="Building entity history…" />;
  if (query.error || !query.data) return <ErrorState error={query.error} onRetry={() => query.refetch()} />;
  const data = query.data;
  return (
    <>
      <PageHeader eyebrow="Entity investigation" title={data.entity_id} description={`${humanize(data.entity_type)} · Behavioral profile and complete activity history`} action={<Score value={data.max_risk} />} />
      <section className="metric-grid entity-metrics">
        <MetricCard icon={Activity} label="Observed events" value={fmtNumber(data.event_count)} detail="Across this dataset" tone="blue" />
        <MetricCard icon={ShieldAlert} label="Associated alerts" value={fmtNumber(data.alert_count)} detail="Behavioral detections" tone="red" />
        <MetricCard icon={Gauge} label="Preferred login hour" value={data.profile.preferred_login_hour == null ? "—" : `${Math.round(data.profile.preferred_login_hour)}:00`} detail="Learned baseline" tone="indigo" />
        <MetricCard icon={CircleGauge} label="Typical session" value={data.profile.typical_session_duration == null ? "—" : `${Math.round(data.profile.typical_session_duration)} min`} detail="Median profile duration" tone="amber" />
      </section>
      <section className="dashboard-grid">
        <article className="panel">
          <PanelHeader title="Behavior baseline" description="Learned context for this entity" />
          <dl className="profile-list"><div><dt>Common subnet</dt><dd>{data.profile.common_ip_subnet || "Unknown"}</dd></div><div><dt>Entity class</dt><dd>{humanize(data.entity_type)}</dd></div><div><dt>Resources observed</dt><dd>{data.resources.length}</dd></div><div><dt>Authentication methods</dt><dd>{data.auth_methods.length}</dd></div></dl>
        </article>
        <article className="panel panel-wide">
          <PanelHeader title="Most accessed resources" description="Frequency within the active dataset" />
          <div className="chart-medium"><ResponsiveContainer width="100%" height="100%"><BarChart data={data.resources.slice(0, 6)} layout="vertical" margin={{ left: 10, right: 20 }}><CartesianGrid strokeDasharray="4 4" horizontal={false} stroke="#e8ebf2" /><XAxis type="number" axisLine={false} tickLine={false} /><YAxis type="category" dataKey="name" width={150} tickFormatter={(v) => v.length > 22 ? `${v.slice(0, 22)}…` : v} axisLine={false} tickLine={false} /><Tooltip /><Bar dataKey="count" fill="#3157d5" radius={[0, 6, 6, 0]} /></BarChart></ResponsiveContainer></div>
        </article>
        <article className="panel full-span">
          <PanelHeader title="Entity event history" description="Chronological access and authentication activity" />
          <TelemetryTable items={data.events} />
        </article>
      </section>
    </>
  );
}

function TelemetryPage() {
  const { runId = "" } = useParams();
  const [pageNumber, setPageNumber] = useState(1);
  const [search, setSearch] = useState("");
  const [evaluation, setEvaluation] = useState(false);
  const query = useQuery({ queryKey: ["telemetry", runId, pageNumber, search, evaluation], queryFn: () => api.telemetry(runId, { page: pageNumber, search, evaluation }) });
  return (
    <>
      <PageHeader eyebrow="Event explorer" title="Telemetry" description="Search the access-log events used by the behavioral models." action={<label className="toggle"><input type="checkbox" checked={evaluation} onChange={(e) => setEvaluation(e.target.checked)} /><span /><div><strong>Evaluation labels</strong><small>Reveal ground truth</small></div></label>} />
      {evaluation && <div className="evaluation-notice"><Info /><span><strong>Evaluation context enabled.</strong> Ground-truth labels are shown for model assessment only and are never used as analyst evidence.</span></div>}
      <div className="filter-bar"><label className="search-field"><Search /><input value={search} onChange={(e) => { setSearch(e.target.value); setPageNumber(1); }} placeholder="Search entity, source IP, resource, command…" /></label></div>
      <article className="panel table-panel">
        {query.isLoading ? <LoadingState label="Loading telemetry…" /> : query.error ? <ErrorState error={query.error} onRetry={() => query.refetch()} /> : query.data && <><TelemetryTable items={query.data.items} showLabel={evaluation} /><Pagination page={query.data.page} pages={query.data.pages} total={query.data.total} onChange={setPageNumber} /></>}
      </article>
    </>
  );
}

function TelemetryTable({ items, showLabel = false, compact = false }: { items: TelemetryEvent[]; showLabel?: boolean; compact?: boolean }) {
  return (
    <div className="table-scroll"><table className={`data-table telemetry-table ${compact ? "compact-table" : ""}`}><thead><tr><th>Timestamp</th><th>Entity</th><th>Source</th><th>Resource</th><th>Authentication</th><th>Session</th>{showLabel && <th>Ground truth</th>}</tr></thead><tbody>
      {items.map((event) => <tr key={event.event_index}><td>{fmtDate(event.timestamp)}</td><td><strong className="table-primary">{event.entity_id}</strong><small>{humanize(event.entity_type)}</small></td><td><code>{event.source_ip}</code><small title={event.device_fingerprint}>{event.geo_location}</small></td><td><span className="resource-cell" title={event.resource_accessed}>{event.resource_accessed}</span><small title={event.command_sequence}>{event.command_sequence}</small></td><td>{humanize(event.auth_method)}</td><td>{event.session_duration} min</td>{showLabel && <td>{event.label?.toLowerCase() === "normal" || event.label?.toLowerCase() === "benign" ? <span className="label-normal">{event.label}</span> : <span className="label-anomaly">{event.label}</span>}</td>}</tr>)}
      {!items.length && <tr><td colSpan={showLabel ? 7 : 6}><div className="empty-table"><Activity /><strong>No telemetry found</strong></div></td></tr>}
    </tbody></table></div>
  );
}

function ModelsPage() {
  const { runId = "" } = useParams();
  const query = useQuery({ queryKey: ["models", runId], queryFn: () => api.models(runId) });
  if (query.isLoading) return <LoadingState label="Loading model evaluation…" />;
  if (query.error || !query.data) return <ErrorState error={query.error} onRetry={() => query.refetch()} />;
  const data = query.data;
  const chartData = Object.entries(data.metrics).map(([name, metric]) => ({ name: name.includes("GRU") ? "Conformal GRU" : "Isolation Forest", Precision: Math.round(metric.precision * 100), Recall: Math.round(metric.recall * 100), "F1 score": Math.round(metric.f1 * 100), "PR-AUC": Math.round(metric.pr_auc * 100) }));
  return (
    <>
      <PageHeader eyebrow="Evaluation laboratory" title="Model insights" description="Detection quality is reported when optional evaluation labels are available." action={<div className="time-pill"><BrainCircuit /> Two-model ensemble</div>} />
      <div className="evaluation-notice"><Info /><span>{data.labels_available ? "These metrics use retained ground-truth labels only after inference. Labels do not influence training, category prediction, or alert explanations." : "No labels were supplied. Inference results are available, while evaluation metrics are intentionally omitted."}</span></div>
      <section className="model-grid">
        {Object.entries(data.metrics).map(([name, metric], index) => <article className="model-card" key={name}><header><div className={`model-icon model-${index}`}><BrainCircuit /></div><div><span>Detection model</span><h2>{name.includes("GRU") ? "Conformal GRU Autoencoder" : "Isolation Forest"}</h2></div></header><div className="model-metrics"><MetricMini label="Precision" value={`${(metric.precision * 100).toFixed(1)}%`} /><MetricMini label="Recall" value={`${(metric.recall * 100).toFixed(1)}%`} /><MetricMini label="F1 score" value={`${(metric.f1 * 100).toFixed(1)}%`} /><MetricMini label="PR-AUC" value={`${(metric.pr_auc * 100).toFixed(1)}%`} /></div><div className="fpr-row"><span>False-positive rate</span><strong>{(metric.false_positive_rate * 100).toFixed(2)}%</strong></div></article>)}
      </section>
      <section className="dashboard-grid">
        <article className="panel panel-wide"><PanelHeader title="Model quality comparison" description="Percentage scores across imbalanced-data metrics" /><div className="chart-large"><ResponsiveContainer width="100%" height="100%"><BarChart data={chartData}><CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e8ebf2" /><XAxis dataKey="name" axisLine={false} tickLine={false} /><YAxis domain={[0, 100]} axisLine={false} tickLine={false} /><Tooltip /><Legend /><Bar dataKey="Precision" fill="#3157d5" radius={[5, 5, 0, 0]} /><Bar dataKey="Recall" fill="#7c3aed" radius={[5, 5, 0, 0]} /><Bar dataKey="F1 score" fill="#0f9f7f" radius={[5, 5, 0, 0]} /><Bar dataKey="PR-AUC" fill="#e59a18" radius={[5, 5, 0, 0]} /></BarChart></ResponsiveContainer></div></article>
        <article className="panel"><PanelHeader title="Explainability status" description="Feature attribution coverage" /><div className={`shap-status ${data.shap.available ? "available" : "unavailable"}`}>{data.shap.available ? <CheckCircle2 /> : <AlertTriangle />}<div><strong>{data.shap.available ? "SHAP available" : "SHAP unavailable"}</strong><span>{data.shap.available ? `${data.shap.explained_incidents} high-scoring events explained` : data.shap.reason || "Fallback behavioral factors are used."}</span></div></div><dl className="profile-list"><div><dt>GRU threshold</dt><dd>{data.adaptive_threshold.toFixed(5)}</dd></div><div><dt>Calibration target</dt><dd>1% FPR</dd></div><div><dt>Risk scale</dt><dd>0–100</dd></div></dl></article>
        <article className="panel full-span"><PanelHeader title="Confusion matrices" description="Predicted versus actual sequence outcomes" /><div className="confusion-wrap">{Object.entries(data.metrics).map(([name, metric]) => <ConfusionMatrix key={name} name={name.includes("GRU") ? "Conformal GRU" : "Isolation Forest"} matrix={metric.confusion_matrix} />)}</div></article>
      </section>
    </>
  );
}

function ConfusionMatrix({ name, matrix }: { name: string; matrix: { tn: number; fp: number; fn: number; tp: number } }) {
  return <div className="confusion"><h3>{name}</h3><div className="matrix-label top">Predicted</div><div className="matrix-label side">Actual</div><div className="matrix-grid"><div className="correct"><strong>{matrix.tn}</strong><span>True negative</span></div><div><strong>{matrix.fp}</strong><span>False positive</span></div><div><strong>{matrix.fn}</strong><span>False negative</span></div><div className="correct"><strong>{matrix.tp}</strong><span>True positive</span></div></div></div>;
}

function MethodologyPage() {
  const steps = [
    { icon: Database, title: "Behavioral telemetry", text: "Access events describe entities, source networks, resources, authentication, session duration, commands, and device identity." },
    { icon: UserRoundSearch, title: "Adaptive profiles", text: "Per-entity baselines learn preferred login time, typical sessions, common subnets, familiar resources, and device continuity." },
    { icon: BrainCircuit, title: "Dual detection", text: "A conformal GRU examines five-event sequences while Isolation Forest evaluates event-level behavioral deviations." },
    { icon: ShieldAlert, title: "Analyst context", text: "Signals become risk-ranked alerts with MITRE ATT&CK techniques, recommended actions, and feature-level explanations." },
  ];
  return (
    <>
      <PageHeader eyebrow="System methodology" title="How CADENCE reaches a decision" description="The assumptions, data flow, and interpretation boundaries behind the dashboard." />
      <section className="method-flow">{steps.map((step, index) => <article key={step.title}><span className="step-number">0{index + 1}</span><div className="method-icon"><step.icon /></div><h2>{step.title}</h2><p>{step.text}</p></article>)}</section>
      <section className="dashboard-grid methodology-grid">
        <article className="panel panel-wide"><PanelHeader title="Behavior taxonomy" description="Categories predicted from behavioral context without reading evaluation labels" /><div className="taxonomy-grid">{["Brute Force","Impossible Travel","Credential Stuffing","Lateral Movement","Device Spoofing","Low-and-Slow Exfiltration","Insider Drift"].map((name) => <div key={name}><ShieldAlert /><strong>{name}</strong><span>{name === "Insider Drift" ? "Edge case" : "Anomaly"}</span></div>)}</div></article>
        <article className="panel"><PanelHeader title="Interpretation guardrails" description="What analysts should keep in mind" /><ul className="guardrail-list"><li>Risk is a prioritization signal, not proof of compromise.</li><li>Ground-truth labels appear only in evaluation views.</li><li>SHAP covers the highest-scoring Isolation Forest events.</li><li>Offline network metadata may return unknown context.</li><li>Models retrain for every local analysis run.</li></ul></article>
        <article className="panel full-span"><PanelHeader title="CSV requirements" description="Official Honeywell schema validated before model work begins" /><div className="schema-chips">{["entity_id","entity_type","timestamp","source_ip","geo_location","resource_accessed","auth_method","session_duration","command_sequence","device_fingerprint","label (optional)"].map((field) => <code key={field}>{field}</code>)}</div><p className="schema-note">Entity type must be user, service_account, or edge_device. Labels are optional and used only after inference for evaluation. CSV files must include at least three five-event entity windows.</p></article>
      </section>
    </>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<StartPage />} />
      <Route path="/runs/:runId" element={<RunWorkspace />}>
        <Route index element={<OverviewPage />} />
        <Route path="alerts" element={<AlertsPage />} />
        <Route path="alerts/:alertId" element={<AlertDetailPage />} />
        <Route path="entities/:entityId" element={<EntityPage />} />
        <Route path="telemetry" element={<TelemetryPage />} />
        <Route path="models" element={<ModelsPage />} />
        <Route path="methodology" element={<MethodologyPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
