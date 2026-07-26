export type RunStatus = {
  run_id: string;
  status: "queued" | "running" | "completed" | "failed";
  source: "generated" | "csv";
  filename?: string | null;
  stage: string;
  progress: number;
  error?: string | null;
  created_at: string;
  completed_at?: string | null;
};

export type ValidationIssue = {
  row?: number | null;
  column?: string | null;
  value?: string | null;
  reason: string;
};

export type ImportResult = {
  import_id: string;
  filename: string;
  valid: boolean;
  summary: Record<string, any>;
  preview: Record<string, any>[];
  errors: ValidationIssue[];
  warnings: ValidationIssue[];
};

export type Alert = {
  alert_id: string;
  entity_id: string;
  window_start: string;
  window_end: string;
  resource_accessed: string;
  resource_category: string;
  auth_method: string;
  command_sequence: string;
  source_ip?: string;
  device_fingerprint?: string;
  severity: "Critical" | "High" | "Medium" | "Low";
  technique_id: string;
  technique_name: string;
  predicted_attack_category: string;
  prediction_confidence: number;
  soc_recommendation: string;
  combined_risk_score: number;
  gru_risk_score: number;
  if_risk_score: number;
  gru_prediction: boolean;
  if_prediction: boolean;
  model_agreement: boolean;
  new_device_indicator: boolean;
  ip_subnet_change: boolean;
  login_frequency: number;
  time_since_previous_login: number;
  network?: Record<string, any> | null;
  explanation?: {
    contributors: { feature: string; value: number }[];
  } | null;
  surrounding_events?: TelemetryEvent[];
  fallback_factors?: Record<string, any>[];
};

export type Summary = {
  total_events: number;
  total_entities: number;
  total_alerts: number;
  high_risk_alerts: number;
  labels_available: boolean;
  anomaly_rate: number | null;
  average_confidence: number;
  detection_precision: number | null;
  false_positive_rate: number | null;
  model_agreement_rate: number;
  adaptive_threshold: number;
  severity_distribution: { name: string; value: number }[];
  technique_distribution: { name: string; value: number }[];
  model_agreement: { name: string; value: number }[];
  risk_timeline: {
    date: string;
    alerts: number;
    average_risk: number;
    max_risk: number;
  }[];
  top_alerts: Alert[];
};

export type TelemetryEvent = {
  event_index: number;
  entity_id: string;
  entity_type: string;
  timestamp: string;
  source_ip: string;
  geo_location: string;
  resource_accessed: string;
  auth_method: string;
  session_duration: number;
  command_sequence: string;
  device_fingerprint: string;
  label?: string;
};

export type Paginated<T> = {
  items: T[];
  page: number;
  page_size: number;
  total: number;
  pages: number;
};

export type EntityDetail = {
  entity_id: string;
  entity_type: string;
  event_count: number;
  alert_count: number;
  max_risk: number;
  profile: Record<string, any>;
  resources: { name: string; count: number }[];
  auth_methods: { name: string; count: number }[];
  events: TelemetryEvent[];
  alerts: Alert[];
  network_history: Record<string, any>[];
};

export type ModelInsights = {
  labels_available: boolean;
  metrics: Record<
    string,
    {
      precision: number;
      recall: number;
      f1: number;
      pr_auc: number;
      false_positive_rate: number;
      confusion_matrix: { tn: number; fp: number; fn: number; tp: number };
    }
  >;
  adaptive_threshold: number;
  shap: {
    available: boolean;
    reason?: string | null;
    explained_incidents: number;
  };
};
