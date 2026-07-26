import type {
  Alert,
  EntityDetail,
  ImportResult,
  ModelInsights,
  Paginated,
  RunStatus,
  Summary,
  TelemetryEvent,
} from "./types";

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch {
      // Use the HTTP status when a body is not JSON.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export const api = {
  latestRun: () => request<RunStatus | null>("/api/runs/latest"),
  getRun: (runId: string) => request<RunStatus>(`/api/runs/${runId}`),
  startRun: (source: "generated" | "csv", importId?: string) =>
    request<RunStatus>("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source, import_id: importId }),
    }),
  uploadCsv: async (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<ImportResult>("/api/imports/csv", {
      method: "POST",
      body,
    });
  },
  summary: (runId: string) =>
    request<Summary>(`/api/runs/${runId}/summary`),
  alerts: (
    runId: string,
    filters: {
      page?: number;
      search?: string;
      severity?: string;
      minRisk?: number;
    },
  ) => {
    const params = new URLSearchParams();
    params.set("page", String(filters.page || 1));
    params.set("page_size", "20");
    if (filters.search) params.set("search", filters.search);
    if (filters.severity) params.set("severity", filters.severity);
    if (filters.minRisk) params.set("min_risk", String(filters.minRisk));
    return request<Paginated<Alert>>(
      `/api/runs/${runId}/alerts?${params.toString()}`,
    );
  },
  alert: (runId: string, alertId: string) =>
    request<Alert>(`/api/runs/${runId}/alerts/${alertId}`),
  entity: (runId: string, entityId: string) =>
    request<EntityDetail>(`/api/runs/${runId}/entities/${entityId}`),
  telemetry: (
    runId: string,
    filters: { page?: number; search?: string; evaluation?: boolean },
  ) => {
    const params = new URLSearchParams({
      page: String(filters.page || 1),
      page_size: "25",
      evaluation: String(Boolean(filters.evaluation)),
    });
    if (filters.search) params.set("search", filters.search);
    return request<Paginated<TelemetryEvent>>(
      `/api/runs/${runId}/telemetry?${params.toString()}`,
    );
  },
  models: (runId: string) =>
    request<ModelInsights>(`/api/runs/${runId}/models`),
};
