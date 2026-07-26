import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";

describe("API client", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("starts a generated analysis run", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          run_id: "run-1",
          status: "queued",
          source: "generated",
          stage: "queued",
          progress: 0,
          created_at: "2026-07-01T00:00:00Z",
        }),
        { status: 202, headers: { "Content-Type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await api.startRun("generated");

    expect(result.run_id).toBe("run-1");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/runs",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("surfaces API detail messages", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "CSV import was not found." }), {
          status: 404,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );

    await expect(api.startRun("csv", "missing")).rejects.toThrow(
      "CSV import was not found.",
    );
  });
});

