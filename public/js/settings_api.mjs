// Pure helpers for the Settings page: payload building + usage-meter math.

const THEMES = new Set(["dark", "light", "auto"]);

function clampInt(value, max = 1_000_000_000) {
  const n = Math.trunc(Number(value));
  if (!Number.isFinite(n) || n < 0) return 0;
  return Math.min(n, max);
}

// Build a normalized PUT /api/settings payload from raw form values.
// Only defined keys are included so partial updates are possible.
export function buildSettingsPayload(values = {}) {
  const payload = {};
  if (values.token_budget_daily !== undefined && values.token_budget_daily !== "") {
    payload.token_budget_daily = clampInt(values.token_budget_daily);
  }
  if (values.token_budget_run !== undefined && values.token_budget_run !== "") {
    payload.token_budget_run = clampInt(values.token_budget_run);
  }
  if (values.max_output_tokens !== undefined && values.max_output_tokens !== "") {
    payload.max_output_tokens = clampInt(values.max_output_tokens, 1_000_000);
  }
  if (values.theme !== undefined && THEMES.has(values.theme)) {
    payload.theme = values.theme;
  }
  if (values.agent_mode_default !== undefined) {
    payload.agent_mode_default = Boolean(values.agent_mode_default);
  }
  return payload;
}

// Given /api/usage, compute a meter for today's paid usage vs the daily cap.
export function usageMeter(usage = {}) {
  const used = Number(usage?.paid_today?.total ?? 0) || 0;
  const cap = Number(usage?.budgets?.daily ?? 0) || 0;
  if (cap <= 0) {
    return { used, cap: 0, pct: 0, unlimited: true, over: false, label: `${used.toLocaleString()} tokens today · no daily cap` };
  }
  const pct = Math.min(100, Math.round((used / cap) * 100));
  const over = used >= cap;
  return {
    used,
    cap,
    pct,
    unlimited: false,
    over,
    label: `${used.toLocaleString()} / ${cap.toLocaleString()} tokens today (${pct}%)`,
  };
}
