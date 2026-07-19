export function createApi(getCsrfToken, fetchImpl = globalThis.fetch) {
  return async function api(path, options = {}) {
    const method = (options.method || "GET").toUpperCase();
    const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
    const csrf = getCsrfToken();
    if (csrf && !["GET", "HEAD", "OPTIONS"].includes(method)) headers["X-CSRF-Token"] = csrf;
    const response = await fetchImpl(path, { credentials: "same-origin", headers, ...options });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || response.statusText);
    return data;
  };
}
