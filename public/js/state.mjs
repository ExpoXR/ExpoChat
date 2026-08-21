export function readPreferences(storage) {
  try {
    const context = JSON.parse(storage.getItem("ollma_context") || "[]");
    return {
      model: storage.getItem("ollma_model") || "",
      target: storage.getItem("ollma_target") || "",
      file: storage.getItem("ollma_file") || "",
      chat: storage.getItem("ollma_chat") || "",
      context: Array.isArray(context) ? context.filter((path) => typeof path === "string").slice(-20) : [],
    };
  } catch (_) {
    return {};
  }
}

export function writePreferences(storage, preferences) {
  storage.setItem("ollma_model", preferences.model || "");
  storage.setItem("ollma_target", preferences.target || "");
  storage.setItem("ollma_file", preferences.file || "");
  storage.setItem("ollma_chat", preferences.chat || "");
  storage.setItem("ollma_context", JSON.stringify(preferences.context || []));
}
