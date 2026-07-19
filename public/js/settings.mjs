export const BRAIN_MODELS = {
  codex: [
    { value: "gpt-5.6-sol", label: "GPT-5.6 Sol" },
    { value: "gpt-5.6-terra", label: "GPT-5.6 Terra" },
    { value: "gpt-5.6-luna", label: "GPT-5.6 Luna" },
    { value: "gpt-5.5", label: "GPT-5.5" },
    { value: "gpt-5.5-pro", label: "GPT-5.5 Pro" },
    { value: "gpt-5.4", label: "GPT-5.4" },
    { value: "gpt-5.4-pro", label: "GPT-5.4 Pro" },
    { value: "gpt-5.4-mini", label: "GPT-5.4 Mini" },
    { value: "gpt-5.4-nano", label: "GPT-5.4 Nano" },
  ],
  claude: [
    { value: "claude-fable-5", label: "Claude Fable 5" },
    { value: "claude-opus-4-8", label: "Claude Opus 4.8" },
    { value: "claude-sonnet-5", label: "Claude Sonnet 5" },
    { value: "claude-sonnet-4-6", label: "Claude Sonnet 4.6" },
    { value: "claude-opus-4-7", label: "Claude Opus 4.7" },
    { value: "claude-opus-4-6", label: "Claude Opus 4.6" },
    { value: "claude-haiku-4-5", label: "Claude Haiku 4.5" },
  ],
};

export function modelOptions(provider, current = "") {
  const options = [...(BRAIN_MODELS[provider] || [])];
  if (current && !options.some((option) => option.value === current)) {
    options.unshift({ value: current, label: `${current} (saved)` });
  }
  return options;
}

export function modelLabel(provider, model) {
  return modelOptions(provider, model).find((option) => option.value === model)?.label || model;
}

export function providerOptions(brains) {
  const labels = { codex: "ChatGPT", claude: "Claude" };
  return brains
    .filter((brain) => brain.enabled && brain.linked)
    .map((brain) => ({
      value: brain.provider,
      label: `${labels[brain.provider] || brain.provider} · ${modelLabel(brain.provider, brain.model)}`,
    }));
}
