export function providerOptions(brains) {
  const labels = { codex: "Codex (OpenAI)", claude: "Claude (Anthropic)" };
  return brains
    .filter((brain) => brain.enabled)
    .map((brain) => ({ value: brain.provider, label: labels[brain.provider] || brain.provider }));
}
