from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"


def test_expoxr_workspace_contract_and_assets():
    html = (PUBLIC / "index.html").read_text()
    script = (PUBLIC / "app.js").read_text()
    tokens = (PUBLIC / "expoxr-system.css").read_text()

    assert 'data-expo-product="expochat"' in html
    assert 'data-expo-profile="workspace"' in html
    assert 'data-theme="dark"' in html
    assert 'data-theme-preference="dark"' in html
    assert 'href="/expoxr-system.css"' in html
    assert 'href="/favicon.svg"' in html
    assert '[data-expo-product="expochat"]' in tokens
    assert "dataset.themePreference" in script

    for name in ("manrope-variable.woff2", "jetbrains-mono-variable.woff2", "LICENSE.md"):
        assert (PUBLIC / "fonts" / name).stat().st_size > 0
    assert (PUBLIC / "favicon.svg").stat().st_size > 0


def test_about_section_has_public_project_links():
    html = (PUBLIC / "index.html").read_text()
    assert "ExpoChat is designed and built by Ayal Othman at ExpoXR." in html
    for url in (
        "https://expoxr.com/",
        "https://expoxr.com/about/",
        "https://github.com/ExpoXR",
        "https://github.com/ExpoXR/ExpoChat",
    ):
        assert f'href="{url}" target="_blank" rel="noopener noreferrer"' in html
    assert 'href="mailto:hallo@expoxr.com"' in html
    assert "MIT License" in html


def test_superseded_chrome_and_unused_styles_are_removed():
    html = (PUBLIC / "index.html").read_text()
    css = (PUBLIC / "app.css").read_text()
    assert "window-dots" not in html
    assert "command-center" not in html
    assert ".window-dots" not in css
    assert ".command-center" not in css
    assert ".plan-result-bar" not in css
