const base = require("./playwright.config.js");
const exe = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";
base.projects = base.projects.map(p => ({...p, use: {...p.use, launchOptions: { executablePath: exe }}}));
module.exports = base;
