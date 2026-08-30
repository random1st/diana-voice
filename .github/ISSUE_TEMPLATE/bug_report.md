---
name: Bug report
about: Something broke — help us reproduce it
labels: bug
---

**What happened**

<!-- What you did, what you expected, what you got instead. -->

**Environment**

```sh
sw_vers -productVersion; uname -m
```

<!-- Paste the output above. Apple Silicon + macOS 13+ are required. -->

**Diagnostics**

```sh
tail -20 ~/Library/Application\ Support/DianaVoice/logs/runtime.log
ls -t ~/Library/Logs/DiagnosticReports/ | grep -i diana | head -3
```

<!-- Paste runtime.log tail. If DiagnosticReports lists a DianaVoice-*.ips
     file, attach it (or paste its first ~45 lines) — it names the crash. -->

**Install method**

- [ ] `brew install --cask random1st/diana-voice/diana-voice`
- [ ] `.dmg` from Releases
- [ ] built from source
