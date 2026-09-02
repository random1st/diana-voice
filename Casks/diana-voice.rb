cask "diana-voice" do
  version "0.1.4"
  sha256 "d5db5f60d0151ff2e8a1d332c8a458819bbba86f6aebb8d75d92c858f54be367"

  url "https://github.com/random1st/diana-voice/releases/download/v#{version}/DianaVoice-#{version}.dmg"
  name "Diana Voice"
  desc "Local push-to-talk voice layer: on-device STT/TTS with an MCP proxy"
  homepage "https://github.com/random1st/diana-voice"

  depends_on macos: :sonoma
  depends_on arch: :arm64

  app "Diana Voice.app"

  caveats <<~EOS
    Diana Voice needs two macOS permissions on first launch:

      1. Microphone — for push-to-talk speech recognition
         (System Settings > Privacy & Security > Microphone)
      2. Accessibility — for the global push-to-talk hotkey
         (System Settings > Privacy & Security > Accessibility)

    Grant both when prompted, or enable them manually in System Settings.
  EOS

  zap trash: [
    "~/Library/Application Support/DianaVoice",
  ]
end
