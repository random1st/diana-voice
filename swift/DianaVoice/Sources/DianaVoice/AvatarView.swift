import SwiftUI
import VoiceFFI

// MARK: - Mood

/// Diana Voice has four moods (the donor's six minus `.thinking`/`.happy`,
/// which had no voice-runtime event to drive them — this product only ever
/// emits "idle" / "listening" / "processing" / "speaking" on `voice-state`,
/// see SSEClient). Renamed from the donor's `.neutral` to `.idle` because
/// that's the literal value the runtime sends for the resting state.
enum AvatarMood: String, CaseIterable {
    case idle, listening, processing, speaking

    /// Pure parse from the runtime's SSE payload. The value arrives either as
    /// a JSON string (`"listening"`) or a bare word; unknown → idle. Kept here
    /// so the live client and the unit tests share one mapping.
    static func parse(_ raw: String) -> AvatarMood {
        let stripped = raw.trimmingCharacters(in: .init(charactersIn: "\" "))
        return AvatarMood(rawValue: stripped) ?? .idle
    }

    // Ring color + glow ported from the donor's Tauri overlay MOOD_STYLES (App.tsx).
    var ringColor: Color {
        switch self {
        case .idle:       return Color(red: 0.376, green: 0.647, blue: 0.980)  // 96,165,250
        case .listening:  return Color(red: 0.133, green: 0.773, blue: 0.369)  // 34,197,94
        case .processing: return Color(red: 0.133, green: 0.773, blue: 0.369)  // 34,197,94
        case .speaking:   return Color(red: 0.659, green: 0.333, blue: 0.969)  // 168,85,247
        }
    }

    // Glow radius (px) — idle subtle, active moods stronger (App.tsx 16→24px).
    var glowRadius: CGFloat {
        switch self {
        case .idle:                   return 8
        case .listening, .processing: return 11
        case .speaking:                return 12
        }
    }

    /// Pulse period in seconds for the breathing animation; nil = static.
    var pulsePeriod: Double? {
        switch self {
        case .idle:                   return 2.5
        case .listening, .processing: return 1.8
        case .speaking:                return 0.9
        }
    }
}

// MARK: - Defaults

/// Diana Voice has no settings API (no persisted `avatar_size` /
/// `bubble_font_size` — the donor's `DaemonSettingsDefaults` backed values
/// fetched from `GET /api/settings`, which this product's voice-runtime
/// doesn't serve; see mcp/server.rs). These are fixed local constants
/// mirroring the donor's defaults instead of a per-install override.
let avatarDiameterDefault: CGFloat = 96
let bubbleFontSizeDefault: CGFloat = 13

// MARK: - AvatarImageResolver

/// Single source of truth for the avatar image priority chain.
/// Priority: SSE override → custom path → bundled `diana.png` → nil (gray
/// circle fallback in AvatarView). The bundled photo is a personal asset:
/// the public build strips it at packaging time (DIANA_PUBLIC_BUILD=1 in
/// package-app.sh, same convention as the donor) and falls to the circle.
enum AvatarImageResolver {
    static let bundled: NSImage? = Bundle.module
        .url(forResource: "diana", withExtension: "png")
        .flatMap { NSImage(contentsOf: $0) }

    static func current(override avatarOverride: NSImage?, customPath: String?) -> NSImage? {
        if let img = avatarOverride { return img }
        if let path = customPath,
           !path.isEmpty,
           FileManager.default.fileExists(atPath: path),
           let nsImage = NSImage(contentsOfFile: path) { return nsImage }
        return bundled
    }
}

// MARK: - AvatarView (pure: renders a given mood + speech, no network)

struct AvatarView: View {
    var mood: AvatarMood = .idle
    var speech: String = ""
    /// Avatar circle diameter. The click-through hit region is computed from
    /// this exact value, so callers must keep it in sync with
    /// ClickThroughHostingView.avatarDiameter and the panel content size.
    var avatarDiameter: CGFloat = avatarDiameterDefault
    /// Padding below the avatar (also feeds the hit-region math).
    var bottomInset: CGFloat = 6
    /// Disable the pulse animation for deterministic offscreen snapshots.
    var animated: Bool = true
    /// Filesystem path to a custom avatar image. Used only by the offscreen
    /// snapshot path; the live overlay passes a pre-resolved `resolvedImage`
    /// instead so no disk read happens per render.
    var customAvatarPath: String? = nil
    /// Pre-loaded NSImage from an external override; preferred over path when set.
    var avatarOverride: NSImage? = nil
    /// Pre-resolved avatar image from the client cache. When set, it is used
    /// directly and the disk-reading resolver is skipped (prevents the
    /// custom↔gray-circle flip under re-render bursts).
    var resolvedImage: NSImage? = nil
    /// Font size for the speech bubble text.
    var bubbleFontSize: CGFloat = bubbleFontSizeDefault

    private let ringWidth: CGFloat = 3

    @State private var pulse = false

    var body: some View {
        VStack(spacing: 8) {
            if !speech.isEmpty {
                SpeechBubble(text: speech, fontSize: bubbleFontSize)
                    .transition(.opacity)
            }
            avatar
        }
        // Avatar anchored bottom-center; bubble grows upward above it. This keeps
        // the click target (the avatar circle) at a fixed, known position so the
        // ClickThroughHostingView radius test stays deterministic.
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
        .padding(.horizontal, 6)
        .padding(.bottom, bottomInset)
    }

    private var avatar: some View {
        let d = avatarDiameter
        let inner = d - ringWidth * 2 - 6
        return ZStack {
            avatarImage
                .frame(width: inner, height: inner)
                .clipShape(Circle())

            Circle()
                .strokeBorder(mood.ringColor, lineWidth: ringWidth)
                .frame(width: d - ringWidth, height: d - ringWidth)
                .shadow(color: mood.ringColor.opacity(0.85), radius: mood.glowRadius)
                .shadow(color: mood.ringColor.opacity(0.45), radius: mood.glowRadius * 1.6)
                .scaleEffect(pulse ? 1.04 : 1.0)
                .opacity(pulse ? 0.85 : 1.0)
        }
        .frame(width: d, height: d)
        .onAppear {
            guard animated, let period = mood.pulsePeriod else { return }
            withAnimation(.easeInOut(duration: period / 2).repeatForever(autoreverses: true)) {
                pulse = true
            }
        }
    }

    @ViewBuilder
    private var avatarImage: some View {
        // Live overlay supplies a pre-resolved (cached) image; the snapshot path
        // leaves it nil and falls back to the disk resolver. Neither resolves to
        // anything by default — no bundled avatar photo — so the common case is
        // the neutral gray circle.
        if let nsImage = resolvedImage ?? AvatarImageResolver.current(override: avatarOverride, customPath: customAvatarPath) {
            Image(nsImage: nsImage)
                .resizable()
                .aspectRatio(contentMode: .fill)
        } else {
            Circle().fill(Color(red: 0.55, green: 0.60, blue: 0.70))
        }
    }
}

// MARK: - Speech bubble (typewriter handled by caller passing partial text)

private struct SpeechBubble: View {
    let text: String
    var fontSize: CGFloat = 12

    var body: some View {
        Text(text)
            .font(.system(size: fontSize))
            .foregroundStyle(.primary)
            .lineLimit(4)
            .multilineTextAlignment(.center)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(.regularMaterial)
                    .shadow(color: .black.opacity(0.18), radius: 6, y: 2)
            )
            .frame(maxWidth: 220)
            .fixedSize(horizontal: false, vertical: true)
    }
}

// MARK: - AvatarPrefs (user-chosen avatar image)

/// The user's custom avatar: `avatar.png` in app support, set from the tray
/// ("Choose Avatar Image…"). Cached as an NSImage so the live overlay never
/// reads disk per render (the resolver comment's custom↔circle flip burst);
/// `reload()` is the only disk touch, called after the file changes.
@MainActor
final class AvatarPrefs: ObservableObject {
    static let shared = AvatarPrefs()

    @Published private(set) var customImage: NSImage?

    static var customPath: String { dataDirPath() + "/avatar.png" }

    private init() {
        reload()
    }

    func reload() {
        customImage = FileManager.default.fileExists(atPath: Self.customPath)
            ? NSImage(contentsOfFile: Self.customPath)
            : nil
    }
}

// MARK: - Live overlay wrapper (observes the runtime SSE stream)

struct AvatarOverlayView: View {
    @ObservedObject var client: SSEClient
    @ObservedObject var prefs = AvatarPrefs.shared

    var body: some View {
        AvatarView(
            mood: client.mood,
            speech: client.speech,
            // nil falls through to the resolver (bundled diana.png → circle).
            resolvedImage: prefs.customImage
        )
    }
}
