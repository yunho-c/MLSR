import Cocoa
import FlutterMacOS
import AVFoundation
import UniformTypeIdentifiers

class MainFlutterWindow: NSWindow {
  private var pcmPlayer: PcmPlayer?
  override func awakeFromNib() {
    let flutterViewController = FlutterViewController()
    self.contentViewController = flutterViewController
    self.setContentSize(NSSize(width: 1120, height: 790))
    self.minSize = NSSize(width: 800, height: 650)
    self.center()
    self.title = "MLSR · Speak"

    RegisterGeneratedPlugins(registry: flutterViewController)
    pcmPlayer = PcmPlayer(messenger: flutterViewController.engine.binaryMessenger)

    super.awakeFromNib()
  }
}

/// Separate host audio adapter. All AVAudioEngine work runs on one serial queue;
/// the Flutter UI and Rust inference workers never wait on that queue directly.
private final class PcmPlayer {
  private let queue = DispatchQueue(label: "dev.mlsr.speak.audio")
  private let engine = AVAudioEngine()
  private let node = AVAudioPlayerNode()
  private let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                     sampleRate: 44100, channels: 1, interleaved: false)!
  private let channel: FlutterMethodChannel
  private let limit = 22050
  private var epoch = 0
  private var queued = 0
  private var played = 0
  private var peakQueued = 0
  private var active = false
  private var pending: (Data, FlutterResult)?
  private var finishing: FlutterResult?
  private var observer: NSObjectProtocol?

  init(messenger: FlutterBinaryMessenger) {
    channel = FlutterMethodChannel(name: "dev.mlsr.speak/pcm", binaryMessenger: messenger)
    engine.attach(node)
    engine.connect(node, to: engine.mainMixerNode, format: format)
    channel.setMethodCallHandler { [weak self] call, result in
      guard let self = self else { result(FlutterError(code: "closed", message: "Player is closed", details: nil)); return }
      if call.method == "save" {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.wav]
        panel.nameFieldStringValue = "MLSR-speech.wav"
        panel.begin { response in result(response == .OK ? panel.url?.path : nil) }
      } else {
        self.queue.async { self.handle(call, result: result) }
      }
    }
    observer = NotificationCenter.default.addObserver(forName: .AVAudioEngineConfigurationChange,
                                                       object: engine, queue: nil) { [weak self] _ in
      self?.queue.async {
        guard let self = self, self.queued > 0 || self.pending != nil else { return }
        self.flush()
      }
    }
  }
  deinit {
    if let observer = observer { NotificationCenter.default.removeObserver(observer) }
    engine.stop()
  }
  private func reply(_ result: @escaping FlutterResult, _ value: Any? = nil) {
    DispatchQueue.main.async { result(value) }
  }
  private func fail(_ result: @escaping FlutterResult, _ message: String) {
    reply(result, FlutterError(code: "playback", message: message, details: nil))
  }
  private func flush() {
    epoch += 1
    node.stop() // Removes scheduled audio; no wait for an inference prediction.
    engine.pause()
    active = false
    queued = 0
    if let (_, result) = pending { fail(result, "Playback stopped") }
    pending = nil
    if let result = finishing { fail(result, "Playback stopped") }
    finishing = nil
  }
  private func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    let args = call.arguments as? [String: Any] ?? [:]
    let requested = args["epoch"] as? Int ?? -1
    switch call.method {
    case "state":
      reply(result, ["epoch": epoch, "queuedFrames": queued, "playedFrames": played,
                     "peakQueuedFrames": peakQueued, "pendingFrames": (pending?.0.count ?? 0) / 4,
                     "active": active, "engineRunning": engine.isRunning,
                     "nodePlaying": node.isPlaying,
                     "renderSampleTime": node.lastRenderTime?.sampleTime ?? -1])
    case "begin":
      guard requested > epoch else { fail(result, "Stale playback request"); return }
      flush()
      epoch = requested
      played = 0; peakQueued = 0
      node.volume = args["muted"] as? Bool == true ? 0 : 1
      do {
        engine.prepare()
        try engine.start()
        node.play()
        active = true
        reply(result)
      } catch { fail(result, error.localizedDescription) }
    case "stop":
      if requested == epoch { flush() }
      reply(result)
    case "enqueue":
      guard active && requested == epoch else { fail(result, "Playback stopped"); return }
      guard pending == nil && finishing == nil,
            let value = args["pcm"] as? FlutterStandardTypedData,
            value.data.count > 0, value.data.count % 4 == 0, value.data.count / 4 <= limit else {
        fail(result, "Invalid PCM or concurrent producer"); return
      }
      let data = value.data
      if queued + data.count / 4 > limit {
        pending = (data, result) // At most one chunk; Dart awaits this reply.
      } else {
        schedule(data, result: result)
      }
    case "finish":
      guard requested == epoch && active && finishing == nil && pending == nil else {
        fail(result, "Playback stopped or concurrent finish"); return
      }
      if queued == 0 {
        active = false; node.pause(); engine.pause(); reply(result)
      } else { finishing = result }
    default: reply(result, FlutterMethodNotImplemented)
    }
  }
  private func schedule(_ data: Data, result: @escaping FlutterResult) {
    let count = data.count / 4
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(count)),
          let target = buffer.floatChannelData?[0] else { fail(result, "Could not allocate audio buffer"); return }
    data.withUnsafeBytes { bytes in
      if let source = bytes.baseAddress { memcpy(target, source, data.count) }
    }
    buffer.frameLength = AVAudioFrameCount(count)
    let captured = epoch
    queued += count
    peakQueued = max(peakQueued, queued)
    node.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
      self?.queue.async { self?.completed(count, epoch: captured) }
    }
    reply(result)
  }
  private func completed(_ frames: Int, epoch captured: Int) {
    guard captured == epoch else { return }
    queued -= frames; played += frames
    if let (data, result) = pending, queued + data.count / 4 <= limit {
      pending = nil
      schedule(data, result: result)
    }
    if queued == 0, let result = finishing {
      finishing = nil; active = false; node.pause(); engine.pause(); reply(result)
    }
  }
}
