import Cocoa
import FlutterMacOS
import UniformTypeIdentifiers

class MainFlutterWindow: NSWindow {
  private var fileChannel: FlutterMethodChannel?
  override func awakeFromNib() {
    let flutterViewController = FlutterViewController()
    self.contentViewController = flutterViewController
    self.setContentSize(NSSize(width: 1120, height: 790))
    self.minSize = NSSize(width: 800, height: 650)
    self.center()
    self.title = "MLSR · Speak"
    RegisterGeneratedPlugins(registry: flutterViewController)
    let channel = FlutterMethodChannel(name: "dev.mlsr.speak/files",
                                       binaryMessenger: flutterViewController.engine.binaryMessenger)
    channel.setMethodCallHandler { call, result in
      guard call.method == "save" else { result(FlutterMethodNotImplemented); return }
      let panel = NSSavePanel()
      panel.allowedContentTypes = [.wav]
      panel.nameFieldStringValue = "MLSR-speech.wav"
      panel.begin { response in result(response == .OK ? panel.url?.path : nil) }
    }
    fileChannel = channel
    super.awakeFromNib()
  }
}
