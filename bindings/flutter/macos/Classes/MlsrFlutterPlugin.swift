import Darwin
import FlutterMacOS
import Foundation

/// Platform-owned paths and bounded file I/O. Model validation remains in Rust.
public final class MlsrFlutterPlugin: NSObject, FlutterPlugin {
    private let registrar: FlutterPluginRegistrar
    private let io = DispatchQueue(label: "dev.mlsr.flutter.model-files", qos: .userInitiated)

    private init(registrar: FlutterPluginRegistrar) { self.registrar = registrar }

    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(name: "dev.mlsr.flutter/models", binaryMessenger: registrar.messenger)
        registrar.addMethodCallDelegate(MlsrFlutterPlugin(registrar: registrar), channel: channel)
    }

    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        do {
            switch call.method {
            case "platform":
                #if arch(arm64)
                result("macos-arm64")
                #else
                result("macos-x64")
                #endif
            case "storeDirectory":
                result(try storeURL().path)
            case "locate":
                let args = try arguments(call)
                let manifest = try assetURL(args, relative: "mlsr-bundle.yaml")
                io.async {
                    let exists = FileManager.default.fileExists(atPath: manifest.path)
                    DispatchQueue.main.async { result(exists ? manifest.deletingLastPathComponent().path : nil) }
                }
            case "copyFile":
                let args = try arguments(call)
                guard let relative = args["relative"] as? String,
                      let staging = args["staging"] as? String,
                      let expected = args["sizeBytes"] as? Int64, expected > 0 else {
                    throw Failure("invalid_model_package", "Invalid copy request")
                }
                try validateRelative(relative)
                let source = try assetURL(args, relative: relative)
                let stagingRoot = try storeURL().appendingPathComponent("staging", isDirectory: true).resolvingSymlinksInPath()
                let stage = URL(fileURLWithPath: staging, isDirectory: true).standardizedFileURL
                guard stage.deletingLastPathComponent().resolvingSymlinksInPath() == stagingRoot,
                      stage.lastPathComponent.hasPrefix("install-") else {
                    throw Failure("model_store_invalid_state", "Copy destination is outside this app's staging store")
                }
                io.async {
                    do {
                        try self.copy(source: source, stage: stage, relative: relative, expected: expected)
                        DispatchQueue.main.async { result(nil) }
                    } catch {
                        let failure = self.flutterError(error)
                        DispatchQueue.main.async { result(failure) }
                    }
                }
            default: result(FlutterMethodNotImplemented)
            }
        } catch { result(flutterError(error)) }
    }

    private struct Failure: Error {
        let code: String
        let message: String
        init(_ code: String, _ message: String) { self.code = code; self.message = message }
    }
    private func flutterError(_ error: Error) -> FlutterError {
        if let failure = error as? Failure { return FlutterError(code: failure.code, message: failure.message, details: nil) }
        return FlutterError(code: "model_resource_error", message: error.localizedDescription, details: nil)
    }
    private func arguments(_ call: FlutterMethodCall) throws -> [String: Any] {
        guard let args = call.arguments as? [String: Any] else { throw Failure("invalid_model_package", "Missing resource arguments") }
        return args
    }
    private func validateRelative(_ path: String) throws {
        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !path.isEmpty, path.utf8.count <= 4096,
              !path.contains("\\"), !path.contains(":"), !path.contains("\0"),
              components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw Failure("invalid_model_package", "Unsafe relative asset path")
        }
    }
    private func assetURL(_ args: [String: Any], relative: String) throws -> URL {
        guard let package = args["package"] as? String,
              package.range(of: "^[a-z][a-z0-9_]{2,63}$", options: .regularExpression) != nil,
              let root = args["assetRoot"] as? String else { throw Failure("invalid_model_package", "Missing package identity") }
        try validateRelative(root)
        try validateRelative(relative)
        let key = registrar.lookupKey(forAsset: "\(root)/\(relative)", fromPackage: package)
        if let path = Bundle.main.path(forResource: key, ofType: nil) { return URL(fileURLWithPath: path) }
        // macOS registrar keys may already be relative to the .app root. This
        // matches Flutter's video_player resource lookup (flutter/flutter#135302).
        return URL(fileURLWithPath: key, relativeTo: Bundle.main.bundleURL).standardizedFileURL
    }
    private func storeURL() throws -> URL {
        guard let id = Bundle.main.bundleIdentifier, !id.isEmpty else { throw Failure("model_resource_error", "Application bundle identifier is missing") }
        let support = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
        return support.appendingPathComponent(id, isDirectory: true).appendingPathComponent("mlsr", isDirectory: true)
    }
    private func requireDirectory(_ url: URL) throws {
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        guard attrs[.type] as? FileAttributeType == .typeDirectory else { throw Failure("model_store_invalid_state", "Staging parent must be a real directory") }
    }
    private func copy(source: URL, stage: URL, relative: String, expected: Int64) throws {
        try requireDirectory(stage)
        var parent = stage
        for component in relative.split(separator: "/").dropLast() {
            parent.appendPathComponent(String(component), isDirectory: true)
            if !FileManager.default.fileExists(atPath: parent.path) {
                try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
            }
            try requireDirectory(parent)
        }
        let destination = stage.appendingPathComponent(relative)
        let attrs = try FileManager.default.attributesOfItem(atPath: source.path)
        guard attrs[.type] as? FileAttributeType == .typeRegular,
              (attrs[.size] as? NSNumber)?.int64Value == expected else {
            throw Failure("invalid_asset", "Bundled asset size or type differs from its inventory")
        }
        let input = try FileHandle(forReadingFrom: source)
        defer { try? input.close() }
        let fd = Darwin.open(destination.path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, S_IRUSR | S_IWUSR)
        guard fd >= 0 else { throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }
        let output = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
        var complete = false
        defer {
            try? output.close()
            if !complete { try? FileManager.default.removeItem(at: destination) }
        }
        var copied: Int64 = 0
        while let bytes = try input.read(upToCount: 64 * 1024), !bytes.isEmpty {
            guard Int64(bytes.count) <= expected - copied else { throw Failure("invalid_asset", "Asset grew while copying") }
            try output.write(contentsOf: bytes)
            copied += Int64(bytes.count)
        }
        guard copied == expected else { throw Failure("invalid_asset", "Asset ended before its declared length") }
        try output.close()
        complete = true
    }
}
