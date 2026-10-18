import 'package:flutter/services.dart';

/// The only app-specific native operation: choosing a user-owned destination.
class FileDialog {
  static const _channel = MethodChannel('dev.mlsr.speak/files');
  static Future<String?> destination() => _channel.invokeMethod<String>('save');
}
