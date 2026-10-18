Pod::Spec.new do |s|
  s.name = 'mlsr_flutter'
  s.version = '0.1.0'
  s.summary = 'Native model resource access for MLSR.'
  s.description = 'Locates Flutter model assets and streams them into Rust-owned staging directories.'
  s.homepage = 'https://github.com/yunho-c/MLSR'
  s.license = { :file => '../LICENSE' }
  s.author = { 'MLSR contributors' => 'https://github.com/yunho-c/MLSR' }
  s.source = { :path => '.' }
  s.source_files = 'Classes/**/*'
  s.dependency 'FlutterMacOS'
  s.platform = :osx, '15.0'
  s.swift_version = '5.0'
end
