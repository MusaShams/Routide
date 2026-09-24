import CryptoKit
import Darwin
import Foundation

public enum ExpertPackError: Error, LocalizedError, Sendable {
    case invalidManifest(String)
    case invalidPath(String)
    case missingFile(String)
    case fileSizeMismatch(String)
    case hashMismatch(String)
    case unknownResidentTensor(String)
    case layerOutOfRange(Int)
    case expertOutOfRange(Int)
    case ioError(path: String, code: Int32)
    case truncatedRead(path: String, expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidManifest(let reason):
            "Invalid expert-pack manifest: \(reason)"
        case .invalidPath(let path):
            "Pack path escapes its root: \(path)"
        case .missingFile(let path):
            "Pack file is missing: \(path)"
        case .fileSizeMismatch(let path):
            "Pack file size does not match the manifest: \(path)"
        case .hashMismatch(let path):
            "Pack file hash does not match the manifest: \(path)"
        case .unknownResidentTensor(let name):
            "Unknown resident tensor: \(name)"
        case .layerOutOfRange(let layer):
            "Layer is out of range: \(layer)"
        case .expertOutOfRange(let expert):
            "Expert is out of range: \(expert)"
        case .ioError(let path, let code):
            "I/O failed for \(path) with errno \(code)"
        case .truncatedRead(let path, let expected, let actual):
            "Read \(actual) of \(expected) bytes from \(path)"
        }
    }
}

private final class FileDescriptorPool: @unchecked Sendable {
    private let lock = NSLock()
    private var descriptors: [String: Int32] = [:]

    func descriptor(for url: URL) throws -> Int32 {
        let path = url.path
        lock.lock()
        defer { lock.unlock() }
        if let descriptor = descriptors[path] {
            return descriptor
        }
        let descriptor = Darwin.open(path, O_RDONLY)
        guard descriptor >= 0 else {
            throw ExpertPackError.ioError(path: path, code: errno)
        }
        descriptors[path] = descriptor
        return descriptor
    }

    deinit {
        for descriptor in descriptors.values {
            Darwin.close(descriptor)
        }
    }
}

public final class ExpertPackReader: @unchecked Sendable {
    public static let format = "routide-expert-pack"
    public static let version = 1

    public let rootURL: URL
    public let manifest: ExpertPackManifest

    private let descriptors = FileDescriptorPool()
    private let residentURL: URL
    private let layerURLs: [URL]

    public init(rootURL: URL, verifyHashes: Bool = false) throws {
        let rootURL = rootURL.resolvingSymlinksInPath().standardizedFileURL
        let manifestURL = rootURL.appendingPathComponent("manifest.json")
        let manifestData: Data
        do {
            manifestData = try Data(contentsOf: manifestURL)
        } catch {
            throw ExpertPackError.missingFile(manifestURL.path)
        }
        let manifest: ExpertPackManifest
        do {
            manifest = try JSONDecoder().decode(ExpertPackManifest.self, from: manifestData)
        } catch {
            throw ExpertPackError.invalidManifest(error.localizedDescription)
        }

        self.rootURL = rootURL
        self.manifest = manifest
        self.residentURL = try Self.resolve(manifest.resident.file, under: rootURL)
        self.layerURLs = try manifest.experts.layers.map {
            try Self.resolve($0.file, under: rootURL)
        }

        try validateStructure()
        try validateFiles(verifyHashes: verifyHashes)
    }

    public func readResidentTensor(named name: String) async throws -> Data {
        guard let tensor = manifest.resident.tensors[name] else {
            throw ExpertPackError.unknownResidentTensor(name)
        }

        return try await read(
            from: residentURL,
            offset: tensor.offset,
            count: tensor.length
        )
    }

    public func readResidentTensor(
        named name: String,
        relativeOffset: Int,
        count: Int
    ) async throws -> Data {
        guard let tensor = manifest.resident.tensors[name] else {
            throw ExpertPackError.unknownResidentTensor(name)
        }
        let (end, overflow) = relativeOffset.addingReportingOverflow(count)
        guard relativeOffset >= 0, count >= 0, !overflow, end <= tensor.length else {
            throw ExpertPackError.invalidManifest("resident tensor subrange is out of bounds")
        }
        return try await read(
            from: residentURL,
            offset: tensor.offset + relativeOffset,
            count: count
        )
    }

    public func readExpert(layer: Int, expert: Int) async throws -> ExpertBlock {
        guard manifest.model.numLayers > 0, (0 ..< manifest.model.numLayers).contains(layer)
        else {
            throw ExpertPackError.layerOutOfRange(layer)
        }
        guard manifest.model.numExperts > 0,
            (0 ..< manifest.model.numExperts).contains(expert)
        else {
            throw ExpertPackError.expertOutOfRange(expert)
        }
        let (offset, overflow) = expert.multipliedReportingOverflow(
            by: manifest.experts.blockStride
        )
        guard !overflow else {
            throw ExpertPackError.invalidManifest("expert block offset overflows")
        }
        let data = try await read(
            from: layerURLs[layer],
            offset: offset,
            count: manifest.experts.blockPayloadBytes
        )
        return ExpertBlock(
            layer: layer,
            expert: expert,
            data: data,
            tensorLayout: manifest.experts.tensorLayout
        )
    }

    private static func resolve(_ relativePath: String, under rootURL: URL) throws -> URL {
        guard !relativePath.isEmpty, !relativePath.hasPrefix("/") else {
            throw ExpertPackError.invalidPath(relativePath)
        }
        let resolved = rootURL.appendingPathComponent(relativePath)
            .resolvingSymlinksInPath()
            .standardizedFileURL
        let rootPath = rootURL.path.hasSuffix("/") ? rootURL.path : rootURL.path + "/"
        guard resolved.path.hasPrefix(rootPath) else {
            throw ExpertPackError.invalidPath(relativePath)
        }
        return resolved
    }

    private func validateStructure() throws {
        guard manifest.format == Self.format, manifest.version == Self.version else {
            throw ExpertPackError.invalidManifest("unsupported format or version")
        }
        guard manifest.model.architecture == "qwen3_5_moe" else {
            throw ExpertPackError.invalidManifest("unsupported model architecture")
        }
        guard manifest.model.numLayers > 0,
            manifest.model.numExperts > 0,
            manifest.model.topK > 0,
            manifest.model.topK <= manifest.model.numExperts,
            manifest.model.hiddenSize > 0,
            manifest.model.attentionHeads > 0,
            manifest.model.kvHeads > 0,
            manifest.model.headDim > 0,
            manifest.model.ropeDimensions > 0,
            manifest.model.ropeDimensions <= manifest.model.headDim,
            manifest.model.ropeTheta > 0,
            manifest.model.rmsNormEps > 0,
            manifest.model.fullAttentionInterval > 0,
            manifest.model.attentionHeads % manifest.model.kvHeads == 0,
            manifest.model.linearValueHeads > 0,
            manifest.model.linearKeyHeads > 0,
            manifest.model.linearValueHeads % manifest.model.linearKeyHeads == 0,
            manifest.model.linearKeyHeadDim > 0,
            manifest.model.linearValueHeadDim > 0,
            manifest.model.linearConvKernelDim > 0
        else {
            throw ExpertPackError.invalidManifest("invalid model dimensions")
        }
        guard manifest.experts.layers.count == manifest.model.numLayers else {
            throw ExpertPackError.invalidManifest("layer count does not match model")
        }
        guard manifest.experts.blockAlignment > 0,
            manifest.experts.blockAlignment.nonzeroBitCount == 1,
            manifest.experts.blockPayloadBytes > 0,
            manifest.experts.blockStride >= manifest.experts.blockPayloadBytes,
            manifest.experts.blockStride % manifest.experts.blockAlignment == 0
        else {
            throw ExpertPackError.invalidManifest("invalid expert block geometry")
        }
        guard manifest.experts.quantization.groupSize > 0,
            manifest.experts.quantization.bits > 0,
            32 % manifest.experts.quantization.bits == 0,
            manifest.experts.quantization.mode == "affine"
        else {
            throw ExpertPackError.invalidManifest("unsupported expert quantization")
        }

        var expectedOffset = 0
        for tensor in manifest.experts.tensorLayout {
            guard tensor.offset == expectedOffset, tensor.length > 0 else {
                throw ExpertPackError.invalidManifest("expert tensor layout is not contiguous")
            }
            let (nextOffset, overflow) = expectedOffset.addingReportingOverflow(tensor.length)
            guard !overflow else {
                throw ExpertPackError.invalidManifest("expert tensor layout overflows")
            }
            expectedOffset = nextOffset
        }
        guard expectedOffset == manifest.experts.blockPayloadBytes else {
            throw ExpertPackError.invalidManifest("expert tensor layout length changed")
        }

        guard manifest.resident.tensorAlignment > 0,
            manifest.resident.tensorAlignment.nonzeroBitCount == 1
        else {
            throw ExpertPackError.invalidManifest("invalid resident alignment")
        }
        try validateQuantization(manifest.resident.quantization.default)
        for quantization in manifest.resident.quantization.overrides.values {
            try validateQuantization(quantization)
        }
        for (name, tensor) in manifest.resident.tensors {
            let (end, overflow) = tensor.offset.addingReportingOverflow(tensor.length)
            guard tensor.offset >= 0,
                tensor.length >= 0,
                tensor.offset % manifest.resident.tensorAlignment == 0,
                !overflow,
                end <= manifest.resident.size
            else {
                throw ExpertPackError.invalidManifest("invalid resident tensor \(name)")
            }
        }
        for (index, layer) in manifest.experts.layers.enumerated() {
            let (expectedSize, overflow) = manifest.experts.blockStride.multipliedReportingOverflow(
                by: manifest.model.numExperts
            )
            guard layer.layer == index, !overflow, layer.size == expectedSize else {
                throw ExpertPackError.invalidManifest("invalid expert layer \(index)")
            }
        }
    }

    private func validateQuantization(_ quantization: ExpertPackManifest.Quantization) throws {
        guard quantization.groupSize > 0,
            quantization.bits > 0,
            32 % quantization.bits == 0,
            quantization.mode == "affine"
        else {
            throw ExpertPackError.invalidManifest("unsupported resident quantization")
        }
    }

    private func validateFiles(verifyHashes: Bool) throws {
        try validateFile(
            residentURL,
            expectedSize: manifest.resident.size,
            expectedHash: manifest.resident.sha256,
            verifyHash: verifyHashes
        )
        for (layer, url) in zip(manifest.experts.layers, layerURLs) {
            try validateFile(
                url,
                expectedSize: layer.size,
                expectedHash: layer.sha256,
                verifyHash: verifyHashes
            )
        }
    }

    private func validateFile(
        _ url: URL,
        expectedSize: Int,
        expectedHash: String,
        verifyHash: Bool
    ) throws {
        let attributes: [FileAttributeKey: Any]
        do {
            attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        } catch {
            throw ExpertPackError.missingFile(url.path)
        }
        guard let size = attributes[.size] as? NSNumber, size.intValue == expectedSize else {
            throw ExpertPackError.fileSizeMismatch(url.path)
        }
        if verifyHash, try Self.sha256(of: url) != expectedHash {
            throw ExpertPackError.hashMismatch(url.path)
        }
    }

    private func read(from url: URL, offset: Int, count: Int) async throws -> Data {
        guard offset >= 0, count >= 0 else {
            throw ExpertPackError.invalidManifest("negative read range")
        }
        let descriptors = descriptors
        return try await Task.detached(priority: .utility) {
            let descriptor = try descriptors.descriptor(for: url)
            var data = Data(count: count)
            let bytesRead = try data.withUnsafeMutableBytes { buffer -> Int in
                guard let baseAddress = buffer.baseAddress else { return 0 }
                var total = 0
                while total < count {
                    let result = Darwin.pread(
                        descriptor,
                        baseAddress.advanced(by: total),
                        count - total,
                        off_t(offset + total)
                    )
                    if result < 0 {
                        if errno == EINTR { continue }
                        throw ExpertPackError.ioError(path: url.path, code: errno)
                    }
                    if result == 0 { break }
                    total += result
                }
                return total
            }
            guard bytesRead == count else {
                throw ExpertPackError.truncatedRead(
                    path: url.path,
                    expected: count,
                    actual: bytesRead
                )
            }
            return data
        }.value
    }

    private static func sha256(of url: URL) throws -> String {
        guard let stream = InputStream(url: url) else {
            throw ExpertPackError.missingFile(url.path)
        }
        stream.open()
        defer { stream.close() }
        var hasher = SHA256()
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: 8 * 1024 * 1024)
        defer { buffer.deallocate() }
        while true {
            let count = stream.read(buffer, maxLength: 8 * 1024 * 1024)
            if count < 0 {
                let code = (stream.streamError as NSError?)?.code ?? Int(EIO)
                throw ExpertPackError.ioError(path: url.path, code: Int32(code))
            }
            if count == 0 { break }
            hasher.update(data: Data(bytesNoCopy: buffer, count: count, deallocator: .none))
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
