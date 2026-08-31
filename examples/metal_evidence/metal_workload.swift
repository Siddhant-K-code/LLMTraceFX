import Foundation
import Metal

enum WorkloadError: Error {
    case invalidDispatchCount
    case metalUnavailable
    case commandFailure(String)
}

let arguments = CommandLine.arguments
guard arguments.count == 2,
      let dispatchCount = Int(arguments[1]),
      dispatchCount > 0,
      dispatchCount <= 10_000 else {
    throw WorkloadError.invalidDispatchCount
}

guard let device = MTLCreateSystemDefaultDevice(),
      let queue = device.makeCommandQueue() else {
    throw WorkloadError.metalUnavailable
}

let source = """
#include <metal_stdlib>
using namespace metal;

kernel void evidence_step(
    device float *values [[buffer(0)]],
    constant uint &salt [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    float value = values[index];
    for (uint round = 0; round < 64; ++round) {
        value = fma(value, 1.000001f, float((salt + round) & 7u));
    }
    values[index] = value;
}
"""

let library = try device.makeLibrary(source: source, options: nil)
guard let function = library.makeFunction(name: "evidence_step") else {
    throw WorkloadError.commandFailure("kernel function unavailable")
}
let pipeline = try device.makeComputePipelineState(function: function)

let elementCount = 262_144
let byteCount = elementCount * MemoryLayout<Float>.stride
guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
    throw WorkloadError.commandFailure("buffer allocation failed")
}

let threadWidth = pipeline.threadExecutionWidth
let threadsPerGroup = MTLSize(width: threadWidth, height: 1, depth: 1)
let threadsPerGrid = MTLSize(width: elementCount, height: 1, depth: 1)

for dispatchIndex in 0..<dispatchCount {
    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
        throw WorkloadError.commandFailure("command creation failed")
    }
    var salt = UInt32(dispatchIndex)
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(buffer, offset: 0, index: 0)
    encoder.setBytes(&salt, length: MemoryLayout<UInt32>.stride, index: 1)
    encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if commandBuffer.status == .error {
        throw WorkloadError.commandFailure(
            commandBuffer.error?.localizedDescription ?? "unknown Metal error"
        )
    }
}

print("completed_dispatches=\(dispatchCount)")
