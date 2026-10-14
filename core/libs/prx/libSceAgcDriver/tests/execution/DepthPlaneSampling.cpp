#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "DepthFastClearHarness.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace AgcDriver {
struct VulkanDeviceTestAccess {
    static Graphics::Context ContextFor(VulkanDevice& device) { return device.graphicsContext(); }
};
}

namespace {

using AgcDriver::Graphics::Require;
using DepthFastClearHarness::BufferDescriptor;
using DepthFastClearHarness::PixelCode;
using DepthFastClearHarness::VertexCode;
using ShaderRecompiler::ShaderStage;

void SetResolvePolicy(const char* value) {
#ifdef _WIN32
    _putenv_s("APS5_EXPERIMENTAL_DEPTH_RESOLVE", value);
#else
    setenv("APS5_EXPERIMENTAL_DEPTH_RESOLVE", value, 1);
#endif
}

void SetTerminalMips(const char* value) {
#ifdef _WIN32
    _putenv_s("APS5_APPROXIMATE_TERMINAL_MIPS", value);
#else
    setenv("APS5_APPROXIMATE_TERMINAL_MIPS", value, 1);
#endif
}

constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t Format16Uint = 11;
constexpr std::uint32_t Format32Float = 22;
constexpr std::uint32_t Type2D = 9;
constexpr std::uint32_t TypeCube = 11;
constexpr std::uint32_t Type2DArray = 13;
constexpr std::uint32_t Swizzle64KbS = 9;
constexpr std::uint32_t Swizzle64KbZX = 24;
constexpr std::uint64_t SliceBytes = 65536;
constexpr std::size_t BlockBytes = 2u << 20u;
constexpr std::size_t OutputOffset = 0;
constexpr std::size_t ArrayOffset = 0x10000;
constexpr std::size_t CubeOffset = 0x40000;
constexpr std::size_t IntegerOffset = 0xa0000;
constexpr std::size_t DepthBitsOffset = 0xb0000;
constexpr std::size_t OneSliceOffset = 0xd0000;
constexpr std::size_t FirstSliceOffset = 0xe0000;
constexpr float DrawnDepth = 0.25f;
constexpr float DrawnD16Depth = 0.2f;
constexpr std::uint32_t DrawnD16Bits = 13107;

// image_load of row 0, one texel per thread (x = thread id), slice / face from user data, stored as a dword per
// thread into the buffer at s[0:3]. The T# is s[4:11], the slice s12.
constexpr std::array<std::uint32_t, 3> LoadDims{1u, 3u, 5u};

std::array<std::uint32_t, 11> LoadCode(std::uint32_t dim) {
    return {
        0x7e020280,                        // v_mov_b32 v1, 0 (y)
        0x7e04020c,                        // v_mov_b32 v2, s12 (slice or face)
        0xf0001100u | (dim << 3u),         // image_load v3, v[0:2], s[4:11] dmask:0x1 unorm, dim 2D / CUBE / 2D_ARRAY
        0x00010300,
        0xbf8c3f70,                        // s_waitcnt vmcnt(0)
        0x34080082,                        // v_lshlrev_b32 v4, 2, v0
        0xe0701000,                        // buffer_store_dword v3, v4, s[0:3], 0 offen
        0x80000304,
        0xbf810000,                        // s_endpgm
        0xbf810000,
        0xbf810000,
    };
}

alignas(256) std::array<std::uint32_t, 11> Load2D = LoadCode(LoadDims[0]);
alignas(256) std::array<std::uint32_t, 11> LoadCube = LoadCode(LoadDims[1]);
alignas(256) std::array<std::uint32_t, 11> LoadArray = LoadCode(LoadDims[2]);
alignas(256) std::array<std::uint32_t, 11> LoadMultisample = LoadCode(6u);
alignas(256) std::array<std::uint32_t, 13> Sample2D{
    0x7e2802ffu, 0x3b800000u, 0x7e2a02ffu, 0x3b800000u, 0xf09c0108u, 0x00610314u,
    0xbf8c3f70u, 0x34080082u, 0xe0701000u, 0x80000304u, 0xbf810000u, 0xbf810000u, 0xbf810000u
};
alignas(256) std::array<std::uint32_t, 7> Store2D{
    0x7e020280u, 0x7e06020cu, 0xf0200108u, 0x00010300u, 0xbf8c3f70u, 0xbf810000u, 0xbf810000u
};
alignas(256) std::array<std::uint32_t, 12> LoadMultisampleArray{
    0x7e020280u, 0x7e040280u, 0x7e06020cu, 0xf0001138u, 0x00010300u, 0xbf8c3f70u,
    0x34080082u, 0xe0701000u, 0x80000304u, 0xbf810000u, 0xbf810000u, 0xbf810000u
};

alignas(256) std::array<std::byte, 256 * 128 * 4> Pixels{};
alignas(256) std::array<std::uint32_t, 4> AlphaPixelCode{0x7e000200u, 0xf800180fu, 0u, 0xbf810000u};
alignas(256) std::array<std::byte, 256 * 256 * 4> MixedPixels{};

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(BlockBytes, 65536));
        if (block != nullptr) GuestArena::GuestArenaCommit_nid_postfix(block, BlockBytes, PAGE_READWRITE, BlockBytes);
#else
        constexpr std::uintptr_t alignment = 65536;
        void* mapped = mmap(nullptr, BlockBytes + alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapped != MAP_FAILED) {
            const auto begin = reinterpret_cast<std::uintptr_t>(mapped);
            const auto aligned = (begin + alignment - 1) & ~(alignment - 1);
            if (aligned != begin) munmap(mapped, aligned - begin);
            if (aligned + BlockBytes != begin + BlockBytes + alignment) munmap(reinterpret_cast<void*>(aligned + BlockBytes), begin + alignment - aligned);
            block = reinterpret_cast<std::uint8_t*>(aligned);
            GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, BlockBytes);
        }
#endif
        Require(block != nullptr, "depth plane sampling: cannot allocate the guest block");
        std::memset(block, 0, BlockBytes);
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
        AgcDriver::GuestMemory::CollectWritesUncached(reinterpret_cast<std::uintptr_t>(block), BlockBytes);
    }

    ~GuestBlock() {
        GuestAllocations::Mutation().Remove(block);
#ifdef _WIN32
        GuestArena::GuestArenaReset_nid_postfix(block, BlockBytes);
        GuestArena::GuestArenaRelease_nid_postfix(block, BlockBytes);
#else
        munmap(block, BlockBytes);
        GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(block, BlockBytes);
#endif
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    std::uint8_t* At(std::size_t offset) { return block + offset; }

private:
    std::uint8_t* block = nullptr;
};

struct Shape {
    std::uint32_t format;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t type;
    std::uint32_t lastArray;
    std::uint32_t swizzle;
    std::uint32_t baseArray = 0;
    std::uint32_t samples = 1;
    std::uint64_t metadataAddress = 0;
    std::uint32_t terminalLevel = 0;
};

std::array<std::uint32_t, 8> TextureDescriptor(const void* data, const Shape& shape) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (shape.format << 20u) | (((shape.width - 1u) & 3u) << 30u),
        ((shape.width - 1u) >> 2u) | ((shape.height - 1u) << 14u),
        0xfacu | (shape.swizzle << 20u) | (shape.type << 28u) | (std::countr_zero(shape.samples) << 16u) | (shape.terminalLevel << 12u) | (shape.terminalLevel << 16u),
        shape.lastArray | (shape.baseArray << 16u),
        static_cast<std::uint32_t>(std::countr_zero(shape.samples)) << 4u,
        (shape.samples > 1 ? 1u << 10u : 0u) | (shape.metadataAddress != 0 ? 1u << 21u : 0u) | (static_cast<std::uint32_t>((shape.metadataAddress >> 8u) & 0xffu) << 24u),
        static_cast<std::uint32_t>(shape.metadataAddress >> 16u),
    };
}

std::uint32_t Bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// Draws a full-target triangle at depth `z` into the depth surface at `address`, which makes the driver create the
// surface (the base slice a later plane copy reads from the surface instead of from guest memory).
AgcDriver::Graphics::ColorTarget DrawDepth(AgcDriver::VulkanDevice& device, std::uint64_t address, VkExtent2D extent, VkFormat format, float z, std::uint32_t samples = 1, AgcDriver::Graphics::MixedSampleApproximation approximation = AgcDriver::Graphics::MixedSampleApproximation::None, VkCompareOp compare = VK_COMPARE_OP_ALWAYS, bool edge = false, std::optional<float> alpha = std::nullopt) {
    static std::deque<std::vector<std::array<float, 4>>> vertexData;
    vertexData.push_back({{-1.0f, -1.0f, z, 1.0f}, {3.0f, -1.0f, z, 1.0f}, {-1.0f, 3.0f, z, 1.0f}});
    if (edge) {
        const auto x = -1.0f + 1.0f / static_cast<float>(extent.width);
        vertexData.back() = {{-3.0f, -1.0f, z, 1.0f}, {x, -1.0f, z, 1.0f}, {x, 3.0f, z, 1.0f}};
    }
    const auto& vertices = vertexData.back();
    const auto target = device.Target();
    std::vector<std::uint32_t> vertexUserData(4, 0u);
    const auto vertexBuffer = BufferDescriptor(vertices.data(), 16u, static_cast<std::uint32_t>(vertices.size()));
    std::copy(vertexBuffer.begin(), vertexBuffer.end(), vertexUserData.begin());
    const std::array<ShaderRecompiler::MemoryRegion, 1> vertexMemory{{{reinterpret_cast<std::uintptr_t>(VertexCode.data()), std::as_bytes(std::span(VertexCode))}}};
    ShaderRecompiler::RecompileRequest vertexRequest{
        {ShaderStage::Vertex, reinterpret_cast<std::uintptr_t>(VertexCode.data()), VertexCode, 0, {}},
        {32, 0, vertexUserData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, vertexMemory},
        target,
        {0, 0, 0, 64}
    };
    vertexRequest.useCache = false;
    const auto vertex = ShaderRecompiler::Recompile(vertexRequest);
    const auto vertexPush = static_cast<std::uint32_t>(vertex.pushConstants.size());
    ShaderRecompiler::ShaderPixelStageInfo pixelInfo{};
    pixelInfo.wave32 = true;
    pixelInfo.inputAddr = ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionX) | ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionY);
    pixelInfo.posX = true;
    pixelInfo.posY = true;
    pixelInfo.targetOutputMode[0] = 9;
    pixelInfo.alphaToCoverage = alpha.has_value();
    pixelInfo.targetExportMapping.fill(0xe4u);
    std::vector<std::uint32_t> pixelUserData(4, 0u);
    if (alpha) pixelUserData[0] = Bits(*alpha);
    const std::span<const std::uint32_t> pixelCode = alpha ? std::span<const std::uint32_t>(AlphaPixelCode) : std::span<const std::uint32_t>(PixelCode);
    const std::array<ShaderRecompiler::MemoryRegion, 1> pixelMemory{{{reinterpret_cast<std::uintptr_t>(pixelCode.data()), std::as_bytes(pixelCode)}}};
    ShaderRecompiler::RecompileRequest fragmentRequest{
        {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(pixelCode.data()), pixelCode, 0, {}},
        {32, 0, pixelUserData, std::nullopt, pixelInfo, std::nullopt, pixelMemory},
        target,
        PixelPushLayout(vertexPush, target)
    };
    fragmentRequest.useCache = false;
    const auto pixel = ShaderRecompiler::Recompile(fragmentRequest);
    const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{
        {ShaderStage::Vertex, &vertex, 0},
        {ShaderStage::Fragment, &pixel, PixelPushOffset(vertexPush, target)}
    }};
    AgcDriver::Graphics::State state{};
    state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0u, 32u, 32u, std::nullopt, std::nullopt};
    state.color = {reinterpret_cast<std::uintptr_t>(Pixels.data()), extent, VK_FORMAT_R8G8B8A8_UNORM, static_cast<std::size_t>(extent.width) * extent.height * 4u, 0xe4u};
    state.colors = {state.color};
    state.hasColorTarget = true;
    state.depth = AgcDriver::Graphics::DepthTarget{address, 0, extent, format, 1.0f, 0};
    state.samples = samples;
    state.depth->samples = samples;
    if (samples > 1 || alpha) {
        state.hasColorTarget = false;
        state.colors.clear();
    }
    state.alphaToCoverage = alpha.has_value();
    state.alphaWithoutColor = alpha.has_value();
    if (approximation == AgcDriver::Graphics::MixedSampleApproximation::WithoutDepth) {
        state.color.address = reinterpret_cast<std::uintptr_t>(MixedPixels.data());
        state.color.samples = 4;
        state.color.tileMode = AgcDriver::Graphics::ColorTileMode::RenderTarget;
        state.colors = {state.color};
        state.hasColorTarget = true;
    }
    state.depthTest = true;
    state.depthWrite = true;
    state.depthCompare = compare;
    state.renderExtent = extent;
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, static_cast<float>(extent.height), static_cast<float>(extent.width), -static_cast<float>(extent.height), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, extent};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blend.colorWriteMask = 15;
    state.blends = {state.blend};
    if (!state.hasColorTarget) state.blends.clear();
    state.blendConstants = {};
    if (approximation != AgcDriver::Graphics::MixedSampleApproximation::None) {
        state.samples = 4;
        AgcDriver::Graphics::ApplyMixedSampleApproximation(state);
        Require(state.mixedSampleApproximation == approximation, "GPU mixed-sample test selected the wrong approximation");
    }
    const AgcDriver::Pm4::DrawParameters draw{0, static_cast<std::uint32_t>(vertices.size()), 0, 1, 0, false};
    device.Draw(state, draw, shaders);
    device.WaitIdle();
    return state.color;
}

// Loads row 0 of `slice` of the texture `shape` describes at `texels` into `output` (one dword per thread). With
// `shaderImage2D` the shader declares a 2D image whatever the T#'s type is.
void Load(AgcDriver::VulkanDevice& device, std::uint8_t* texels, const Shape& shape, std::uint32_t slice, std::span<std::uint32_t> output, bool shaderImage2D = false, bool resolveDepth = false, bool sampleDepth = false, bool store = false) {
    const std::span<const std::uint32_t> code = store ? std::span<const std::uint32_t>(Store2D) : sampleDepth ? std::span<const std::uint32_t>(Sample2D) : resolveDepth ? std::span<const std::uint32_t>(Load2D) : shape.samples > 1 && shape.type == 15u && !shaderImage2D ? std::span<const std::uint32_t>(LoadMultisampleArray) : std::span<const std::uint32_t>(shape.samples > 1 ? LoadMultisample : shaderImage2D ? Load2D : shape.type == TypeCube ? LoadCube : shape.type == Type2DArray ? LoadArray : Load2D);
    std::vector<std::uint32_t> userData(16, 0u);
    const auto out = std::array<std::uint32_t, 4>{static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(output.data())), static_cast<std::uint32_t>((reinterpret_cast<std::uintptr_t>(output.data()) >> 32u) & 0xffffu), static_cast<std::uint32_t>(output.size_bytes()), 0x31016facu};
    const auto texture = TextureDescriptor(texels, shape);
    std::copy(out.begin(), out.end(), userData.begin());
    std::copy(texture.begin(), texture.end(), userData.begin() + 4);
    userData[12] = slice;
    if (sampleDepth) {
        const std::array<std::uint32_t, 4> sampler{0x92u, 0x00fff000u, 0u, 0u};
        std::copy(sampler.begin(), sampler.end(), userData.begin() + 12);
    }
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(std::span(code))}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{store ? 1u : Threads, 1, 1}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    std::fill(output.begin(), output.end(), 0xdeadbeefu);
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
}

void Expect(std::span<const std::uint32_t> output, std::uint32_t expected, const std::string& what) {
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        char text[160];
        std::snprintf(text, sizeof(text), "%s: texel %u read 0x%08x, expected 0x%08x", what.c_str(), tid, output[tid], expected);
        Require(output[tid] == expected, text);
    }
}

void FillSlice(std::uint8_t* slice, std::uint64_t bytes, std::uint32_t value, std::uint32_t elementBytes) {
    for (std::uint64_t offset = 0; offset < bytes; offset += elementBytes) std::memcpy(slice + offset, &value, elementBytes);
}

// A 3-slice D32 array: slice 0 from the live depth surface, slices 1 and 2 from guest memory (detiled).
void ArrayCase(AgcDriver::VulkanDevice& device, GuestBlock& block, std::span<std::uint32_t> output) {
    auto* base = block.At(ArrayOffset);
    DrawDepth(device, reinterpret_cast<std::uintptr_t>(base), {128, 128}, VK_FORMAT_D32_SFLOAT, DrawnDepth);
    FillSlice(base + SliceBytes, SliceBytes, Bits(0.5f), 4u);
    FillSlice(base + 2u * SliceBytes, SliceBytes, Bits(0.75f), 4u);
    const Shape shape{Format32Float, 128, 128, Type2DArray, 2, Swizzle64KbZX};
    const std::array<float, 3> expected{DrawnDepth, 0.5f, 0.75f};
    for (std::uint32_t slice = 0; slice < 3u; ++slice) {
        Load(device, base, shape, slice, output);
        Expect(output, Bits(expected[slice]), "D32 2D array slice " + std::to_string(slice));
    }
}

// A cube over six square D32 slices: face 0 from the live surface, faces 1-5 from guest memory.
void CubeCase(AgcDriver::VulkanDevice& device, GuestBlock& block, std::span<std::uint32_t> output) {
    auto* base = block.At(CubeOffset);
    DrawDepth(device, reinterpret_cast<std::uintptr_t>(base), {128, 128}, VK_FORMAT_D32_SFLOAT, DrawnDepth);
    for (std::uint32_t face = 1; face < 6u; ++face) FillSlice(base + face * SliceBytes, SliceBytes, Bits(static_cast<float>(face) / 8.0f), 4u);
    const Shape shape{Format32Float, 128, 128, TypeCube, 0, Swizzle64KbZX};
    for (std::uint32_t face = 0; face < 6u; ++face) {
        Load(device, base, shape, face, output);
        Expect(output, Bits(face == 0 ? DrawnDepth : static_cast<float>(face) / 8.0f), "D32 cube face " + std::to_string(face));
    }
}

// A D16 plane read through an R16_UINT view (a non-depth swizzle, so not flagged as depth bits): the raw depth value.
void IntegerCase(AgcDriver::VulkanDevice& device, GuestBlock& block, std::span<std::uint32_t> output) {
    auto* base = block.At(IntegerOffset);
    DrawDepth(device, reinterpret_cast<std::uintptr_t>(base), {256, 128}, VK_FORMAT_D16_UNORM, DrawnD16Depth);
    Load(device, base, {Format16Uint, 256, 128, Type2D, 0, Swizzle64KbS}, 0, output);
    Expect(output, DrawnD16Bits, "D16 plane as R16_UINT");
}

// A D16 2D array whose R16_UINT T# has a depth swizzle (flagged as depth bits): both slices' raw values.
void DepthBitsArrayCase(AgcDriver::VulkanDevice& device, GuestBlock& block, std::span<std::uint32_t> output) {
    auto* base = block.At(DepthBitsOffset);
    DrawDepth(device, reinterpret_cast<std::uintptr_t>(base), {256, 128}, VK_FORMAT_D16_UNORM, DrawnD16Depth);
    FillSlice(base + SliceBytes, SliceBytes, 0x1234u, 2u);
    const Shape shape{Format16Uint, 256, 128, Type2DArray, 1, Swizzle64KbZX};
    Load(device, base, shape, 0, output);
    Expect(output, DrawnD16Bits, "D16 depth-bits array slice 0");
    Load(device, base, shape, 1, output);
    Expect(output, 0x1234u, "D16 depth-bits array slice 1");
}

// A 2D shader image over a 2D-array T# reads the array's first slice: a one-slice array, which samples the depth
// image itself, and a 3-slice array, which samples the plane copy.
void FirstSliceCase(AgcDriver::VulkanDevice& device, GuestBlock& block, std::span<std::uint32_t> output) {
    auto* one = block.At(OneSliceOffset);
    DrawDepth(device, reinterpret_cast<std::uintptr_t>(one), {128, 128}, VK_FORMAT_D32_SFLOAT, DrawnDepth);
    Load(device, one, {Format32Float, 128, 128, Type2DArray, 0, Swizzle64KbZX}, 0, output, true);
    Expect(output, Bits(DrawnDepth), "one-slice D32 2D array through a 2D shader image");
    auto* base = block.At(FirstSliceOffset);
    DrawDepth(device, reinterpret_cast<std::uintptr_t>(base), {128, 128}, VK_FORMAT_D32_SFLOAT, DrawnDepth);
    FillSlice(base + SliceBytes, SliceBytes, Bits(0.5f), 4u);
    FillSlice(base + 2u * SliceBytes, SliceBytes, Bits(0.75f), 4u);
    Load(device, base, {Format32Float, 128, 128, Type2DArray, 2, Swizzle64KbZX}, 0, output, true);
    Expect(output, Bits(DrawnDepth), "3-slice D32 2D array through a 2D shader image");
}

// A plane array from a later base slice is still refused (its slice offsets are not modeled here).
void MultisampleCase(AgcDriver::VulkanDevice& device, GuestBlock& block, std::span<std::uint32_t> output) {
    auto* base = block.At(0x100000);
    DrawDepth(device, reinterpret_cast<std::uintptr_t>(base), {128, 128}, VK_FORMAT_D32_SFLOAT, 0.375f, 4u);
    for (const auto type : {14u, 15u}) {
        Shape shape{Format32Float, 128, 128, type, 0, Swizzle64KbZX};
        shape.samples = 4u;
        for (std::uint32_t sample = 0; sample < shape.samples; ++sample) {
            Load(device, base, shape, sample, output);
            Expect(output, Bits(0.375f), "D32 multisample depth sample " + std::to_string(sample));
        }
        if (type == 15u) {
            Load(device, base, shape, 3u, output, true);
            Expect(output, Bits(0.375f), "D32 multisample array first layer through a 2D shader");
        }
    }
    Shape mismatch{Format32Float, 128, 128, 14u, 0, Swizzle64KbZX};
    mismatch.samples = 2u;
    bool refused = false;
    try { Load(device, base, mismatch, 0, output); }
    catch (const std::runtime_error& error) { refused = std::string(error.what()).find("sample count disagrees") != std::string::npos; }
    Require(refused, "a multisample depth view with a different sample count was accepted");
}

void TerminalMipCase(AgcDriver::VulkanDevice& device, GuestBlock& block, std::span<std::uint32_t> output) {
    SetTerminalMips("1");
    for (const auto format : {Format32Float, 62u, 75u}) {
        auto* base = block.At(format == Format32Float ? 0x1c0000 : format == 62u ? 0x1d0000 : 0x190000);
        const auto value = Bits(0.875f);
        std::memcpy(base, &value, sizeof(value));
        Shape shape{format, 1, 1, Type2DArray, 0, 1};
        shape.terminalLevel = 8;
        Load(device, base, shape, 0, output, true, false, true);
        Expect(output, Bits(0.875f), "sampled terminal-mip placeholder");
        Load(device, base, shape, Bits(0.5f), output, true, false, false, true);
        Load(device, base, shape, 0, output, true, false, true);
        Expect(output, Bits(0.5f), "terminal-mip placeholder refreshed after a storage write");
    }
    SetTerminalMips("0");
}

void AlphaCoverageCase(AgcDriver::VulkanDevice& device, GuestBlock& block, std::span<std::uint32_t> output) {
    using Approximation = AgcDriver::Graphics::MixedSampleApproximation;
    for (const auto samples : {1u, 4u}) {
        auto* base = block.At(samples == 1 ? 0x130000 : 0x150000);
        const auto address = reinterpret_cast<std::uintptr_t>(base);
        Shape shape{Format32Float, 128, 128, samples == 1 ? Type2D : 14u, 0, Swizzle64KbZX};
        shape.samples = samples;
        DrawDepth(device, address, {128, 128}, VK_FORMAT_D32_SFLOAT, 0.25f, samples, Approximation::None, VK_COMPARE_OP_ALWAYS, false, 0.25f);
        Load(device, base, shape, 0, output);
        Expect(output, Bits(1.0f), "depth-only alpha below threshold did not discard coverage");
        DrawDepth(device, address, {128, 128}, VK_FORMAT_D32_SFLOAT, 0.375f, samples, Approximation::None, VK_COMPARE_OP_ALWAYS, false, 0.75f);
        for (std::uint32_t sample = 0; sample < samples; ++sample) {
            Load(device, base, shape, sample, output);
            Expect(output, Bits(0.375f), "depth-only alpha above threshold did not retain depth writes");
        }
    }
}

void ResolvedMultisampleCase(AgcDriver::VulkanDevice& device, GuestBlock& block, std::span<std::uint32_t> output) {
    using Approximation = AgcDriver::Graphics::MixedSampleApproximation;
    auto* base = block.At(0x1a0000);
    const auto address = reinterpret_cast<std::uintptr_t>(base);
    Shape shape{Format32Float, 128, 128, 14u, 0, Swizzle64KbZX};
    shape.samples = 4;
    DrawDepth(device, address, {128, 128}, VK_FORMAT_D32_SFLOAT, 0.3125f, 4, Approximation::None, VK_COMPARE_OP_ALWAYS, true);
    std::array<std::array<std::uint32_t, Threads>, 4> samples{};
    for (std::uint32_t sample = 0; sample < 4; ++sample) {
        Load(device, base, shape, sample, output);
        std::copy(output.begin(), output.end(), samples[sample].begin());
    }
    Require(std::ranges::any_of(samples, [&](const auto& values) { return values != samples[0]; }), "depth resolve fixture did not produce different sample values");
    SetResolvePolicy("clear");
    bool refused = false;
    try { Load(device, base, shape, 0, output, false, true); }
    catch (const std::runtime_error& error) { refused = std::string(error.what()).find("cannot represent rendered depth") != std::string::npos; }
    Require(refused, "a clear-only proxy admitted rendered depth");
    for (const auto* policy : {"sample0", "min", "max"}) {
        SetResolvePolicy(policy);
        Load(device, base, shape, 0, output, false, true);
        for (std::uint32_t tid = 0; tid < Threads; ++tid) {
            auto expected = samples[0][tid];
            for (std::uint32_t sample = 1; sample < 4; ++sample) {
                if (std::string_view(policy) == "min") expected = std::min(expected, samples[sample][tid]);
                if (std::string_view(policy) == "max") expected = std::max(expected, samples[sample][tid]);
            }
            Require(output[tid] == expected, std::string("depth resolve ") + policy + " produced the wrong GPU value at texel " + std::to_string(tid));
        }
        const auto first = output[0];
        Load(device, base, shape, 0, output, false, true, true);
        Expect(output, first, std::string("ordinary image_sample_lz over resolved depth ") + policy);
    }
    DrawDepth(device, address, {128, 128}, VK_FORMAT_D32_SFLOAT, 0.625f, 4);
    Load(device, base, shape, 0, output, false, true);
    Expect(output, Bits(0.625f), "depth proxy refreshed after a new draw");
    Load(device, base, shape, 3, output);
    Expect(output, Bits(0.625f), "depth resolve preserved the original multisample plane");
    SetResolvePolicy("off");
}

void FastClearedMultisampleCase(AgcDriver::VulkanDevice& device, GuestBlock& block, std::span<std::uint32_t> output) {
    const auto context = AgcDriver::VulkanDeviceTestAccess::ContextFor(device);
    auto* base = block.At(0x170000);
    auto* metadata = block.At(0x180000);
    AgcDriver::Graphics::DepthTarget target{reinterpret_cast<std::uintptr_t>(base), reinterpret_cast<std::uintptr_t>(block.At(0x160000)), {128, 128}, VK_FORMAT_D32_SFLOAT_S8_UINT, 0.625f, 0, reinterpret_cast<std::uintptr_t>(metadata), true, 4};
    FillSlice(metadata, AgcDriver::Graphics::HtileMetadataBytes(target.extent), 0xfffffff0u, 4u);
    Shape shape{Format32Float, 128, 128, 14u, 0, Swizzle64KbZX};
    shape.samples = 4;
    shape.metadataAddress = target.htileAddress;
    AgcDriver::Graphics::RememberBoundDepthSurface(context.device, target);
    for (std::uint32_t sample = 0; sample < shape.samples; ++sample) {
        Load(device, base, shape, sample, output);
        Expect(output, Bits(target.clearDepth), "undrawn fast-cleared multisample depth");
    }
    target.clearDepth = 0.875f;
    SetResolvePolicy("clear");
    Load(device, base, shape, 0, output, false, true);
    Expect(output, Bits(0.625f), "proven-clear single-sample depth proxy");
    Load(device, base, shape, 0, output, false, true, true);
    Expect(output, Bits(0.625f), "ordinary image_sample_lz over proven-clear depth");
    bool stencilRefused = false;
    try { AgcDriver::Graphics::DepthSurfaceView(context, target); }
    catch (const std::runtime_error& error) { stencilRefused = std::string(error.what()).find("stencil plane") != std::string::npos; }
    Require(stencilRefused, "an undefined stencil plane was accepted after initializing only fast-cleared depth");
    AgcDriver::Graphics::RememberBoundDepthSurface(context.device, target);
    Load(device, base, shape, 3, output);
    Expect(output, Bits(target.clearDepth), "updated bound multisample depth clear value");
    Load(device, base, shape, 0, output, false, true);
    Expect(output, Bits(target.clearDepth), "updated proven-clear depth proxy");
    const std::uint32_t uncompressed = 0xffffffffu;
    std::memcpy(metadata + 1024, &uncompressed, sizeof(uncompressed));
    bool refused = false;
    try { Load(device, base, shape, 0, output, false, true); }
    catch (const std::runtime_error& error) { refused = std::string(error.what()).find("not fast-cleared") != std::string::npos; }
    Require(refused, "partly uncompressed undrawn multisample depth was accepted");
    SetResolvePolicy("off");
}

void RefusalCase(AgcDriver::VulkanDevice& device, GuestBlock& block, std::span<std::uint32_t> output) {
    bool refused = false;
    try {
        Shape shape{Format32Float, 128, 128, Type2DArray, 2, Swizzle64KbZX};
        shape.baseArray = 1;
        Load(device, block.At(ArrayOffset), shape, 0, output);
    } catch (const std::runtime_error& error) {
        refused = std::string(error.what()).find("sampling the depth planes of depth surface") != std::string::npos;
        if (!refused) throw;
    }
    Require(refused, "a depth plane array from base slice 1 was not refused");
}

void MixedSampleCase(AgcDriver::VulkanDevice& device, GuestBlock& block, std::span<std::uint32_t> output) {
    using Approximation = AgcDriver::Graphics::MixedSampleApproximation;
    auto* base = block.At(0x120000);
    const auto address = reinterpret_cast<std::uintptr_t>(base);
    const Shape shape{Format32Float, 128, 128, Type2D, 0, Swizzle64KbZX};
    DrawDepth(device, address, {128, 128}, VK_FORMAT_D32_SFLOAT, 0.25f);
    DrawDepth(device, address, {128, 128}, VK_FORMAT_D32_SFLOAT, 0.75f, 1, Approximation::SingleSample, VK_COMPARE_OP_LESS);
    Load(device, base, shape, 0, output);
    Expect(output, Bits(0.25f), "single-sample approximation retained failing depth tests");
    DrawDepth(device, address, {128, 128}, VK_FORMAT_D32_SFLOAT, 0.125f, 1, Approximation::SingleSample, VK_COMPARE_OP_LESS);
    Load(device, base, shape, 0, output);
    Expect(output, Bits(0.125f), "single-sample approximation retained passing depth writes");
    const auto color = DrawDepth(device, address, {128, 128}, VK_FORMAT_D32_SFLOAT, 0.0f, 1, Approximation::WithoutDepth);
    Load(device, base, shape, 0, output);
    Expect(output, Bits(0.125f), "depth-free approximation changed the omitted depth plane");
    const std::span<std::uint8_t> resolvedPixels(block.At(0x1e0000), 128 * 128 * 4);
    AgcDriver::Graphics::ColorTarget resolved{reinterpret_cast<std::uintptr_t>(resolvedPixels.data()), {128, 128}, VK_FORMAT_R8G8B8A8_UNORM, resolvedPixels.size(), 0xe4u};
    resolved.tileMode = AgcDriver::Graphics::ColorTileMode::RenderTarget;
    device.ColorMetadataPass({AgcDriver::Graphics::ColorMetadataPass::Mode::Resolve, {resolved}, color});
    AgcDriver::Graphics::StorageTexture::FlushPending(resolved.address, resolved.bytes);
    device.WaitIdle();
    Require(std::ranges::any_of(resolvedPixels, [](std::uint8_t pixel) { return pixel == 255; }), "depth-free approximation did not produce color pixels");
}

}

int main() {
    try {
        SetResolvePolicy("off");
        SetTerminalMips("0");
        GuestBlock block;
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const std::span<std::uint32_t> output(reinterpret_cast<std::uint32_t*>(block.At(OutputOffset)), Threads);
        ArrayCase(*device, block, output);
        CubeCase(*device, block, output);
        IntegerCase(*device, block, output);
        DepthBitsArrayCase(*device, block, output);
        FirstSliceCase(*device, block, output);
        RefusalCase(*device, block, output);
        MultisampleCase(*device, block, output);
        ResolvedMultisampleCase(*device, block, output);
        TerminalMipCase(*device, block, output);
        AlphaCoverageCase(*device, block, output);
        FastClearedMultisampleCase(*device, block, output);
        MixedSampleCase(*device, block, output);
        std::puts("depth plane sampling tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
