#include "SceTypes.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

extern "C" {
int APS5_VABI sceAudioOutInit();
int APS5_VABI sceAudioOutGetSystemState(AudioOutSystemState*);
int APS5_VABI sceAudioOutOpen(int, int, int, std::uint32_t, std::uint32_t, std::uint32_t);
int APS5_VABI sceAudioOutOutput(int, const void*);
int APS5_VABI sceAudioOutClose(int);
}

static void Require(bool value, const char* message) {
    if (value) return;
    std::fprintf(stderr, "%s\n", message);
    std::abort();
}

int main() {
    static_assert(sizeof(AudioOutSystemState) == 32);
#ifdef _WIN32
    _putenv_s("SDL_AUDIODRIVER", "dummy");
#else
    setenv("SDL_AUDIODRIVER", "dummy", 1);
#endif
    struct GuardedState {
        std::uint64_t before;
        AudioOutSystemState state;
        std::uint64_t after;
    } guarded;
    std::memset(&guarded, 0xA5, sizeof(guarded));
    const auto untouched = guarded;
    Require(sceAudioOutGetSystemState(&guarded.state) == static_cast<int>(0x8026000F), "uninitialized query must fail");
    Require(std::memcmp(&guarded, &untouched, sizeof(guarded)) == 0, "failed query must preserve the output");
    Require(sceAudioOutInit() == 0, "initialization must succeed");
    Require(sceAudioOutGetSystemState(nullptr) == static_cast<int>(0x80260004), "null output must be rejected");
    Require(sceAudioOutGetSystemState(&guarded.state) == 0, "idle query must succeed");
    const std::array<unsigned char, 32> zero{};
    Require(std::memcmp(&guarded.state, zero.data(), zero.size()) == 0, "idle query must report silence and clear reserved fields");
    Require(guarded.before == untouched.before && guarded.after == untouched.after, "query must only write its 32-byte output");
    const int port = sceAudioOutOpen(0x10000000, 0, 0, 48000, 48000, 0);
    Require(port > 0, "audio port must open");
    const std::vector<std::int16_t> block(48000, 8192);
    Require(sceAudioOutOutput(port, block.data()) == 48000, "audio must queue");
    std::memset(&guarded.state, 0xA5, sizeof(guarded.state));
    bool threw = false;
    try {
        sceAudioOutGetSystemState(&guarded.state);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    Require(threw, "live loudness must fail explicitly until implemented");
    Require(std::memcmp(&guarded.state, &untouched.state, sizeof(guarded.state)) == 0, "unsupported query must preserve the output");
    Require(sceAudioOutClose(port) == 0, "audio port must close");
    Require(sceAudioOutGetSystemState(&guarded.state) == 0, "query must recover after the port closes");
    return 0;
}
