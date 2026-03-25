/**
 * NeuralNote Guitar2MIDI — LV2 Plugin
 *
 * Audio in (guitar) → NeuralNote transcription engine → MIDI Atom out
 *
 * Uses NeuralNote's Lib/Model unchanged. The plugin accumulates audio into
 * a 22050 Hz buffer, runs transcription when enough samples have arrived,
 * and emits note-on / note-off events as LV2 MIDI Atoms.
 *
 * Ports:
 *   0  — AudioIn  (lv2:AudioPort, lv2:InputPort)
 *   1  — MidiOut  (atom:AtomPort, lv2:OutputPort, atom:Sequence of midi:MidiEvent)
 *   2  — Threshold (lv2:ControlPort, lv2:InputPort) — onset sensitivity 0..1
 */

#include <lv2/core/lv2.h>
#include <lv2/atom/atom.h>
#include <lv2/atom/forge.h>
#include <lv2/atom/util.h>
#include <lv2/midi/midi.h>
#include <lv2/urid/urid.h>
#include <lv2/log/log.h>
#include <lv2/log/logger.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>

// BinaryData.h must be included before any Lib/Model headers so the LV2
// file-loading substitute is used instead of the JUCE-generated version.
#include "BinaryData.h"

// NeuralNote's transcription engine (Lib/Model — unchanged)
#include "BasicPitch.h"   // wraps Features + BasicPitchCNN + Notes

#define PLUGIN_URI "https://github.com/DamRsn/NeuralNote/guitar2midi"

// ── Port indices ─────────────────────────────────────────────────────────────
enum PortIndex {
    PORT_AUDIO_IN   = 0,
    PORT_MIDI_OUT   = 1,
    PORT_THRESHOLD  = 2,
};

// ── Mapped URIDs ─────────────────────────────────────────────────────────────
struct URIs {
    LV2_URID atom_Sequence;
    LV2_URID atom_EventTransfer;
    LV2_URID midi_MidiEvent;
};

static void mapURIs(LV2_URID_Map* map, URIs* uris)
{
    uris->atom_Sequence      = map->map(map->handle, LV2_ATOM__Sequence);
    uris->atom_EventTransfer = map->map(map->handle, LV2_ATOM__eventTransfer);
    uris->midi_MidiEvent     = map->map(map->handle, LV2_MIDI__MidiEvent);
}

// ── Pending MIDI event ────────────────────────────────────────────────────────
struct PendingNote {
    bool    noteOn;
    uint8_t pitch;
    uint8_t velocity;
};

// ── Plugin instance ───────────────────────────────────────────────────────────
struct NeuralNotePlugin {
    // LV2 infrastructure
    LV2_URID_Map*   map;
    LV2_Log_Logger  logger;
    LV2_Atom_Forge  forge;
    URIs            uris;

    // Ports
    const float*        audioIn;
    LV2_Atom_Sequence*  midiOut;
    const float*        threshold;

    // NeuralNote engine
    std::unique_ptr<BasicPitch> basicPitch;

    // Audio accumulation at 22050 Hz (BasicPitch expects this sample rate)
    double             sampleRate;
    std::vector<float> accumulator;
    static constexpr int BLOCK_SAMPLES = 22050 * 2; // 2-second inference window

    // Sensitivity parameters (updated from control port each block)
    float noteSensitivity  = 0.7f;
    float splitSensitivity = 0.5f;
    float minNoteLengthMs  = 50.0f;

    // Note output queue (written by inference, read same thread — mutex for clarity)
    std::vector<PendingNote> pendingNotes;
    std::mutex               notesMutex;

    bool noteActive[128] = {};
};

// ── LV2 lifecycle ─────────────────────────────────────────────────────────────

static LV2_Handle instantiate(const LV2_Descriptor*     descriptor,
                               double                    rate,
                               const char*               bundlePath,
                               const LV2_Feature* const* features)
{
    NeuralNotePlugin* self = new NeuralNotePlugin();
    self->sampleRate = rate;
    self->map        = nullptr;

    for (int i = 0; features[i]; ++i) {
        if (!strcmp(features[i]->URI, LV2_URID__map))
            self->map = static_cast<LV2_URID_Map*>(features[i]->data);
        else if (!strcmp(features[i]->URI, LV2_LOG__log))
            lv2_log_logger_init(&self->logger, self->map,
                                static_cast<LV2_Log_Log*>(features[i]->data));
    }

    if (!self->map) {
        delete self;
        return nullptr;
    }

    mapURIs(self->map, &self->uris);
    lv2_atom_forge_init(&self->forge, self->map);

    // Load model weights from the bundle's ModelData/ directory.
    // BinaryData::init() fills the in-memory pointers used by Features and
    // BasicPitchCNN — must happen before constructing BasicPitch.
    try {
        BinaryData::init(bundlePath);
    } catch (const std::exception& e) {
        lv2_log_error(&self->logger, "NeuralNote: %s\n", e.what());
        delete self;
        return nullptr;
    }

    self->basicPitch = std::make_unique<BasicPitch>();
    self->basicPitch->setParameters(self->noteSensitivity,
                                    self->splitSensitivity,
                                    self->minNoteLengthMs);

    self->accumulator.reserve(static_cast<size_t>(self->BLOCK_SAMPLES) + 4096);
    memset(self->noteActive, 0, sizeof(self->noteActive));

    lv2_log_note(&self->logger,
                 "NeuralNote Guitar2MIDI: instantiated at %.0f Hz\n", rate);
    return static_cast<LV2_Handle>(self);
}

static void connectPort(LV2_Handle instance, uint32_t port, void* data)
{
    NeuralNotePlugin* self = static_cast<NeuralNotePlugin*>(instance);
    switch (static_cast<PortIndex>(port)) {
        case PORT_AUDIO_IN:  self->audioIn   = static_cast<const float*>(data);       break;
        case PORT_MIDI_OUT:  self->midiOut   = static_cast<LV2_Atom_Sequence*>(data); break;
        case PORT_THRESHOLD: self->threshold = static_cast<const float*>(data);       break;
    }
}

static void activate(LV2_Handle instance)
{
    NeuralNotePlugin* self = static_cast<NeuralNotePlugin*>(instance);
    self->accumulator.clear();
    memset(self->noteActive, 0, sizeof(self->noteActive));
    self->basicPitch->reset();
}

// ── Helper: linear resample srcRate → 22050 Hz ───────────────────────────────
static void resampleLinear(const float* in, int inLen, double srcRate,
                            std::vector<float>& out)
{
    if (srcRate == 22050.0) {
        out.insert(out.end(), in, in + inLen);
        return;
    }
    const double ratio  = 22050.0 / srcRate;
    const int    outLen = static_cast<int>(inLen * ratio);
    for (int i = 0; i < outLen; ++i) {
        const double srcPos = i / ratio;
        const int    s0     = static_cast<int>(srcPos);
        const double frac   = srcPos - s0;
        const int    s1     = std::min(s0 + 1, inLen - 1);
        out.push_back(static_cast<float>((1.0 - frac) * in[s0] + frac * in[s1]));
    }
}

// ── Helper: write a 3-byte MIDI event into the LV2 Atom sequence ──────────────
static void writeMidi(LV2_Atom_Forge* forge, uint32_t frames,
                       LV2_URID midiType, uint8_t b0, uint8_t b1, uint8_t b2)
{
    uint8_t msg[3] = {b0, b1, b2};
    lv2_atom_forge_frame_time(forge, frames);
    lv2_atom_forge_atom(forge, 3, midiType);
    lv2_atom_forge_write(forge, msg, 3);
}

// ── run() — called every audio block ─────────────────────────────────────────
static void run(LV2_Handle instance, uint32_t nSamples)
{
    NeuralNotePlugin* self = static_cast<NeuralNotePlugin*>(instance);

    // Set up the Atom forge to write into midiOut
    const uint32_t outCapacity = self->midiOut->atom.size;
    lv2_atom_forge_set_buffer(&self->forge,
                               reinterpret_cast<uint8_t*>(self->midiOut),
                               outCapacity);
    LV2_Atom_Forge_Frame seqFrame;
    lv2_atom_forge_sequence_head(&self->forge, &seqFrame, 0);

    // Flush pending notes from last inference run
    {
        std::lock_guard<std::mutex> lock(self->notesMutex);
        for (const auto& pn : self->pendingNotes) {
            if (pn.noteOn) {
                if (!self->noteActive[pn.pitch]) {
                    writeMidi(&self->forge, 0, self->uris.midi_MidiEvent,
                              0x90, pn.pitch, pn.velocity);
                    self->noteActive[pn.pitch] = true;
                }
            } else {
                if (self->noteActive[pn.pitch]) {
                    writeMidi(&self->forge, 0, self->uris.midi_MidiEvent,
                              0x80, pn.pitch, 0);
                    self->noteActive[pn.pitch] = false;
                }
            }
        }
        self->pendingNotes.clear();
    }

    // Update onset sensitivity from control port
    if (self->threshold)
        self->noteSensitivity = *self->threshold;

    // Resample and accumulate audio
    resampleLinear(self->audioIn, static_cast<int>(nSamples),
                   self->sampleRate, self->accumulator);

    // When we have enough audio, run inference
    if (static_cast<int>(self->accumulator.size()) >= self->BLOCK_SAMPLES) {
        self->basicPitch->setParameters(self->noteSensitivity,
                                        self->splitSensitivity,
                                        self->minNoteLengthMs);
        self->basicPitch->transcribeToMIDI(self->accumulator.data(),
                                           static_cast<int>(self->accumulator.size()));

        const auto& notes = self->basicPitch->getNoteEvents();

        {
            std::lock_guard<std::mutex> lock(self->notesMutex);
            for (const auto& note : notes) {
                const uint8_t vel = static_cast<uint8_t>(
                    std::clamp(static_cast<int>(note.amplitude * 127.0), 1, 127));
                self->pendingNotes.push_back({true,  static_cast<uint8_t>(note.pitch), vel});
                self->pendingNotes.push_back({false, static_cast<uint8_t>(note.pitch), 0});
            }
        }

        self->accumulator.clear();
    }

    lv2_atom_forge_pop(&self->forge, &seqFrame);
}

static void deactivate(LV2_Handle /*instance*/) {}

static void cleanup(LV2_Handle instance)
{
    delete static_cast<NeuralNotePlugin*>(instance);
}

static const void* extensionData(const char* /*uri*/) { return nullptr; }

// ── LV2 descriptor ────────────────────────────────────────────────────────────
static const LV2_Descriptor descriptor = {
    PLUGIN_URI,
    instantiate,
    connectPort,
    activate,
    run,
    deactivate,
    cleanup,
    extensionData,
};

LV2_SYMBOL_EXPORT const LV2_Descriptor* lv2_descriptor(uint32_t index)
{
    return (index == 0) ? &descriptor : nullptr;
}
