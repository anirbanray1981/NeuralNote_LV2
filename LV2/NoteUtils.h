/**
 * NoteUtils.h — LV2 build substitute (JUCE-free).
 *
 * Provides only the math utilities used by Notes.cpp; the JUCE
 * StringArray/String UI helpers are omitted as they are not needed
 * in the LV2 context.
 *
 * This file is found first on the include path (LV2/ precedes Lib/Utils/)
 * so that #include "NoteUtils.h" in Notes.h resolves here instead of
 * Lib/Utils/NoteUtils.h which depends on JuceHeader.h.
 */
#pragma once
#include <algorithm>
#include <cmath>

namespace NoteUtils {

static inline int hzToMidi(float hz)
{
    return static_cast<int>(std::round(12.0f * std::log2(hz / 440.0f) + 69.0f));
}

static inline float midiToHz(float inMidiNote)
{
    return 440.0f * std::pow(2.0f, (inMidiNote - 69.0f) / 12.0f);
}

} // namespace NoteUtils
