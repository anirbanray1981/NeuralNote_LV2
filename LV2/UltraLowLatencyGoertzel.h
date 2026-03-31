#pragma once
/**
 * UltraLowLatencyGoertzel — sample-by-sample polyphonic pitch detection.
 *
 * Processes audio at native sample rate (48 kHz) using NEON SIMD.
 * 4 bins per SIMD lane × 13 groups = 52 slots (49 notes E2–C6 + 3 padding).
 *
 * Features:
 * - Hann window applied as block pre-pass (reduces spectral leakage)
 * - Exponential decay prevents unbounded energy accumulation
 * - Separate activeCount/inactiveCount debounce (clean hysteresis)
 * - Winner-takes-all per octave (strongest bin in 12-note range wins)
 * - Multi-interval harmonic suppression
 */

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>

class UltraLowLatencyGoertzel {
public:
#ifdef __aarch64__
    struct alignas(64) BinGroup {
        float32x4_t s1;
        float32x4_t s2;
        float32x4_t coeff;
        float32x4_t mag;
        float32x4_t decay;
    };
#else
    struct alignas(64) BinGroup {
        float s1[4];
        float s2[4];
        float coeff[4];
        float mag[4];
        float decay[4];
    };
#endif

    struct NoteState {
        int   activeCount   = 0;
        int   inactiveCount = 0;
        bool  isMidiOn      = false;
        bool  triggerPending = false;
        int   velocity       = 0;
        float currentMag     = 0.0f;

        // Aliases for external code
        bool isActive() const { return isMidiOn; }
    };

    static constexpr int   CONFIDENCE_BLOCKS = 3;       // ~4ms at 64-sample blocks
    static constexpr int   MAX_BLOCK_SIZE    = 256;      // max supported block size
    static constexpr float DECAY_FACTOR      = 0.99985f; // ~50ms half-life at 48kHz
    static constexpr float ON_THRESHOLD      = 0.08f;    // magnitude to trigger note-ON
    static constexpr float OFF_THRESHOLD     = 0.02f;    // magnitude to trigger note-OFF
    static constexpr float HANN_ENERGY_COMP  = 2.0f;     // compensate Hann ~50% energy loss

    UltraLowLatencyGoertzel() = default;

    UltraLowLatencyGoertzel(float fs, int startMidi = 40, int endMidi = 84)
    {
        init(fs, startMidi, endMidi);
    }

    void init(float fs, int startMidi = 40, int endMidi = 84)
    {
        sampleRate_ = fs;
        startMidi_  = startMidi;
        numNotes_   = (endMidi - startMidi) + 1;
        numGroups_  = (numNotes_ + 3) / 4;

        groups_.resize(numGroups_);
        noteStates_.resize(numNotes_);
        magBuf_.resize(numGroups_ * 4, 0.0f);

        // Pre-compute Hann window (up to MAX_BLOCK_SIZE)
        hannWindow_.resize(MAX_BLOCK_SIZE);
        for (int i = 0; i < MAX_BLOCK_SIZE; ++i)
            hannWindow_[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (MAX_BLOCK_SIZE - 1)));
        windowedBuf_.resize(MAX_BLOCK_SIZE, 0.0f);

        for (int i = 0; i < numGroups_; ++i) {
            float c[4] = {0, 0, 0, 0};
            for (int j = 0; j < 4; ++j) {
                int midi = startMidi + (i * 4) + j;
                if (midi <= endMidi) {
                    float f = 440.0f * std::pow(2.0f, (midi - 69) / 12.0f);
                    c[j] = 2.0f * std::cos(2.0f * M_PI * f / fs);
                }
            }
#ifdef __aarch64__
            groups_[i].coeff = vld1q_f32(c);
            groups_[i].s1    = vdupq_n_f32(0.0f);
            groups_[i].s2    = vdupq_n_f32(0.0f);
            groups_[i].mag   = vdupq_n_f32(0.0f);
            groups_[i].decay = vdupq_n_f32(DECAY_FACTOR);
#else
            for (int j = 0; j < 4; ++j) {
                groups_[i].coeff[j] = c[j];
                groups_[i].s1[j] = groups_[i].s2[j] = groups_[i].mag[j] = 0.0f;
                groups_[i].decay[j] = DECAY_FACTOR;
            }
#endif
        }

        // Recompute Hann for actual block size on first processBlock call
        hannReady_ = false;
    }

    // Process a block of audio: apply Hann window, run Goertzel, update notes.
    // Call once per audio callback with the full buffer.
    // transientRatio: PickDetector's fast/slow ratio
    void processBlock(const float* input, int nSamples, float transientRatio = 1.0f)
    {
        if (nSamples <= 0 || nSamples > MAX_BLOCK_SIZE) return;

        // Recompute Hann window if block size changed
        if (!hannReady_ || nSamples != lastBlockSize_) {
            for (int i = 0; i < nSamples; ++i)
                hannWindow_[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (nSamples - 1)));
            lastBlockSize_ = nSamples;
            hannReady_ = true;
        }

        // 1. Apply Hann window with energy compensation
        for (int i = 0; i < nSamples; ++i)
            windowedBuf_[i] = input[i] * hannWindow_[i] * HANN_ENERGY_COMP;

        // 2. Feed windowed samples through Goertzel IIR
        for (int n = 0; n < nSamples; ++n) {
            const float sample = windowedBuf_[n];
#ifdef __aarch64__
            const float32x4_t inp = vdupq_n_f32(sample);
            for (int i = 0; i < numGroups_; ++i) {
                BinGroup& g = groups_[i];
                float32x4_t s0 = vmlaq_f32(inp, g.coeff, g.s1);
                s0 = vsubq_f32(s0, g.s2);
                s0 = vmulq_f32(s0, g.decay);
                g.s2 = vmulq_f32(g.s1, g.decay);
                g.s1 = s0;
            }
#else
            for (int i = 0; i < numGroups_; ++i) {
                BinGroup& g = groups_[i];
                for (int j = 0; j < 4; ++j) {
                    float s0 = sample + g.coeff[j] * g.s1[j] - g.s2[j];
                    s0 *= g.decay[j];
                    g.s2[j] = g.s1[j] * g.decay[j];
                    g.s1[j] = s0;
                }
            }
#endif
        }

        // 3. Compute magnitudes
#ifdef __aarch64__
        for (int i = 0; i < numGroups_; ++i) {
            BinGroup& g = groups_[i];
            float32x4_t s1_2 = vmulq_f32(g.s1, g.s1);
            float32x4_t s2_2 = vmulq_f32(g.s2, g.s2);
            float32x4_t prod = vmulq_f32(vmulq_f32(g.s1, g.s2), g.coeff);
            g.mag = vsubq_f32(vaddq_f32(s1_2, s2_2), prod);
            vst1q_f32(&magBuf_[i * 4], g.mag);
        }
#else
        for (int i = 0; i < numGroups_; ++i) {
            BinGroup& g = groups_[i];
            for (int j = 0; j < 4; ++j) {
                g.mag[j] = g.s1[j] * g.s1[j] + g.s2[j] * g.s2[j]
                           - g.coeff[j] * g.s1[j] * g.s2[j];
                magBuf_[i * 4 + j] = g.mag[j];
            }
        }
#endif

        // 4. Apply harmonic suppression, winner-takes-all, and note tracking
        updateNotes(transientRatio);
    }

    void reset()
    {
        for (int i = 0; i < numGroups_; ++i) {
#ifdef __aarch64__
            groups_[i].s1  = vdupq_n_f32(0.0f);
            groups_[i].s2  = vdupq_n_f32(0.0f);
            groups_[i].mag = vdupq_n_f32(0.0f);
#else
            for (int j = 0; j < 4; ++j)
                groups_[i].s1[j] = groups_[i].s2[j] = groups_[i].mag[j] = 0.0f;
#endif
        }
        for (auto& s : noteStates_) {
            s.activeCount = s.inactiveCount = 0;
            s.isMidiOn = s.triggerPending = false;
            s.velocity = 0;
            s.currentMag = 0.0f;
        }
    }

    int startMidi() const { return startMidi_; }
    int numNotes()  const { return numNotes_; }
    std::vector<NoteState>& getNoteStates() { return noteStates_; }
    const std::vector<NoteState>& getNoteStates() const { return noteStates_; }

private:
    void updateNotes(float transientRatio)
    {
        float* m = magBuf_.data();
        const float dynamicOn = ON_THRESHOLD * (1.0f + transientRatio * 2.0f);

        // Harmonic suppression pass
        for (int i = 0; i < numNotes_; ++i) {
            float val = m[i];
            if (i >= 12 && m[i - 12] > val * 0.25f) val *= 0.02f;  // octave
            if (i >= 24 && m[i - 24] > val * 0.25f) val *= 0.02f;  // 2 octaves
            if (i >= 7  && m[i - 7]  > val * 0.4f)  val *= 0.05f;  // fifth
            if (i >= 19 && m[i - 19] > val * 0.4f)  val *= 0.05f;  // octave+fifth
            if (i >= 4  && m[i - 4]  > val * 0.5f)  val *= 0.1f;   // major third
            if (i >= 5  && m[i - 5]  > val * 0.5f)  val *= 0.1f;   // fourth
            if (i >= 16 && m[i - 16] > val * 0.4f)  val *= 0.05f;  // octave+fourth
            if (i >= 28 && m[i - 28] > val * 0.3f)  val *= 0.02f;  // 2oct+third
            m[i] = val;  // write back suppressed value
        }

        // Winner-takes-all per octave: bin must be the strongest in its
        // 12-note range to claim detection.
        for (int i = 0; i < numNotes_; ++i) {
            if (m[i] <= 0.0f) continue;
            // Find max in the octave centered on this note
            const int lo = std::max(0, i - 6);
            const int hi = std::min(numNotes_ - 1, i + 5);
            float maxInOctave = 0.0f;
            for (int k = lo; k <= hi; ++k)
                if (k != i && m[k] > maxInOctave) maxInOctave = m[k];
            // Must be the winner by a clear margin
            if (maxInOctave > 0.0f && m[i] < maxInOctave)
                m[i] = 0.0f;
        }

        // Note tracking with separate active/inactive counters
        for (int i = 0; i < numNotes_; ++i) {
            const float val = m[i];
            NoteState& s = noteStates_[i];
            s.currentMag = val;

            if (val > dynamicOn) {
                s.activeCount++;
                s.inactiveCount = 0;

                if (s.activeCount >= CONFIDENCE_BLOCKS && !s.isMidiOn) {
                    s.isMidiOn       = true;
                    s.triggerPending = true;
                    float norm = std::clamp(val / (ON_THRESHOLD * 20.0f), 0.0f, 1.0f);
                    s.velocity = static_cast<int>(127.0f * std::pow(norm, 1.0f / 1.2f));
                    s.velocity = std::max(10, s.velocity);
                }
            } else if (val < OFF_THRESHOLD) {
                s.inactiveCount++;
                s.activeCount = 0;

                if (s.inactiveCount >= CONFIDENCE_BLOCKS && s.isMidiOn) {
                    s.isMidiOn       = false;
                    s.triggerPending = false;
                }
            }
            // Between OFF_THRESHOLD and dynamicOn: no counter changes (hysteresis hold)
        }
    }

    float sampleRate_ = 48000.0f;
    int   startMidi_  = 40;
    int   numNotes_   = 0;
    int   numGroups_  = 0;
    bool  hannReady_  = false;
    int   lastBlockSize_ = 0;
    std::vector<BinGroup>   groups_;
    std::vector<NoteState>  noteStates_;
    std::vector<float>      magBuf_;
    std::vector<float>      hannWindow_;
    std::vector<float>      windowedBuf_;
};
