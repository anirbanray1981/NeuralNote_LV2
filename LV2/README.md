# NeuralNote Guitar2MIDI — LV2 Plugin

Audio-to-MIDI transcription for LV2 hosts (Ardour, Carla, Zynthian, Pisound, etc.).
Guitar (or any mono tonal audio) → NeuralNote's BasicPitch engine → MIDI note events.

## Ports

| Index | Type | Direction | Description |
|-------|------|-----------|-------------|
| 0 | AudioPort | Input | Mono audio in (guitar / instrument) |
| 1 | AtomPort (MIDI Sequence) | Output | Transcribed MIDI note events |
| 2 | ControlPort | Input | Onset sensitivity 0.1 – 1.0 (default 0.7) |

---

## Prerequisites

### All platforms

| Tool | Minimum version |
|------|----------------|
| CMake | 3.16 |
| C++17 compiler (GCC / Clang) | GCC 10 / Clang 12 |
| LV2 development headers | 1.18 |
| pkg-config | any |

### Linux (including Raspberry Pi 4 / Pi 5 / Pisound)

```bash
# Debian / Raspberry Pi OS
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake pkg-config \
    lv2-dev liblilv-dev lilv-utils
```

#### ONNX Runtime — Linux aarch64 (Raspberry Pi)

The bundled `ThirdParty/onnxruntime/lib/libonnxruntime.a` is a **macOS-only** universal binary.
A pre-built Linux aarch64 shared library is included in:

```
ThirdParty/onnxruntime/lib-linux-aarch64/libonnxruntime.so
```

This was sourced from the official `onnxruntime` pip package (v1.24.4, generic aarch64 build).
If you want to re-fetch it yourself:

```bash
pip3 install onnxruntime --break-system-packages
ORT_CAPI=$(python3 -c "import onnxruntime,os; print(os.path.dirname(onnxruntime.__file__))")/capi
mkdir -p ThirdParty/onnxruntime/lib-linux-aarch64
cp  "$ORT_CAPI"/libonnxruntime.so.*.* \
    ThirdParty/onnxruntime/lib-linux-aarch64/libonnxruntime.so
ln -sf libonnxruntime.so \
    ThirdParty/onnxruntime/lib-linux-aarch64/libonnxruntime.so.1
```

CMake will find the library automatically in that staging directory.

> **Note:** The Debian package `libonnxruntime-dev` (v1.21+) as of 2025 is compiled for
> ARMv8.2-A and **causes SIGILL on Raspberry Pi 4** (Cortex-A72 / ARMv8-A).
> Use the pip package instead.

---

## Building

```bash
# From the repository root:
cmake -B build_lv2 \
      -DBUILD_LV2=ON \
      -DCMAKE_BUILD_TYPE=Release

cmake --build build_lv2 --target NeuralNoteGuitar2Midi_LV2 -j$(nproc)
```

On success the bundle is assembled at:

```
build_lv2/neuralnote_guitar2midi.lv2/
├── manifest.ttl
├── plugin.ttl
├── neuralnote_guitar2midi.so
└── ModelData/
    ├── cnn_contour_model.json
    ├── cnn_note_model.json
    ├── cnn_onset_1_model.json
    ├── cnn_onset_2_model.json
    └── features_model.ort
```

---

## Testing the build

### 1. Validate the bundle metadata

```bash
LV2_PATH=build_lv2 lv2info "https://github.com/DamRsn/NeuralNote/guitar2midi"
```

Expected output (abbreviated):

```
https://github.com/DamRsn/NeuralNote/guitar2midi
    Name:    NeuralNote Guitar2MIDI
    Port 0:  audio_in   (AudioPort, Input)
    Port 1:  midi_out   (AtomPort, Output)
    Port 2:  threshold  (ControlPort, Input, default=0.7)
```

### 2. Verify the shared library loads and exports `lv2_descriptor`

```bash
LD_LIBRARY_PATH=$(pwd)/ThirdParty/onnxruntime/lib-linux-aarch64 \
python3 - <<'EOF'
import ctypes
lib = ctypes.CDLL("build_lv2/neuralnote_guitar2midi.lv2/neuralnote_guitar2midi.so")
lib.lv2_descriptor.restype  = ctypes.c_void_p
lib.lv2_descriptor.argtypes = [ctypes.c_uint32]
assert lib.lv2_descriptor(0) != 0, "descriptor(0) must be non-null"
assert lib.lv2_descriptor(1) is None, "descriptor(1) must be null"
print("PASS — lv2_descriptor entry point is correct")
EOF
```

### 3. Scan with lv2lint (optional, requires `lv2lint` package)

```bash
sudo apt-get install -y lv2lint
LD_LIBRARY_PATH=$(pwd)/ThirdParty/onnxruntime/lib-linux-aarch64 \
LV2_PATH=build_lv2 \
lv2lint "https://github.com/DamRsn/NeuralNote/guitar2midi"
```

---

## Installing

Copy the bundle to an LV2 search path that your host scans:

```bash
# User install (recommended)
cp -r build_lv2/neuralnote_guitar2midi.lv2 ~/.lv2/

# System-wide install
sudo cp -r build_lv2/neuralnote_guitar2midi.lv2 /usr/local/lib/lv2/
```

The plugin needs the onnxruntime shared library at runtime.
Either install it system-wide:

```bash
sudo cp ThirdParty/onnxruntime/lib-linux-aarch64/libonnxruntime.so   /usr/local/lib/
sudo cp ThirdParty/onnxruntime/lib-linux-aarch64/libonnxruntime.so.1 /usr/local/lib/
sudo ldconfig
```

Or set `LD_LIBRARY_PATH` before launching your LV2 host:

```bash
export LD_LIBRARY_PATH=/path/to/NeuralNote/ThirdParty/onnxruntime/lib-linux-aarch64:$LD_LIBRARY_PATH
ardour7  # or carla, jalv, etc.
```

---

## Quick functional test with jalv

```bash
sudo apt-get install -y jalv

# Install the bundle and onnxruntime first (see above), then:
jalv.gtk "https://github.com/DamRsn/NeuralNote/guitar2midi"
```

Connect a guitar/microphone audio source to Port 0 and route Port 1 (MIDI) to
a synthesizer or MIDI recorder. Play and watch notes appear.

---

## Architecture overview

```
lv2_run()
  ├─ flush pending MIDI note events → Atom sequence output
  ├─ update noteSensitivity from control port
  ├─ resample audio block → 22050 Hz accumulation buffer
  └─ when buffer ≥ 2 s of audio:
       ├─ BasicPitch::setParameters(sensitivity, splitSensitivity, minNoteMs)
       ├─ BasicPitch::transcribeToMIDI(buffer, nSamples)
       │     ├─ Features (ONNX CQT)  → stacked CQT frames
       │     └─ BasicPitchCNN (RTNeural) → note/onset/contour posteriorgrams
       ├─ BasicPitch::getNoteEvents() → Notes::Event vector
       └─ convert Events → pending {note-on, note-off} pairs
```

The inference runs **synchronously** in `lv2_run()` every 2 seconds of audio.
For a future hard-real-time version the 2-second batch can be moved to an
`lv2:WorkerInterface` thread.

---

## Key source files

| File | Purpose |
|------|---------|
| `LV2/neuralnote_guitar2midi.cpp` | LV2 plugin entry point |
| `LV2/BinaryData.h` | File-loading substitute for JUCE BinaryData (loads CNN/ONNX model weights from the bundle's `ModelData/`) |
| `LV2/NoteUtils.h` | JUCE-free stub for `Lib/Utils/NoteUtils.h` |
| `Lib/Model/BasicPitch.{h,cpp}` | Top-level transcription pipeline |
| `Lib/Model/Features.{h,cpp}` | ONNX CQT feature extraction |
| `Lib/Model/BasicPitchCNN.{h,cpp}` | RTNeural CNN inference |
| `Lib/Model/Notes.{h,cpp}` | Posteriorgram → note event conversion |

---

## CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_LV2` | `OFF` | Enable the LV2 plugin target |
| `BUILD_UNIT_TESTS` | `OFF` | Build unit tests |
| `RTNeural_Release` | `OFF` | Force Release optimisation for RTNeural in Debug builds |
| `LTO` | `ON` | Link-Time Optimisation |
