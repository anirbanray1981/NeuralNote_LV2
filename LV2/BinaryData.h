/**
 * BinaryData.h — LV2 build substitute for JUCE BinaryData.
 *
 * Provides the same BinaryData:: symbols that Features.cpp and
 * BasicPitchCNN.cpp reference, but backed by runtime file I/O instead
 * of JUCE-embedded binary blobs.
 *
 * Call BinaryData::init(bundlePath) once (from lv2_instantiate) before
 * constructing any Features or BasicPitchCNN objects.
 */
#pragma once
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace BinaryData {

// ── internal storage ─────────────────────────────────────────────────────────
inline std::vector<char> _buf_cnn_contour;
inline std::vector<char> _buf_cnn_note;
inline std::vector<char> _buf_cnn_onset1;
inline std::vector<char> _buf_cnn_onset2;
inline std::vector<char> _buf_features;

// ── public symbols (same names JUCE BinaryData generates) ────────────────────
inline const char* cnn_contour_model_json     = nullptr;
inline int         cnn_contour_model_jsonSize  = 0;

inline const char* cnn_note_model_json        = nullptr;
inline int         cnn_note_model_jsonSize     = 0;

inline const char* cnn_onset_1_model_json     = nullptr;
inline int         cnn_onset_1_model_jsonSize  = 0;

inline const char* cnn_onset_2_model_json     = nullptr;
inline int         cnn_onset_2_model_jsonSize  = 0;

inline const char* features_model_ort         = nullptr;
inline int         features_model_ortSize      = 0;

// ── internal helper ──────────────────────────────────────────────────────────
inline void _loadFile(const std::string& path, std::vector<char>& buf,
                      const char*& ptr, int& size)
{
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
        throw std::runtime_error("BinaryData: cannot open " + path);
    std::fseek(f, 0, SEEK_END);
    long len = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    buf.resize(static_cast<size_t>(len));
    std::fread(buf.data(), 1, static_cast<size_t>(len), f);
    std::fclose(f);
    ptr  = buf.data();
    size = static_cast<int>(len);
}

// ── public init ──────────────────────────────────────────────────────────────
/**
 * Load all model files from <bundlePath>/ModelData/.
 * Must be called before constructing Features or BasicPitchCNN.
 */
inline void init(const std::string& bundlePath)
{
    std::string dir = bundlePath;
    if (!dir.empty() && dir.back() != '/')
        dir += '/';
    dir += "ModelData/";

    _loadFile(dir + "cnn_contour_model.json", _buf_cnn_contour,
              cnn_contour_model_json,  cnn_contour_model_jsonSize);
    _loadFile(dir + "cnn_note_model.json",    _buf_cnn_note,
              cnn_note_model_json,     cnn_note_model_jsonSize);
    _loadFile(dir + "cnn_onset_1_model.json", _buf_cnn_onset1,
              cnn_onset_1_model_json,  cnn_onset_1_model_jsonSize);
    _loadFile(dir + "cnn_onset_2_model.json", _buf_cnn_onset2,
              cnn_onset_2_model_json,  cnn_onset_2_model_jsonSize);
    _loadFile(dir + "features_model.ort",     _buf_features,
              features_model_ort,      features_model_ortSize);
}

} // namespace BinaryData
