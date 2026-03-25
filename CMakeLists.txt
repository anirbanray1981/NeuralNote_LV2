cmake_minimum_required(VERSION 3.16)
project(NeuralNotePlugin VERSION 1.1.0)

# C++ settings
set(CMAKE_CXX_STANDARD 17)
enable_language(CXX)

#Compile commands, useful for some IDEs like VS-Code
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

#Minimum MacOS target, set globally
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.11" CACHE STRING "Minimum OS X deployment version" FORCE)

option(UniversalBinary "Build universal binary for mac" OFF)
option(RTNeural_Release "When CMAKE_BUILD_TYPE=Debug, overwrite it to Release for RTNeural only" OFF)
option(LTO "Enable Link Time Optimization" ON)

if (UniversalBinary)
    set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE INTERNAL "")
    message(STATUS "Building Universal Binary")
endif ()

#static linking in Windows
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

add_subdirectory(ThirdParty/JUCE)
add_subdirectory(ThirdParty/RTNeural)

set(BaseTargetName NeuralNote)

# Set COPY_PLUGIN_AFTER_BUILD to TRUE on Mac and to FALSE on Windows due to permission issues
if (APPLE)
    set(COPY_PLUGIN_AFTER_BUILD TRUE)
else ()
    set(COPY_PLUGIN_AFTER_BUILD FALSE)
endif ()

juce_add_plugin("${BaseTargetName}"
        # VERSION ...                               # Set this if the plugin version is different to the project version
        ICON_BIG "${CMAKE_CURRENT_LIST_DIR}/NeuralNote/Assets/logo.png"
        ICON_SMALL "${CMAKE_CURRENT_LIST_DIR}/NeuralNote/Assets/logo.png"
        COMPANY_NAME "Dr. Audio"
        BUNDLE_ID "com.draudio.neuralnote"
        IS_SYNTH FALSE
        NEEDS_MIDI_INPUT FALSE
        NEEDS_MIDI_OUTPUT TRUE
        IS_MIDI_EFFECT FALSE
        EDITOR_WANTS_KEYBOARD_FOCUS FALSE
        COPY_PLUGIN_AFTER_BUILD ${COPY_PLUGIN_AFTER_BUILD}
        MICROPHONE_PERMISSION_ENABLED TRUE
        MICROPHONE_PERMISSION_TEXT "Need access to Microphone"
        HARDENED_RUNTIME_ENABLED TRUE
        PLUGIN_MANUFACTURER_CODE DrAu
        PLUGIN_CODE NRNT
        FORMATS AU VST3 Standalone
        PRODUCT_NAME "NeuralNote")

juce_generate_juce_header(${BaseTargetName})

# Source files
file(GLOB_RECURSE SOURCES_PLUGIN ${CMAKE_CURRENT_LIST_DIR}/NeuralNote/*.cpp ${CMAKE_CURRENT_LIST_DIR}/Lib/*.cpp)
list(REMOVE_ITEM SOURCES_PLUGIN ${CMAKE_CURRENT_LIST_DIR}/Lib/Model/BasicPitchCNN.cpp)
file(GLOB_RECURSE HEADERS_PLUGIN ${CMAKE_CURRENT_LIST_DIR}/NeuralNote/*.h ${CMAKE_CURRENT_LIST_DIR}/Lib/*.h)
list(REMOVE_ITEM HEADERS_PLUGIN ${CMAKE_CURRENT_LIST_DIR}/Lib/Model/BasicPitchCNN.h)

target_sources(${BaseTargetName} PRIVATE ${SOURCES_PLUGIN} ${HEADERS_PLUGIN})

file(GLOB_RECURSE lib_dirs LIST_DIRECTORIES true ${CMAKE_CURRENT_LIST_DIR}/Lib/*)

foreach (dir ${lib_dirs})
    IF (IS_DIRECTORY ${dir})
        target_include_directories(${BaseTargetName} PRIVATE ${dir})
    ELSE ()
        CONTINUE()
    ENDIF ()
endforeach ()

file(GLOB_RECURSE source_dirs LIST_DIRECTORIES true ${CMAKE_CURRENT_LIST_DIR}/NeuralNote/*)

foreach (dir ${source_dirs})
    IF (IS_DIRECTORY ${dir})
        target_include_directories(${BaseTargetName} PRIVATE ${dir})
    ELSE ()
        CONTINUE()
    ENDIF ()
endforeach ()

#Binary data
file(GLOB RESOURCES_FILES ${CMAKE_CURRENT_LIST_DIR}/Lib/ModelData/*.json
        ${CMAKE_CURRENT_LIST_DIR}/Lib/ModelData/*.ort
        ${CMAKE_CURRENT_LIST_DIR}/NeuralNote/Assets/*.ttf
        ${CMAKE_CURRENT_LIST_DIR}/NeuralNote/Assets/*.png
        ${CMAKE_CURRENT_LIST_DIR}/NeuralNote/Assets/*.svg)

juce_add_binary_data(bin_data SOURCES ${RESOURCES_FILES})

add_library(BasicPitchCNN STATIC ${CMAKE_CURRENT_LIST_DIR}/Lib/Model/BasicPitchCNN.cpp)
target_include_directories(BasicPitchCNN PRIVATE ${CMAKE_CURRENT_LIST_DIR}/ThirdParty/RTNeural)
if ((CMAKE_BUILD_TYPE STREQUAL "Debug") AND RTNeural_Release)
    if (MSVC)
        target_compile_options(BasicPitchCNN PUBLIC /O2) # or maybe /Ox
    else () # clang or gcc
        target_compile_options(BasicPitchCNN PUBLIC -O3) # or maybe -Ofast
    endif ()
endif ()
target_link_libraries(BasicPitchCNN PUBLIC RTNeural bin_data)

target_include_directories(${BaseTargetName} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/ONNXRuntime/${ONNXRUNTIME_DIRNAME}/include)
target_include_directories(${BaseTargetName} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/ThirdParty/minimp3)

target_compile_definitions(${BaseTargetName}
        PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0)

add_library(onnxruntime STATIC IMPORTED)

if (APPLE)
    set_property(TARGET onnxruntime PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_LIST_DIR}/ThirdParty/onnxruntime/lib/libonnxruntime.a)

elseif (WIN32)
    set_property(TARGET onnxruntime APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)

    set_target_properties(onnxruntime PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
            IMPORTED_LOCATION_RELEASE "${CMAKE_CURRENT_LIST_DIR}/ThirdParty/onnxruntime/lib/onnxruntime.lib"
    )
    set_target_properties(onnxruntime PROPERTIES
            MAP_IMPORTED_CONFIG_DEBUG Release
            MAP_IMPORTED_CONFIG_MINSIZEREL Release
            MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release
    )
elseif (UNIX)
    set_property(TARGET onnxruntime PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_LIST_DIR}/ThirdParty/onnxruntime/lib/libonnxruntime.a)
endif ()


target_include_directories("${BaseTargetName}" PRIVATE ${CMAKE_CURRENT_LIST_DIR}/ThirdParty/onnxruntime/include)


target_link_libraries(${BaseTargetName}
        PRIVATE
        juce::juce_audio_utils
        juce::juce_dsp
        BasicPitchCNN
        onnxruntime
        bin_data
        PUBLIC
        juce_recommended_config_flags)

if (${LTO})
    target_link_libraries(${BaseTargetName} PUBLIC juce_recommended_lto_flags)
    message(STATUS "Link Time Optimization (LTO) enabled")
endif ()

if (BUILD_UNIT_TESTS)
    add_subdirectory(Tests)
endif ()

target_compile_definitions(${BaseTargetName}
        PRIVATE
        SAVE_DOWNSAMPLED_AUDIO=0
        USE_TEST_NOTE_FRAME_TO_TIME=0
)

if (WIN32)
    target_compile_definitions(${BaseTargetName} PUBLIC JUCE_ASIO=1)
    target_include_directories(${BaseTargetName} PRIVATE ThirdParty/ASIO/common)
endif ()

# ── LV2 Plugin target ────────────────────────────────────────────────────────
option(BUILD_LV2 "Build LV2 plugin" OFF)

if(BUILD_LV2)
    set(LV2_BUNDLE_NAME "neuralnote_guitar2midi.lv2")
    set(LV2_BUNDLE_DIR  "${CMAKE_BINARY_DIR}/${LV2_BUNDLE_NAME}")

    # Collect NeuralNote's Lib/Model sources (unchanged — same as main plugin)
    file(GLOB_RECURSE LIB_MODEL_SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/Lib/Model/*.cpp"
    )
    # Exclude BasicPitchCNN.h's companion .cpp if it conflicts — NeuralNote
    # already links it via RTNeural; adjust if your build differs.

    add_library(NeuralNoteGuitar2Midi_LV2 MODULE
        LV2/neuralnote_guitar2midi.cpp
        ${LIB_MODEL_SOURCES}
    )

    # Match the prefix convention LV2 hosts expect (.so, not lib*.so)
    set_target_properties(NeuralNoteGuitar2Midi_LV2 PROPERTIES
        PREFIX ""
        OUTPUT_NAME "neuralnote_guitar2midi"
        LIBRARY_OUTPUT_DIRECTORY "${LV2_BUNDLE_DIR}"
    )

    target_compile_features(NeuralNoteGuitar2Midi_LV2 PRIVATE cxx_std_17)

    # ── SIMD auto-detection ──────────────────────────────────────────────────
    # cmake_cxx_source_runs() compiles *and executes* a small probe program on
    # the build host.  Because we build directly on the target Pi (not cross-
    # compiling), this correctly distinguishes:
    #
    #   Raspberry Pi 4  (Cortex-A72, ARMv8-A)    →  armv8-a+neon
    #   Raspberry Pi 5  (Cortex-A76, ARMv8.2-A)  →  armv8.2-a+dotprod+fp16
    #   x86-64 with AVX2 + FMA                    →  avx2+fma
    #   x86-64 with SSE4.2                        →  sse4.2
    #
    # The detected level is baked into NEURALNOTE_SIMD_LEVEL (a string) so the
    # plugin can log it at instantiation time.  Results are CMake-cached; wipe
    # the build directory or run cmake with -UNEURALNOTE_HAVE_* to re-probe.
    include(CheckCXXSourceRuns)

    set(_NN_MARCH "")          # -march=<value>; empty = let compiler choose
    set(_NN_SIMD  "baseline")  # human-readable label passed to the plugin

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        # ── AArch64 ───────────────────────────────────────────────────────────
        # Probe ARMv8.2-A + dotprod + fp16 (Raspberry Pi 5 / Cortex-A76).
        # The sdot instruction raises SIGILL on Pi 4 (ARMv8-A / Cortex-A72),
        # so the run-test naturally fails there and we fall back to plain NEON.
        set(CMAKE_REQUIRED_FLAGS "-march=armv8.2-a+dotprod+fp16")
        check_cxx_source_runs([[
            #include <arm_neon.h>
            int main() {
                int32x4_t acc = vdupq_n_s32(0);
                int8x16_t a = vdupq_n_s8(3), b = vdupq_n_s8(2);
                acc = vdotq_s32(acc, a, b);  /* sdot: ARMv8.2-A only */
                return (vgetq_lane_s32(acc, 0) == 24) ? 0 : 1;
            }
        ]] NEURALNOTE_HAVE_ARMV82_DOTPROD)
        unset(CMAKE_REQUIRED_FLAGS)

        if(NEURALNOTE_HAVE_ARMV82_DOTPROD)
            set(_NN_MARCH "armv8.2-a+dotprod+fp16")
            set(_NN_SIMD  "armv8.2-a+dotprod+fp16")
        else()
            # Probe baseline ARMv8-A NEON (Raspberry Pi 4 / Cortex-A72)
            set(CMAKE_REQUIRED_FLAGS "-march=armv8-a+simd")
            check_cxx_source_runs([[
                #include <arm_neon.h>
                int main() {
                    float32x4_t a = vdupq_n_f32(1.0f), b = vdupq_n_f32(2.0f);
                    float32x4_t c = vmulq_f32(a, b);
                    return (vgetq_lane_f32(c, 0) > 1.9f) ? 0 : 1;
                }
            ]] NEURALNOTE_HAVE_NEON)
            unset(CMAKE_REQUIRED_FLAGS)
            if(NEURALNOTE_HAVE_NEON)
                set(_NN_MARCH "armv8-a+simd")
                set(_NN_SIMD  "armv8-a+neon")
            endif()
        endif()

    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        # ── x86-64 ────────────────────────────────────────────────────────────
        set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma")
        check_cxx_source_runs([[
            #include <immintrin.h>
            int main() {
                __m256 a = _mm256_set1_ps(1.5f), b = _mm256_set1_ps(2.0f);
                __m256 c = _mm256_mul_ps(a, b);
                float r[8]; _mm256_storeu_ps(r, c);
                return (r[0] > 2.9f) ? 0 : 1;
            }
        ]] NEURALNOTE_HAVE_AVX2)
        unset(CMAKE_REQUIRED_FLAGS)

        if(NEURALNOTE_HAVE_AVX2)
            target_compile_options(NeuralNoteGuitar2Midi_LV2 PRIVATE -mavx2 -mfma)
            set(_NN_SIMD "avx2+fma")
        else()
            set(CMAKE_REQUIRED_FLAGS "-msse4.2")
            check_cxx_source_runs([[
                #include <nmmintrin.h>
                int main() {
                    __m128i a = _mm_set1_epi32(42);
                    return (_mm_extract_epi32(a, 0) == 42) ? 0 : 1;
                }
            ]] NEURALNOTE_HAVE_SSE42)
            unset(CMAKE_REQUIRED_FLAGS)
            if(NEURALNOTE_HAVE_SSE42)
                target_compile_options(NeuralNoteGuitar2Midi_LV2 PRIVATE -msse4.2)
                set(_NN_SIMD "sse4.2")
            endif()
        endif()
    endif()

    message(STATUS "LV2 SIMD: ${_NN_SIMD}  (march=${_NN_MARCH})")

    target_compile_options(NeuralNoteGuitar2Midi_LV2 PRIVATE
        -O2 -ffast-math -fPIC
        $<$<BOOL:${_NN_MARCH}>:-march=${_NN_MARCH}>
    )

    # Include paths — LV2/ must come first so BinaryData.h and NoteUtils.h
    # stubs shadow the JUCE-dependent versions in Lib/Utils and the build dir.
    target_include_directories(NeuralNoteGuitar2Midi_LV2 PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/LV2
        ${CMAKE_CURRENT_LIST_DIR}/Lib/Model
        ${CMAKE_CURRENT_LIST_DIR}/Lib/Utils
        ${CMAKE_CURRENT_LIST_DIR}/ThirdParty/RTNeural
        ${CMAKE_CURRENT_LIST_DIR}/ThirdParty/onnxruntime/include
    )

    target_compile_definitions(NeuralNoteGuitar2Midi_LV2 PRIVATE
        SAVE_DOWNSAMPLED_AUDIO=0
        USE_TEST_NOTE_FRAME_TO_TIME=0
        NEURALNOTE_SIMD_LEVEL="${_NN_SIMD}"
    )

    # Link onnxruntime static lib + RTNeural + system LV2
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LV2 REQUIRED lv2)

    # The bundled .a is a macOS universal binary and cannot link on Linux.
    # Search for the shared library; on aarch64 also check the local staging
    # directory (ThirdParty/onnxruntime/lib-linux-aarch64/) populated by the
    # setup step documented in LV2/README.md.
    find_library(LV2_ONNXRUNTIME_LIB
        NAMES onnxruntime
        PATHS
            "${CMAKE_CURRENT_LIST_DIR}/ThirdParty/onnxruntime/lib-linux-aarch64"
            /usr/local/lib
        NO_DEFAULT_PATH
    )
    if(NOT LV2_ONNXRUNTIME_LIB)
        find_library(LV2_ONNXRUNTIME_LIB onnxruntime REQUIRED)
    endif()
    message(STATUS "LV2: using onnxruntime: ${LV2_ONNXRUNTIME_LIB}")

    target_link_libraries(NeuralNoteGuitar2Midi_LV2 PRIVATE
        RTNeural
        ${LV2_ONNXRUNTIME_LIB}
        ${LV2_LIBRARIES}
        pthread dl
    )
    target_include_directories(NeuralNoteGuitar2Midi_LV2 PRIVATE
        ${LV2_INCLUDE_DIRS}
    )

    # Copy TTL files and ModelData into the bundle directory
    add_custom_command(TARGET NeuralNoteGuitar2Midi_LV2 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_LIST_DIR}/LV2/manifest.ttl
            ${LV2_BUNDLE_DIR}/manifest.ttl
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_LIST_DIR}/LV2/plugin.ttl
            ${LV2_BUNDLE_DIR}/plugin.ttl
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_LIST_DIR}/Lib/ModelData
            ${LV2_BUNDLE_DIR}/ModelData
        COMMENT "Assembling LV2 bundle: ${LV2_BUNDLE_DIR}"
    )

    message(STATUS "LV2 bundle will be assembled at: ${LV2_BUNDLE_DIR}")
endif()
