#ifndef NAPTRAP_H
#define NAPTRAP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NAPTRAP_ABI_VERSION 1u
#define NAPTRAP_SAMPLE_RATE 16000u

typedef struct naptrap_model naptrap_model;
typedef struct naptrap_keyword naptrap_keyword;
typedef struct naptrap_stream naptrap_stream;

/* Every status-returning function returns one of these values as an int. */
typedef enum naptrap_status {
    NAPTRAP_OK = 0,
    NAPTRAP_INVALID_ARGUMENT = 1,
    NAPTRAP_INVALID_AUDIO = 2,
    NAPTRAP_INVALID_KEYWORD = 3,
    NAPTRAP_UNKNOWN_FAR = 4,
    NAPTRAP_MODEL_ERROR = 5,
    NAPTRAP_OUT_OF_MEMORY = 6,
    NAPTRAP_BUSY = 7,
    NAPTRAP_INTERNAL_ERROR = 8
} naptrap_status;

typedef struct naptrap_prediction {
    float score;
    float logit;
    double threshold;
    double target_far;
    uint32_t accepted;
    uint32_t reserved;
} naptrap_prediction;

typedef struct naptrap_stream_config {
    uint32_t window_samples;
    uint32_t hop_samples;
    uint32_t trigger_windows;
    uint32_t release_windows;
    uint64_t cooldown_samples;
    double far;
} naptrap_stream_config;

typedef struct naptrap_event {
    uint64_t start_sample;
    uint64_t end_sample; /* exclusive, measured from construction/reset */
    naptrap_prediction prediction;
    uint32_t triggered;
    uint32_t reserved;
} naptrap_event;

/* Called synchronously for every complete window, not just wake events.
 * The event pointer is valid only during the callback. Copy it to retain it.
 * Do not call functions on handles sharing this model from the callback: every
 * status-returning call on them (including releases, thresholds and keyword
 * creation) returns NAPTRAP_BUSY. */
typedef void (*naptrap_event_callback)(void *context, const naptrap_event *event);

uint32_t naptrap_abi_version(void);
uint32_t naptrap_has_embedded_models(void);
/* No filesystem access. NAPTRAP_MODEL_ERROR when not built with embedded models. */
naptrap_status naptrap_model_load_embedded(naptrap_model **out);
/* Thread-local diagnostic, valid until the next status-returning call on this
 * thread: the error name, optionally followed by ": " and details, such as
 * "WrongArtifactFormat: models/google is not a naptrap-phonmatchnet-v1 artifact".
 * Success clears it. Never free this pointer. The library never writes to
 * stdout or stderr. */
const char *naptrap_last_error(void);

/* Paths are NUL-terminated UTF-8. Returned handles are NULL on failure.
 * A keyword retains its model, and a stream retains its keyword. Thus an owner
 * may release its model/keyword reference after creating dependent handles.
 * Every returned handle must be released exactly once. NULL releases succeed.
 * A released reference must never be used again, even if dependents remain.
 * Serialize operations on handles sharing a model, including releases. Separate
 * models may be used concurrently, so load one model per thread to detect in
 * parallel. Overlapping calls on one model are detected and fail with
 * NAPTRAP_BUSY without changing any handle; out-parameters are still reset
 * (handles to NULL, numbers and predictions to zero) as on any failure.
 * Foreign pointers must refer to valid, suitably aligned, live memory. */
naptrap_status naptrap_model_load(const char *model_dir, const char *google_dir,
                                 const char *g2p_dir, naptrap_model **out);
naptrap_status naptrap_model_release(naptrap_model *model);
naptrap_status naptrap_model_threshold(naptrap_model *model, double far, double *out);
/* utf8 need not be NUL-terminated. Keywords longer than 1024 bytes, or that
 * produce more than 128 feature rows, fail with NAPTRAP_INVALID_KEYWORD. */
naptrap_status naptrap_keyword_create(naptrap_model *model, const char *utf8,
                                     size_t length, naptrap_keyword **out);
/* Borrowed NUL-terminated normalized text, valid until keyword release. */
const char *naptrap_keyword_text(const naptrap_keyword *keyword);
naptrap_status naptrap_keyword_release(naptrap_keyword *keyword);

/* Audio: 8000-32000 finite mono float32 samples in [-1,1], sample_rate=16000;
 * otherwise NAPTRAP_INVALID_AUDIO (NULL samples: NAPTRAP_INVALID_ARGUMENT).
 * No resampling or clipping is performed. Output is zeroed on failure.
 * Keyword must belong to the supplied model. */
naptrap_status naptrap_detect(naptrap_model *model, const naptrap_keyword *keyword,
                             const float *samples, size_t count, uint32_t sample_rate,
                             double far, naptrap_prediction *out);

void naptrap_stream_config_default(naptrap_stream_config *out);
/* NULL config selects defaults. The stream retains keyword and model. */
naptrap_status naptrap_stream_create(naptrap_keyword *keyword,
                                    const naptrap_stream_config *config, naptrap_stream **out);
/* Any chunk size, including zero. NULL samples is allowed only for count=0.
 * callback must be non-NULL. Invalid chunks do not change state. Reset after
 * other errors. No final-window padding; release/reset discards partial audio. */
naptrap_status naptrap_stream_push(naptrap_stream *stream, const float *samples,
                                  size_t count, uint32_t sample_rate,
                                  naptrap_event_callback callback, void *context);
naptrap_status naptrap_stream_reset(naptrap_stream *stream);
naptrap_status naptrap_stream_release(naptrap_stream *stream);

#ifdef __cplusplus
}
#endif
#endif
