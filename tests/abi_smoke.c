/* C ABI smoke test, linked against the static library by `zig build test`.
 * Usage: abi_smoke [MODELS_DIR]. Without a models directory only the checks
 * that need no model files run. */
#include <naptrap.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            fprintf(stderr, "%s:%d: CHECK failed: %s (last error: %s)\n", \
                    __FILE__, __LINE__, #cond, naptrap_last_error());     \
            failures++;                                                   \
        }                                                                 \
    } while (0)

static void without_models(void) {
    CHECK(naptrap_abi_version() == NAPTRAP_ABI_VERSION);

    naptrap_model *model = (naptrap_model *)1;
    CHECK(naptrap_model_load(NULL, "g", "p", &model) == NAPTRAP_INVALID_ARGUMENT);
    CHECK(model == NULL);
    CHECK(naptrap_model_load("m", "g", "p", NULL) == NAPTRAP_INVALID_ARGUMENT);
    CHECK(naptrap_model_load("/nonexistent/m", "/nonexistent/g", "/nonexistent/p", &model) == NAPTRAP_MODEL_ERROR);
    CHECK(model == NULL);
    CHECK(strstr(naptrap_last_error(), "FileNotFound") != NULL);

    if (!naptrap_has_embedded_models()) {
        CHECK(naptrap_model_load_embedded(&model) == NAPTRAP_MODEL_ERROR);
        CHECK(strcmp(naptrap_last_error(), "ModelsNotEmbedded") == 0);
    }

    /* NULL releases succeed and clear the diagnostic. */
    CHECK(naptrap_model_release(NULL) == NAPTRAP_OK);
    CHECK(naptrap_keyword_release(NULL) == NAPTRAP_OK);
    CHECK(naptrap_stream_release(NULL) == NAPTRAP_OK);
    CHECK(naptrap_last_error()[0] == '\0');
    CHECK(naptrap_keyword_text(NULL) == NULL);

    double threshold = 1;
    CHECK(naptrap_model_threshold(NULL, 0.01, &threshold) == NAPTRAP_INVALID_ARGUMENT);
    CHECK(threshold == 0);

    naptrap_prediction prediction;
    memset(&prediction, 0xff, sizeof prediction);
    CHECK(naptrap_detect(NULL, NULL, NULL, 0, NAPTRAP_SAMPLE_RATE, 0.01, &prediction) == NAPTRAP_INVALID_ARGUMENT);
    CHECK(prediction.score == 0 && prediction.accepted == 0);

    naptrap_stream_config config;
    naptrap_stream_config_default(&config);
    CHECK(config.window_samples == 24000 && config.hop_samples == 4000);
    CHECK(config.trigger_windows == 2 && config.release_windows == 2);
    CHECK(config.cooldown_samples == 16000 && config.far == 0.01);
    naptrap_stream_config_default(NULL);
}

struct callback_state {
    naptrap_stream *stream;
    int windows;
    naptrap_status reentrant_release;
};

static void on_window(void *context, const naptrap_event *event) {
    struct callback_state *state = context;
    state->windows++;
    CHECK(event->end_sample - event->start_sample == 16000);
    /* Re-entering a handle that shares the model is refused, not undefined. */
    state->reentrant_release = naptrap_stream_release(state->stream);
}

static void with_models(const char *root) {
    char main_dir[4096], google_dir[4096], g2p_dir[4096];
    snprintf(main_dir, sizeof main_dir, "%s/phonmatchnet", root);
    snprintf(google_dir, sizeof google_dir, "%s/google", root);
    snprintf(g2p_dir, sizeof g2p_dir, "%s/g2p", root);

    naptrap_model *model = NULL;
    CHECK(naptrap_model_load(google_dir, google_dir, g2p_dir, &model) == NAPTRAP_MODEL_ERROR);
    CHECK(strstr(naptrap_last_error(), "WrongArtifactFormat: ") != NULL);
    CHECK(naptrap_model_load(main_dir, google_dir, g2p_dir, &model) == NAPTRAP_OK);
    if (model == NULL) return;
    CHECK(naptrap_last_error()[0] == '\0');

    double threshold = 0, unknown = 1;
    CHECK(naptrap_model_threshold(model, 0.01, &threshold) == NAPTRAP_OK && threshold > 0);
    CHECK(naptrap_model_threshold(model, 0.5, &unknown) == NAPTRAP_UNKNOWN_FAR && unknown == 0);

    naptrap_keyword *keyword = NULL;
    CHECK(naptrap_keyword_create(model, "route 66", 8, &keyword) == NAPTRAP_INVALID_KEYWORD);
    CHECK(keyword == NULL);
    CHECK(naptrap_keyword_create(model, NULL, 3, &keyword) == NAPTRAP_INVALID_ARGUMENT);
    CHECK(naptrap_keyword_create(model, "Hey Computer!", 13, &keyword) == NAPTRAP_OK);
    CHECK(keyword != NULL && strcmp(naptrap_keyword_text(keyword), "hey computer") == 0);

    static float samples[32001];
    naptrap_prediction prediction;
    CHECK(naptrap_detect(model, keyword, samples, 16000, NAPTRAP_SAMPLE_RATE, 0.01, &prediction) == NAPTRAP_OK);
    CHECK(prediction.accepted == 0 && prediction.threshold == threshold);
    CHECK(naptrap_detect(model, keyword, samples, 16000, 44100, 0.01, &prediction) == NAPTRAP_INVALID_AUDIO);
    CHECK(naptrap_detect(model, keyword, samples, 7999, NAPTRAP_SAMPLE_RATE, 0.01, &prediction) == NAPTRAP_INVALID_AUDIO);
    CHECK(naptrap_detect(model, keyword, samples, 32001, NAPTRAP_SAMPLE_RATE, 0.01, &prediction) == NAPTRAP_INVALID_AUDIO);
    CHECK(naptrap_detect(model, keyword, NULL, 16000, NAPTRAP_SAMPLE_RATE, 0.01, &prediction) == NAPTRAP_INVALID_ARGUMENT);
    CHECK(naptrap_detect(model, keyword, samples, 16000, NAPTRAP_SAMPLE_RATE, 0.5, &prediction) == NAPTRAP_UNKNOWN_FAR);
    samples[100] = 2.0f;
    CHECK(naptrap_detect(model, keyword, samples, 16000, NAPTRAP_SAMPLE_RATE, 0.01, &prediction) == NAPTRAP_INVALID_AUDIO);
    samples[100] = 0;

    naptrap_stream_config config;
    naptrap_stream_config_default(&config);
    config.window_samples = 16000;
    config.hop_samples = 8000;
    naptrap_stream *stream = NULL;
    config.hop_samples = 0;
    CHECK(naptrap_stream_create(keyword, &config, &stream) == NAPTRAP_INVALID_ARGUMENT);
    config.hop_samples = 8000;
    CHECK(naptrap_stream_create(keyword, &config, &stream) == NAPTRAP_OK);

    /* Dependents keep their parents alive. */
    CHECK(naptrap_keyword_release(keyword) == NAPTRAP_OK);
    CHECK(naptrap_model_release(model) == NAPTRAP_OK);

    struct callback_state state = {stream, 0, NAPTRAP_OK};
    CHECK(naptrap_stream_push(stream, samples, 0, NAPTRAP_SAMPLE_RATE, on_window, &state) == NAPTRAP_OK);
    CHECK(naptrap_stream_push(stream, NULL, 0, NAPTRAP_SAMPLE_RATE, on_window, &state) == NAPTRAP_OK);
    CHECK(naptrap_stream_push(stream, samples, 24000, NAPTRAP_SAMPLE_RATE, NULL, &state) == NAPTRAP_INVALID_ARGUMENT);
    CHECK(naptrap_stream_push(stream, samples, 24000, NAPTRAP_SAMPLE_RATE, on_window, &state) == NAPTRAP_OK);
    CHECK(state.windows == 2);
    CHECK(state.reentrant_release == NAPTRAP_BUSY);
    CHECK(naptrap_stream_reset(stream) == NAPTRAP_OK);
    CHECK(naptrap_stream_release(stream) == NAPTRAP_OK);
}

int main(int argc, char **argv) {
    without_models();
    if (argc > 1) with_models(argv[1]);
    if (failures) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    printf("abi_smoke: ok%s\n", argc > 1 ? " (with models)" : "");
    return EXIT_SUCCESS;
}
