#ifndef AG_CAPI_H
#define AG_CAPI_H

/* capi — the only surface the WebAssembly build exports. plain C, fixed-width
 * fields, integer handles. every function returns a status code and never
 * throws or aborts; details of the last failure are read with ag_last_error.
 *
 * handles carry a generation number, so using a freed or stale handle returns
 * AG_NOT_FOUND instead of touching memory. */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AG_ABI_VERSION 1u
#define AG_MAX_RANK 4
#define AG_HIST_BINS 32

typedef int32_t  ag_status;   /* ag::Code */
typedef uint32_t ag_handle;   /* 0 is never valid */

typedef struct { int32_t rank; int32_t dim[AG_MAX_RANK]; } ag_shape;

typedef struct {
    ag_shape shape;
    int32_t  axis;
    int32_t  keep_dim;
    float    scalar;
    int32_t  role;
    int32_t  init;
    float    init_scale;
    int32_t  requires_grad;
} ag_attrs;

typedef struct { int32_t kind; int32_t slot; int32_t node; } ag_step;

typedef struct {
    float    min, max, mean, std, l2, frac_zero;
    uint32_t non_finite;
    float    hist_lo, hist_hi;
    uint32_t hist[AG_HIST_BINS];
} ag_stats;

typedef struct {
    int32_t kind;
    int32_t lr_kind;
    float   base_lr, min_lr, gamma;
    int32_t period, warmup_steps, total_steps;
    float   momentum, beta1, beta2, eps, weight_decay, clip_norm;
} ag_optim_config;

typedef struct {
    int32_t  batch_size, eval_batch_size;
    uint32_t seed_lo, seed_hi;
    int32_t  with_tangents;
    int32_t  dataset_kind, dataset_n;
    float    dataset_noise, dataset_val_fraction;
    ag_optim_config optim;
} ag_session_config;

typedef struct {
    int32_t step_lo, step_hi;
    int32_t steps_run;
    float   loss, grad_norm, lr;
} ag_train_stats;

typedef struct {
    float   x0, x1, y0, y1;
    int32_t res, node, element;
} ag_grid;

typedef struct {
    int32_t category, min_inputs, max_inputs, saves;
    int32_t has_vjp, has_jvp, has_scalarize;
} ag_op_desc;

/* ── errors and metadata ─────────────────────────────────────────────────── */
uint32_t    ag_abi_version(void);
const char* ag_last_error(void);        /* valid until the next call */
int32_t     ag_last_error_node(void);
int32_t     ag_last_error_slot(void);

int32_t     ag_op_count(void);
const char* ag_op_name(int32_t op);     /* null if out of range */
const char* ag_op_symbol(int32_t op);
ag_status   ag_op_describe(int32_t op, ag_op_desc* out);

/* scratch memory for the caller to pass arrays and structs in and out */
void* ag_alloc(uint32_t bytes);         /* null on failure */
void  ag_free(void* p);

/* ── graph ───────────────────────────────────────────────────────────────── */
ag_status ag_graph_new(ag_handle* out);
ag_status ag_graph_free(ag_handle g);
ag_status ag_graph_add_node(ag_handle g, int32_t op, const int32_t* inputs, int32_t n_inputs,
                            const ag_attrs* attrs, const char* name, const char* module,
                            int32_t* out_id);
ag_status ag_graph_set_loss(ag_handle g, int32_t node);
ag_status ag_graph_add_output(ag_handle g, int32_t node);

/* ── session ─────────────────────────────────────────────────────────────── */
ag_status ag_session_new(ag_handle g, const ag_session_config* cfg, ag_handle* out);
ag_status ag_session_free(ag_handle s);
ag_status ag_session_reset(ag_handle s, uint32_t seed_lo, uint32_t seed_hi);
ag_status ag_session_set_optimizer(ag_handle s, const ag_optim_config* cfg);

/* compiled plan: shapes and the schedule for each pass */
ag_status ag_session_node_count(ag_handle s, int32_t* out);
ag_status ag_session_node_shape(ag_handle s, int32_t node, ag_shape* out, int32_t* requires_grad);
ag_status ag_session_schedule_len(ag_handle s, int32_t pass, int32_t* out);
ag_status ag_session_schedule(ag_handle s, int32_t pass, ag_step* out, int32_t cap);

/* structure of the compiled graph, for sessions the page did not build
 * itself (the scalar zoom). inputs must hold AG_MAX_RANK entries */
ag_status   ag_session_node_op(ag_handle s, int32_t node, int32_t* op, int32_t* inputs, int32_t* n_inputs);
const char* ag_session_node_name(ag_handle s, int32_t node);   /* null if out of range */

/* stepping */
ag_status ag_session_begin(ag_handle s, int32_t pass, int32_t seed_node, int32_t seed_element);
ag_status ag_session_advance(ag_handle s, int32_t n, int32_t* executed);
ag_status ag_session_seek(ag_handle s, int32_t index);
ag_status ag_session_cursor(ag_handle s, int32_t* out);

/* training */
ag_status ag_session_train(ag_handle s, int32_t n_steps, ag_train_stats* out);

/* reading; which: 0 value, 1 grad, 2 tangent */
ag_status ag_session_read(ag_handle s, int32_t node, int32_t which, float* out, int32_t cap,
                          int32_t* written);
ag_status ag_session_stats(ag_handle s, int32_t node, int32_t which, ag_stats* out);
ag_status ag_session_eval_loss(ag_handle s, int32_t split, float* out);
ag_status ag_session_eval_grid(ag_handle s, const ag_grid* grid, float* out, int32_t cap);
ag_status ag_session_loss_surface(ag_handle s, int32_t param_x, int32_t elem_x, int32_t param_y,
                                  int32_t elem_y, const ag_grid* grid, float* out, int32_t cap);

/* the session's dataset, so the page can draw it: dims[0..3] receive n,
 * n_train, x_dim, y_dim. x and y may be null to query the sizes only */
ag_status ag_session_dataset(ag_handle s, int32_t* dims, float* x, int32_t x_cap, float* y, int32_t y_cap);

/* optimizer state of one parameter; which: 0 first moment, 1 second moment */
ag_status ag_session_read_optim(ag_handle s, int32_t param, int32_t which, float* out, int32_t cap);

/* parameters, for snapshots and side-by-side runs from identical weights */
ag_status ag_session_param_count(ag_handle s, int32_t* out);
ag_status ag_session_export_params(ag_handle s, float* out, int32_t cap);
ag_status ag_session_import_params(ag_handle s, const float* in, int32_t n);

/* scalar zoom: a new session over the scalar expansion; map_* arrays are
 * indexed by scalar node id and may be null */
ag_status ag_session_scalarize(ag_handle s, ag_handle* out, int32_t* map_source,
                               int32_t* map_element, int32_t cap);

#ifdef __cplusplus
}
#endif

#endif /* AG_CAPI_H */
