#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_set.h>

#include <onnxruntime_cxx_api.h>
#include <tokenizers_cpp.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <memory>
#include <limits>
#include <stdexcept>
#include <cctype>
#include <cstring>
#include <unordered_set>

namespace nb = nanobind;

static std::string LoadBytesFromFile(const std::string& path) {
    std::ifstream fs(path, std::ios::binary | std::ios::ate);
    if (!fs.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    auto size = fs.tellg();
    fs.seekg(0, std::ios::beg);
    std::string buf(size, '\0');
    fs.read(buf.data(), size);
    return buf;
}

// Some HF ColBERT tokenizer.json files bake in query-style MASK padding /
// truncation (e.g. length 31). We apply ColBERT packing ourselves, so clear
// those so Encode() returns only real tokens (+ specials).
static void NullifyJsonObjectKey(std::string& json, const std::string& key) {
    const std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) {
        return;
    }
    size_t val = pos + pattern.size();
    while (val < json.size() && std::isspace(static_cast<unsigned char>(json[val]))) {
        ++val;
    }
    if (val >= json.size() || json.compare(val, 4, "null") == 0) {
        return;
    }
    if (json[val] != '{') {
        return;
    }
    int depth = 0;
    size_t end = val;
    for (; end < json.size(); ++end) {
        if (json[end] == '{') {
            ++depth;
        } else if (json[end] == '}') {
            --depth;
            if (depth == 0) {
                ++end;
                break;
            }
        }
    }
    json.replace(val, end - val, "null");
}

class LateEmbedder {
public:
    int query_marker_id_ = -1;
    int doc_marker_id_ = -1;
    std::unordered_set<int> skiplist_ids_;

    LateEmbedder(
        const std::string& model_path,
        const std::string& tokenizer_path,
        bool do_lower_case,
        int num_threads,
        int query_marker_id,
        int doc_marker_id,
        int cls_token_id,
        int sep_token_id,
        int pad_token_id,
        int mask_token_id,
        int vocab_size,
        const std::vector<int>& skip_list
    ) : do_lower_case_(do_lower_case),
        query_marker_id_(query_marker_id),
        doc_marker_id_(doc_marker_id),
        cls_token_id_(cls_token_id),
        sep_token_id_(sep_token_id),
        pad_token_id_(pad_token_id),
        mask_token_id_(mask_token_id) {

        // 1. Initialize tokenizer (disable baked-in padding/truncation)
        std::string tok_blob = LoadBytesFromFile(tokenizer_path);
        NullifyJsonObjectKey(tok_blob, "padding");
        NullifyJsonObjectKey(tok_blob, "truncation");
        tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(tok_blob);

        // 2. Initialize ONNX runtime session
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(num_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
        int size_needed = MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, NULL, 0);
        std::wstring model_path_w(size_needed, 0);
        MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, &model_path_w[0], size_needed);
        if (size_needed > 0) {
            model_path_w.resize(size_needed - 1);
        }
        session_ = std::make_unique<Ort::Session>(env_, model_path_w.c_str(), session_options);
#else
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
#endif

        // Discover input/output names and cache their C-string pointers
        Ort::AllocatorWithDefaultOptions allocator;
        size_t input_count = session_->GetInputCount();
        for (size_t i = 0; i < input_count; ++i) {
            auto ptr = session_->GetInputNameAllocated(i, allocator);
            std::string name(ptr.get());
            if (name == "token_type_ids") {
                has_token_type_ = true;
            }
            input_names_.push_back(std::move(name));
        }
        for (const auto& name : input_names_) {
            input_names_cstr_.push_back(name.c_str());
        }

        if (session_->GetOutputCount() > 0) {
            auto ptr = session_->GetOutputNameAllocated(0, allocator);
            output_name_ = std::string(ptr.get());
            output_names_cstr_ = { output_name_.c_str() };
        }

        // Initialize is_punct_ lookup table
        int max_token_id = vocab_size - 1;
        for (int id : skip_list) {
            if (id > max_token_id) max_token_id = id;
            skiplist_ids_.insert(id);
        }

        // If skip_list is empty, dynamically detect punctuation tokens using the C++ tokenizer
        if (skip_list.empty()) {
            std::string punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
            for (char c : punctuation) {
                std::string symbol(1, c);
                std::vector<int> ids = tokenizer_->Encode(symbol);
                for (int id : ids) {
                    if (id != cls_token_id && id != sep_token_id && id != pad_token_id && id != mask_token_id) {
                        if (id > max_token_id) max_token_id = id;
                        skiplist_ids_.insert(id);
                    }
                }
            }
        }

        is_punct_.assign(max_token_id + 1, false);
        for (int id : skiplist_ids_) {
            is_punct_[id] = true;
        }
    }

    nb::ndarray<float, nb::numpy, nb::device::cpu> encode_queries(
        const std::vector<std::string>& queries,
        size_t max_length,
        bool normalize,
        bool query_attn_mask_all_1s = false
    ) {
        return Encode(queries, query_marker_id_, max_length, normalize, false, query_attn_mask_all_1s);
    }

    nb::ndarray<float, nb::numpy, nb::device::cpu> encode_docs(
        const std::vector<std::string>& docs,
        size_t max_length,
        bool normalize
    ) {
        return Encode(docs, doc_marker_id_, max_length, normalize, true, false);
    }

private:

    struct PreparedInputs {
        std::vector<int64_t> input_ids;
        std::vector<int64_t> attention_mask;
        std::vector<int64_t> token_type_ids; // only populated when needed
        size_t batch_size;
        size_t seq_len;
    };

    PreparedInputs PrepareInputs(
        const std::vector<std::string>& texts,
        int marker_id,
        size_t max_length,
        bool is_doc,
        bool query_attn_mask_all_1s = false
    ) {
        size_t batch_size = texts.size();
        
        std::vector<std::vector<int64_t>> batch_clean_ids(batch_size);
        size_t max_req_len = 0;

        int64_t cls_id = cls_token_id_;
        int64_t sep_id = sep_token_id_;
        int64_t pad_id = pad_token_id_;
        int64_t mask_id = mask_token_id_;

        for (size_t b = 0; b < batch_size; ++b) {
            const std::string* text_ptr = &texts[b];
            std::string lower_buf;
            if (do_lower_case_) {
                lower_buf = texts[b];
                std::transform(lower_buf.begin(), lower_buf.end(), lower_buf.begin(),
                    [](unsigned char c) { return std::tolower(c); });
                text_ptr = &lower_buf;
            }

            std::vector<int> raw_ids = tokenizer_->Encode(*text_ptr);

            // Strip specials and any residual PAD/MASK from tokenizer config.
            // We re-insert CLS/marker/SEP and apply ColBERT padding ourselves.
            std::vector<int64_t>& clean_ids = batch_clean_ids[b];
            clean_ids.reserve(raw_ids.size());
            for (int id : raw_ids) {
                if (id != cls_id && id != sep_id && id != pad_id && id != mask_id) {
                    clean_ids.push_back(id);
                }
            }

            size_t req_len = 1 + (marker_id != -1 ? 1 : 0) + clean_ids.size() + 1;
            if (req_len > max_length) req_len = max_length;
            if (req_len > max_req_len) max_req_len = req_len;
        }

        size_t seq_len = is_doc ? max_req_len : max_length;
        if (seq_len == 0) seq_len = 1;

        size_t flat_size = batch_size * seq_len;

        int64_t fill_id = is_doc ? pad_id : mask_id;
        std::vector<int64_t> flat_ids(flat_size, fill_id);
        std::vector<int64_t> flat_mask(flat_size, 0);
        std::vector<int64_t> flat_ttype;
        if (has_token_type_) {
            flat_ttype.assign(flat_size, 0);
        }

        for (size_t b = 0; b < batch_size; ++b) {
            const std::vector<int64_t>& clean_ids = batch_clean_ids[b];

            // Build: [CLS] [Marker] [Tokens...] [SEP] [PAD...] or [MASK...]
            std::vector<int64_t> seq;
            seq.reserve(seq_len);
            seq.push_back(cls_id);
            if (marker_id != -1) {
                seq.push_back(marker_id);
            }
            for (int64_t id : clean_ids) {
                seq.push_back(id);
            }

            // Truncate or pad
            if (seq.size() >= seq_len) {
                seq.resize(seq_len - 1);
                seq.push_back(sep_id);
            } else {
                seq.push_back(sep_id);
                seq.resize(seq_len, fill_id);
            }

            size_t non_pad = std::min(
                clean_ids.size() + (marker_id != -1 ? 2u : 1u) + 1u,
                seq_len
            );

            size_t base = b * seq_len;
            std::memcpy(&flat_ids[base], seq.data(), seq_len * sizeof(int64_t));
            // Set attention mask: 1 for real tokens, 0 for padding (already zeroed)
            size_t limit = (query_attn_mask_all_1s && !is_doc) ? seq_len : non_pad;
            for (size_t i = 0; i < limit; ++i) {
                flat_mask[base + i] = 1;
            }
        }

        return {std::move(flat_ids), std::move(flat_mask), std::move(flat_ttype), batch_size, seq_len};
    }

    nb::ndarray<float, nb::numpy, nb::device::cpu> Encode(
        const std::vector<std::string>& texts,
        int marker_id,
        size_t max_length,
        bool normalize,
        bool is_doc,
        bool query_attn_mask_all_1s = false
    ) {
        if (texts.empty()) {
            float* empty_data = new float[0];
            size_t shape[3] = {0, 0, 0};
            nb::capsule owner(empty_data, [](void* p) noexcept {
                delete[] static_cast<float*>(p);
            });
            return nb::ndarray<float, nb::numpy, nb::device::cpu>(empty_data, 3, shape, owner);
        }

        PreparedInputs prep = PrepareInputs(texts, marker_id, max_length, is_doc, query_attn_mask_all_1s);

        std::vector<int64_t> input_shape = {
            static_cast<int64_t>(prep.batch_size),
            static_cast<int64_t>(prep.seq_len)
        };

        std::vector<Ort::Value> input_tensors;
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault
        );

        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, prep.input_ids.data(), prep.input_ids.size(),
            input_shape.data(), input_shape.size()
        ));

        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, prep.attention_mask.data(), prep.attention_mask.size(),
            input_shape.data(), input_shape.size()
        ));

        if (has_token_type_) {
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, prep.token_type_ids.data(), prep.token_type_ids.size(),
                input_shape.data(), input_shape.size()
            ));
        }

        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_cstr_.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names_cstr_.data(),
            output_names_cstr_.size()
        );

        if (output_tensors.empty()) {
            throw std::runtime_error("ONNX model execution returned no outputs.");
        }

        auto& out_val = output_tensors[0];
        auto out_info = out_val.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> out_shape = out_info.GetShape();

        if (out_shape.size() != 3) {
            throw std::runtime_error("Expected 3D output tensor [batch, seq, dim].");
        }

        size_t batch_size = out_shape[0];
        size_t seq_len = out_shape[1];
        size_t dim = out_shape[2];

        float* result_data = new float[batch_size * seq_len * dim];
        const float* model_out = out_val.GetTensorData<float>();
        std::memcpy(result_data, model_out, batch_size * seq_len * dim * sizeof(float));

        size_t punct_limit = is_punct_.size();

        {
            nb::gil_scoped_release release;
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t s = 0; s < seq_len; ++s) {
                    size_t flat_idx = b * seq_len + s;
                    int64_t token_id = prep.input_ids[flat_idx];
                    float* vec = result_data + flat_idx * dim;

                    // Zero out punctuation embeddings for documents
                    if (is_doc && token_id >= 0 &&
                        static_cast<size_t>(token_id) < punct_limit &&
                        is_punct_[static_cast<size_t>(token_id)]) {
                        std::memset(vec, 0, dim * sizeof(float));
                        continue;
                    }

                    if (normalize) {
                        float norm_sq = 0.0f;
                        for (size_t k = 0; k < dim; ++k) {
                            norm_sq += vec[k] * vec[k];
                        }
                        float norm = std::sqrt(norm_sq);
                        float scale = norm > 1e-12f ? 1.0f / norm : 0.0f;
                        for (size_t k = 0; k < dim; ++k) {
                            vec[k] *= scale;
                        }
                    }
                }
            }
        }

        size_t shape[3] = {batch_size, seq_len, dim};
        nb::capsule owner(result_data, [](void* p) noexcept {
            delete[] static_cast<float*>(p);
        });
        return nb::ndarray<float, nb::numpy, nb::device::cpu>(result_data, 3, shape, owner);
    }

    // --- members ---
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "limbed"};
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;

    std::vector<std::string> input_names_;
    std::string output_name_;
    std::vector<const char*> input_names_cstr_;
    std::vector<const char*> output_names_cstr_;

    std::vector<bool> is_punct_;
    bool has_token_type_ = false;
    bool do_lower_case_;

    int cls_token_id_ = -1;
    int sep_token_id_ = -1;
    int pad_token_id_ = -1;
    int mask_token_id_ = -1;
};

// ---- MaxSim ----

float compute_maxsim(
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu> query,
    nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu> doc
) {
    size_t q_tokens = query.shape(0);
    size_t d_tokens = doc.shape(0);
    size_t dim = query.shape(1);

    if (dim != doc.shape(1)) {
        throw std::invalid_argument("Query and document embeddings must have the same dimension.");
    }

    const float* q_data = query.data();
    const float* d_data = doc.data();
    float total = 0.0f;

    {
        nb::gil_scoped_release release;
        for (size_t q = 0; q < q_tokens; ++q) {
            float best = -std::numeric_limits<float>::infinity();
            const float* qv = q_data + q * dim;

            for (size_t d = 0; d < d_tokens; ++d) {
                const float* dv = d_data + d * dim;
                float dot = 0.0f;
                for (size_t k = 0; k < dim; ++k) {
                    dot += qv[k] * dv[k];
                }
                if (dot > best) best = dot;
            }
            total += best;
        }
    }
    return total;
}

// ---- Module bindings ----

NB_MODULE(_core, m) {
    m.doc() = "limbed native C++ core";

    m.def("compute_maxsim", &compute_maxsim,
        "MaxSim late-interaction score", nb::arg("query"), nb::arg("doc"));

    nb::class_<LateEmbedder>(m, "LateEmbedder")
        .def(nb::init<const std::string&, const std::string&, bool, int, int, int, int, int, int, int, int, const std::vector<int>&>(),
            nb::arg("model_path"), nb::arg("tokenizer_path"),
            nb::arg("do_lower_case"), nb::arg("num_threads"),
            nb::arg("query_marker_id"), nb::arg("doc_marker_id"),
            nb::arg("cls_token_id"), nb::arg("sep_token_id"),
            nb::arg("pad_token_id"), nb::arg("mask_token_id"),
            nb::arg("vocab_size"), nb::arg("skip_list"))
        .def("encode_queries", &LateEmbedder::encode_queries,
            nb::arg("queries"), nb::arg("max_length"), nb::arg("normalize"), nb::arg("query_attn_mask_all_1s") = false)
        .def("encode_docs", &LateEmbedder::encode_docs,
            nb::arg("docs"), nb::arg("max_length"), nb::arg("normalize"))
        .def_prop_ro("query_marker_id", [](const LateEmbedder& e) { return e.query_marker_id_; })
        .def_prop_ro("doc_marker_id", [](const LateEmbedder& e) { return e.doc_marker_id_; })
        .def_prop_ro("skiplist_arr", [](const LateEmbedder& e) { return e.skiplist_ids_; });
}
