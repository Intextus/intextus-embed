#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_set.h>

#include <onnxruntime_cxx_api.h>
#include <tokenizers_cpp.h>

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

class IntextusEncoder {
public:
    int query_marker_id_ = -1;
    int doc_marker_id_ = -1;
    std::unordered_set<int> skiplist_ids_;

    IntextusEncoder(
        const std::string& model_path,
        const std::string& tokenizer_path,
        const std::string& query_marker,
        const std::string& doc_marker,
        bool do_lower_case
    ) : do_lower_case_(do_lower_case) {

        // 1. Initialize tokenizer
        std::string tok_blob = LoadBytesFromFile(tokenizer_path);
        tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(tok_blob);

        // 2. Initialize ONNX runtime session
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

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

        // Resolve marker token IDs
        query_marker_id_ = ResolveTokenId(query_marker);
        doc_marker_id_ = ResolveTokenId(doc_marker);
        if (query_marker_id_ == -1)
            query_marker_id_ = ResolveTokenId(query_marker + " ");
        if (doc_marker_id_ == -1)
            doc_marker_id_ = ResolveTokenId(doc_marker + " ");

        // Resolve special token IDs
        cls_token_id_ = ResolveTokenId("[CLS]");
        if (cls_token_id_ == -1) cls_token_id_ = ResolveTokenId("<s>");

        sep_token_id_ = ResolveTokenId("[SEP]");
        if (sep_token_id_ == -1) sep_token_id_ = ResolveTokenId("</s>");

        pad_token_id_ = ResolveTokenId("[PAD]");
        if (pad_token_id_ == -1) pad_token_id_ = ResolveTokenId("<pad>");

        PrecomputePunctuationTokens();
    }

    nb::ndarray<float, nb::numpy, nb::device::cpu> encode_queries(
        const std::vector<std::string>& queries,
        size_t max_length,
        bool normalize
    ) {
        return Encode(queries, query_marker_id_, max_length, normalize, false);
    }

    nb::ndarray<float, nb::numpy, nb::device::cpu> encode_docs(
        const std::vector<std::string>& docs,
        size_t max_length,
        bool normalize
    ) {
        return Encode(docs, doc_marker_id_, max_length, normalize, true);
    }

private:
    int ResolveTokenId(const std::string& token) {
        std::vector<int> ids = tokenizer_->Encode(token);
        if (ids.empty()) return -1;
        // If tokenizer wraps with special tokens (CLS/SEP), take the middle one
        if (ids.size() == 3 && (ids[0] == cls_token_id_ || ids[0] == 101 || ids[0] == 0)) {
            return ids[1];
        }
        return ids[0];
    }

    void PrecomputePunctuationTokens() {
        std::vector<std::string> markers = {"##", "\xC4\xA0", " ", "</w>"};
        int consecutive_failures = 0;
        int max_token_id = 0;

        for (int id = 0; id < 250000; ++id) {
            try {
                std::string token = tokenizer_->Decode({id});
                consecutive_failures = 0;
                if (token.empty()) continue;

                // Strip subword markers
                std::string clean = token;
                for (const auto& marker : markers) {
                    size_t pos = 0;
                    while ((pos = clean.find(marker, pos)) != std::string::npos) {
                        clean.erase(pos, marker.length());
                    }
                }
                if (clean.empty()) continue;

                // Skip special tokens like [MASK], <pad>, etc.
                if (clean.size() > 1 && (
                    (clean.front() == '[' && clean.back() == ']') ||
                    (clean.front() == '<' && clean.back() == '>'))) {
                    continue;
                }

                bool is_punct = true;
                for (char c : clean) {
                    if (!std::ispunct(static_cast<unsigned char>(c))) {
                        is_punct = false;
                        break;
                    }
                }
                if (is_punct) {
                    skiplist_ids_.insert(id);
                    if (id > max_token_id) max_token_id = id;
                }
            } catch (...) {
                if (++consecutive_failures > 100) break;
            }
        }

        // Build flat boolean lookup for O(1) masking
        is_punct_.assign(static_cast<size_t>(max_token_id) + 1, false);
        for (int id : skiplist_ids_) {
            is_punct_[static_cast<size_t>(id)] = true;
        }
    }

    struct PreparedInputs {
        std::vector<int64_t> input_ids;
        std::vector<int64_t> attention_mask;
        std::vector<int64_t> token_type_ids; // only populated when needed
        size_t batch_size;
        size_t seq_len;
    };

    PreparedInputs PrepareInputs(const std::vector<std::string>& texts, int marker_id, size_t max_length) {
        size_t batch_size = texts.size();
        size_t seq_len = max_length;
        size_t flat_size = batch_size * seq_len;

        std::vector<int64_t> flat_ids(flat_size, 0);
        std::vector<int64_t> flat_mask(flat_size, 0);
        std::vector<int64_t> flat_ttype;
        if (has_token_type_) {
            flat_ttype.assign(flat_size, 0);
        }

        int64_t cls_id = cls_token_id_ != -1 ? cls_token_id_ : 101;
        int64_t sep_id = sep_token_id_ != -1 ? sep_token_id_ : 102;
        int64_t pad_id = pad_token_id_ != -1 ? pad_token_id_ : 0;

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

            // Strip auto-inserted CLS/SEP
            std::vector<int64_t> clean_ids;
            clean_ids.reserve(raw_ids.size());
            for (int id : raw_ids) {
                if (id != cls_id && id != sep_id) {
                    clean_ids.push_back(id);
                }
            }

            // Build: [CLS] [Marker] [Tokens...] [SEP] [PAD...]
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
                seq.resize(seq_len, pad_id);
            }

            size_t non_pad = std::min(
                clean_ids.size() + (marker_id != -1 ? 2u : 1u) + 1u,
                seq_len
            );

            size_t base = b * seq_len;
            std::memcpy(&flat_ids[base], seq.data(), seq_len * sizeof(int64_t));
            // Set attention mask: 1 for real tokens, 0 for padding (already zeroed)
            for (size_t i = 0; i < non_pad; ++i) {
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
        bool is_doc
    ) {
        size_t batch_size = texts.size();
        float* result_data = nullptr;
        size_t dim = 0;

        {
            nb::gil_scoped_release release;

            PreparedInputs prep = PrepareInputs(texts, marker_id, max_length);

            int64_t input_shape[2] = {
                static_cast<int64_t>(batch_size),
                static_cast<int64_t>(max_length)
            };
            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            std::vector<Ort::Value> inputs;
            inputs.reserve(3);
            inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                mem_info, prep.input_ids.data(), prep.input_ids.size(), input_shape, 2));
            inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                mem_info, prep.attention_mask.data(), prep.attention_mask.size(), input_shape, 2));

            if (has_token_type_) {
                inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                    mem_info, prep.token_type_ids.data(), prep.token_type_ids.size(), input_shape, 2));
            }

            auto outputs = session_->Run(
                Ort::RunOptions{nullptr},
                input_names_cstr_.data(), inputs.data(), inputs.size(),
                output_names_cstr_.data(), 1
            );

            float* raw = outputs[0].GetTensorMutableData<float>();
            auto shape_info = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            dim = static_cast<size_t>(shape_info[2]);
            size_t total = batch_size * max_length * dim;

            result_data = new float[total];
            std::memcpy(result_data, raw, total * sizeof(float));

            // Post-processing: punctuation masking + L2 normalization
            size_t punct_limit = is_punct_.size();
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t s = 0; s < max_length; ++s) {
                    int64_t token_id = prep.input_ids[b * max_length + s];
                    float* vec = result_data + (b * max_length + s) * dim;

                    // Zero out punctuation embeddings for documents
                    if (is_doc && token_id >= 0 &&
                        static_cast<size_t>(token_id) < punct_limit &&
                        is_punct_[static_cast<size_t>(token_id)]) {
                        std::memset(vec, 0, dim * sizeof(float));
                        continue; // already zero, skip normalization
                    }

                    // L2 normalize: precompute reciprocal to use multiply instead of divide
                    if (normalize) {
                        float sum_sq = 0.0f;
                        for (size_t d = 0; d < dim; ++d) {
                            sum_sq += vec[d] * vec[d];
                        }
                        if (sum_sq > 0.0f) {
                            float inv_norm = 1.0f / std::sqrt(sum_sq);
                            for (size_t d = 0; d < dim; ++d) {
                                vec[d] *= inv_norm;
                            }
                        }
                    }
                }
            }
        }

        size_t shape[3] = { batch_size, max_length, dim };
        nb::capsule owner(result_data, [](void* p) noexcept {
            delete[] static_cast<float*>(p);
        });
        return nb::ndarray<float, nb::numpy, nb::device::cpu>(result_data, 3, shape, owner);
    }

    // --- members ---
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "intextus"};
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
    m.doc() = "intextus native C++ core";

    m.def("compute_maxsim", &compute_maxsim,
        "MaxSim late-interaction score", nb::arg("query"), nb::arg("doc"));

    nb::class_<IntextusEncoder>(m, "IntextusEncoder")
        .def(nb::init<const std::string&, const std::string&, const std::string&, const std::string&, bool>(),
            nb::arg("model_path"), nb::arg("tokenizer_path"),
            nb::arg("query_marker"), nb::arg("doc_marker"), nb::arg("do_lower_case"))
        .def("encode_queries", &IntextusEncoder::encode_queries,
            nb::arg("queries"), nb::arg("max_length"), nb::arg("normalize"))
        .def("encode_docs", &IntextusEncoder::encode_docs,
            nb::arg("docs"), nb::arg("max_length"), nb::arg("normalize"))
        .def_prop_ro("query_marker_id", [](const IntextusEncoder& e) { return e.query_marker_id_; })
        .def_prop_ro("doc_marker_id", [](const IntextusEncoder& e) { return e.doc_marker_id_; })
        .def_prop_ro("skiplist_arr", [](const IntextusEncoder& e) { return e.skiplist_ids_; });
}