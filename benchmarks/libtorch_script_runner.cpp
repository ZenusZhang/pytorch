#include <torch/script.h>
#include <torch/torch.h>

#include <ATen/Parallel.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr const char* kInputIdsKey = "libtorch_input_ids.bin";
constexpr const char* kAttentionMaskKey = "libtorch_attention_mask.bin";

std::string usage() {
    return "Usage: libtorch_script_runner <module.pt> <warmup> <iterations> <num_threads> <output.pt>";
}

std::string json_escape(const std::string& value) {
    std::ostringstream escaped;
    for (char c : value) {
        switch (c) {
            case '"':
                escaped << "\\\"";
                break;
            case '\\':
                escaped << "\\\\";
                break;
            case '\n':
                escaped << "\\n";
                break;
            case '\r':
                escaped << "\\r";
                break;
            case '\t':
                escaped << "\\t";
                break;
            default:
                escaped << c;
        }
    }
    return escaped.str();
}

void write_results(const std::vector<double>& timings, const std::string& output_path) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(9);
    out << "{\"timings_s\":[";
    for (std::size_t i = 0; i < timings.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << timings[i];
    }
    out << "],\"output_path\":\"" << json_escape(output_path) << "\"}";
    std::cout << out.str() << std::endl;
}

torch::ScalarType decode_scalar_type(int32_t code) {
    switch (code) {
        case 0:
            return torch::kFloat32;
        case 1:
            return torch::kFloat16;
        case 2:
            return torch::kBFloat16;
        case 3:
            return torch::kFloat64;
        case 4:
            return torch::kInt64;
        case 5:
            return torch::kInt32;
        case 6:
            return torch::kInt16;
        case 7:
            return torch::kInt8;
        case 8:
            return torch::kUInt8;
        case 9:
            return torch::kBool;
        default:
            throw std::runtime_error("Unsupported dtype code in extra file");
    }
}

std::size_t element_size(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kFloat32:
        case torch::kInt32:
            return 4;
        case torch::kFloat16:
        case torch::kBFloat16:
        case torch::kInt16:
            return 2;
        case torch::kFloat64:
        case torch::kInt64:
            return 8;
        case torch::kInt8:
        case torch::kUInt8:
        case torch::kBool:
            return 1;
        default:
            throw std::runtime_error("Unsupported dtype for element size");
    }
}

torch::Tensor tensor_from_blob(const std::string& blob) {
    if (blob.size() < 8) {
        throw std::runtime_error("Tensor payload too small");
    }

    const char* cursor = blob.data();
    int32_t dtype_code;
    std::memcpy(&dtype_code, cursor, sizeof(int32_t));
    cursor += sizeof(int32_t);

    int32_t rank;
    std::memcpy(&rank, cursor, sizeof(int32_t));
    cursor += sizeof(int32_t);

    if (rank < 0) {
        throw std::runtime_error("Invalid tensor rank in payload");
    }

    const std::size_t header_bytes = sizeof(int32_t) * 2 + sizeof(int64_t) * static_cast<std::size_t>(rank);
    if (blob.size() < header_bytes) {
        throw std::runtime_error("Tensor payload truncated (missing shape data)");
    }

    std::vector<int64_t> shape(static_cast<std::size_t>(rank));
    for (int32_t i = 0; i < rank; ++i) {
        std::memcpy(&shape[i], cursor, sizeof(int64_t));
        cursor += sizeof(int64_t);
        if (shape[i] < 0) {
            throw std::runtime_error("Negative dimension in tensor payload");
        }
    }

    torch::ScalarType dtype = decode_scalar_type(dtype_code);
    std::size_t elem_size = element_size(dtype);
    int64_t numel = 1;
    for (int64_t dim : shape) {
        if (dim == 0) {
            numel = 0;
            break;
        }
        numel *= dim;
    }

    const std::size_t data_bytes = elem_size * static_cast<std::size_t>(numel);
    const std::size_t expected_size = header_bytes + data_bytes;
    if (blob.size() != expected_size) {
        throw std::runtime_error("Tensor payload size mismatch");
    }

    torch::Tensor tensor = torch::empty(shape, torch::TensorOptions().dtype(dtype).device(torch::kCPU));
    if (data_bytes > 0) {
        std::memcpy(tensor.data_ptr(), cursor, data_bytes);
    }
    return tensor;
}

int32_t encode_scalar_type(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kFloat32:
            return 0;
        case torch::kFloat16:
            return 1;
        case torch::kBFloat16:
            return 2;
        case torch::kFloat64:
            return 3;
        case torch::kInt64:
            return 4;
        case torch::kInt32:
            return 5;
        case torch::kInt16:
            return 6;
        case torch::kInt8:
            return 7;
        case torch::kUInt8:
            return 8;
        case torch::kBool:
            return 9;
        default:
            throw std::runtime_error("Unsupported dtype for serialization");
    }
}

void write_tensor_to_file(const torch::Tensor& tensor, const std::string& path) {
    torch::Tensor cpu_tensor = tensor.detach().to(torch::kCPU).contiguous();
    const int32_t dtype_code = encode_scalar_type(cpu_tensor.scalar_type());
    const int32_t rank = static_cast<int32_t>(cpu_tensor.dim());

    std::ofstream out(path, std::ios::binary | std::ios::out);
    if (!out) {
        throw std::runtime_error("Failed to open output file: " + path);
    }

    out.write(reinterpret_cast<const char*>(&dtype_code), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&rank), sizeof(int32_t));

    for (int64_t dim : cpu_tensor.sizes()) {
        out.write(reinterpret_cast<const char*>(&dim), sizeof(int64_t));
    }

    const std::size_t bytes = element_size(cpu_tensor.scalar_type()) * static_cast<std::size_t>(cpu_tensor.numel());
    if (bytes > 0) {
        out.write(static_cast<const char*>(cpu_tensor.data_ptr()), bytes);
    }
    out.close();
    if (!out) {
        throw std::runtime_error("Failed to write tensor payload");
    }
}
} // namespace

int main(int argc, const char* argv[]) {
    if (argc != 6) {
        std::cerr << usage() << std::endl;
        return 1;
    }

    const std::string module_path = argv[1];
    const int warmup = std::stoi(argv[2]);
    const int iterations = std::stoi(argv[3]);
    const int num_threads = std::stoi(argv[4]);
    const std::string output_path = argv[5];

    torch::NoGradGuard no_grad;

    torch::jit::ExtraFilesMap extra_files;
    extra_files.emplace(kInputIdsKey, std::string());
    extra_files.emplace(kAttentionMaskKey, std::string());

    torch::jit::Module module = torch::jit::load(module_path, std::nullopt, extra_files);
    module.eval();
    module.to(at::kCPU);

    if (num_threads > 0) {
        at::set_num_threads(num_threads);
        at::set_num_interop_threads(std::max(1, num_threads));
    }

    const auto& input_ids_blob = extra_files.at(kInputIdsKey);
    const auto& attention_mask_blob = extra_files.at(kAttentionMaskKey);

    torch::Tensor input_ids = tensor_from_blob(input_ids_blob);
    torch::Tensor attention_mask = tensor_from_blob(attention_mask_blob);

    std::vector<torch::jit::IValue> inputs;
    inputs.reserve(2);
    inputs.emplace_back(std::move(input_ids));
    inputs.emplace_back(std::move(attention_mask));

    for (int i = 0; i < warmup; ++i) {
        module.forward(inputs);
    }

    std::vector<double> timings;
    timings.reserve(iterations);
    torch::Tensor last_output;
    for (int i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        torch::Tensor result = module.forward(inputs).toTensor();
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = end - start;
        timings.push_back(elapsed.count());
        last_output = result.detach();
    }

    if (iterations > 0) {
        write_tensor_to_file(last_output, output_path);
    } else {
        write_tensor_to_file(torch::empty({0}), output_path);
    }

    write_results(timings, output_path);

    return 0;
}
