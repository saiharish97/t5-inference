#include "model.h"
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <random>
#include <cmath>
#include <algorithm>

T5Model::T5Model() {
    graph_ = TF_NewGraph();
    status_ = TF_NewStatus();
    if (!graph_ || !status_) {
        throw std::runtime_error("Failed to initialize TensorFlow objects");
    }
    session_ = nullptr;
}

T5Model::~T5Model() {
    if (session_) {
        TF_DeleteSession(session_, status_);
    }
    if (graph_) TF_DeleteGraph(graph_);
    if (status_) TF_DeleteStatus(status_);
}

bool T5Model::loadModel(const std::string& model_path) {
    TF_SessionOptions* opts = TF_NewSessionOptions();
    const char* tags[] = {"serve"};
    int ntags = 1;
    
    session_ = TF_LoadSessionFromSavedModel(opts, nullptr, model_path.c_str(),
                                          tags, ntags, graph_, nullptr, status_);
    
    TF_DeleteSessionOptions(opts);
    
    if (TF_GetCode(status_) != TF_OK) {
        std::cerr << "Error loading model: " << TF_Message(status_) << std::endl;
        return false;
    }
    return true;
}

TF_Tensor* T5Model::createInputTensor(const std::vector<int>& ids) {
    const int64_t dims[] = {1, static_cast<int64_t>(ids.size())};
    const size_t num_bytes = ids.size() * sizeof(int32_t);
    
    TF_Tensor* tensor = TF_AllocateTensor(TF_INT32, dims, 2, num_bytes);
    if (!tensor) {
        throw std::runtime_error("Failed to allocate input tensor");
    }
    
    int32_t* tensor_data = static_cast<int32_t*>(TF_TensorData(tensor));
    for (size_t i = 0; i < ids.size(); ++i) {
        tensor_data[i] = static_cast<int32_t>(ids[i]);
    }
    return tensor;
}

std::vector<float> T5Model::getNextTokenLogits(const std::vector<int>& input_ids,
                                             const std::vector<int>& decoder_ids) {
    // Create input tensors
    TF_Tensor* input_tensor = createInputTensor(input_ids);
    TF_Tensor* decoder_tensor = createInputTensor(decoder_ids);
    
    // Setup inputs
    TF_Output input_ops[2] = {
        {TF_GraphOperationByName(graph_, "serving_default_input_ids"), 0},
        {TF_GraphOperationByName(graph_, "serving_default_decoder_input_ids"), 0}
    };
    
    if (!input_ops[0].oper || !input_ops[1].oper) {
        TF_DeleteTensor(input_tensor);
        TF_DeleteTensor(decoder_tensor);
        throw std::runtime_error("Could not find input operations in the model");
    }
    
    TF_Tensor* input_values[2] = {input_tensor, decoder_tensor};
    
    // Setup output
    TF_Output output_op = {TF_GraphOperationByName(graph_, "StatefulPartitionedCall"), 0};
    TF_Tensor* output_tensor = nullptr;
    
    // Run inference
    TF_SessionRun(session_,
                  nullptr,
                  input_ops, input_values, 2,
                  &output_op, &output_tensor, 1,
                  nullptr, 0,
                  nullptr,
                  status_);
    
    checkStatus("running inference");
    
    if (!output_tensor) {
        TF_DeleteTensor(input_tensor);
        TF_DeleteTensor(decoder_tensor);
        throw std::runtime_error("No output tensor produced");
    }
    
    // Get output dimensions
    if (TF_NumDims(output_tensor) != 3) {
        throw std::runtime_error("Expected output tensor with 3 dimensions");
    }
    
    const int64_t batch_size = TF_Dim(output_tensor, 0);
    const int64_t seq_len = TF_Dim(output_tensor, 1);
    const int64_t vocab_size = TF_Dim(output_tensor, 2);
    
    // Get logits for the last token
    const float* data = static_cast<const float*>(TF_TensorData(output_tensor));
    std::vector<float> last_token_logits(vocab_size);
    
    // Copy logits for last position
    const float* logits_start = data + (seq_len - 1) * vocab_size;
    std::copy(logits_start, logits_start + vocab_size, last_token_logits.begin());
    
    // Cleanup
    TF_DeleteTensor(input_tensor);
    TF_DeleteTensor(decoder_tensor);
    TF_DeleteTensor(output_tensor);
    
    return last_token_logits;
}

int T5Model::sampleFromLogits(const std::vector<float>& logits, float temperature) {
    std::vector<float> scaled_logits(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());
    
    // Apply temperature and compute softmax
    float sum_exp = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        scaled_logits[i] = std::exp((logits[i] - max_logit) / temperature);
        sum_exp += scaled_logits[i];
    }
    
    // Normalize to get probabilities
    for (float& prob : scaled_logits) {
        prob /= sum_exp;
    }
    
    // Sample from distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(scaled_logits.begin(), scaled_logits.end());
    
    return dist(gen);
}

std::vector<int> T5Model::generate(const std::vector<int>& input_ids,
                                 int max_length,
                                 float temperature) {
    std::vector<int> decoder_ids = {0};  // Start token
    std::vector<int> output_ids;
    
    try {
        for (int i = 0; i < max_length; ++i) {
            // Get next token logits
            auto logits = getNextTokenLogits(input_ids, decoder_ids);
            
            // Sample next token
            int next_token = sampleFromLogits(logits, temperature);
            output_ids.push_back(next_token);
            decoder_ids.push_back(next_token);
            
            // Break if we hit the end token (1)
            if (next_token == 1) {
                break;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in generate: " << e.what() << std::endl;
        throw;
    }
    
    return output_ids;
}

void T5Model::checkStatus(const char* operation) {
    if (TF_GetCode(status_) != TF_OK) {
        throw std::runtime_error(std::string("Error during ") + operation + ": " + 
                               TF_Message(status_));
    }
}