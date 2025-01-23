#include "tflitemodel.h"
#include <random>
#include <algorithm>
#include <iostream>

T5ModelTFLite::T5ModelTFLite() 
    : input_ids_idx_(-1), attention_mask_idx_(-1), 
      decoder_input_ids_idx_(-1), output_logits_idx_(-1) {}

bool T5ModelTFLite::loadModel(const std::string& model_path) {
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model_) return false;

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model_, resolver);
    builder(&interpreter_);
    if (!interpreter_) return false;

    // Get tensor indices
    for (size_t i = 0; i < interpreter_->inputs().size(); i++) {
        const std::string tensor_name = interpreter_->input_tensor(i)->name;
        if (tensor_name.find("input_ids") != std::string::npos) {
            input_ids_idx_ = i;
        } else if (tensor_name.find("attention_mask") != std::string::npos) {
            attention_mask_idx_ = i;
        } else if (tensor_name.find("decoder_input_ids") != std::string::npos) {
            decoder_input_ids_idx_ = i;
        }
    }
    output_logits_idx_ = interpreter_->outputs()[0];

    return (input_ids_idx_ != -1 && attention_mask_idx_ != -1 && 
            decoder_input_ids_idx_ != -1 && output_logits_idx_ != -1);
}

std::vector<float> T5ModelTFLite::getLogits(
    const std::vector<int>& input_ids,
    const std::vector<int>& attention_mask,
    const std::vector<int>& decoder_input_ids) {

    std::vector<int> input_shape = {BATCH_SIZE, static_cast<int>(input_ids.size())};
    std::vector<int> attention_shape = {BATCH_SIZE, static_cast<int>(attention_mask.size())};
    std::vector<int> decoder_shape = {BATCH_SIZE, static_cast<int>(decoder_input_ids.size())};

    interpreter_->ResizeInputTensor(input_ids_idx_, input_shape);
    interpreter_->ResizeInputTensor(attention_mask_idx_, attention_shape);
    interpreter_->ResizeInputTensor(decoder_input_ids_idx_, decoder_shape);
    
    if (interpreter_->AllocateTensors() != kTfLiteOk) 
        throw std::runtime_error("Failed to allocate tensors");

    std::copy(input_ids.begin(), input_ids.end(), 
              interpreter_->typed_input_tensor<int>(input_ids_idx_));
    std::copy(attention_mask.begin(), attention_mask.end(),
              interpreter_->typed_input_tensor<int>(attention_mask_idx_));
    std::copy(decoder_input_ids.begin(), decoder_input_ids.end(),
              interpreter_->typed_input_tensor<int>(decoder_input_ids_idx_));

    if (interpreter_->Invoke() != kTfLiteOk)
        throw std::runtime_error("Failed to invoke interpreter");

    float* output = interpreter_->typed_output_tensor<float>(output_logits_idx_);
    std::vector<float> logits(VOCAB_SIZE);
    std::copy(output + (decoder_input_ids.size() - 1) * VOCAB_SIZE,
              output + decoder_input_ids.size() * VOCAB_SIZE,
              logits.begin());

    return logits;
}

int T5ModelTFLite::sampleFromLogits(const std::vector<float>& logits, float temperature) {
    if (temperature == 0.0f) 
        return std::max_element(logits.begin(), logits.end()) - logits.begin();

    std::vector<float> probs(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());
    
    float sum_exp = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp((logits[i] - max_logit) / temperature);
        sum_exp += probs[i];
    }
    
    for (float& prob : probs) prob /= sum_exp;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    return dist(gen);
}

std::vector<int> T5ModelTFLite::generate(
    const std::vector<int>& input_ids,
    const std::vector<int>& attention_mask,
    const std::vector<int>& initial_decoder_ids,
    int max_length,
    int min_length,
    float temperature) {
    
    std::vector<int> decoder_ids = initial_decoder_ids;
    std::vector<int> output_ids;
    
    for (int i = 0; i < max_length; ++i) {
        auto logits = getLogits(input_ids, attention_mask, decoder_ids);
        int next_token = sampleFromLogits(logits, temperature);
        output_ids.push_back(next_token);
        
        if (next_token == 1 && i >= min_length) break;
        decoder_ids.push_back(next_token);
    }
    
    return output_ids;
}