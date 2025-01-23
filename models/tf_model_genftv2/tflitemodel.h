#pragma once
#include <string>
#include <vector>
#include <memory>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"

class T5ModelTFLite {
public:
    T5ModelTFLite();
    ~T5ModelTFLite() = default;

    bool loadModel(const std::string& model_path);
    
    std::vector<int> generate(
        const std::vector<int>& input_ids,
        const std::vector<int>& attention_mask,
        const std::vector<int>& decoder_input_ids,
        int max_length = 350,
        int min_length = 30,
        float temperature = 1.0
    );

private:
    static constexpr int BATCH_SIZE = 1;
    static constexpr int MAX_LENGTH = 1024;
    static constexpr int VOCAB_SIZE = 32128;
    
    std::vector<float> getLogits(
        const std::vector<int>& input_ids,
        const std::vector<int>& attention_mask,
        const std::vector<int>& decoder_input_ids
    );
    
    int sampleFromLogits(const std::vector<float>& logits, float temperature);
    
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
    
    int input_ids_idx_;
    int attention_mask_idx_;
    int decoder_input_ids_idx_;
    int output_logits_idx_;
};