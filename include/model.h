#pragma once
#include <string>
#include <vector>
#include <memory>
#include <tensorflow/c/c_api.h>

class T5Model {
public:
    T5Model();
    ~T5Model();
    
    bool loadModel(const std::string& model_path);
    
    // Main generate method
    std::vector<int> generate(const std::vector<int>& input_ids,
                            int max_length = 50,
                            float temperature = 0.7);

private:
    TF_Graph* graph_;
    TF_Session* session_;
    TF_Status* status_;
    
    // Get next token logits
    std::vector<float> getNextTokenLogits(const std::vector<int>& input_ids,
                                        const std::vector<int>& decoder_ids);
    
    // Helper methods
    TF_Tensor* createInputTensor(const std::vector<int>& ids);
    int sampleFromLogits(const std::vector<float>& logits, float temperature);
    void checkStatus(const char* operation);
};