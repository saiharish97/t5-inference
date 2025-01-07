#pragma once
#include <string>
#include <vector>
#include <memory>
#include <sentencepiece_processor.h>

class T5Tokenizer {
public:
    T5Tokenizer();
    bool loadModel(const std::string& model_path);
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& ids);
    
private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> processor_;
};