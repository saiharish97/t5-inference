#include "tokenizer.h"
#include <iostream>

T5Tokenizer::T5Tokenizer() : processor_(std::make_unique<sentencepiece::SentencePieceProcessor>()) {}

bool T5Tokenizer::loadModel(const std::string& model_path) {
    const auto status = processor_->Load(model_path);
    if (!status.ok()) {
        std::cerr << "Error loading tokenizer model: " << status.ToString() << std::endl;
        return false;
    }
    return true;
}

std::vector<int> T5Tokenizer::encode(const std::string& text) {
    std::vector<int> ids;
    const auto status = processor_->Encode(text, &ids);
    if (!status.ok()) {
        std::cerr << "Error encoding text: " << status.ToString() << std::endl;
        return {};
    }
    return ids;
}

std::string T5Tokenizer::decode(const std::vector<int>& ids) {
    std::string text;
    const auto status = processor_->Decode(ids, &text);
    if (!status.ok()) {
        std::cerr << "Error decoding ids: " << status.ToString() << std::endl;
        return {};
    }
    return text;
}