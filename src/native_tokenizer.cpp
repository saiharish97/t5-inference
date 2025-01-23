// native_tokenizer.cpp - C interface implementation
#include "native_tokenizer.h"
#include "tokenizer.h"
#include <cstring>

struct T5TokenizerWrapper {
    T5Tokenizer tokenizer;
};

extern "C" {

T5TokenizerWrapper* create_tokenizer() {
    try {
        return new T5TokenizerWrapper();
    } catch (const std::exception&) {
        return nullptr;
    }
}

void delete_tokenizer(T5TokenizerWrapper* tokenizer) {
    delete tokenizer;
}

int load_tokenizer_model(T5TokenizerWrapper* tokenizer, const char* model_path) {
    if (!tokenizer || !model_path) return 0;
    return tokenizer->tokenizer.loadModel(model_path) ? 1 : 0;
}

int encode_text(T5TokenizerWrapper* tokenizer,
               const char* text,
               int* token_ids,
               int max_length,
               int* actual_length) {
    if (!tokenizer || !text || !token_ids || !actual_length) return 0;
    
    try {
        std::vector<int> ids = tokenizer->tokenizer.encode(text);
        if (ids.empty()) {
            return 0;
        }
        
        *actual_length = static_cast<int>(ids.size());
        int copy_length = std::min(max_length, *actual_length);
        std::copy(ids.begin(), ids.begin() + copy_length, token_ids);
        return 1;
    } catch (const std::exception&) {
        return 0;
    }
}

int decode_tokens(T5TokenizerWrapper* tokenizer,
                const int* token_ids,
                int length,
                char* text,
                int max_text_length,
                int* actual_text_length) {
    if (!tokenizer || !token_ids || !text || !actual_text_length || length <= 0) return 0;
    
    try {
        std::vector<int> ids(token_ids, token_ids + length);
        std::string result = tokenizer->tokenizer.decode(ids);
        if (result.empty()) {
            return 0;
        }
        
        *actual_text_length = static_cast<int>(result.length());
        int copy_length = std::min(max_text_length - 1, *actual_text_length);
        std::strncpy(text, result.c_str(), copy_length);
        text[copy_length] = '\0';
        return 1;
    } catch (const std::exception&) {
        return 0;
    }
}

} // extern "C"