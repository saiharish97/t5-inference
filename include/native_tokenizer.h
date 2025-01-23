/* native_tokenizer.h - C interface header */
#ifndef NATIVE_TOKENIZER_H
#define NATIVE_TOKENIZER_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque pointer to hide C++ implementation */
typedef struct T5TokenizerWrapper T5TokenizerWrapper;

/* Create a new tokenizer instance */
T5TokenizerWrapper* create_tokenizer(void);

/* Delete tokenizer instance */
void delete_tokenizer(T5TokenizerWrapper* tokenizer);

/* Load model from file */
int load_tokenizer_model(T5TokenizerWrapper* tokenizer, const char* model_path);

/* Encode text into tokens */
int encode_text(T5TokenizerWrapper* tokenizer, 
                const char* text,
                int* token_ids,
                int max_length,
                int* actual_length);

/* Decode tokens back to text */
int decode_tokens(T5TokenizerWrapper* tokenizer,
                 const int* token_ids,
                 int length,
                 char* text,
                 int max_text_length,
                 int* actual_text_length);

#ifdef __cplusplus
}
#endif

#endif /* NATIVE_TOKENIZER_H */