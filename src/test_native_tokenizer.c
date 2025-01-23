/* test_tokenizer.c */
#include "native_tokenizer.h"
#include <stdio.h>
#include <string.h>

#define MAX_TOKENS 1024
#define MAX_TEXT_LENGTH 4096

void test_tokenizer_creation() {
    printf("\nTest 1: Tokenizer Creation\n");
    printf("---------------------------\n");
    
    T5TokenizerWrapper* tokenizer = create_tokenizer();
    if (tokenizer) {
        printf("✓ Successfully created tokenizer\n");
        delete_tokenizer(tokenizer);
        printf("✓ Successfully deleted tokenizer\n");
    } else {
        printf("✗ Failed to create tokenizer\n");
    }
}

void test_model_loading(const char* model_path) {
    printf("\nTest 2: Model Loading\n");
    printf("---------------------\n");
    
    T5TokenizerWrapper* tokenizer = create_tokenizer();
    if (!tokenizer) {
        printf("✗ Failed to create tokenizer\n");
        return;
    }

    if (load_tokenizer_model(tokenizer, model_path)) {
        printf("✓ Successfully loaded model from: %s\n", model_path);
    } else {
        printf("✗ Failed to load model from: %s\n", model_path);
    }

    delete_tokenizer(tokenizer);
}

void test_encode_decode(const char* model_path) {
    printf("\nTest 3: Encode and Decode\n");
    printf("-------------------------\n");
    
    T5TokenizerWrapper* tokenizer = create_tokenizer();
    if (!tokenizer) {
        printf("✗ Failed to create tokenizer\n");
        return;
    }

    if (!load_tokenizer_model(tokenizer, model_path)) {
        printf("✗ Failed to load model\n");
        delete_tokenizer(tokenizer);
        return;
    }

    const char* test_texts[] = {
        "Hello, world!",
        "Testing the tokenizer",
        "This is a longer piece of text to test the tokenizer's behavior with longer sequences."
    };
    int num_tests = sizeof(test_texts) / sizeof(test_texts[0]);

    for (int i = 0; i < num_tests; i++) {
        printf("\nTest case %d: \"%s\"\n", i + 1, test_texts[i]);
        
        // Encode
        int token_ids[MAX_TOKENS];
        int actual_length;
        
        if (!encode_text(tokenizer, test_texts[i], token_ids, MAX_TOKENS, &actual_length)) {
            printf("✗ Failed to encode text\n");
            continue;
        }
        
        printf("✓ Encoded to %d tokens: ", actual_length);
        for (int j = 0; j < actual_length && j < 10; j++) {
            printf("%d ", token_ids[j]);
        }
        if (actual_length > 10) printf("...");
        printf("\n");

        // Decode
        char decoded_text[MAX_TEXT_LENGTH];
        int actual_text_length;
        
        if (!decode_tokens(tokenizer, token_ids, actual_length, 
                         decoded_text, MAX_TEXT_LENGTH, &actual_text_length)) {
            printf("✗ Failed to decode tokens\n");
            continue;
        }

        printf("✓ Decoded text (%d chars): \"%s\"\n", actual_text_length, decoded_text);
        
        // Compare original and decoded
        if (strcmp(test_texts[i], decoded_text) == 0) {
            printf("✓ Decoded text matches original\n");
        } else {
            printf("✗ Decoded text differs from original\n");
        }
    }

    delete_tokenizer(tokenizer);
}

void test_error_handling() {
    printf("\nTest 4: Error Handling\n");
    printf("----------------------\n");
    
    T5TokenizerWrapper* tokenizer = create_tokenizer();
    if (!tokenizer) {
        printf("✗ Failed to create tokenizer\n");
        return;
    }

    // Test null/invalid inputs
    int token_ids[10];
    int actual_length;
    char text[100];
    int actual_text_length;

    printf("Testing null/invalid inputs:\n");
    
    if (!encode_text(NULL, "test", token_ids, 10, &actual_length)) {
        printf("✓ Properly handled null tokenizer in encode\n");
    }
    
    if (!encode_text(tokenizer, NULL, token_ids, 10, &actual_length)) {
        printf("✓ Properly handled null input text in encode\n");
    }
    
    if (!decode_tokens(NULL, token_ids, 5, text, 100, &actual_text_length)) {
        printf("✓ Properly handled null tokenizer in decode\n");
    }
    
    if (!decode_tokens(tokenizer, NULL, 5, text, 100, &actual_text_length)) {
        printf("✓ Properly handled null token array in decode\n");
    }

    delete_tokenizer(tokenizer);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <model_path>\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    
    printf("Running T5Tokenizer C Interface Tests\n");
    printf("====================================\n");

    test_tokenizer_creation();
    test_model_loading(model_path);
    test_encode_decode(model_path);
    test_error_handling();

    printf("\nAll tests completed!\n");
    return 0;
}