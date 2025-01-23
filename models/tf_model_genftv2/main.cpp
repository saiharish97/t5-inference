#include "tokenizer.h"
#include "tflitemodel.h"
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

void printTokens(const char* label, const std::vector<int>& tokens) {
    std::cout << label << " [" << tokens.size() << "]: ";
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << tokens[i];
        if (i < tokens.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <tflite_model_path> <tokenizer_path>" << std::endl;
        return 1;
    }

    try {
        // Initialize tokenizer and model
        T5Tokenizer tokenizer;
        if (!tokenizer.loadModel(argv[2])) {
            std::cerr << "Failed to load tokenizer model" << std::endl;
            return 1;
        }

        T5ModelTFLite model;
        if (!model.loadModel(argv[1])) {
            std::cerr << "Failed to load T5 TFLite model" << std::endl;
            return 1;
        }

        std::cout << "Models loaded successfully.\n" << std::endl;

        // Interactive loop
        std::string input_text;
        std::cout << "Enter text to summarize (or 'quit' to exit).\n"
                  << "Use Ctrl+D (Unix) or Ctrl+Z (Windows) to finish multi-line input:\n> ";
        
        std::string full_text;
        while (std::getline(std::cin, input_text)) {
            if (input_text == "quit") break;
            
            if (input_text.empty()) {
                if (full_text.empty()) {
                    std::cout << "> ";
                    continue;
                }
                
                try {
                    // Add task prefix and tokenize
                    std::string full_input = "summarize: " + full_text;
                    std::cout << "\nGenerating summary for input text of length: " 
                             << full_text.length() << " characters..." << std::endl;
                    
                    auto input_ids = tokenizer.encode(full_input);
                    auto attention_mask = std::vector<int>(input_ids.size(), 1);
                    auto decoder_input_ids = std::vector<int>(1024, 0);

                    printTokens("Input tokens", input_ids);
                    
                    // Generate summary
                    std::cout << "Generating summary..." << std::endl;
                    auto generated_ids = model.generate(
                        input_ids, 
                        attention_mask,
                        decoder_input_ids,
                        100,  // max_length 
                        30,   // min_length
                        0.8   // temperature
                    );
                    
                    printTokens("Output tokens", generated_ids);
                    
                    // Decode output
                    std::string output_text = tokenizer.decode(generated_ids);
                    std::cout << "\nSummary: " << output_text << std::endl;

                } catch (const std::exception& e) {
                    std::cerr << "Error during summarization: " << e.what() << std::endl;
                }

                full_text.clear();
                std::cout << "\nEnter text to summarize (or 'quit' to exit):\n> ";
            } else {
                if (!full_text.empty()) {
                    full_text += "\n";
                }
                full_text += input_text;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}







// #include "tokenizer.h"
// #include "model.h"
// #include <iostream>
// #include <string>
// #include <vector>
// #include <iomanip>
// #include <google/protobuf/stubs/common.h>

// void printTokens(const char* label, const std::vector<int>& tokens) {
//     std::cout << label << " [" << tokens.size() << "]: ";
//     for (size_t i = 0; i < tokens.size(); ++i) {
//         std::cout << tokens[i];
//         if (i < tokens.size() - 1) std::cout << ", ";
//     }
//     std::cout << std::endl;
// }

// int main(int argc, char* argv[]) {
//     if (argc != 3) {
//         std::cerr << "Usage: " << argv[0] << " <saved_model_path> <spm_model_path>" << std::endl;
//         return 1;
//     }

//     try {

//         std::cout << "Protobuf version: " 
//           << GOOGLE_PROTOBUF_VERSION / 1000000 << "."  // Major version
//           << (GOOGLE_PROTOBUF_VERSION / 1000) % 1000 << "."  // Minor version
//           << GOOGLE_PROTOBUF_VERSION % 1000  // Patch version
//           << std::endl;

//         // Initialize tokenizer and model
//         T5Tokenizer tokenizer;
//         if (!tokenizer.loadModel(argv[2])) {
//             std::cerr << "Failed to load tokenizer model" << std::endl;
//             return 1;
//         }

//         T5Model model;
//         if (!model.loadModel(argv[1])) {
//             std::cerr << "Failed to load T5 model" << std::endl;
//             return 1;
//         }

//         std::cout << "Models loaded successfully.\n" << std::endl;

//         // Interactive loop
//         std::string input_text;
//         std::cout << "Enter text to summarize (or 'quit' to exit).\nUse Ctrl+D (Unix) or Ctrl+Z (Windows) to finish multi-line input:\n> ";
        
//         std::string full_text;
//         while (std::getline(std::cin, input_text)) {
//             if (input_text == "quit") break;
            
//             // Empty line marks the end of input
//             if (input_text.empty()) {
//                 if (full_text.empty()) {
//                     std::cout << "> ";
//                     continue;
//                 }
                
//                 try {
//                     // Add task prefix for summarization
//                     std::string full_input = "summarize: " + full_text;
//                     std::cout << "\nGenerating summary for input text of length: " 
//                              << full_text.length() << " characters..." << std::endl;
                    
//                     // Tokenize input
//                     auto input_ids = tokenizer.encode(full_input);
//                     printTokens("Input tokens", input_ids);
                    
//                     // Generate summary
//                     std::cout << "Generating summary..." << std::endl;
//                     auto output_ids = model.generate(input_ids, 100, 0.8); // Increased max length for summaries
//                     printTokens("Output tokens", output_ids);
                    
//                     // Decode output
//                     std::string output_text = tokenizer.decode(output_ids);
//                     std::cout << "\nSummary: " << output_text << std::endl;

//                 } catch (const std::exception& e) {
//                     std::cerr << "Error during summarization: " << e.what() << std::endl;
//                 }

//                 // Reset for next input
//                 full_text.clear();
//                 std::cout << "\nEnter text to summarize (or 'quit' to exit):\n> ";
//             } else {
//                 // Accumulate multi-line input
//                 if (!full_text.empty()) {
//                     full_text += "\n";
//                 }
//                 full_text += input_text;
//             }
//         }

//     } catch (const std::exception& e) {
//         std::cerr << "Fatal error: " << e.what() << std::endl;
//         return 1;
//     }

//     return 0;
// }