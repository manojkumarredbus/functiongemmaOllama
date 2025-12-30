package com.rblab;

import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.ModelSupport;
import com.github.tjake.jlama.model.functions.Generator;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.prompt.PromptContext;
import com.github.tjake.jlama.util.Downloader;

import java.io.File;
import java.io.IOException;
import java.util.UUID;

public class RunJlama {
    public static void main(String[] args) throws IOException {
        if (args.length > 0) {
            System.out.println("Using provided model path: " + args[0]);
            String prompt = args.length > 1 ? args[1] : "What is the reimbursement limit for travel meals?";
            runLocalModel(new File(args[0]), prompt);
        } else {
            System.out.println("No model provided. Running sample...");
            sample();
        }
    }

    public static void runLocalModel(File localModelPath, String prompt) throws IOException {
        System.out.println("Loading local model from: " + localModelPath.getAbsolutePath());
        
        // Loads the model. Assuming F32 for weights and I8 for working memory/quantization if supported.
        // Adjust DType based on your specific quantization (e.g. if model is Q4, Jlama detects it).
        AbstractModel m = ModelSupport.loadModel(localModelPath, DType.F32, DType.I8);

        runInference(m, prompt);
    }
    
    public static void sample() throws IOException {
         String model = "tjake/Llama-3.2-1B-Instruct-JQ4";//"google/t5gemma-2-1b-1b";//"meta-llama/Llama-3.2-1B-Instruct";//"HuggingFaceTB/SmolLM3-3B" ;//"tjake/Llama-3.2-1B-Instruct-JQ4";
        String workingDirectory = "./models";
        String prompt = "Ways to design gears";

        System.out.println("Downloading/Loading sample model: " + model);
        // Downloads the model or just returns the local path if it's already downloaded
        File localModelPath = new Downloader(workingDirectory, model).huggingFaceModel();
        System.out.println(localModelPath);

        // Loads the quantized model
        AbstractModel m = ModelSupport.loadModel(localModelPath, DType.F32, DType.I8);
        
        runInference(m, prompt);
    }

    private static void runInference(AbstractModel m, String prompt) {
        PromptContext ctx;
        // Checks if the model supports chat prompting and adds prompt in the expected format for this model
        if (m.promptSupport().isPresent()) {
            ctx = m.promptSupport()
                    .get()
                    .builder()
                    .addSystemMessage("You are a helpful chatbot who writes short responses.")
                    .addUserMessage(prompt)
                    .build();
        } else {
            ctx = PromptContext.of(prompt);
        }

        System.out.println("Prompt: " + ctx.getPrompt() + "\n");
        System.out.println("--- Response ---");
        
        // Generates a response to the prompt and prints it
        // We use a simple non-streaming callback here, but could print token-by-token
        Generator.Response r = m.generate(UUID.randomUUID(), ctx, 0.7f, 256, (s, f) -> System.out.print(s));
        System.out.println("\n----------------");
    }
}
