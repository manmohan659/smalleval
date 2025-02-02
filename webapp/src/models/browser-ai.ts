//my current file
import { BrowserAI } from "@browserai/browserai";
import { Model, ModelInfo, GenerationOptions, EvaluationMetrics, EvaluationLog } from "./types";

export interface BrowserAIConfig {
    model: string;
}

export class BrowserAIModel implements Model {
    modelInfo: ModelInfo;
    private browserAI: BrowserAI;
    private modelConfig: BrowserAIConfig;
    private modelLoaded: boolean = false;
    private evaluationCallbacks?: {
        onProgress: (progress: number, metrics: EvaluationMetrics) => void;
        onComplete: (metrics: EvaluationMetrics) => void;
        onLog: (message: string, type: "info" | "error" | "success", clear?: boolean) => void;
    };

    constructor(modelConfig: BrowserAIConfig, callbacks?: typeof this.evaluationCallbacks) {
        this.modelConfig = modelConfig;
        this.modelInfo = {
            modelName: modelConfig.model,
            modelSha: null,
            modelDtype: null,
            modelSize: null
        };
        this.browserAI = new BrowserAI();
        this.evaluationCallbacks = callbacks;
    }

    async generate(context: string, options?: GenerationOptions): Promise<string> {
        if (!this.modelLoaded) {
            await this.browserAI.loadModel(this.modelConfig.model);
            this.modelLoaded = true;
        }
        return await this.browserAI.generateText(context, {
            stopSequence: options?.stopSequence,
            generationSize: options?.generationSize,
            doSample: options?.doSample
        }) as string;
    }

    async evaluate(dataset: string): Promise<EvaluationMetrics> {
        this.evaluationCallbacks?.onLog("", "info", true);
    
        if (!this.modelLoaded) {
            this.evaluationCallbacks?.onLog("Loading model...", "info");
            await this.browserAI.loadModel(this.modelConfig.model);
            this.modelLoaded = true;
        }
    
        this.evaluationCallbacks?.onLog(`Starting evaluation on dataset: ${dataset}`, "info");
        try {
            const startTime = Date.now();
            let correctAnswers = 0;
            let totalTokens = 0;
            const evaluationLogs: EvaluationLog[] = [];
            
            // Read from local datasets folder
            const response = await fetch(`/datasets/${dataset.split(':')[1]}`);
            
            if (!response.ok) {
                throw new Error(`Failed to load dataset: ${response.statusText}`);
            }
    
            const text = await response.text();
            const examples = text
                .trim()
                .split('\n')
                .map(line => JSON.parse(line));
    
            // --- Helper functions for parsing ---
            const parseMultipleChoiceResponse = (raw: string, numChoices: number): string | null => {
                const allowedLetters = Array.from({ length: numChoices }, (_, i) => String.fromCharCode(65 + i));
                
                // 1. First try to find a standalone letter
                const standaloneMatch = raw.match(new RegExp(`\\b([${allowedLetters.join('')}])\\b`, 'i'));
                if (standaloneMatch) return standaloneMatch[1].toUpperCase();
                
                // 2. Look for letter with parentheses or period
                const withPuncMatch = raw.match(new RegExp(`[\\(\\[]?([${allowedLetters.join('')}])[\\)\\]\\.]`, 'i'));
                if (withPuncMatch) return withPuncMatch[1].toUpperCase();
                
                // 3. Look for "Answer: X" or "Choice: X" pattern
                const answerMatch = raw.match(new RegExp(`(?:answer|choice)\\s*(?:is)?\\s*:?\\s*([${allowedLetters.join('')}])`, 'i'));
                if (answerMatch) return answerMatch[1].toUpperCase();
                
                // 4. If nothing else works, just take the first valid letter that appears
                const firstLetterMatch = raw.match(new RegExp(`([${allowedLetters.join('')}])`, 'i'));
                if (firstLetterMatch) return firstLetterMatch[1].toUpperCase();
                
                return null;
            };
    
            const parseMathResponse = (raw: string): string | null => {
                const match = raw.match(/[-+]?(\d*\.\d+|\d+)/);
                return match ? match[0] : null;
            };
            // --- End helper functions ---
    
            for (let i = 0; i < examples.length; i++) {
                const example = examples[i];
                let prompt = '';
                let choices: string[] = [];
                let expectedAnswer = '';
                let type: 'math' | 'multiple-choice' = 'multiple-choice';
    
                if (example.gold_index !== undefined && Array.isArray(example.choices)) {
                    // TruthfulQA format
                    choices = example.choices;
                    const allowed = Array.from({ length: choices.length }, (_, idx) => String.fromCharCode(65 + idx)).join(', ');
                    prompt = `Question: ${example.question}\nChoices:\n` + 
                        choices.map((choice, idx) => `${String.fromCharCode(65 + idx)}) ${choice}`).join('\n') + 
                        `\nAnswer: Please respond with only one of the following letters: ${allowed}.DO NOT provide explanations or additional text.\nAnswer: `;
                    expectedAnswer = String.fromCharCode(65 + example.gold_index);
                } else if (example.choices && example.choices.text) {
                    // ARC format
                    choices = example.choices.text;
                    const allowed = Array.from({ length: choices.length }, (_, idx) => String.fromCharCode(65 + idx)).join(', ');
                    prompt = `Question: ${example.question}\nChoices:\n` + 
                        choices.map((choice, idx) => `${String.fromCharCode(65 + idx)}) ${choice}`).join('\n') + 
                        `\nAnswer: Please respond with only one of the following letters: ${allowed}. DO NOT provide explanations or additional text.\nAnswer: `;
                    expectedAnswer = example.answerKey;
                }  else if (example.answer && example.equation) {
                    // MathQA format
                    type = 'math';
                    const isTurkishMath = dataset.includes('MathQA-TR') || /[çğıöşü]/.test(example.question);
                    
                    if (isTurkishMath) {
                        prompt = `Question: ${example.question}\nProvide ONLY a single number as answer. For example: If asked '2+2=?', just write '4'.\nAnswer: `;
                    } else {
                        prompt = `Question: ${example.question}\nProvide only the numeric answer, no explanation:`;
                    }
                    expectedAnswer = example.answer;
                } else if (Array.isArray(example.choices) && typeof example.answer === 'number') {
                    // Standard multiple choice format
                    choices = example.choices;
                    const allowed = Array.from({ length: choices.length }, (_, idx) => String.fromCharCode(65 + idx)).join(', ');
                    prompt = `Question: ${example.question}\nChoices:\n` + 
                        choices.map((choice, idx) => `${String.fromCharCode(65 + idx)}) ${choice}`).join('\n') + 
                        `\nAnswer: Please respond with only one of the following letters: ${allowed}. DO NOT provide explanations or additional text.\nAnswer: `;
                    expectedAnswer = String.fromCharCode(65 + example.answer);
                }
                else if (example.activity_label) {  // This is specific to hellaswag Thai dataset
                    // Thai Hellaswag format
                    type = 'multiple-choice';
                    choices = example.endings;
                    const allowed = Array.from({ length: choices.length }, (_, idx) => String.fromCharCode(65 + idx)).join(', ');
                    prompt = `Question: Select the most appropriate ending for this context.\nContext: ${example.ctx}\nChoices:\n` + 
                        choices.map((choice, idx) => `${String.fromCharCode(65 + idx)}) ${choice}`).join('\n') + 
                        `\nIMPORTANT: You must respond with only a single letter (${allowed}). DO NOT write any explanation or additional text.\nAnswer: `;
                    expectedAnswer = String.fromCharCode(65 + example.label);
                }
    
                const rawResponse = await this.generate(prompt);
                let predictedAnswer = '';
                let isCorrect = false;
    
                if (type === 'math') {
                    const parsedNumber = parseMathResponse(rawResponse);
                    if (parsedNumber) {
                        predictedAnswer = parsedNumber;
                        const numericResponse = parseFloat(parsedNumber);
                        const expectedNumeric = parseFloat(expectedAnswer);
                        isCorrect = Math.abs(numericResponse - expectedNumeric) < 0.01;
                    } else {
                        predictedAnswer = "No numeric answer found";
                        isCorrect = false;
                    }
                } else {
                    // Use improved multiple choice parsing
                    const parsedLetter = parseMultipleChoiceResponse(rawResponse, choices.length);
                    if (parsedLetter) {
                        predictedAnswer = parsedLetter;
                        isCorrect = (predictedAnswer === expectedAnswer.toUpperCase());
                    } else {
                        // Fallback: try using the raw response if nothing valid is found
                        predictedAnswer = rawResponse.trim().toUpperCase();
                        isCorrect = (predictedAnswer === expectedAnswer.toUpperCase());
                    }
                }
    
                if (isCorrect) correctAnswers++;
                totalTokens += prompt.length + rawResponse.length;
                if (evaluationLogs.length >= 10) {
                    evaluationLogs.shift(); 
                  }
                
                evaluationLogs.push({
                    prompt,
                    predictedAnswer,
                    expectedAnswer,
                    isCorrect,
                    choices,
                    question: example.question,
                    type,
                    subject: example.subject,
                    latency: totalTokens * 1000 / (Date.now() - startTime),
                    tokenCount: prompt.length + rawResponse.length
                });
    
                const progress = ((i + 1) / examples.length) * 100;
                const currentMetrics: EvaluationMetrics = {
                    latency: totalTokens * 1000 / (Date.now() - startTime),
                    accuracy: correctAnswers / (i + 1),
                    tokensProcessed: totalTokens,
                    memoryUsage: performance.memory?.usedJSHeapSize || 0,
                    evalTime: (Date.now() - startTime) / 1000,
                    logs: evaluationLogs
                };
    
                this.evaluationCallbacks?.onProgress(progress, currentMetrics);
                this.evaluationCallbacks?.onLog(`Processed ${i + 1}/${examples.length} examples`, "info");
                
                if (i === examples.length - 1) {
                    this.evaluationCallbacks?.onLog("Evaluation complete!", "success");
                    this.evaluationCallbacks?.onComplete(currentMetrics);
                    return currentMetrics;
                }
            }
    
            throw new Error("Evaluation failed to complete");
        } catch (error) {
            this.evaluationCallbacks?.onLog(`Evaluation error: ${error.message}`, "error");
            throw error;
        }
    }

    cleanup(): void {
        this.modelLoaded = false;
    }
}