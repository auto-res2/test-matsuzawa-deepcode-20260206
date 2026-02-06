"""
Model wrapper for CoT generation using transformer models.
Provides utilities for generating chain-of-thought reasoning and answer extraction.
"""

import re
import torch
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter


class CoTModel:
    """Wrapper for causal LM models that generate chain-of-thought reasoning."""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        cache_dir: str = ".cache/",
        max_memory_gb: Optional[int] = None
    ):
        """
        Initialize the CoT model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
            dtype: Data type for model weights
            cache_dir: Directory for caching models
            max_memory_gb: Maximum memory in GB per device
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        
        # Convert dtype string to torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.dtype = dtype_map.get(dtype, torch.bfloat16)
        
        print(f"Loading model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Assert tokenizer has required attributes
        assert self.tokenizer.pad_token_id is not None, "Tokenizer must have pad_token_id"
        assert self.tokenizer.eos_token_id is not None, "Tokenizer must have eos_token_id"
        
        # Load model
        device_map = "auto" if device == "cuda" else device
        max_memory = None
        if max_memory_gb is not None and device == "cuda":
            max_memory = {0: f"{max_memory_gb}GB"}
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=self.dtype,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=True
        )
        
        self.model.eval()
        
        # Compile number extraction regex
        self.num_pattern = re.compile(r"-?\d+(?:\.\d+)?")
        
        print(f"Model loaded successfully on {device}")
    
    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        num_return_sequences: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
        do_sample: bool = True
    ) -> List[str]:
        """
        Generate text completions for a prompt.
        
        Args:
            prompt: Input prompt
            num_return_sequences: Number of sequences to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_new_tokens: Maximum new tokens to generate
            do_sample: Whether to use sampling (vs greedy)
        
        Returns:
            List of generated text strings
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        if do_sample and num_return_sequences > 1:
            generation_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            })
        elif do_sample:
            generation_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            })
        else:
            generation_kwargs["do_sample"] = False
        
        outputs = self.model.generate(**inputs, **generation_kwargs)
        
        # Decode outputs, removing the input prompt
        input_length = inputs["input_ids"].shape[1]
        decoded = []
        for output in outputs:
            # Extract only the generated part
            generated_ids = output[input_length:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            decoded.append(text)
        
        return decoded
    
    def extract_numbers(self, text: str) -> List[str]:
        """Extract all numbers from text."""
        return self.num_pattern.findall(text)
    
    def extract_final_answer(self, text: str) -> Optional[str]:
        """
        Extract the final numeric answer from generated text.
        
        Args:
            text: Generated chain-of-thought text
        
        Returns:
            Final answer as string, or None if no numbers found
        """
        numbers = self.extract_numbers(text)
        if not numbers:
            return None
        return numbers[-1]
    
    def format_cot_prompt(
        self,
        question: str,
        demonstrations: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Format a chain-of-thought prompt.
        
        Args:
            question: The question to answer
            demonstrations: Optional list of demo dicts with 'question', 'reasoning', 'answer'
        
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Add instruction
        prompt_parts.append(
            "Solve the following math word problems step by step. "
            "Show your reasoning and then provide the final numerical answer.\n"
        )
        
        # Add demonstrations if provided
        if demonstrations:
            for demo in demonstrations:
                prompt_parts.append(f"\nQ: {demo['question']}")
                prompt_parts.append(f"A: Let's think step by step. {demo['reasoning']}")
                prompt_parts.append(f"The answer is {demo['answer']}.\n")
        
        # Add the actual question
        prompt_parts.append(f"\nQ: {question}")
        prompt_parts.append("A: Let's think step by step.")
        
        return "\n".join(prompt_parts)
    
    def compute_self_consistency(
        self,
        question: str,
        num_samples: int = 4,
        temperature: float = 0.7,
        demonstrations: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[float, Optional[str], List[str]]:
        """
        Compute self-consistency by sampling multiple CoT chains.
        
        Args:
            question: Question to answer
            num_samples: Number of chains to sample
            temperature: Sampling temperature
            demonstrations: Optional demonstrations for prompt
        
        Returns:
            Tuple of (consistency_score, majority_answer, all_outputs)
        """
        prompt = self.format_cot_prompt(question, demonstrations)
        
        # Generate multiple chains
        outputs = self.generate(
            prompt,
            num_return_sequences=num_samples,
            temperature=temperature,
            do_sample=True
        )
        
        # Extract answers
        answers = []
        for output in outputs:
            answer = self.extract_final_answer(output)
            if answer is not None:
                answers.append(answer)
        
        # Compute consistency
        if not answers:
            return 0.0, None, outputs
        
        counter = Counter(answers)
        majority_answer, count = counter.most_common(1)[0]
        consistency = count / len(answers)
        
        return consistency, majority_answer, outputs
    
    def generate_paraphrase(
        self,
        question: str,
        temperature: float = 0.8
    ) -> str:
        """
        Generate a paraphrase of a question.
        
        Args:
            question: Original question
            temperature: Sampling temperature
        
        Returns:
            Paraphrased question
        """
        paraphrase_prompt = (
            f"Paraphrase the following question while keeping the exact same numbers and meaning:\n\n"
            f"Original: {question}\n\n"
            f"Paraphrase:"
        )
        
        paraphrases = self.generate(
            paraphrase_prompt,
            num_return_sequences=1,
            temperature=temperature,
            max_new_tokens=256,
            do_sample=True
        )
        
        return paraphrases[0].strip()
    
    def reconstruct_question(
        self,
        reasoning: str,
        temperature: float = 0.0
    ) -> str:
        """
        Reconstruct the original question from a reasoning chain.
        
        Args:
            reasoning: Chain-of-thought reasoning
            temperature: Sampling temperature (0 for deterministic)
        
        Returns:
            Reconstructed question
        """
        reconstruct_prompt = (
            f"Given the following step-by-step solution, reconstruct the original word problem:\n\n"
            f"Solution: {reasoning}\n\n"
            f"Original problem:"
        )
        
        reconstructions = self.generate(
            reconstruct_prompt,
            num_return_sequences=1,
            temperature=temperature,
            max_new_tokens=256,
            do_sample=temperature > 0
        )
        
        return reconstructions[0].strip()
    
    def compute_embedding_similarity(self, text1: str, text2: str) -> float:
        """
        Compute simple token-based similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score between 0 and 1
        """
        # Tokenize both texts
        tokens1 = set(self.tokenizer.tokenize(text1.lower()))
        tokens2 = set(self.tokenizer.tokenize(text2.lower()))
        
        # Compute Jaccard similarity
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0


def create_model(cfg) -> CoTModel:
    """
    Create a CoT model from Hydra config.
    
    Args:
        cfg: Hydra configuration object
    
    Returns:
        Initialized CoTModel instance
    """
    return CoTModel(
        model_name=cfg.model.name,
        device=cfg.model.device,
        dtype=cfg.compute.dtype,
        cache_dir=cfg.cache_dir,
        max_memory_gb=cfg.compute.get("max_memory_gb", None)
    )
