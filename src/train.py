"""
Training script for C3-AutoCoT and baseline methods.
Implements demonstration construction with reliability filtering and evaluation on test set.
"""

import os
import json
import random
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from collections import Counter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available")

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from .model import CoTModel, create_model
from .preprocess import (
    SVAMPDataset,
    load_data,
    DataExample,
    DemonstrationBank,
    create_demonstration_dict,
    numbers_match,
    extract_numbers_from_text
)


class C3AutoCoT:
    """
    Cycle-Consistent & Paraphrase-Invariant Reliability Auto-CoT.
    
    Implements demonstration construction with three reliability checks:
    1. Self-consistency (r_sc)
    2. Paraphrase invariance (r_pi)
    3. Cycle-consistency grounding (r_cc)
    """
    
    def __init__(self, cfg, model: CoTModel, dataset: SVAMPDataset):
        """
        Initialize C3-AutoCoT.
        
        Args:
            cfg: Hydra configuration
            model: CoT model instance
            dataset: Dataset instance
        """
        self.cfg = cfg
        self.model = model
        self.dataset = dataset
        
        # Extract method parameters
        self.num_clusters = cfg.method_params.num_clusters
        self.max_candidates = cfg.method_params.max_candidates_per_cluster
        self.reliability_threshold = cfg.method_params.reliability_threshold
        self.num_samples = cfg.method_params.num_samples
        self.num_paraphrases = cfg.method_params.num_paraphrases
        
        # Component flags
        self.use_self_consistency = cfg.method_params.use_self_consistency
        self.use_paraphrase_invariance = cfg.method_params.use_paraphrase_invariance
        self.use_cycle_consistency = cfg.method_params.use_cycle_consistency
        
        # Initialize embedding model for clustering
        print(f"Loading embedding model: {cfg.method_params.embedding_model}")
        self.embedding_model = SentenceTransformer(
            cfg.method_params.embedding_model,
            cache_folder=cfg.cache_dir
        )
        
        # Demonstration bank
        self.demo_bank = DemonstrationBank(self.num_clusters)
        
        # Clustering assignments
        self.cluster_assignments = None
        self.cluster_centers = None
    
    def cluster_questions(self, questions: List[str]) -> np.ndarray:
        """
        Cluster questions using embeddings.
        
        Args:
            questions: List of question strings
        
        Returns:
            Array of cluster assignments
        """
        print(f"Computing embeddings for {len(questions)} questions...")
        embeddings = self.embedding_model.encode(
            questions,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"Clustering into {self.num_clusters} clusters...")
        kmeans = KMeans(
            n_clusters=self.num_clusters,
            random_state=self.cfg.dataset.split_seed,
            n_init=10
        )
        cluster_labels = kmeans.fit_predict(embeddings)
        
        self.cluster_assignments = cluster_labels
        self.cluster_centers = kmeans.cluster_centers_
        
        return cluster_labels
    
    def compute_self_consistency(
        self,
        question: str
    ) -> Tuple[float, Optional[str], List[str]]:
        """
        Compute self-consistency score (r_sc).
        
        Args:
            question: Question to evaluate
        
        Returns:
            Tuple of (consistency_score, majority_answer, all_outputs)
        """
        return self.model.compute_self_consistency(
            question,
            num_samples=self.num_samples,
            temperature=self.cfg.model.temperature
        )
    
    def compute_paraphrase_invariance(
        self,
        question: str,
        original_answer: str
    ) -> Tuple[float, List[str]]:
        """
        Compute paraphrase invariance score (r_pi).
        
        Args:
            question: Original question
            original_answer: Answer from original question
        
        Returns:
            Tuple of (invariance_score, valid_paraphrases)
        """
        valid_paraphrases = []
        paraphrase_answers = []
        
        for _ in range(self.num_paraphrases):
            # Generate paraphrase
            paraphrase = self.model.generate_paraphrase(
                question,
                temperature=0.8
            )
            
            # Check number preservation
            if not numbers_match(question, paraphrase):
                continue
            
            valid_paraphrases.append(paraphrase)
            
            # Get answer for paraphrase
            consistency, answer, _ = self.model.compute_self_consistency(
                paraphrase,
                num_samples=self.num_samples,
                temperature=self.cfg.model.temperature
            )
            
            if answer is not None:
                paraphrase_answers.append(answer)
        
        # Compute invariance: proportion that match original answer
        if not paraphrase_answers:
            return 0.0, valid_paraphrases
        
        matching = sum(1 for ans in paraphrase_answers if ans == original_answer)
        invariance = matching / len(paraphrase_answers)
        
        return invariance, valid_paraphrases
    
    def compute_cycle_consistency(
        self,
        question: str,
        reasoning: str
    ) -> float:
        """
        Compute cycle-consistency grounding score (r_cc).
        
        Args:
            question: Original question
            reasoning: Generated reasoning chain
        
        Returns:
            Grounding score
        """
        # Reconstruct question from reasoning
        reconstructed = self.model.reconstruct_question(
            reasoning,
            temperature=0.0
        )
        
        # Compute similarity
        text_similarity = self.model.compute_embedding_similarity(
            question,
            reconstructed
        )
        
        # Check number match
        number_match = 1.0 if numbers_match(question, reconstructed) else 0.0
        
        # Combined score
        grounding_score = 0.5 * text_similarity + 0.5 * number_match
        
        return grounding_score
    
    def compute_reliability(
        self,
        example: DataExample
    ) -> Tuple[float, Dict[str, float], Optional[Dict[str, str]]]:
        """
        Compute overall reliability score for a candidate demonstration.
        
        Args:
            example: Data example to evaluate
        
        Returns:
            Tuple of (overall_reliability, component_scores, demo_dict or None)
        """
        components = {}
        
        # 1. Self-consistency
        r_sc = 1.0
        majority_answer = None
        representative_chain = None
        
        if self.use_self_consistency:
            r_sc, majority_answer, outputs = self.compute_self_consistency(
                example.question
            )
            components["r_sc"] = r_sc
            
            if outputs:
                representative_chain = outputs[0]
        else:
            # Still need to generate one chain
            outputs = self.model.generate(
                self.model.format_cot_prompt(example.question),
                num_return_sequences=1,
                temperature=self.cfg.model.temperature,
                do_sample=True
            )
            if outputs:
                representative_chain = outputs[0]
                majority_answer = self.model.extract_final_answer(representative_chain)
        
        if majority_answer is None or representative_chain is None:
            return 0.0, components, None
        
        # 2. Paraphrase invariance
        r_pi = 1.0
        if self.use_paraphrase_invariance:
            r_pi, valid_paraphrases = self.compute_paraphrase_invariance(
                example.question,
                majority_answer
            )
            components["r_pi"] = r_pi
            
            # Require at least one valid paraphrase
            if not valid_paraphrases:
                return 0.0, components, None
        
        # 3. Cycle-consistency grounding
        r_cc = 1.0
        if self.use_cycle_consistency:
            r_cc = self.compute_cycle_consistency(
                example.question,
                representative_chain
            )
            components["r_cc"] = r_cc
        
        # Combined reliability
        reliability = r_sc * r_pi * r_cc
        
        # Create demonstration dict
        demo = create_demonstration_dict(
            question=example.question,
            reasoning=representative_chain,
            answer=majority_answer
        )
        
        return reliability, components, demo
    
    def construct_demonstrations(self) -> DemonstrationBank:
        """
        Construct demonstrations by clustering and reliability filtering.
        
        Returns:
            DemonstrationBank with selected demonstrations
        """
        print("\n=== Demonstration Construction ===")
        
        # Get demo pool
        demo_pool = self.dataset.get_demo_pool()
        questions = [ex.question for ex in demo_pool]
        
        # Cluster questions
        cluster_labels = self.cluster_questions(questions)
        
        # For each cluster, find a reliable demonstration
        for cluster_id in range(self.num_clusters):
            print(f"\nCluster {cluster_id + 1}/{self.num_clusters}:")
            
            # Get examples in this cluster
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_examples = [demo_pool[i] for i in cluster_indices]
            
            # Try candidates in order until one passes threshold
            accepted = False
            for candidate_idx in range(min(self.max_candidates, len(cluster_examples))):
                example = cluster_examples[candidate_idx]
                
                print(f"  Candidate {candidate_idx + 1}: Evaluating...")
                
                # Compute reliability
                reliability, components, demo = self.compute_reliability(example)
                
                print(f"    Reliability: {reliability:.3f}")
                for key, val in components.items():
                    print(f"      {key}: {val:.3f}")
                
                # Check threshold
                if reliability >= self.reliability_threshold and demo is not None:
                    print(f"    ACCEPTED")
                    self.demo_bank.add_demonstration(
                        cluster_id=cluster_id,
                        demo=demo,
                        reliability=reliability,
                        reliability_components=components
                    )
                    accepted = True
                    break
                else:
                    print(f"    REJECTED (below threshold or invalid)")
            
            if not accepted:
                print(f"  No candidates accepted for cluster {cluster_id}")
        
        print(f"\n=== Demo Construction Complete ===")
        print(f"Accepted: {len(self.demo_bank.get_demonstrations())} / {self.num_clusters}")
        print(f"Acceptance rate: {self.demo_bank.get_acceptance_rate():.2%}")
        
        return self.demo_bank
    
    def evaluate_test_set(self, seed: int) -> Dict:
        """
        Evaluate on test set using constructed demonstrations.
        
        Args:
            seed: Random seed for evaluation
        
        Returns:
            Dictionary of metrics
        """
        print(f"\n=== Evaluation (seed={seed}) ===")
        
        # Set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Get demonstrations
        demonstrations = self.demo_bank.get_demonstrations()
        print(f"Using {len(demonstrations)} demonstrations")
        
        # Get test set
        test_set = self.dataset.get_test_set()
        print(f"Test set size: {len(test_set)}")
        
        # Evaluate each test example
        correct = 0
        total = 0
        results = []
        
        for i, example in enumerate(test_set):
            if (i + 1) % 50 == 0:
                print(f"Progress: {i + 1}/{len(test_set)}")
            
            # Generate answer with demonstrations
            prompt = self.model.format_cot_prompt(
                example.question,
                demonstrations=demonstrations
            )
            
            # Greedy decoding for test
            outputs = self.model.generate(
                prompt,
                num_return_sequences=1,
                do_sample=False,
                temperature=0.0
            )
            
            predicted_answer = self.model.extract_final_answer(outputs[0])
            
            # Check correctness
            is_correct = predicted_answer == example.answer
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                "question": example.question,
                "predicted": predicted_answer,
                "gold": example.answer,
                "correct": is_correct
            })
        
        accuracy = correct / total if total > 0 else 0.0
        print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }


def train(cfg):
    """
    Main training function.
    
    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print(f"C3-AutoCoT Experiment: {cfg.method}")
    print(f"Run ID: {cfg.run_id}")
    print("=" * 80)
    
    # Initialize WandB
    use_wandb = WANDB_AVAILABLE and cfg.wandb.mode != "disabled"
    if use_wandb:
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run_id,
            name=cfg.run_id,
            config=dict(cfg),
            mode=cfg.wandb.mode,
            reinit=True
        )
        print(f"WandB initialized: {wandb.run.url}")
    else:
        print("WandB disabled or not available")
    
    # Load model and data
    print("\nLoading model and data...")
    model = create_model(cfg)
    dataset = load_data(cfg)
    
    # Post-init assertions
    assert model.tokenizer.pad_token_id is not None, "Model tokenizer must have pad_token_id"
    assert len(dataset.get_demo_pool()) > 0, "Demo pool must not be empty"
    assert len(dataset.get_test_set()) > 0, "Test set must not be empty"
    
    # Initialize method
    method = C3AutoCoT(cfg, model, dataset)
    
    # Construct demonstrations
    demo_bank = method.construct_demonstrations()
    
    # Log demo construction metrics
    demo_metrics = {
        "demo_acceptance_rate": demo_bank.get_acceptance_rate(),
        "num_demos_accepted": len(demo_bank.get_demonstrations()),
        "mean_reliability": demo_bank.get_mean_reliability(),
    }
    
    # Add component means
    component_means = demo_bank.get_mean_reliability_components()
    for key, val in component_means.items():
        demo_metrics[f"mean_{key}"] = val
    
    print("\nDemo Construction Metrics:")
    for key, val in demo_metrics.items():
        print(f"  {key}: {val:.4f}")
    
    if use_wandb:
        wandb.log(demo_metrics)
    
    # Evaluate on test set with multiple seeds
    all_accuracies = []
    all_results = []
    
    for seed in cfg.evaluation.seeds:
        eval_metrics = method.evaluate_test_set(seed)
        all_accuracies.append(eval_metrics["accuracy"])
        all_results.append(eval_metrics["results"])
        
        # Log per-seed metrics
        if use_wandb:
            wandb.log({
                f"accuracy_seed_{seed}": eval_metrics["accuracy"],
                f"correct_seed_{seed}": eval_metrics["correct"],
                f"total_seed_{seed}": eval_metrics["total"],
            })
    
    # Aggregate metrics
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    
    final_metrics = {
        "accuracy_mean": mean_accuracy,
        "accuracy_std": std_accuracy,
        "accuracy_min": np.min(all_accuracies),
        "accuracy_max": np.max(all_accuracies),
    }
    
    print("\n=== Final Results ===")
    print(f"Mean Accuracy: {mean_accuracy:.4f} +/- {std_accuracy:.4f}")
    print(f"Range: [{np.min(all_accuracies):.4f}, {np.max(all_accuracies):.4f}]")
    
    # Save results
    results_dir = os.path.join(cfg.results_dir, cfg.run_id)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump({**demo_metrics, **final_metrics}, f, indent=2)
    
    # Save detailed results
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump({
            "seeds": cfg.evaluation.seeds,
            "accuracies": all_accuracies,
            "detailed_results": all_results
        }, f, indent=2)
    
    # Log to WandB summary
    if use_wandb:
        wandb.log(final_metrics)
        for key, val in final_metrics.items():
            wandb.run.summary[key] = val
        for key, val in demo_metrics.items():
            wandb.run.summary[key] = val
        
        wandb.finish()
    
    print("\n=== Experiment Complete ===")
    print(f"Results saved to: {results_dir}")
    
    return final_metrics
