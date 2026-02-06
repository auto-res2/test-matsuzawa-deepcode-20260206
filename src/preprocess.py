"""
Data preprocessing for SVAMP and other arithmetic word problem datasets.
Handles dataset loading, splitting, and preparation for CoT experiments.
"""

import re
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datasets import load_dataset, Dataset


@dataclass
class DataExample:
    """Single data example with question, answer, and optional metadata."""
    question: str
    answer: str
    index: int
    equation: Optional[str] = None
    original_split: Optional[str] = None


class SVAMPDataset:
    """SVAMP dataset loader and preprocessor."""
    
    def __init__(
        self,
        dataset_name: str = "MU-NLPC/Calc-svamp",
        cache_dir: str = ".cache/",
        demo_pool_size: int = 500,
        test_size: int = 200,
        split_seed: int = 42
    ):
        """
        Initialize SVAMP dataset.
        
        Args:
            dataset_name: HuggingFace dataset identifier
            cache_dir: Directory for caching datasets
            demo_pool_size: Number of examples for demo pool
            test_size: Number of examples for testing
            split_seed: Random seed for splitting
        """
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.demo_pool_size = demo_pool_size
        self.test_size = test_size
        self.split_seed = split_seed
        
        # Load dataset
        print(f"Loading dataset: {dataset_name}")
        self.raw_dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        
        # Process and split data
        self._prepare_splits()
        
        print(f"Dataset prepared: {len(self.demo_pool)} demo pool, {len(self.test_set)} test")
    
    def _prepare_splits(self):
        """Prepare train (demo pool) and test splits."""
        # Get all examples from the dataset
        # SVAMP typically has a single split or train/test
        if "train" in self.raw_dataset:
            all_examples = self.raw_dataset["train"]
        elif "test" in self.raw_dataset:
            all_examples = self.raw_dataset["test"]
        else:
            # Take the first available split
            split_name = list(self.raw_dataset.keys())[0]
            all_examples = self.raw_dataset[split_name]
        
        # Convert to our format
        examples = []
        for idx, item in enumerate(all_examples):
            # Handle different possible field names
            question = item.get("Question", item.get("question", ""))
            answer = item.get("Answer", item.get("answer", ""))
            equation = item.get("Equation", item.get("equation", None))
            
            # Extract numeric answer if it's in text form
            answer_str = self._extract_numeric_answer(str(answer))
            
            examples.append(DataExample(
                question=question,
                answer=answer_str,
                index=idx,
                equation=equation,
                original_split="train"
            ))
        
        # Shuffle with seed
        random.seed(self.split_seed)
        random.shuffle(examples)
        
        # Split into demo pool and test set
        total_needed = self.demo_pool_size + self.test_size
        if len(examples) < total_needed:
            print(f"Warning: Dataset has only {len(examples)} examples, "
                  f"but {total_needed} requested. Adjusting sizes.")
            # Adjust proportionally
            ratio = len(examples) / total_needed
            self.demo_pool_size = int(self.demo_pool_size * ratio)
            self.test_size = len(examples) - self.demo_pool_size
        
        self.demo_pool = examples[:self.demo_pool_size]
        self.test_set = examples[self.demo_pool_size:self.demo_pool_size + self.test_size]
        
        # Re-index
        for i, ex in enumerate(self.demo_pool):
            ex.index = i
        for i, ex in enumerate(self.test_set):
            ex.index = i
    
    def _extract_numeric_answer(self, answer: str) -> str:
        """Extract numeric answer from text."""
        # Try to parse as number directly
        answer = str(answer).strip()
        
        # Remove common text patterns
        answer = re.sub(r'^[Aa]nswer:?\s*', '', answer)
        answer = re.sub(r'^[Tt]he answer is:?\s*', '', answer)
        
        # Extract number
        num_pattern = re.compile(r"-?\d+(?:\.\d+)?")
        numbers = num_pattern.findall(answer)
        
        if numbers:
            return numbers[-1]
        
        return answer.strip()
    
    def get_demo_pool(self) -> List[DataExample]:
        """Get the demonstration pool."""
        return self.demo_pool
    
    def get_test_set(self) -> List[DataExample]:
        """Get the test set."""
        return self.test_set
    
    def get_example_by_index(self, index: int, split: str = "demo") -> DataExample:
        """
        Get a specific example by index.
        
        Args:
            index: Example index
            split: "demo" or "test"
        
        Returns:
            DataExample
        """
        if split == "demo":
            return self.demo_pool[index]
        else:
            return self.test_set[index]


def load_data(cfg) -> SVAMPDataset:
    """
    Load dataset from Hydra config.
    
    Args:
        cfg: Hydra configuration object
    
    Returns:
        Initialized SVAMPDataset instance
    """
    return SVAMPDataset(
        dataset_name=cfg.dataset.name,
        cache_dir=cfg.cache_dir,
        demo_pool_size=cfg.dataset.demo_pool_size,
        test_size=cfg.dataset.test_size,
        split_seed=cfg.dataset.split_seed
    )


def extract_numbers_from_text(text: str) -> List[str]:
    """
    Extract all numbers from text.
    
    Args:
        text: Input text
    
    Returns:
        List of number strings
    """
    num_pattern = re.compile(r"-?\d+(?:\.\d+)?")
    return num_pattern.findall(text)


def numbers_match(text1: str, text2: str) -> bool:
    """
    Check if two texts contain the same multiset of numbers.
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        True if numbers match, False otherwise
    """
    nums1 = sorted(extract_numbers_from_text(text1))
    nums2 = sorted(extract_numbers_from_text(text2))
    return nums1 == nums2


def create_demonstration_dict(
    question: str,
    reasoning: str,
    answer: str
) -> Dict[str, str]:
    """
    Create a demonstration dictionary.
    
    Args:
        question: Question text
        reasoning: Chain-of-thought reasoning
        answer: Final answer
    
    Returns:
        Dictionary with question, reasoning, answer
    """
    return {
        "question": question,
        "reasoning": reasoning,
        "answer": answer
    }


class DemonstrationBank:
    """Storage for selected demonstrations."""
    
    def __init__(self, num_clusters: int):
        """
        Initialize demonstration bank.
        
        Args:
            num_clusters: Number of clusters (determines max demos)
        """
        self.num_clusters = num_clusters
        self.demonstrations = []
        self.cluster_assignments = {}
        self.reliability_scores = {}
    
    def add_demonstration(
        self,
        cluster_id: int,
        demo: Dict[str, str],
        reliability: float,
        reliability_components: Optional[Dict[str, float]] = None
    ):
        """
        Add a demonstration to the bank.
        
        Args:
            cluster_id: Cluster this demo belongs to
            demo: Demonstration dictionary
            reliability: Overall reliability score
            reliability_components: Individual component scores
        """
        demo_id = len(self.demonstrations)
        self.demonstrations.append(demo)
        self.cluster_assignments[demo_id] = cluster_id
        self.reliability_scores[demo_id] = {
            "overall": reliability,
            "components": reliability_components or {}
        }
    
    def get_demonstrations(self) -> List[Dict[str, str]]:
        """Get all demonstrations."""
        return self.demonstrations
    
    def get_acceptance_rate(self) -> float:
        """Get the acceptance rate (proportion of clusters with demos)."""
        unique_clusters = set(self.cluster_assignments.values())
        return len(unique_clusters) / self.num_clusters if self.num_clusters > 0 else 0.0
    
    def get_mean_reliability_components(self) -> Dict[str, float]:
        """Get mean reliability component scores."""
        if not self.reliability_scores:
            return {}
        
        components = {}
        for demo_id, scores in self.reliability_scores.items():
            for key, value in scores["components"].items():
                if key not in components:
                    components[key] = []
                components[key].append(value)
        
        return {key: sum(values) / len(values) for key, values in components.items()}
    
    def get_mean_reliability(self) -> float:
        """Get mean overall reliability."""
        if not self.reliability_scores:
            return 0.0
        
        scores = [s["overall"] for s in self.reliability_scores.values()]
        return sum(scores) / len(scores)
