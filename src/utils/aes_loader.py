"""AES Dataset Loader — Utilities for loading ASAP 2.0 essay data and formatting long-context LLM prompts."""

import os
import pandas as pd
from typing import List, Dict, Any, Optional


class AESDatasetLoader:
    """Loads student essays, rubrics, and source texts from ASAP 2.0 CSV dataset."""

    def __init__(self, csv_path: str = "dataset/ASAP2_train_sourcetexts.csv"):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"AES dataset CSV not found at: {csv_path}")
        self.csv_path = csv_path
        self._df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Lazy load dataset into pandas DataFrame."""
        if self._df is None:
            self._df = pd.read_csv(self.csv_path)
        return self._df

    def get_samples(
        self,
        num_samples: int = 7,
        min_essay_len: int = 200,
        prompt_name: Optional[str] = None,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """Extract formatted essay evaluation samples across prompt categories.

        Args:
            num_samples: Number of samples to extract
            min_essay_len: Minimum word/character length filter for full_text
            prompt_name: Filter by specific prompt_name if specified
            seed: Random seed for sampling reproducibility

        Returns:
            List of sample dicts containing essay_id, prompt_name, prompt, student_essay, score, etc.
        """
        df = self.load_data()

        # Filter out rows missing essential text or score
        filtered_df = df.dropna(subset=["full_text", "score", "assignment"])
        filtered_df = filtered_df[filtered_df["full_text"].str.len() >= min_essay_len]

        if prompt_name:
            filtered_df = filtered_df[filtered_df["prompt_name"] == prompt_name]
            sample_df = filtered_df.sample(n=min(num_samples, len(filtered_df)), random_state=seed)
        else:
            # Sample evenly across distinct prompt_name categories
            distinct_prompts = filtered_df["prompt_name"].unique()
            per_prompt = max(1, num_samples // len(distinct_prompts))
            sampled_dfs = []
            for p in distinct_prompts:
                sub_df = filtered_df[filtered_df["prompt_name"] == p]
                sampled_dfs.append(sub_df.sample(n=min(per_prompt, len(sub_df)), random_state=seed))
            sample_df = pd.concat(sampled_dfs).head(num_samples)

        samples = []
        for _, row in sample_df.iterrows():
            # Combine available source texts
            source_texts = []
            for i in range(1, 5):
                col_name = f"source_text_{i}"
                if col_name in row and pd.notna(row[col_name]):
                    source_texts.append(str(row[col_name]).strip())
            
            combined_source = "\n\n".join(source_texts)
            
            # Format long context prompt
            prompt_text = self.format_long_context_prompt(
                source_text=combined_source,
                assignment=str(row["assignment"]).strip(),
                student_essay=str(row["full_text"]).strip(),
            )

            samples.append({
                "essay_id": str(row["essay_id"]),
                "prompt_name": str(row.get("prompt_name", "Unknown")),
                "score": float(row["score"]),
                "formatted_prompt": prompt_text,
                "student_essay": str(row["full_text"]).strip(),
                "assignment": str(row["assignment"]).strip(),
            })

        return samples


    @staticmethod
    def format_long_context_prompt(
        source_text: str,
        assignment: str,
        student_essay: str,
    ) -> str:
        """Format a structured long-context prompt for LLM evaluation."""
        prompt = (
            "You are an expert Automated Essay Scoring (AES) evaluator.\n\n"
            "### SOURCE READING TEXT\n"
            f"{source_text if source_text else 'No additional reading text provided.'}\n\n"
            "### GRADING ASSIGNMENT & RUBRIC\n"
            f"{assignment}\n\n"
            "### STUDENT ESSAY TO EVALUATE\n"
            f"{student_essay}\n\n"
            "### EVALUATION INSTRUCTION\n"
            "Output the evaluation result starting with the score format 'Score: X' where X is an integer from 1 to 6."
        )
        return prompt

