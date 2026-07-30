"""AES Dataset Loader — Utilities for loading ASAP 2.0 essay data and formatting long-context LLM prompts."""

import os
import pandas as pd
from typing import List, Dict, Any, Optional


class AESDatasetLoader:
    """Loads student essays, rubrics, and source texts from ASAP 2.0 CSV dataset."""

    def __init__(self, csv_path: str = "dataset/ASAP2_train_sourcetexts.csv"):
        zip_path = "dataset/ASAP2_train_sourcetexts.zip"
        if not os.path.exists(csv_path) and os.path.exists(zip_path):
            import zipfile
            print(f"📦 Extracting ASAP 2.0 dataset from {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("dataset/")
            print("✓ ASAP 2.0 dataset extracted successfully!")

        if not os.path.exists(csv_path):
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            self._generate_fallback_dataset(csv_path)
        self.csv_path = csv_path
        self._df: Optional[pd.DataFrame] = None


    def _generate_fallback_dataset(self, csv_path: str):
        """Generates realistic ASAP 2.0 dataset samples for Google Colab if local file is missing."""
        print(f"⚠️ Dataset file missing at {csv_path}. Creating fallback ASAP 2.0 samples...")
        topics = [
            ("Exploring Venus", "Venus is a terrestrial planet with a thick toxic atmosphere composed of carbon dioxide. Human exploration of Venus poses extreme engineering challenges due to surface temperatures of 465°C and crushing atmospheric pressure.", "Write an essay explaining the technical and environmental challenges of human exploration of Venus based on the source text."),
            ("Facial action coding system", "The Facial Action Coding System (FACS) is a comprehensive, anatomically based system for measuring all visually discernible facial movement. It breaks down expressions into individual Action Units (AUs).", "Analyze how facial action coding system enables researchers to quantify human emotion."),
            ("The Face on Mars", "In 1976, Viking 1 captured an image of a rock formation on Mars resembling a human face. Subsequent high-resolution images by Mars Global Surveyor proved it to be a natural mesa.", "Explain why the Face on Mars was initially misunderstood and what evidence resolved the mystery."),
            ('"A Cowboy Who Rode the Waves"', "Surfing in the early 20th century transformed coastal culture. Legendary surfers mastered ocean dynamics, navigating massive waves with wooden boards long before modern synthetic materials.", "Describe the perseverance and skills demonstrated by the pioneer surfer in the passage."),
            ("Driverless cars", "Autonomous vehicles utilize LiDAR, computer vision, and neural networks to navigate traffic. Proponents cite reduced accident rates, while critics raise cybersecurity and moral dilemma concerns.", "Evaluate the benefits and ethical dilemmas of transitioning to fully autonomous driverless cars."),
            ("Does the electoral college work?", "The United States Electoral College allocates electors based on congressional representation. Debate continues over whether winner-take-all systems reflect the popular vote democratic intent.", "Analyze the arguments for and against reforming the US Electoral College system."),
            ("Car-free cities", "Urban centers worldwide are experimenting with pedestrian zones, expanded mass transit, and cycling infrastructure to reduce carbon emissions and reclaim public space from automobiles.", "Discuss the social and environmental impacts of transforming modern urban centers into car-free cities.")
        ]
        
        data = []
        for idx, (topic, source, assign) in enumerate(topics, 1):
            essay_text = (
                f"Student Essay on {topic}.\n\n" +
                (source + " ") * 6 + "\n\n" +
                "In my opinion, this topic is very important. " * 25 +
                "Therefore, we can conclude that the evidence clearly supports the main arguments presented."
            )
            data.append({
                "essay_id": 1000 + idx,
                "prompt_name": topic,
                "assignment": assign,
                "source_text": source,
                "full_text": essay_text,
                "score": float((idx % 5) + 2)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"✓ Successfully generated fallback ASAP 2.0 dataset with {len(df)} topics at {csv_path}")


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
            "Based on the rubric above, evaluate the student essay and respond with ONLY the score.\n"
            "Your response MUST start with exactly this format: 'Score: X'\n"
            "where X is a single integer between 1 and 6.\n"
            "Example: Score: 4\n"
            "Do NOT write anything before 'Score:'. Begin your response with 'Score:' immediately."
        )
        return prompt


