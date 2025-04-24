#!/usr/bin/env python3

"""
CSV Processor using Hybrid Neural Network Model for PanicSense
This script processes CSV files using the mBERT + Bi-GRU & LSTM hybrid model.
It handles multilingual input (English and Tagalog) and outputs detailed sentiment analysis.
"""

import os
import sys
import json
import argparse
import pandas as pd
import torch
import numpy as np
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from langdetect import detect, LangDetectException

# Import our custom modules
try:
    from hybrid_model import HybridModelProcessor
    from evaluation_metrics import ModelEvaluator
    from emoji_utils import clean_text_preserve_indicators, preprocess_text
except ImportError:
    try:
        from python.hybrid_model import HybridModelProcessor
        from python.evaluation_metrics import ModelEvaluator
        from python.emoji_utils import clean_text_preserve_indicators, preprocess_text
    except ImportError:
        print("Error importing custom modules. Make sure the required files exist.")
        sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CSVProcessorHybrid:
    """
    Process CSV files using the hybrid neural network model
    Specifically designed for sentiment analysis in disaster contexts
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the processor
        
        Args:
            model_path: Path to the pretrained model (if None, the default model will be used)
        """
        # Determine device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize the hybrid model processor
        self.processor = HybridModelProcessor(model_path=model_path, device=self.device)
        
        # Initialize the model evaluator
        self.evaluator = ModelEvaluator(sentiment_labels=self.processor.model.sentiment_labels)
        
        # Define class for extracting location and disaster type from text
        self.location_extractor = LocationExtractor()
        self.disaster_extractor = DisasterTypeExtractor()
        
    def report_progress(self, processed: int, stage: str, total: int = None):
        """Print progress in a format that can be parsed by the Node.js service"""
        progress_data = {"processed": processed, "stage": stage}
        
        # If total is provided, include it in the progress report
        if total is not None:
            progress_data["total"] = total
        
        progress_info = json.dumps(progress_data)
        # Add a unique marker at the end to ensure each progress message is on a separate line
        print(f"PROGRESS:{progress_info}::END_PROGRESS", file=sys.stderr)
        sys.stderr.flush()  # Ensure output is immediately visible
    
    def process_csv(self, input_file: str, output_file: Optional[str] = None, 
                   text_column: str = 'text', batch_size: int = 32, 
                   validate: bool = False) -> Dict[str, Any]:
        """
        Process a CSV file containing text data
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output JSON file (if None, uses input filename with .json extension)
            text_column: Name of column containing text data
            batch_size: Batch size for processing
            validate: Whether to validate model performance (requires 'sentiment' column)
            
        Returns:
            Dictionary with processing results and metrics
        """
        start_time = time.time()
        
        try:
            # Load CSV file
            logger.info(f"Loading CSV file: {input_file}")
            df = pd.read_csv(input_file)
            
            if text_column not in df.columns:
                logger.error(f"Text column '{text_column}' not found in CSV. Available columns: {df.columns.tolist()}")
                return {"error": f"Text column '{text_column}' not found"}
            
            total_rows = len(df)
            logger.info(f"Processing {total_rows} rows from CSV file")
            
            # Initialize results list
            results = []
            
            # Process in batches to avoid memory issues
            true_labels = []  # For validation if available
            predicted_labels = []
            confidences = []
            languages = []
            disaster_types = []
            
            # Check if we have ground truth for validation
            has_ground_truth = 'sentiment' in df.columns
            if has_ground_truth and validate:
                logger.info("Ground truth sentiment labels found. Will validate model performance.")
                true_labels = df['sentiment'].tolist()
            
            # Process batches
            self.report_progress(0, "Hybrid Model Analysis", total_rows)
            
            for i in range(0, total_rows, batch_size):
                batch_end = min(i + batch_size, total_rows)
                batch_df = df.iloc[i:batch_end]
                batch_texts = batch_df[text_column].tolist()
                
                # Process batch with hybrid model
                batch_results = self.processor.process_batch(batch_texts)
                
                # Extract disaster types and locations for each text
                for j, (text, result) in enumerate(zip(batch_texts, batch_results)):
                    idx = i + j
                    
                    # Add row information
                    result_row = {
                        "text": text,
                        "sentiment": result["sentiment"],
                        "confidence": result["confidence"],
                        "explanation": result["explanation"]
                    }
                    
                    # Try to detect language
                    try:
                        language = detect(text)
                        result_row["language"] = language
                        languages.append(language)
                    except LangDetectException:
                        result_row["language"] = "unknown"
                        languages.append("unknown")
                    
                    # Extract disaster type
                    disaster_type = self.disaster_extractor.extract(text)
                    result_row["disaster_type"] = disaster_type
                    disaster_types.append(disaster_type)
                    
                    # Extract location
                    location = self.location_extractor.extract(text)
                    result_row["location"] = location
                    
                    # Add additional columns from original CSV
                    for col in batch_df.columns:
                        if col != text_column and col not in result_row:
                            result_row[col] = batch_df.iloc[j-i][col]
                    
                    # Add timestamp if not present
                    if "timestamp" not in result_row:
                        result_row["timestamp"] = datetime.now().isoformat()
                    
                    # Add to results
                    results.append(result_row)
                    
                    # For validation
                    predicted_labels.append(result["sentiment"])
                    confidences.append(result["confidence"])
                
                # Report progress
                self.report_progress(batch_end, "Hybrid Model Analysis", total_rows)
            
            # Create output file path if not provided
            if output_file is None:
                output_file = os.path.splitext(input_file)[0] + "_hybrid_results.json"
            
            # Prepare final output
            output = {
                "results": results,
                "processing_time": time.time() - start_time,
                "total_processed": total_rows,
                "model_type": "mBERT + Bi-GRU & LSTM Hybrid"
            }
            
            # Validate model performance if ground truth is available
            if has_ground_truth and validate:
                logger.info("Calculating evaluation metrics...")
                metrics = self.evaluator.evaluate_predictions(true_labels, predicted_labels, confidences)
                
                # Add multilingual and cross-disaster metrics
                multilingual_metrics = self.evaluator.evaluate_multilingual_performance(
                    predicted_labels, true_labels, languages
                )
                
                disaster_metrics = self.evaluator.evaluate_cross_disaster_performance(
                    predicted_labels, true_labels, disaster_types
                )
                
                # Combine all metrics
                metrics['multilingual'] = multilingual_metrics
                metrics['cross_disaster'] = disaster_metrics
                
                # Add metrics to output
                output["metrics"] = metrics
                
                # Generate human-readable report
                report = self.evaluator.generate_evaluation_report(metrics)
                logger.info(f"\n{report}")
                
                # Save evaluation report
                report_file = os.path.splitext(output_file)[0] + "_evaluation_report.txt"
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                logger.info(f"Evaluation report saved to {report_file}")
                
                # Generate plots if matplotlib is available
                try:
                    import matplotlib.pyplot as plt
                    plots_dir = os.path.join(os.path.dirname(output_file), "evaluation_plots")
                    os.makedirs(plots_dir, exist_ok=True)
                    
                    # Confusion matrix
                    self.evaluator.plot_confusion_matrix(
                        np.array(metrics['confusion_matrix']),
                        save_path=os.path.join(plots_dir, 'confusion_matrix.png')
                    )
                    
                    # Normalized confusion matrix
                    self.evaluator.plot_confusion_matrix(
                        np.array(metrics['normalized_confusion_matrix']),
                        normalize=True,
                        save_path=os.path.join(plots_dir, 'normalized_confusion_matrix.png')
                    )
                    
                    # Metrics plots
                    self.evaluator.plot_metrics(
                        metrics,
                        save_path=os.path.join(plots_dir, 'performance_metrics.png')
                    )
                    
                    logger.info(f"Evaluation plots saved to {plots_dir}")
                except ImportError:
                    logger.warning("Matplotlib not available. Skipping plot generation.")
            
            # Save results to output file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Results saved to {output_file}")
            return output
            
        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}", exc_info=True)
            return {"error": str(e)}


class LocationExtractor:
    """
    Extract location information from text with a focus on Philippine locations
    """
    
    def __init__(self):
        # Define regions and major cities/provinces in the Philippines
        self.ph_locations = [
            # Regions
            "NCR", "Metro Manila", "National Capital Region",
            "CAR", "Cordillera Administrative Region",
            "CALABARZON", "MIMAROPA", "SOCCSKSARGEN",
            "Ilocos Region", "Cagayan Valley", "Central Luzon",
            "Bicol Region", "Western Visayas", "Central Visayas",
            "Eastern Visayas", "Zamboanga Peninsula", "Northern Mindanao",
            "Davao Region", "CARAGA", "ARMM",
            "Bangsamoro Autonomous Region",
            
            # Major cities and provinces
            "Manila", "Quezon City", "Davao", "Cebu", "Makati", "Taguig",
            "Pasig", "Cagayan de Oro", "Zamboanga", "Baguio", "Iloilo",
            "Batangas", "Laguna", "Cavite", "Rizal", "Bulacan", "Pampanga",
            "Pangasinan", "La Union", "Ilocos", "Isabela", "Bicol", "Albay",
            "Camarines", "Palawan", "Mindoro", "Marinduque", "Leyte", "Samar",
            "Negros", "Panay", "Iloilo", "Bohol", "Tacloban", "Tagbilaran",
            "Lanao", "Cotabato", "Maguindanao", "Sulu", "Basilan", "Tawi-Tawi",
            
            # Common place terms in Filipino
            "Maynila", "Lungsod", "Lalawigan", "Probinsya", "Barangay",
            "Munisipyo", "Bayan", "Nayon", "Lungsod", "Kabisera", "Rehiyon"
        ]
        
        # Common location prefixes and indicators
        self.location_indicators = [
            "in", "at", "near", "around", "from", "to", "sa", "malapit sa",
            "dito sa", "diyan sa", "papunta sa", "galing sa", "nasa", "bandang"
        ]
    
    def extract(self, text: str) -> str:
        """
        Extract location from text, focusing on Philippine locations
        
        Args:
            text: Input text to analyze
            
        Returns:
            Extracted location or empty string if none found
        """
        # Preprocess text
        text_lower = text.lower()
        
        # Check for direct mentions of locations
        for location in self.ph_locations:
            if location.lower() in text_lower:
                # Check if it's a full word or part of a word
                location_lower = location.lower()
                if (f" {location_lower} " in f" {text_lower} " or
                    text_lower.startswith(f"{location_lower} ") or
                    text_lower.endswith(f" {location_lower}") or
                    text_lower == location_lower):
                    return location
        
        # Try to find location with indicators
        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower()
            # If we find a location indicator, check the next word(s)
            if word_lower in self.location_indicators and i < len(words) - 1:
                # Check next 1-3 words (for multi-word locations)
                for j in range(1, min(4, len(words) - i)):
                    potential_location = " ".join(words[i+1:i+j+1])
                    # Check against known locations
                    for location in self.ph_locations:
                        if location.lower() in potential_location.lower():
                            return location
        
        # No location found
        return ""


class DisasterTypeExtractor:
    """
    Extract disaster type from text
    """
    
    def __init__(self):
        # Define disaster types and their keywords (English and Filipino)
        self.disaster_types = {
            "Earthquake": [
                "earthquake", "quake", "tremor", "seismic", "lindol",
                "magnitude", "aftershock", "shaking", "lumindol", "pagyanig",
                "paglindol", "ground shaking", "magnitude"
            ],
            "Flood": [
                "flood", "flooding", "inundation", "baha", "tubig", "binaha",
                "flash flood", "rising water", "bumabaha", "nagbaha",
                "high water level", "water rising", "overflowing", "pagbaha",
                "underwater", "submerged", "nabahaan"
            ],
            "Typhoon": [
                "typhoon", "storm", "cyclone", "hurricane", "bagyo",
                "super typhoon", "habagat", "ulan", "buhos", "storm surge",
                "malakas na hangin", "heavy rain", "signal no", "strong wind",
                "malakas na ulan", "flood warning", "storm warning",
                "evacuate due to storm", "matinding ulan"
            ],
            "Fire": [
                "fire", "blaze", "burning", "sunog", "nasusunog", "nasunog",
                "may sunog", "may nasusunog", "meron sunog", "may nasunog",
                "nagliliyab", "flame", "apoy", "burning building", 
                "burning house", "tulong sunog", "house fire", "fire truck",
                "fire fighter", "building fire", "fire alarm"
            ],
            "Volcanic Eruption": [
                "volcano", "eruption", "lava", "ash", "bulkan", "ashfall",
                "magma", "volcanic", "bulkang", "active volcano",
                "phivolcs alert", "taal", "mayon", "pinatubo",
                "volcanic activity", "phivolcs", "volcanic ash",
                "evacuate volcano", "erupting", "erupted", "abo ng bulkan"
            ],
            "Landslide": [
                "landslide", "mudslide", "avalanche", "guho", "pagguho",
                "pagguho ng lupa", "collapsed", "erosion", "land collapse",
                "soil erosion", "rock slide", "debris flow", "mountainside",
                "nagkaroong ng guho", "rumble", "bangin", "bumagsak na lupa"
            ]
        }
    
    def extract(self, text: str) -> str:
        """
        Extract disaster type from text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Extracted disaster type or "Not Specified" if none found
        """
        # Preprocess text
        text_lower = text.lower()
        
        # Score each disaster type based on keyword matches
        scores = {disaster_type: 0 for disaster_type in self.disaster_types}
        
        for disaster_type, keywords in self.disaster_types.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    # Check if it's a full word or part of a word
                    keyword_lower = keyword.lower()
                    if (f" {keyword_lower} " in f" {text_lower} " or
                        text_lower.startswith(f"{keyword_lower} ") or
                        text_lower.endswith(f" {keyword_lower}") or
                        text_lower == keyword_lower):
                        scores[disaster_type] += 2  # Full word match
                    else:
                        scores[disaster_type] += 1  # Partial match
        
        # Get disaster type with highest score, if any
        max_score = max(scores.values())
        if max_score > 0:
            max_disaster_types = [dt for dt, score in scores.items() if score == max_score]
            return max_disaster_types[0]  # Return the first one if there are ties
        
        return "Not Specified"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CSV files using Hybrid Neural Network Model')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    parser.add_argument('--text_column', type=str, default='text', help='Column name containing text data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--model_path', type=str, help='Path to pretrained model')
    parser.add_argument('--validate', action='store_true', help='Validate model performance (requires sentiment column)')
    
    args = parser.parse_args()
    
    processor = CSVProcessorHybrid(model_path=args.model_path)
    result = processor.process_csv(
        args.input, 
        args.output, 
        text_column=args.text_column, 
        batch_size=args.batch_size,
        validate=args.validate
    )
    
    if "error" in result:
        logger.error(f"Processing failed: {result['error']}")
        sys.exit(1)
    else:
        logger.info(f"Processed {result['total_processed']} rows in {result['processing_time']:.2f} seconds")
        
        # If metrics are available, print summary
        if "metrics" in result:
            logger.info(f"Model accuracy: {result['metrics']['accuracy']:.4f}")
            logger.info(f"F1 score (weighted): {result['metrics']['f1_score']['weighted']:.4f}")
        
        sys.exit(0)