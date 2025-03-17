import re
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternSentimentAnalyzer:
    def __init__(self):
        # Initialize pattern dictionaries for both English and Filipino
        self.patterns = {
            'en': {
                'Panic': [
                    r'help.*!+', r'emergency', r'sos', r'trapped', r'urgent',
                    r'rescue', r'dying', r'evacuate', r'danger', r'critical'
                ],
                'Fear/Anxiety': [
                    r'scared', r'afraid', r'worried', r'fear', r'terrified',
                    r'nervous', r'anxious', r'frightened', r'scared', r'panic'
                ],
                'Disbelief': [
                    r'cannot believe', r'unbelievable', r"can't believe",
                    r'shocking', r'shocked', r'unexpected', r'surprised',
                    r'how could', r'how can', r'impossible'
                ],
                'Resilience': [
                    r'strong', r'together', r'help.*others', r'community',
                    r'rebuild', r'recover', r'support', r'brave', r'strength',
                    r'survive', r'overcome'
                ]
            },
            'tl': {
                'Panic': [
                    r'tulong.*!+', r'saklolo', r'naiipit', r'nanganganib',
                    r'nasugatan', r'nakulong', r'nasusunog', r'nalulunod',
                    r'hindi.*makalabas', r'hindi.*makahinga'
                ],
                'Fear/Anxiety': [
                    r'takot', r'natatakot', r'nag-aalala', r'kabado',
                    r'kinakabahan', r'nakakatakot', r'nakakanerbyos',
                    r'nakakapanginig', r'hindi.*mapakali', r'nangangamba'
                ],
                'Disbelief': [
                    r'hindi.*kapani-paniwala', r'gulat', r'nagulat',
                    r'nagugulat', r'di.*makapaniwala', r'grabe',
                    r'nakakaloka', r'nakakagulat', r'paano.*nangyari'
                ],
                'Resilience': [
                    r'kakayanin', r'magbayanihan', r'tulong.*sa.*kapwa',
                    r'sama-sama', r'magtulungan', r'magkaisa', r'malalagpasan',
                    r'babangon', r'lalaban', r'matatag'
                ]
            }
        }

        # Weight multipliers for pattern matching
        self.weights = {
            'exact': 1.0,
            'partial': 0.7,
            'multiple': 1.2,
            'exclamation': 1.1,
            'question': 0.9
        }

    def detect_language(self, text: str) -> str:
        """Detect if text is Filipino or English based on common Filipino words"""
        filipino_markers = ['ang', 'ng', 'mga', 'sa', 'po', 'ko', 'na', 'ay', 'mga', 'namin', 'kami']
        text_lower = text.lower()
        
        # Count Filipino markers
        marker_count = sum(1 for marker in filipino_markers if f' {marker} ' in f' {text_lower} ')
        
        return 'tl' if marker_count >= 1 else 'en'

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment in text and return detailed analysis"""
        # Detect language
        lang = self.detect_language(text)
        text_lower = text.lower()
        
        # Initialize scores for each sentiment
        scores = {sentiment: 0.0 for sentiment in self.patterns[lang].keys()}
        
        # Count pattern matches for each sentiment
        for sentiment, patterns in self.patterns[lang].items():
            match_count = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    base_score = self.weights['exact']
                    
                    # Apply multipliers
                    if len(matches) > 1:
                        base_score *= self.weights['multiple']
                    if '!' in text:
                        base_score *= self.weights['exclamation']
                    if '?' in text:
                        base_score *= self.weights['question']
                        
                    scores[sentiment] += base_score
                    match_count += 1
            
            # Normalize score based on number of patterns matched
            if match_count > 0:
                scores[sentiment] = scores[sentiment] / len(patterns)

        # Determine dominant sentiment
        dominant_sentiment = max(scores.items(), key=lambda x: x[1])
        confidence = dominant_sentiment[1]

        # If no strong sentiment detected, return Neutral
        if confidence < 0.3:
            return {
                'sentiment': 'Neutral',
                'confidence': 0.5,
                'explanation': self._generate_explanation('Neutral', lang),
                'language': 'Filipino' if lang == 'tl' else 'English'
            }

        return {
            'sentiment': dominant_sentiment[0],
            'confidence': min(confidence, 0.95),  # Cap confidence at 95%
            'explanation': self._generate_explanation(dominant_sentiment[0], lang),
            'language': 'Filipino' if lang == 'tl' else 'English'
        }

    def _generate_explanation(self, sentiment: str, lang: str) -> str:
        """Generate explanation in the appropriate language"""
        explanations = {
            'Panic': {
                'en': 'Shows immediate distress or urgent need for help',
                'tl': 'Nagpapakita ng matinding pangamba o pangangailangan ng tulong'
            },
            'Fear/Anxiety': {
                'en': 'Expresses worry or concern about the situation',
                'tl': 'Nagpapahiwatig ng takot o pag-aalala sa sitwasyon'
            },
            'Disbelief': {
                'en': 'Shows shock or surprise about events',
                'tl': 'Nagpapakita ng pagkagulat o pagkabigla sa mga pangyayari'
            },
            'Resilience': {
                'en': 'Demonstrates community support and determination',
                'tl': 'Nagpapakita ng pagkakaisa at determinasyon ng komunidad'
            },
            'Neutral': {
                'en': 'Provides factual information without strong emotion',
                'tl': 'Nagbibigay ng impormasyon nang walang matinding damdamin'
            }
        }
        
        return explanations[sentiment][lang]

# Create a global instance
analyzer = PatternSentimentAnalyzer()
