import re
from langdetect import detect
import logging

class DisasterSentimentAnalyzer:
    def __init__(self):
        self.sentiment_patterns = {
            'Panic': {
                'en': [
                    r'help.*urgently', r'sos', r'emergency', r'trapped', r'desperate',
                    r'no food', r'no water', r'dying', r'danger', r'stranded'
                ],
                'tl': [
                    r'tulong.*po', r'naiipit', r'nangangailangan', r'walang (pagkain|tubig)',
                    r'hindi.*makalabas', r'nasasayang', r'nawawalan ng.*', r'nasiraan'
                ]
            },
            'Fear/Anxiety': {
                'en': [
                    r'scared', r'afraid', r'worried', r'terrified', r'frightened',
                    r'fear', r'anxiety', r'concerned', r'nervous', r'unsafe'
                ],
                'tl': [
                    r'takot', r'natatakot', r'kabado', r'nag-aalala', r'hindi.*makatulog',
                    r'delikado', r'mapanganib', r'nakakatakot', r'kinakabahan'
                ]
            },
            'Disbelief': {
                'en': [
                    r'cannot believe', r"can't believe", r'unbelievable', r'shocked',
                    r'surprised', r'unexpected', r'how could', r'never thought'
                ],
                'tl': [
                    r'hindi.*makapaniwala', r'gulat', r'nagulat', r'mengubra',
                    r'grabe', r'sobrang lakas', r'ganito pala', r'sino.*mag-aakala'
                ]
            },
            'Resilience': {
                'en': [
                    r'helping', r'support', r'together', r'strong', r'recover',
                    r'rebuild', r'assist', r'volunteer', r'donate', r'community'
                ],
                'tl': [
                    r'tulong', r'magtulungan', r'kakayanin', r'babangon', r'lalaban',
                    r'magkaisa', r'magbayanihan', r'magmalasakit', r'magdasal'
                ]
            }
        }
        
        # Additional sentiment markers
        self.sentiment_words = {
            'Panic': {
                'en': ['urgent', 'emergency', 'help', 'sos', 'save', 'rescue', 'trapped'],
                'tl': ['tulong', 'saklolo', 'naiipit', 'nangangailangan', 'walang']
            },
            'Fear/Anxiety': {
                'en': ['scared', 'afraid', 'worried', 'fear', 'anxiety', 'terrified'],
                'tl': ['takot', 'natatakot', 'kabado', 'nag-aalala', 'delikado']
            },
            'Disbelief': {
                'en': ['unbelievable', 'shocked', 'surprised', 'unexpected', 'how'],
                'tl': ['hindi makapaniwala', 'gulat', 'grabe', 'sobra', 'ganito']
            },
            'Resilience': {
                'en': ['help', 'support', 'together', 'strong', 'recover', 'assist'],
                'tl': ['tulong', 'sama', 'kakayanin', 'babangon', 'lalaban', 'bayanihan']
            }
        }

    def analyze_sentiment(self, text):
        """
        Analyze sentiment using pattern matching and word presence
        Returns sentiment label and confidence score
        """
        try:
            # Detect language
            lang = detect(text)
            lang = 'tl' if lang in ['tl', 'fil'] else 'en'
            
            # Initialize scores for each sentiment
            scores = {
                'Panic': 0,
                'Fear/Anxiety': 0,
                'Disbelief': 0,
                'Resilience': 0,
                'Neutral': 0
            }
            
            text_lower = text.lower()
            
            # Check patterns
            for sentiment in self.sentiment_patterns:
                patterns = self.sentiment_patterns[sentiment][lang]
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        scores[sentiment] += 1.5  # Patterns have higher weight
            
            # Check word presence
            for sentiment in self.sentiment_words:
                words = self.sentiment_words[sentiment][lang]
                for word in words:
                    if word in text_lower:
                        scores[sentiment] += 1
            
            # If no strong sentiment is detected, increase neutral score
            if max(scores.values()) < 1:
                scores['Neutral'] = 1
            
            # Get the dominant sentiment
            max_score = max(scores.values())
            if max_score == 0:
                dominant_sentiment = 'Neutral'
                confidence = 0.7
            else:
                dominant_sentiment = max(scores.items(), key=lambda x: x[1])[0]
                confidence = min(0.95, max_score / (sum(scores.values()) + 0.1))
            
            # Generate explanation
            explanation = self._generate_explanation(dominant_sentiment, lang)
            
            return {
                'sentiment': dominant_sentiment,
                'confidence': confidence,
                'explanation': explanation,
                'language': 'Filipino' if lang == 'tl' else 'English'
            }
            
        except Exception as e:
            logging.error(f"Sentiment analysis error: {e}")
            return {
                'sentiment': 'Neutral',
                'confidence': 0.7,
                'explanation': 'Fallback due to analysis error',
                'language': 'English'
            }
    
    def _generate_explanation(self, sentiment, lang):
        """Generate a human-readable explanation for the sentiment"""
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
                'tl': 'Nagpapakita ng pagkagulat o hindi makapaniwala sa nangyayari'
            },
            'Resilience': {
                'en': 'Demonstrates community support and determination',
                'tl': 'Nagpapakita ng pagkakaisa at determinasyon ng komunidad'
            },
            'Neutral': {
                'en': 'Provides factual information without strong emotion',
                'tl': 'Nagbibigay ng impormasyon na walang matinding damdamin'
            }
        }
        return explanations.get(sentiment, {}).get(lang, explanations[sentiment]['en'])

# Create global instance
analyzer = DisasterSentimentAnalyzer()
