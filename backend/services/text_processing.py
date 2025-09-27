"""
NLP and content analysis services for the Space Biology Knowledge Engine.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import asyncio

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from config.settings import get_settings
from services.llm_service import LLMService

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Main text processing service for NLP tasks.
    """

    def __init__(self):
        self.settings = get_settings()
        self.nlp = None
        self.llm_service = None
        self._initialize_components()

    def _initialize_components(self):
        """Initialize NLP components."""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.error("spaCy model 'en_core_web_sm' not found. Please install with: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Initialize LLM service
        self.llm_service = LLMService()

    async def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract named entities from text using spaCy NLP pipeline.

        Args:
            text: Input text to process

        Returns:
            Dictionary with entity types and their details
        """
        if not self.nlp:
            logger.warning("spaCy model not available for entity extraction")
            return {}

        try:
            # Process text with spaCy
            doc = self.nlp(text)

            entities = {
                'organisms': [],
                'chemicals': [],
                'locations': [],
                'persons': [],
                'organizations': [],
                'biological_processes': [],
                'equipment': []
            }

            # Extract standard named entities
            for ent in doc.ents:
                entity_info = {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 1.0  # spaCy doesn't provide confidence scores
                }

                # Categorize entities for space biology context
                if ent.label_ in ['PERSON']:
                    entities['persons'].append(entity_info)
                elif ent.label_ in ['ORG']:
                    entities['organizations'].append(entity_info)
                elif ent.label_ in ['GPE', 'LOC']:
                    entities['locations'].append(entity_info)

            # Extract domain-specific entities
            bio_entities = self._extract_biological_entities(text)
            entities.update(bio_entities)

            # Filter and deduplicate
            entities = self._filter_and_deduplicate_entities(entities)

            logger.debug(f"Extracted {sum(len(v) for v in entities.values())} entities from text")
            return entities

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {}

    def _extract_biological_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract biology-specific entities using pattern matching and domain knowledge.

        Args:
            text: Input text

        Returns:
            Dictionary with biological entities
        """
        entities = {
            'organisms': [],
            'chemicals': [],
            'biological_processes': [],
            'equipment': []
        }

        # Organism patterns
        organism_patterns = [
            r'\b(?:Homo sapiens|humans?|astronauts?)\b',
            r'\b(?:Mus musculus|mice|mouse)\b',
            r'\b(?:Rattus norvegicus|rats?)\b',
            r'\b(?:Drosophila|fruit flies?)\b',
            r'\b(?:Caenorhabditis elegans|C\. elegans|nematodes?)\b',
            r'\b(?:Saccharomyces cerevisiae|S\. cerevisiae|yeast)\b',
            r'\b(?:Escherichia coli|E\. coli|bacteria)\b',
            r'\b(?:Arabidopsis|plants?|seedlings?)\b'
        ]

        # Chemical/molecular patterns
        chemical_patterns = [
            r'\b(?:DNA|RNA|proteins?|genes?|chromosomes?)\b',
            r'\b(?:calcium|potassium|sodium|magnesium|iron)\b',
            r'\b(?:glucose|ATP|ADP|cortisol|adrenaline)\b',
            r'\b(?:radiation|cosmic rays?|gamma rays?)\b'
        ]

        # Biological process patterns
        process_patterns = [
            r'\b(?:gene expression|transcription|translation)\b',
            r'\b(?:cell division|mitosis|meiosis|apoptosis)\b',
            r'\b(?:metabolism|photosynthesis|respiration)\b',
            r'\b(?:bone loss|muscle atrophy|cardiac deconditioning)\b',
            r'\b(?:immune response|inflammation|stress response)\b'
        ]

        # Equipment patterns
        equipment_patterns = [
            r'\b(?:centrifuge|microscope|spectrophotometer)\b',
            r'\b(?:ISS|International Space Station|space station)\b',
            r'\b(?:bioreactor|incubator|cell culture)\b'
        ]

        pattern_groups = [
            (organism_patterns, 'organisms'),
            (chemical_patterns, 'chemicals'),
            (process_patterns, 'biological_processes'),
            (equipment_patterns, 'equipment')
        ]

        for patterns, entity_type in pattern_groups:
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_info = {
                        'text': match.group(),
                        'label': entity_type.upper(),
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.8
                    }
                    entities[entity_type].append(entity_info)

        return entities

    def _filter_and_deduplicate_entities(self, entities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Filter and deduplicate extracted entities.

        Args:
            entities: Raw entities dictionary

        Returns:
            Filtered and deduplicated entities
        """
        filtered_entities = {}

        for entity_type, entity_list in entities.items():
            # Remove duplicates based on text and position
            seen = set()
            unique_entities = []

            for entity in entity_list:
                key = (entity['text'].lower(), entity['start'], entity['end'])
                if key not in seen:
                    seen.add(key)
                    unique_entities.append(entity)

            # Filter out very short entities (less than 2 characters)
            unique_entities = [e for e in unique_entities if len(e['text']) >= 2]

            # Sort by confidence and position
            unique_entities.sort(key=lambda x: (-x['confidence'], x['start']))

            filtered_entities[entity_type] = unique_entities

        return filtered_entities

    async def summarize_text(self, text: str, max_length: int = 200,
                           summary_type: str = 'brief') -> str:
        """
        Generate a summary of the input text.

        Args:
            text: Text to summarize
            max_length: Maximum summary length in characters
            summary_type: Type of summary ('brief' or 'detailed')

        Returns:
            Generated summary
        """
        if len(text) <= max_length:
            return text

        try:
            # Use extractive summarization for short summaries
            if summary_type == 'brief' and self.nlp:
                summary = self._extractive_summarization(text, max_length)
                if summary:
                    return summary

            # Use LLM for more sophisticated summarization
            if self.llm_service:
                prompt = self._create_summarization_prompt(text, summary_type, max_length)
                summary = await self.llm_service.generate_text(prompt)
                return summary[:max_length] if len(summary) > max_length else summary

            # Fallback: simple truncation with sentence boundary
            return self._simple_truncate(text, max_length)

        except Exception as e:
            logger.error(f"Text summarization failed: {e}")
            return self._simple_truncate(text, max_length)

    def _extractive_summarization(self, text: str, max_length: int) -> Optional[str]:
        """
        Perform extractive summarization using sentence scoring.

        Args:
            text: Input text
            max_length: Maximum summary length

        Returns:
            Extractive summary or None if processing fails
        """
        if not self.nlp:
            return None

        try:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]

            if len(sentences) <= 2:
                return " ".join(sentences)

            # Score sentences based on keyword frequency and position
            sentence_scores = {}
            word_freq = Counter()

            # Calculate word frequencies (excluding stop words)
            for token in doc:
                if token.text.lower() not in STOP_WORDS and token.is_alpha:
                    word_freq[token.text.lower()] += 1

            # Score sentences
            for i, sentence in enumerate(sentences):
                sent_doc = self.nlp(sentence)
                score = 0

                # Keyword frequency score
                for token in sent_doc:
                    if token.text.lower() in word_freq:
                        score += word_freq[token.text.lower()]

                # Position score (earlier sentences get higher scores)
                position_score = 1.0 - (i / len(sentences)) * 0.5
                score *= position_score

                sentence_scores[sentence] = score

            # Select top sentences
            sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
            summary_sentences = []
            current_length = 0

            for sentence, score in sorted_sentences:
                if current_length + len(sentence) <= max_length:
                    summary_sentences.append(sentence)
                    current_length += len(sentence)
                else:
                    break

            # Reorder sentences by original position
            original_order = []
            for sentence in sentences:
                if sentence in summary_sentences:
                    original_order.append(sentence)

            return " ".join(original_order)

        except Exception as e:
            logger.error(f"Extractive summarization failed: {e}")
            return None

    def _create_summarization_prompt(self, text: str, summary_type: str, max_length: int) -> str:
        """Create a prompt for LLM-based summarization."""
        if summary_type == 'brief':
            return f"""
            Please provide a brief summary of the following space biology research text in {max_length} characters or less:

            {text}

            Summary:
            """
        else:
            return f"""
            Please provide a detailed but concise summary of the following space biology research text,
            highlighting key findings, methodology, and implications for space missions:

            {text}

            Summary:
            """

    def _simple_truncate(self, text: str, max_length: int) -> str:
        """Simple truncation at sentence boundaries."""
        if len(text) <= max_length:
            return text

        # Find the last complete sentence within the limit
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        last_exclamation = truncated.rfind('!')
        last_question = truncated.rfind('?')

        last_sentence_end = max(last_period, last_exclamation, last_question)

        if last_sentence_end > max_length * 0.6:  # At least 60% of desired length
            return truncated[:last_sentence_end + 1]
        else:
            return truncated + "..."

    async def detect_methodology(self, text: str) -> Dict[str, Any]:
        """
        Extract experimental methodology information from text.

        Args:
            text: Research text to analyze

        Returns:
            Dictionary with methodology details
        """
        methodology = {
            'study_type': 'unknown',
            'experimental_design': [],
            'techniques': [],
            'duration': None,
            'sample_size': None,
            'statistical_methods': []
        }

        try:
            # Study type patterns
            study_type_patterns = {
                'controlled_experiment': r'\b(?:controlled\s+(?:experiment|trial|study)|randomized|placebo)\b',
                'observational': r'\b(?:observational|longitudinal|cohort|cross-sectional)\b',
                'case_study': r'\b(?:case\s+study|case\s+report)\b',
                'review': r'\b(?:systematic\s+review|meta-analysis|literature\s+review)\b'
            }

            for study_type, pattern in study_type_patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    methodology['study_type'] = study_type
                    break

            # Experimental techniques
            technique_patterns = [
                r'\b(?:PCR|qPCR|RT-PCR|western\s+blot|ELISA)\b',
                r'\b(?:RNA-seq|DNA\s+sequencing|gene\s+expression)\b',
                r'\b(?:flow\s+cytometry|microscopy|immunofluorescence)\b',
                r'\b(?:cell\s+culture|tissue\s+culture|organoids?)\b',
                r'\b(?:centrifugation|microgravity\s+simulation)\b'
            ]

            for pattern in technique_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                methodology['techniques'].extend([match.lower() for match in matches])

            # Duration extraction
            duration_patterns = [
                r'\b(\d+)\s*(?:days?|weeks?|months?|years?)\b',
                r'\b(?:for|during|over)\s+(\d+)\s*(?:days?|weeks?|months?)\b'
            ]

            for pattern in duration_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    methodology['duration'] = match.group(1)
                    break

            # Sample size extraction
            sample_size_patterns = [
                r'\bn\s*=\s*(\d+)\b',
                r'\b(\d+)\s+(?:subjects?|participants?|samples?|mice|rats)\b',
                r'\bsample\s+size\s*:\s*(\d+)\b'
            ]

            for pattern in sample_size_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    methodology['sample_size'] = int(match.group(1))
                    break

            # Statistical methods
            stats_patterns = [
                r'\b(?:t-test|ANOVA|chi-square|regression|correlation)\b',
                r'\b(?:p\s*<\s*0\.05|significant|statistical\s+analysis)\b',
                r'\b(?:mean|median|standard\s+deviation|confidence\s+interval)\b'
            ]

            for pattern in stats_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                methodology['statistical_methods'].extend([match.lower() for match in matches])

            # Remove duplicates
            methodology['techniques'] = list(set(methodology['techniques']))
            methodology['statistical_methods'] = list(set(methodology['statistical_methods']))

            return methodology

        except Exception as e:
            logger.error(f"Methodology detection failed: {e}")
            return methodology

    async def classify_research_type(self, text: str, title: str = "") -> Dict[str, float]:
        """
        Classify research type based on text content.

        Args:
            text: Research text content
            title: Research title (optional)

        Returns:
            Dictionary with research type classifications and confidence scores
        """
        # Combine title and text for classification
        full_text = f"{title} {text}".lower()

        research_categories = {
            'genomics': [
                'gene', 'dna', 'rna', 'genome', 'transcription', 'expression',
                'sequencing', 'pcr', 'mutation', 'genetic', 'allele'
            ],
            'proteomics': [
                'protein', 'proteome', 'enzyme', 'amino acid', 'peptide',
                'western blot', 'mass spectrometry', 'protein expression'
            ],
            'physiology': [
                'physiological', 'cardiovascular', 'respiratory', 'muscular',
                'bone density', 'metabolism', 'hormone', 'blood pressure'
            ],
            'cell_biology': [
                'cell culture', 'cellular', 'mitosis', 'apoptosis', 'organelle',
                'cytoplasm', 'membrane', 'cell division', 'growth'
            ],
            'microbiology': [
                'bacteria', 'microbial', 'pathogen', 'microorganism', 'infection',
                'antibiotic', 'culture', 'microgravity microbes'
            ],
            'radiation': [
                'radiation', 'cosmic ray', 'ionizing', 'uv', 'gamma',
                'dose', 'exposure', 'radioprotection', 'dna damage'
            ],
            'psychology': [
                'psychological', 'behavioral', 'cognitive', 'stress', 'isolation',
                'crew dynamics', 'mental health', 'performance'
            ],
            'biomedical': [
                'medical', 'clinical', 'therapeutic', 'drug', 'treatment',
                'diagnosis', 'biomarker', 'health monitoring'
            ]
        }

        scores = {}
        total_words = len(full_text.split())

        for category, keywords in research_categories.items():
            category_score = 0
            for keyword in keywords:
                # Count occurrences of each keyword
                count = full_text.count(keyword.lower())
                category_score += count

            # Normalize by total words and keyword count
            normalized_score = category_score / (total_words + 1) * 100
            scores[category] = min(normalized_score, 1.0)  # Cap at 1.0

        # Ensure scores sum to reasonable total
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v / total_score for k, v in scores.items()}

        return scores

    async def extract_key_findings(self, text: str) -> List[str]:
        """
        Extract key findings and conclusions from research text.

        Args:
            text: Research text

        Returns:
            List of key findings
        """
        findings = []

        try:
            # Look for common conclusion indicators
            conclusion_patterns = [
                r'(?:we found|results showed?|findings indicate|conclusion|in summary)',
                r'(?:significantly|notably|importantly|remarkably)',
                r'(?:demonstrated|revealed|suggested|indicated)',
                r'(?:effects? of|impact of|influence of|relationship between)'
            ]

            if self.nlp:
                doc = self.nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents]

                for sentence in sentences:
                    # Check if sentence contains conclusion indicators
                    for pattern in conclusion_patterns:
                        if re.search(pattern, sentence, re.IGNORECASE):
                            findings.append(sentence)
                            break

            # Limit to most relevant findings
            findings = findings[:5]

            # If no findings found, use LLM to extract key points
            if not findings and self.llm_service:
                prompt = f"""
                Extract the top 3 key findings or conclusions from this space biology research text:

                {text}

                Key findings:
                1.
                2.
                3.
                """

                llm_findings = await self.llm_service.generate_text(prompt)
                if llm_findings:
                    # Parse LLM response into list
                    lines = llm_findings.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and (line.startswith(('1.', '2.', '3.', '-', '•'))):
                            # Clean up the finding
                            finding = re.sub(r'^\d+\.\s*|^[-•]\s*', '', line)
                            if finding:
                                findings.append(finding)

            return findings

        except Exception as e:
            logger.error(f"Key findings extraction failed: {e}")
            return []

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for processing.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())

        # Remove special characters but keep basic punctuation
        cleaned = re.sub(r'[^\w\s.,;:!?()-]', '', cleaned)

        # Remove multiple consecutive punctuation
        cleaned = re.sub(r'([.,;:!?]){2,}', r'\1', cleaned)

        return cleaned