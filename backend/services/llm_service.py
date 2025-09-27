"""
Large Language Model integration service for the Space Biology Knowledge Engine.
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import openai

from config.settings import get_settings, get_data_paths

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for integrating with Large Language Models (OpenAI).
    """

    def __init__(self):
        self.settings = get_settings()
        self.data_paths = get_data_paths()
        self.openai_client = None
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize LLM clients based on available API keys."""
        api_keys = self.settings.model_dump()

        # Initialize OpenAI client
        if api_keys.get('openai_api_key'):
            try:
                openai.api_key = api_keys['openai_api_key']
                self.openai_client = openai
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.openai_client = None
        else:
            logger.warning("OpenAI API key not found in configuration")

    async def generate_text(self, prompt: str, model: Optional[str] = None,
                          max_tokens: Optional[int] = None,
                          temperature: Optional[float] = None) -> str:
        """
        Generate text using the specified LLM.

        Args:
            prompt: Input prompt for text generation
            model: Model to use (if None, uses default from settings)
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature

        Returns:
            Generated text
        """
        # Use defaults from settings if not specified
        model = model or self.settings.default_llm_model
        max_tokens = max_tokens or self.settings.llm_max_tokens
        temperature = temperature or self.settings.llm_temperature

        try:
            # Log the request for monitoring
            await self._log_llm_request(prompt, model, max_tokens, temperature)

            # Route to OpenAI (only supported LLM provider)
            if model.startswith('gpt-') or not model.startswith('claude-'):
                return await self._generate_openai(prompt, model, max_tokens, temperature)
            else:
                logger.error(f"Unsupported model: {model}. Only OpenAI models are supported.")
                return f"Error: Unsupported model {model}. Please use an OpenAI model (e.g., gpt-3.5-turbo, gpt-4)."

        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return f"Error generating text: {str(e)}"

    async def _generate_openai(self, prompt: str, model: str,
                             max_tokens: int, temperature: float) -> str:
        """Generate text using OpenAI API."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")

        try:
            response = await asyncio.to_thread(
                self.openai_client.ChatCompletion.create,
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert in space biology research."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=30
            )

            generated_text = response.choices[0].message.content.strip()

            # Log usage statistics
            await self._log_llm_response(
                model, response.usage.total_tokens,
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )

            return generated_text

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise


    async def summarize_research(self, text: str, summary_type: str = 'brief') -> str:
        """
        Generate a research summary using LLM.

        Args:
            text: Research text to summarize
            summary_type: Type of summary ('brief', 'detailed', 'executive')

        Returns:
            Generated summary
        """
        if summary_type == 'brief':
            prompt = f"""
            Please provide a brief 2-3 sentence summary of this space biology research:

            {text}

            Focus on the main findings and their implications for space missions.
            """
            max_tokens = 150

        elif summary_type == 'detailed':
            prompt = f"""
            Please provide a detailed summary of this space biology research including:
            1. Research objectives and methodology
            2. Key findings
            3. Implications for space missions
            4. Limitations and future research directions

            Research text:
            {text}
            """
            max_tokens = 400

        elif summary_type == 'executive':
            prompt = f"""
            Please provide an executive summary suitable for mission planners and decision makers:
            - What was studied and why it matters for space missions
            - Key findings in plain language
            - Recommended actions or considerations
            - Risk assessment implications

            Research text:
            {text}
            """
            max_tokens = 300

        else:
            raise ValueError(f"Unknown summary type: {summary_type}")

        return await self.generate_text(prompt, max_tokens=max_tokens)

    async def extract_key_insights(self, text: str) -> Dict[str, List[str]]:
        """
        Extract key insights and categorize them.

        Args:
            text: Research text to analyze

        Returns:
            Dictionary with categorized insights
        """
        prompt = f"""
        Analyze this space biology research and extract key insights in the following categories:

        1. Health Risks: Any health risks or concerns identified
        2. Countermeasures: Proposed solutions or mitigation strategies
        3. Mission Implications: How findings affect mission planning
        4. Research Gaps: Areas needing further investigation
        5. Practical Applications: Direct applications for space missions

        Please format your response as JSON with these categories as keys and lists of insights as values.

        Research text:
        {text}
        """

        try:
            response = await self.generate_text(prompt, max_tokens=500)

            # Try to parse as JSON
            try:
                insights = json.loads(response)
                return insights
            except json.JSONDecodeError:
                # Fallback: parse text format
                return self._parse_text_insights(response)

        except Exception as e:
            logger.error(f"Failed to extract insights: {e}")
            return {
                "Health Risks": [],
                "Countermeasures": [],
                "Mission Implications": [],
                "Research Gaps": [],
                "Practical Applications": []
            }

    def _parse_text_insights(self, text: str) -> Dict[str, List[str]]:
        """Parse insights from text format when JSON parsing fails."""
        insights = {
            "Health Risks": [],
            "Countermeasures": [],
            "Mission Implications": [],
            "Research Gaps": [],
            "Practical Applications": []
        }

        current_category = None
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line is a category header
            for category in insights.keys():
                if category.lower() in line.lower():
                    current_category = category
                    break

            # If line starts with bullet point or number and we have a current category
            if current_category and (line.startswith(('-', '"', '*')) or
                                   line[0].isdigit() and line[1] in '.):'):
                # Clean up the line
                cleaned_line = line.lstrip('-"*0123456789.): ').strip()
                if cleaned_line:
                    insights[current_category].append(cleaned_line)

        return insights

    async def generate_research_questions(self, topic: str, context: str = "") -> List[str]:
        """
        Generate research questions for a given topic.

        Args:
            topic: Research topic
            context: Additional context about the research area

        Returns:
            List of generated research questions
        """
        prompt = f"""
        Generate 5-7 important research questions for space biology research on the topic: {topic}

        {f"Context: {context}" if context else ""}

        The questions should:
        - Address practical concerns for space missions
        - Consider both short-term (ISS) and long-term (Mars) missions
        - Focus on actionable research that could inform mission planning
        - Consider ethical and safety implications

        Please provide questions in a numbered list format.
        """

        try:
            response = await self.generate_text(prompt, max_tokens=400)

            # Parse questions from response
            questions = []
            lines = response.split('\n')

            for line in lines:
                line = line.strip()
                # Look for numbered questions
                if line and (line[0].isdigit() or line.startswith(('"', '-', '*'))):
                    # Clean up the question
                    question = line.lstrip('0123456789."-* ').strip()
                    if question and question.endswith('?'):
                        questions.append(question)

            return questions

        except Exception as e:
            logger.error(f"Failed to generate research questions: {e}")
            return []

    async def detect_contradictions(self, finding1: str, finding2: str) -> Dict[str, Any]:
        """
        Analyze two findings for potential contradictions.

        Args:
            finding1: First research finding
            finding2: Second research finding

        Returns:
            Dictionary with contradiction analysis
        """
        prompt = f"""
        Analyze these two space biology research findings for potential contradictions:

        Finding 1: {finding1}

        Finding 2: {finding2}

        Please assess:
        1. Are these findings contradictory? (Yes/No)
        2. If contradictory, what is the nature of the contradiction?
        3. What could explain the difference (methodology, subjects, conditions)?
        4. How confident are you in this assessment? (0-100%)
        5. What additional research would help resolve any contradiction?

        Provide your response in a structured format.
        """

        try:
            response = await self.generate_text(prompt, max_tokens=300)

            # Extract key information
            analysis = {
                'contradictory': 'yes' in response.lower(),
                'confidence': self._extract_confidence(response),
                'explanation': response,
                'recommendations': []
            }

            # Look for recommendations in the response
            if 'additional research' in response.lower():
                # Extract the recommendations section
                parts = response.split('additional research')
                if len(parts) > 1:
                    recommendations_text = parts[1]
                    analysis['recommendations'] = [
                        line.strip() for line in recommendations_text.split('\n')
                        if line.strip() and not line.strip().startswith(('?', '.'))
                    ]

            return analysis

        except Exception as e:
            logger.error(f"Failed to detect contradictions: {e}")
            return {
                'contradictory': False,
                'confidence': 0,
                'explanation': f"Error in analysis: {str(e)}",
                'recommendations': []
            }

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence percentage from text."""
        import re
        # Look for percentage patterns
        patterns = [
            r'(\d+)%',
            r'(\d+)\s*percent',
            r'confidence.*?(\d+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1)) / 100.0

        # Default moderate confidence if not specified
        return 0.7

    async def _log_llm_request(self, prompt: str, model: str,
                             max_tokens: int, temperature: float):
        """Log LLM request for monitoring and debugging."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'request',
            'model': model,
            'prompt_length': len(prompt),
            'max_tokens': max_tokens,
            'temperature': temperature,
            'prompt_preview': prompt[:200] + "..." if len(prompt) > 200 else prompt
        }

        log_file = self.data_paths['logs'] / 'llm_requests.jsonl'
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    async def _log_llm_response(self, model: str, total_tokens: int,
                              prompt_tokens: int, completion_tokens: int):
        """Log LLM response for usage tracking."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'response',
            'model': model,
            'total_tokens': total_tokens,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens
        }

        log_file = self.data_paths['logs'] / 'llm_usage.jsonl'
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    async def get_usage_statistics(self) -> Dict[str, Any]:
        """Get LLM usage statistics."""
        try:
            usage_file = self.data_paths['logs'] / 'llm_usage.jsonl'

            if not usage_file.exists():
                return {'error': 'No usage data found'}

            total_tokens = 0
            model_usage = {}
            request_count = 0

            with open(usage_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get('type') == 'response':
                            total_tokens += entry.get('total_tokens', 0)
                            model = entry.get('model', 'unknown')
                            model_usage[model] = model_usage.get(model, 0) + 1
                            request_count += 1
                    except json.JSONDecodeError:
                        continue

            return {
                'total_requests': request_count,
                'total_tokens': total_tokens,
                'model_usage': model_usage,
                'average_tokens_per_request': total_tokens / request_count if request_count > 0 else 0
            }

        except Exception as e:
            logger.error(f"Failed to get usage statistics: {e}")
            return {'error': str(e)}