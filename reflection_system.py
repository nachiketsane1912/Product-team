from typing import Callable, Any, Dict, Optional

class ReflectionWorkflow:
    """
    A configurable reflection workflow that allows agents to self-reflect 
    on their outputs before finalizing responses.
    """
    
    def __init__(self, client, model: str, num_reflections: int = 0):
        """
        Initialize the reflection workflow.
        
        Args:
            client: The AI client (e.g., genai.Client)
            model: The model to use for reflection
            num_reflections: Number of reflection cycles (default: 0)
        """
        self.client = client
        self.model = model
        self.num_reflections = num_reflections
        self.reflection_history = []
    
    def execute(self, 
                initial_query: str,
                generate_response_fn: Callable[[str], str],
                system_instruction: str = "",
                reflection_prompt_template: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the reflection workflow.
        
        Args:
            initial_query: The user's original query
            generate_response_fn: Function that generates a response given a query
            system_instruction: System instruction for the agent
            reflection_prompt_template: Custom reflection prompt (optional)
        
        Returns:
            Dict containing final response, reflection history, and metadata
        """
        self.reflection_history = []
        
        # Use default reflection prompt if none provided
        if not reflection_prompt_template:
            reflection_prompt_template = self._default_reflection_prompt()
        
        # Initial response
        current_response = generate_response_fn(initial_query)
        
        self.reflection_history.append({
            'iteration': 0,
            'type': 'initial_response',
            'query': initial_query,
            'response': current_response
        })
        
        # Reflection cycles
        for i in range(self.num_reflections):
            # Generate reflection
            reflection_query = reflection_prompt_template.format(
                original_query=initial_query,
                current_response=current_response,
                iteration=i + 1
            )
            
            reflection = self._generate_reflection(reflection_query, system_instruction)
            
            self.reflection_history.append({
                'iteration': i + 1,
                'type': 'reflection',
                'content': reflection
            })
            
            # Generate improved response based on reflection
            improvement_query = f"""Original query: {initial_query}

Previous response: {current_response}

Reflection on the response: {reflection}

Based on this reflection, provide an improved response that addresses the identified issues and incorporates the suggestions:"""
            
            current_response = generate_response_fn(improvement_query)
            
            self.reflection_history.append({
                'iteration': i + 1,
                'type': 'improved_response',
                'response': current_response
            })
        
        return {
            'final_response': current_response,
            'reflection_history': self.reflection_history,
            'num_reflections': self.num_reflections,
            'total_iterations': len(self.reflection_history)
        }
    
    def _generate_reflection(self, reflection_query: str, system_instruction: str) -> str:
        """Generate a reflection using the AI model."""
        reflection_system = f"""{system_instruction}

You are now in reflection mode. Critically analyze the response and provide constructive feedback on:
1. Accuracy and completeness
2. Clarity and structure
3. Potential improvements or missing information
4. Alignment with the user's intent
5. Any errors or inconsistencies

Be specific and actionable in your feedback."""
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=reflection_query,
            config={'system_instruction': reflection_system}
        )
        
        return response.text
    
    def _default_reflection_prompt(self) -> str:
        """Default reflection prompt template."""
        return """Original Query: {original_query}

Current Response:
{current_response}

Reflection #{iteration}:
Please critically evaluate this response. Consider:
- Is the response accurate and complete?
- Does it fully address the user's question?
- Is it clear and well-structured?
- Are there any errors or areas for improvement?
- What additional information or context might be helpful?

Provide specific, actionable feedback for improvement:"""

    def get_reflection_summary(self) -> str:
        """Get a formatted summary of the reflection process."""
        if not self.reflection_history:
            return "No reflections performed yet."
        
        summary = f"Reflection Workflow Summary (Total iterations: {len(self.reflection_history)})\n"
        summary += "=" * 60 + "\n\n"
        
        for entry in self.reflection_history:
            iteration = entry['iteration']
            entry_type = entry['type']
            
            if entry_type == 'initial_response':
                summary += f"[Initial Response]\n{entry['response'][:200]}...\n\n"
            elif entry_type == 'reflection':
                summary += f"[Reflection #{iteration}]\n{entry['content'][:200]}...\n\n"
            elif entry_type == 'improved_response':
                summary += f"[Improved Response #{iteration}]\n{entry['response'][:200]}...\n\n"
        
        return summary
    
    def clear_history(self):
        """Clear reflection history."""
        self.reflection_hi