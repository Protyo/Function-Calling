from openai import OpenAI


class Conductor:
    def __init__(self, api_key, system_prompt, model="gpt-"):
        """
        Expand this into configurations to support Mistral, OpenAI, Gemini, Nous-Hermes
        
        
        """
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = system_prompt
        self.variables = {}
        self.state = {}

    def register(self, name, update_function, starting_value, value_range=None):
        """
        Registers a variable with its update function and value range.

        :param name: The name of the variable.
        :param update_function: A function that takes the current value and returns an updated value.
        :param value_range: The permissible range or set of values for the variable.
        """
        self.variables[name] = {'update_function': update_function, 'value': starting_value, 'value_range': value_range}

    def log(self, state_dict):
        """
        Updates the internal state of the system based on the provided dictionary.

        :param state_dict: A dictionary containing key-value pairs of system state indicators.
        """
        self.state.update(state_dict)

    def update(self):
        """
        Uses the OpenAI client to suggest new values for registered variables.
        Returns suggestions as JSON and updates the values.
        """
        response = self.client.Completion.create(
            engine="davinci",
            prompt=self._generate_prompt(),
            max_tokens=100
        )
        suggestions = self._parse_response(response)
        self._apply_suggestions(suggestions)

    def _generate_prompt(self):
        """
        Generates a prompt for the LLM based on the current state and system description.
        """
        prompt = self.system_prompt
        for name, value in self.state.items():
            prompt += f"\n{name}: {value}"
        prompt += "\nSuggested adjustments:"
        return prompt

    def _parse_response(self, response):
        """
        Parses the LLM response and extracts variable adjustments.
        """
        # Implementation depends on the response format
        # ...

    def _apply_suggestions(self, suggestions):
        """
        Applies the suggested adjustments to the variables.
        """
        for name, suggested_value in suggestions.items():
            if name in self.variables:
                value_range = self.variables[name]['value_range']
                if suggested_value in value_range:
                    self.state[name] = suggested_value
                else:
                    print(f"Suggested value for {name} is out of range.")
