import os
from src.domain_handlers import TextAnalysisHandler # Import the new handler

class AgentCore:
    def __init__(self, config):
        self.config = config
        self.knowledge_base = {} # Placeholder for some internal state/knowledge
        self.text_analyzer = TextAnalysisHandler() # Initialize the text analysis handler

    def process_command(self, command_text):
        """Processes a natural language command."""
        command_text = command_text.lower().strip()

        if "list files" in command_text:
            return self._list_files()
        elif "read file" in command_text:
            # Simple example, needs more robust parsing
            parts = command_text.split("read file ")
            if len(parts) > 1:
                filename = parts[1].strip()
                return self._read_file(filename)
            else:
                return "Please specify a filename to read."
        elif "analyze text" in command_text:
            parts = command_text.split("analyze text ")
            if len(parts) > 1:
                filename = parts[1].strip()
                analysis_result = self.text_analyzer.analyze_file(filename)
                if analysis_result["status"] == "success":
                    return (f"Text analysis of '{analysis_result['filename']}':\n"
                            f"Lines: {analysis_result['lines']}\n"
                            f"Words: {analysis_result['words']}")
                else:
                    return analysis_result["message"]
            else:
                return "Please specify a filename to analyze text."
        elif "execute" in command_text:
            return "Execution capability is handled by the higher-level agent environment."
        else:
            return f"Command '{command_text}' not recognized by AgentCore."

    def _list_files(self):
        """Lists files in the current directory."""
        files = [f for f in os.listdir('.') if os.path.isfile(f)]
        dirs = [d for d in os.listdir('.') if os.path.isdir(d)]
        return f"Files: {', '.join(files)}\nDirectories: {', '.join(dirs)}"

    def _read_file(self, filename):
        """Reads the content of a specified file."""
        try:
            with open(filename, 'r') as f:
                content = f.read()
            return f"Content of {filename}:\n{content}"
        except FileNotFoundError:
            return f"Error: File '{filename}' not found."
        except Exception as e:
            return f"Error reading file '{filename}': {e}"

    def update_knowledge(self, key, value):
        """Updates the agent's internal knowledge base."""
        self.knowledge_base[key] = value
        return f"Knowledge updated for '{key}'."

    def get_knowledge(self, key):
        """Retrieves information from the agent's internal knowledge base."""
        return self.knowledge_base.get(key, "Knowledge not found.")
