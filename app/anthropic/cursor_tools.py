"""Standard tool definitions for Cursor Code / Claude Code.

When Cursor Code doesn't send tools in the API request, we auto-inject these
standard tools so Claude knows they're available and returns proper tool_use
blocks instead of XML tags.
"""

# Standard Cursor Code / Claude Code tools
# Based on the Claude Code tool schema
CURSOR_CODE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Reads a file from the local filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to read"
                    },
                    "offset": {
                        "type": "number",
                        "description": "The line number to start reading from"
                    },
                    "limit": {
                        "type": "number",
                        "description": "The number of lines to read"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Writes a file to the local filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                },
                "required": ["file_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Edit",
            "description": "Performs exact string replacements in files",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to modify"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The text to replace"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The text to replace it with"
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences of old_string"
                    }
                },
                "required": ["file_path", "old_string", "new_string"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Executes a bash command in a persistent shell session",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute"
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of what this command does"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Optional timeout in milliseconds"
                    },
                    "run_in_background": {
                        "type": "boolean",
                        "description": "Set to true to run in background"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Glob",
            "description": "Fast file pattern matching tool using glob patterns",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The glob pattern to match files against"
                    },
                    "path": {
                        "type": "string",
                        "description": "The directory to search in"
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Grep",
            "description": "Search tool built on ripgrep for finding patterns in files",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The regular expression pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in"
                    },
                    "output_mode": {
                        "type": "string",
                        "description": "Output mode: content, files_with_matches, or count"
                    },
                    "glob": {
                        "type": "string",
                        "description": "Glob pattern to filter files"
                    },
                    "-i": {
                        "type": "boolean",
                        "description": "Case insensitive search"
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "TodoWrite",
            "description": "Create and manage a structured task list",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "The updated todo list",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "Task description"
                                },
                                "status": {
                                    "type": "string",
                                    "description": "Task status: pending, in_progress, or completed"
                                },
                                "activeForm": {
                                    "type": "string",
                                    "description": "Present continuous form of the task"
                                }
                            },
                            "required": ["content", "status", "activeForm"]
                        }
                    }
                },
                "required": ["todos"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Task",
            "description": "Launch a specialized agent for complex tasks",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Short description of the task"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The task for the agent to perform"
                    },
                    "subagent_type": {
                        "type": "string",
                        "description": "The type of specialized agent to use"
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional model to use for this agent"
                    }
                },
                "required": ["description", "prompt", "subagent_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "AskUserQuestion",
            "description": "Ask the user questions during execution",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "description": "Questions to ask the user (1-4 questions)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": "The complete question to ask"
                                },
                                "header": {
                                    "type": "string",
                                    "description": "Short label (max 12 chars)"
                                },
                                "multiSelect": {
                                    "type": "boolean",
                                    "description": "Allow multiple selections"
                                },
                                "options": {
                                    "type": "array",
                                    "description": "Available choices (2-4 options)",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "label": {
                                                "type": "string",
                                                "description": "Display text for this option"
                                            },
                                            "description": {
                                                "type": "string",
                                                "description": "Explanation of this option"
                                            }
                                        },
                                        "required": ["label", "description"]
                                    }
                                }
                            },
                            "required": ["question", "header", "options", "multiSelect"]
                        }
                    }
                },
                "required": ["questions"]
            }
        }
    }
]
