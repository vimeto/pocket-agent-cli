"""BFCL (Berkeley Function Calling Leaderboard) dataset implementation.

Measures tool-calling proficiency: can the model produce the correct function
call (name + arguments) given a user request and a set of function definitions?

Dataset source: https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Dataset, Problem
from .registry import DatasetRegistry

# ---------------------------------------------------------------------------
# Category helpers
# ---------------------------------------------------------------------------

# BFCL v3 file names on HuggingFace (parquet) and the categories we use.
# We focus on "simple" and "multiple" as the most relevant for on-device
# models.  "parallel" and "relevance" are also supported if present.
CATEGORIES = ["simple", "multiple", "parallel", "relevance"]

# HuggingFace dataset id
HF_DATASET_ID = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"


def _build_bfcl_test_data() -> List[Dict[str, Any]]:
    """Build a curated set of function-calling test examples.

    Each example is a dict with:
        - id: unique identifier
        - category: simple | multiple | parallel | relevance
        - prompt: user request text
        - functions: list of OpenAI-compatible function tool defs
        - expected: list of expected function call dicts
                    (empty list for relevance = no call expected)
    """
    examples: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # SIMPLE: single function call with correct arguments
    # -----------------------------------------------------------------------
    simple_cases = [
        {
            "prompt": "What's the weather in San Francisco?",
            "functions": [
                {"type": "function", "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"],
                                     "description": "Temperature unit"},
                        },
                        "required": ["location"],
                    },
                }},
            ],
            "expected": [{"name": "get_weather",
                          "arguments": {"location": "San Francisco"}}],
        },
        {
            "prompt": "Calculate 15% tip on a $85.50 bill.",
            "functions": [
                {"type": "function", "function": {
                    "name": "calculate_tip",
                    "description": "Calculate tip amount",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bill_amount": {"type": "number", "description": "Total bill"},
                            "tip_percentage": {"type": "number", "description": "Tip percentage"},
                        },
                        "required": ["bill_amount", "tip_percentage"],
                    },
                }},
            ],
            "expected": [{"name": "calculate_tip",
                          "arguments": {"bill_amount": 85.50, "tip_percentage": 15}}],
        },
        {
            "prompt": "Send an email to john@example.com with subject 'Meeting Tomorrow' and body 'Hi John, can we meet at 3pm?'",
            "functions": [
                {"type": "function", "function": {
                    "name": "send_email",
                    "description": "Send an email",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string", "description": "Recipient email"},
                            "subject": {"type": "string", "description": "Email subject"},
                            "body": {"type": "string", "description": "Email body"},
                        },
                        "required": ["to", "subject", "body"],
                    },
                }},
            ],
            "expected": [{"name": "send_email",
                          "arguments": {"to": "john@example.com",
                                        "subject": "Meeting Tomorrow",
                                        "body": "Hi John, can we meet at 3pm?"}}],
        },
        {
            "prompt": "Create a reminder for 'Buy groceries' at 5pm today.",
            "functions": [
                {"type": "function", "function": {
                    "name": "create_reminder",
                    "description": "Create a reminder",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Reminder title"},
                            "time": {"type": "string", "description": "Time for reminder"},
                        },
                        "required": ["title", "time"],
                    },
                }},
            ],
            "expected": [{"name": "create_reminder",
                          "arguments": {"title": "Buy groceries", "time": "5pm today"}}],
        },
        {
            "prompt": "Convert 100 USD to EUR.",
            "functions": [
                {"type": "function", "function": {
                    "name": "convert_currency",
                    "description": "Convert between currencies",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "amount": {"type": "number", "description": "Amount to convert"},
                            "from_currency": {"type": "string", "description": "Source currency"},
                            "to_currency": {"type": "string", "description": "Target currency"},
                        },
                        "required": ["amount", "from_currency", "to_currency"],
                    },
                }},
            ],
            "expected": [{"name": "convert_currency",
                          "arguments": {"amount": 100, "from_currency": "USD",
                                        "to_currency": "EUR"}}],
        },
        {
            "prompt": "Search for 'machine learning tutorials' on the web.",
            "functions": [
                {"type": "function", "function": {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "num_results": {"type": "integer", "description": "Number of results",
                                            "default": 10},
                        },
                        "required": ["query"],
                    },
                }},
            ],
            "expected": [{"name": "web_search",
                          "arguments": {"query": "machine learning tutorials"}}],
        },
        {
            "prompt": "Set an alarm for 7:30 AM tomorrow.",
            "functions": [
                {"type": "function", "function": {
                    "name": "set_alarm",
                    "description": "Set an alarm",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "time": {"type": "string", "description": "Alarm time"},
                            "label": {"type": "string", "description": "Optional label"},
                        },
                        "required": ["time"],
                    },
                }},
            ],
            "expected": [{"name": "set_alarm",
                          "arguments": {"time": "7:30 AM tomorrow"}}],
        },
        {
            "prompt": "Translate 'Hello, how are you?' to Spanish.",
            "functions": [
                {"type": "function", "function": {
                    "name": "translate_text",
                    "description": "Translate text between languages",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to translate"},
                            "target_language": {"type": "string",
                                                "description": "Target language"},
                            "source_language": {"type": "string",
                                                "description": "Source language"},
                        },
                        "required": ["text", "target_language"],
                    },
                }},
            ],
            "expected": [{"name": "translate_text",
                          "arguments": {"text": "Hello, how are you?",
                                        "target_language": "Spanish"}}],
        },
        {
            "prompt": "Read the file '/home/user/report.txt'.",
            "functions": [
                {"type": "function", "function": {
                    "name": "read_file",
                    "description": "Read contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path"},
                        },
                        "required": ["path"],
                    },
                }},
            ],
            "expected": [{"name": "read_file",
                          "arguments": {"path": "/home/user/report.txt"}}],
        },
        {
            "prompt": "Get the current time in Tokyo.",
            "functions": [
                {"type": "function", "function": {
                    "name": "get_time",
                    "description": "Get current time in a timezone",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timezone": {"type": "string", "description": "Timezone name"},
                        },
                        "required": ["timezone"],
                    },
                }},
            ],
            "expected": [{"name": "get_time",
                          "arguments": {"timezone": "Tokyo"}}],
        },
        {
            "prompt": "Add 'Buy milk' to my todo list.",
            "functions": [
                {"type": "function", "function": {
                    "name": "add_todo",
                    "description": "Add an item to the todo list",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "item": {"type": "string", "description": "Todo item text"},
                            "priority": {"type": "string", "enum": ["low", "medium", "high"],
                                         "description": "Priority level"},
                        },
                        "required": ["item"],
                    },
                }},
            ],
            "expected": [{"name": "add_todo",
                          "arguments": {"item": "Buy milk"}}],
        },
        {
            "prompt": "Calculate the square root of 144.",
            "functions": [
                {"type": "function", "function": {
                    "name": "calculator",
                    "description": "Perform mathematical calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string",
                                           "description": "Mathematical expression"},
                        },
                        "required": ["expression"],
                    },
                }},
            ],
            "expected": [{"name": "calculator",
                          "arguments": {"expression": "sqrt(144)"}}],
        },
        {
            "prompt": "Play the song 'Bohemian Rhapsody' by Queen.",
            "functions": [
                {"type": "function", "function": {
                    "name": "play_music",
                    "description": "Play a song",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "song": {"type": "string", "description": "Song name"},
                            "artist": {"type": "string", "description": "Artist name"},
                        },
                        "required": ["song"],
                    },
                }},
            ],
            "expected": [{"name": "play_music",
                          "arguments": {"song": "Bohemian Rhapsody",
                                        "artist": "Queen"}}],
        },
        {
            "prompt": "Book a table for 4 at 'La Trattoria' at 7pm tonight.",
            "functions": [
                {"type": "function", "function": {
                    "name": "book_restaurant",
                    "description": "Book a restaurant reservation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "restaurant": {"type": "string",
                                           "description": "Restaurant name"},
                            "party_size": {"type": "integer",
                                           "description": "Number of guests"},
                            "time": {"type": "string", "description": "Reservation time"},
                        },
                        "required": ["restaurant", "party_size", "time"],
                    },
                }},
            ],
            "expected": [{"name": "book_restaurant",
                          "arguments": {"restaurant": "La Trattoria",
                                        "party_size": 4,
                                        "time": "7pm tonight"}}],
        },
        {
            "prompt": "Create a new folder called 'projects' in /home/user/.",
            "functions": [
                {"type": "function", "function": {
                    "name": "create_directory",
                    "description": "Create a new directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Directory path"},
                        },
                        "required": ["path"],
                    },
                }},
            ],
            "expected": [{"name": "create_directory",
                          "arguments": {"path": "/home/user/projects"}}],
        },
        {
            "prompt": "Find all Python files in /home/user/code.",
            "functions": [
                {"type": "function", "function": {
                    "name": "search_files",
                    "description": "Search for files matching a pattern",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {"type": "string",
                                          "description": "Directory to search"},
                            "pattern": {"type": "string",
                                        "description": "File pattern (glob)"},
                        },
                        "required": ["directory", "pattern"],
                    },
                }},
            ],
            "expected": [{"name": "search_files",
                          "arguments": {"directory": "/home/user/code",
                                        "pattern": "*.py"}}],
        },
        {
            "prompt": "Compress the file /tmp/data.csv using gzip.",
            "functions": [
                {"type": "function", "function": {
                    "name": "compress_file",
                    "description": "Compress a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File to compress"},
                            "format": {"type": "string",
                                       "enum": ["gzip", "zip", "tar.gz"],
                                       "description": "Compression format"},
                        },
                        "required": ["path", "format"],
                    },
                }},
            ],
            "expected": [{"name": "compress_file",
                          "arguments": {"path": "/tmp/data.csv",
                                        "format": "gzip"}}],
        },
        {
            "prompt": "Get stock price for AAPL.",
            "functions": [
                {"type": "function", "function": {
                    "name": "get_stock_price",
                    "description": "Get current stock price",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string",
                                       "description": "Stock ticker symbol"},
                        },
                        "required": ["symbol"],
                    },
                }},
            ],
            "expected": [{"name": "get_stock_price",
                          "arguments": {"symbol": "AAPL"}}],
        },
        {
            "prompt": "Resize the image /photos/vacation.jpg to 800x600 pixels.",
            "functions": [
                {"type": "function", "function": {
                    "name": "resize_image",
                    "description": "Resize an image",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Image file path"},
                            "width": {"type": "integer", "description": "New width"},
                            "height": {"type": "integer", "description": "New height"},
                        },
                        "required": ["path", "width", "height"],
                    },
                }},
            ],
            "expected": [{"name": "resize_image",
                          "arguments": {"path": "/photos/vacation.jpg",
                                        "width": 800, "height": 600}}],
        },
        {
            "prompt": "Schedule a meeting titled 'Sprint Planning' for next Monday at 10am.",
            "functions": [
                {"type": "function", "function": {
                    "name": "create_event",
                    "description": "Create a calendar event",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Event title"},
                            "date": {"type": "string", "description": "Event date"},
                            "time": {"type": "string", "description": "Event time"},
                        },
                        "required": ["title", "date", "time"],
                    },
                }},
            ],
            "expected": [{"name": "create_event",
                          "arguments": {"title": "Sprint Planning",
                                        "date": "next Monday",
                                        "time": "10am"}}],
        },
    ]

    for i, case in enumerate(simple_cases):
        examples.append({
            "id": f"simple_{i}",
            "category": "simple",
            **case,
        })

    # -----------------------------------------------------------------------
    # MULTIPLE: choose the right function from several options
    # -----------------------------------------------------------------------
    multiple_cases = [
        {
            "prompt": "What is the weather in Paris right now?",
            "functions": [
                {"type": "function", "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "get_forecast",
                    "description": "Get weather forecast for upcoming days",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                            "days": {"type": "integer", "description": "Number of days"},
                        },
                        "required": ["location", "days"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "get_historical_weather",
                    "description": "Get historical weather data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "date": {"type": "string"},
                        },
                        "required": ["location", "date"],
                    },
                }},
            ],
            "expected": [{"name": "get_weather",
                          "arguments": {"location": "Paris"}}],
        },
        {
            "prompt": "Delete the file /tmp/old_log.txt.",
            "functions": [
                {"type": "function", "function": {
                    "name": "read_file",
                    "description": "Read file contents",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path"},
                        },
                        "required": ["path"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "delete_file",
                    "description": "Delete a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path to delete"},
                        },
                        "required": ["path"],
                    },
                }},
            ],
            "expected": [{"name": "delete_file",
                          "arguments": {"path": "/tmp/old_log.txt"}}],
        },
        {
            "prompt": "Add 25 and 17.",
            "functions": [
                {"type": "function", "function": {
                    "name": "add",
                    "description": "Add two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number", "description": "First number"},
                            "b": {"type": "number", "description": "Second number"},
                        },
                        "required": ["a", "b"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "subtract",
                    "description": "Subtract two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "multiply",
                    "description": "Multiply two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "divide",
                    "description": "Divide two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                }},
            ],
            "expected": [{"name": "add",
                          "arguments": {"a": 25, "b": 17}}],
        },
        {
            "prompt": "Find restaurants near me.",
            "functions": [
                {"type": "function", "function": {
                    "name": "search_restaurants",
                    "description": "Search for restaurants nearby",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cuisine": {"type": "string", "description": "Cuisine type"},
                            "radius_km": {"type": "number", "description": "Search radius in km"},
                        },
                        "required": [],
                    },
                }},
                {"type": "function", "function": {
                    "name": "book_restaurant",
                    "description": "Book a restaurant reservation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "restaurant": {"type": "string"},
                            "party_size": {"type": "integer"},
                            "time": {"type": "string"},
                        },
                        "required": ["restaurant", "party_size", "time"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "get_restaurant_reviews",
                    "description": "Get reviews for a specific restaurant",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "restaurant": {"type": "string"},
                        },
                        "required": ["restaurant"],
                    },
                }},
            ],
            "expected": [{"name": "search_restaurants",
                          "arguments": {}}],
        },
        {
            "prompt": "Multiply 12 by 8.",
            "functions": [
                {"type": "function", "function": {
                    "name": "add",
                    "description": "Add two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"}, "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "multiply",
                    "description": "Multiply two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"}, "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                }},
            ],
            "expected": [{"name": "multiply",
                          "arguments": {"a": 12, "b": 8}}],
        },
        {
            "prompt": "Send a text message to Alice saying 'Running late, be there in 10.'",
            "functions": [
                {"type": "function", "function": {
                    "name": "send_sms",
                    "description": "Send an SMS text message",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "recipient": {"type": "string", "description": "Recipient name or number"},
                            "message": {"type": "string", "description": "Message text"},
                        },
                        "required": ["recipient", "message"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "send_email",
                    "description": "Send an email",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string"},
                            "subject": {"type": "string"},
                            "body": {"type": "string"},
                        },
                        "required": ["to", "subject", "body"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "make_call",
                    "description": "Make a phone call",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "contact": {"type": "string"},
                        },
                        "required": ["contact"],
                    },
                }},
            ],
            "expected": [{"name": "send_sms",
                          "arguments": {"recipient": "Alice",
                                        "message": "Running late, be there in 10."}}],
        },
        {
            "prompt": "Turn on the living room lights.",
            "functions": [
                {"type": "function", "function": {
                    "name": "control_light",
                    "description": "Control a smart light",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "room": {"type": "string", "description": "Room name"},
                            "action": {"type": "string", "enum": ["on", "off", "dim"],
                                       "description": "Action to perform"},
                        },
                        "required": ["room", "action"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "set_thermostat",
                    "description": "Set thermostat temperature",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "temperature": {"type": "number"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["temperature"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "lock_door",
                    "description": "Lock or unlock a door",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "door": {"type": "string"},
                            "action": {"type": "string", "enum": ["lock", "unlock"]},
                        },
                        "required": ["door", "action"],
                    },
                }},
            ],
            "expected": [{"name": "control_light",
                          "arguments": {"room": "living room", "action": "on"}}],
        },
        {
            "prompt": "Get the 5-day weather forecast for London.",
            "functions": [
                {"type": "function", "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "get_forecast",
                    "description": "Get weather forecast for upcoming days",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "days": {"type": "integer", "description": "Number of forecast days"},
                        },
                        "required": ["location", "days"],
                    },
                }},
            ],
            "expected": [{"name": "get_forecast",
                          "arguments": {"location": "London", "days": 5}}],
        },
        {
            "prompt": "Show me the last 20 lines of /var/log/syslog.",
            "functions": [
                {"type": "function", "function": {
                    "name": "read_file",
                    "description": "Read entire file contents",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                        },
                        "required": ["path"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "tail_file",
                    "description": "Read last N lines of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "lines": {"type": "integer", "description": "Number of lines"},
                        },
                        "required": ["path", "lines"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "search_file",
                    "description": "Search for a pattern in a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "pattern": {"type": "string"},
                        },
                        "required": ["path", "pattern"],
                    },
                }},
            ],
            "expected": [{"name": "tail_file",
                          "arguments": {"path": "/var/log/syslog", "lines": 20}}],
        },
        {
            "prompt": "What's the distance between New York and Los Angeles?",
            "functions": [
                {"type": "function", "function": {
                    "name": "get_distance",
                    "description": "Calculate distance between two locations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "origin": {"type": "string", "description": "Starting location"},
                            "destination": {"type": "string", "description": "End location"},
                            "unit": {"type": "string", "enum": ["km", "miles"]},
                        },
                        "required": ["origin", "destination"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "get_directions",
                    "description": "Get driving directions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "origin": {"type": "string"},
                            "destination": {"type": "string"},
                        },
                        "required": ["origin", "destination"],
                    },
                }},
            ],
            "expected": [{"name": "get_distance",
                          "arguments": {"origin": "New York",
                                        "destination": "Los Angeles"}}],
        },
        {
            "prompt": "List all running processes.",
            "functions": [
                {"type": "function", "function": {
                    "name": "list_processes",
                    "description": "List running processes",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filter": {"type": "string",
                                       "description": "Optional name filter"},
                        },
                        "required": [],
                    },
                }},
                {"type": "function", "function": {
                    "name": "kill_process",
                    "description": "Kill a process by name or PID",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "identifier": {"type": "string"},
                        },
                        "required": ["identifier"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "get_system_info",
                    "description": "Get system information (CPU, RAM, etc.)",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                }},
            ],
            "expected": [{"name": "list_processes",
                          "arguments": {}}],
        },
        {
            "prompt": "Download the file from https://example.com/data.csv and save it to /tmp/data.csv.",
            "functions": [
                {"type": "function", "function": {
                    "name": "download_file",
                    "description": "Download a file from a URL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL to download from"},
                            "save_path": {"type": "string",
                                          "description": "Local path to save to"},
                        },
                        "required": ["url", "save_path"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "upload_file",
                    "description": "Upload a file to a URL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "url": {"type": "string"},
                        },
                        "required": ["path", "url"],
                    },
                }},
            ],
            "expected": [{"name": "download_file",
                          "arguments": {"url": "https://example.com/data.csv",
                                        "save_path": "/tmp/data.csv"}}],
        },
        {
            "prompt": "Rename the file /tmp/old_name.txt to /tmp/new_name.txt.",
            "functions": [
                {"type": "function", "function": {
                    "name": "rename_file",
                    "description": "Rename or move a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "old_path": {"type": "string", "description": "Current file path"},
                            "new_path": {"type": "string", "description": "New file path"},
                        },
                        "required": ["old_path", "new_path"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "copy_file",
                    "description": "Copy a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "destination": {"type": "string"},
                        },
                        "required": ["source", "destination"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "delete_file",
                    "description": "Delete a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                        },
                        "required": ["path"],
                    },
                }},
            ],
            "expected": [{"name": "rename_file",
                          "arguments": {"old_path": "/tmp/old_name.txt",
                                        "new_path": "/tmp/new_name.txt"}}],
        },
        {
            "prompt": "Set the thermostat to 72 degrees Fahrenheit.",
            "functions": [
                {"type": "function", "function": {
                    "name": "control_light",
                    "description": "Control a smart light",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "room": {"type": "string"},
                            "action": {"type": "string", "enum": ["on", "off", "dim"]},
                        },
                        "required": ["room", "action"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "set_thermostat",
                    "description": "Set thermostat temperature",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "temperature": {"type": "number", "description": "Target temperature"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["temperature"],
                    },
                }},
            ],
            "expected": [{"name": "set_thermostat",
                          "arguments": {"temperature": 72, "unit": "fahrenheit"}}],
        },
        {
            "prompt": "Take a screenshot.",
            "functions": [
                {"type": "function", "function": {
                    "name": "take_screenshot",
                    "description": "Capture a screenshot",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "save_path": {"type": "string",
                                          "description": "Path to save screenshot"},
                        },
                        "required": [],
                    },
                }},
                {"type": "function", "function": {
                    "name": "record_screen",
                    "description": "Start screen recording",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "duration": {"type": "integer", "description": "Duration in seconds"},
                        },
                        "required": ["duration"],
                    },
                }},
            ],
            "expected": [{"name": "take_screenshot",
                          "arguments": {}}],
        },
    ]

    for i, case in enumerate(multiple_cases):
        examples.append({
            "id": f"multiple_{i}",
            "category": "multiple",
            **case,
        })

    # -----------------------------------------------------------------------
    # PARALLEL: call multiple functions in one response
    # -----------------------------------------------------------------------
    parallel_cases = [
        {
            "prompt": "What's the weather in Tokyo AND New York?",
            "functions": [
                {"type": "function", "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                        },
                        "required": ["location"],
                    },
                }},
            ],
            "expected": [
                {"name": "get_weather", "arguments": {"location": "Tokyo"}},
                {"name": "get_weather", "arguments": {"location": "New York"}},
            ],
        },
        {
            "prompt": "Add 3 and 5, and also multiply 4 and 7.",
            "functions": [
                {"type": "function", "function": {
                    "name": "add",
                    "description": "Add two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"}, "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "multiply",
                    "description": "Multiply two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"}, "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                }},
            ],
            "expected": [
                {"name": "add", "arguments": {"a": 3, "b": 5}},
                {"name": "multiply", "arguments": {"a": 4, "b": 7}},
            ],
        },
        {
            "prompt": "Turn on the bedroom lights and set thermostat to 68F.",
            "functions": [
                {"type": "function", "function": {
                    "name": "control_light",
                    "description": "Control a smart light",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "room": {"type": "string"},
                            "action": {"type": "string", "enum": ["on", "off", "dim"]},
                        },
                        "required": ["room", "action"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "set_thermostat",
                    "description": "Set thermostat temperature",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "temperature": {"type": "number"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["temperature"],
                    },
                }},
            ],
            "expected": [
                {"name": "control_light",
                 "arguments": {"room": "bedroom", "action": "on"}},
                {"name": "set_thermostat",
                 "arguments": {"temperature": 68, "unit": "fahrenheit"}},
            ],
        },
        {
            "prompt": "Send an email to bob@example.com about the meeting and add 'Follow up with Bob' to my todo list.",
            "functions": [
                {"type": "function", "function": {
                    "name": "send_email",
                    "description": "Send an email",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string"},
                            "subject": {"type": "string"},
                            "body": {"type": "string"},
                        },
                        "required": ["to", "subject", "body"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "add_todo",
                    "description": "Add a todo item",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "item": {"type": "string"},
                        },
                        "required": ["item"],
                    },
                }},
            ],
            "expected": [
                {"name": "send_email",
                 "arguments": {"to": "bob@example.com",
                               "subject": "Meeting",
                               "body": "Meeting"}},
                {"name": "add_todo",
                 "arguments": {"item": "Follow up with Bob"}},
            ],
        },
        {
            "prompt": "Get stock prices for AAPL, GOOGL, and MSFT.",
            "functions": [
                {"type": "function", "function": {
                    "name": "get_stock_price",
                    "description": "Get current stock price",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "Ticker symbol"},
                        },
                        "required": ["symbol"],
                    },
                }},
            ],
            "expected": [
                {"name": "get_stock_price", "arguments": {"symbol": "AAPL"}},
                {"name": "get_stock_price", "arguments": {"symbol": "GOOGL"}},
                {"name": "get_stock_price", "arguments": {"symbol": "MSFT"}},
            ],
        },
    ]

    for i, case in enumerate(parallel_cases):
        examples.append({
            "id": f"parallel_{i}",
            "category": "parallel",
            **case,
        })

    # -----------------------------------------------------------------------
    # RELEVANCE: no function should be called
    # -----------------------------------------------------------------------
    relevance_cases = [
        {
            "prompt": "What is the meaning of life?",
            "functions": [
                {"type": "function", "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                }},
            ],
            "expected": [],
        },
        {
            "prompt": "Tell me a joke.",
            "functions": [
                {"type": "function", "function": {
                    "name": "calculator",
                    "description": "Perform mathematical calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                        },
                        "required": ["expression"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "send_email",
                    "description": "Send an email",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string"},
                            "subject": {"type": "string"},
                            "body": {"type": "string"},
                        },
                        "required": ["to", "subject", "body"],
                    },
                }},
            ],
            "expected": [],
        },
        {
            "prompt": "Explain how photosynthesis works.",
            "functions": [
                {"type": "function", "function": {
                    "name": "search_files",
                    "description": "Search for files",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {"type": "string"},
                            "pattern": {"type": "string"},
                        },
                        "required": ["directory", "pattern"],
                    },
                }},
            ],
            "expected": [],
        },
        {
            "prompt": "What is 2+2? Just tell me without using any tools.",
            "functions": [
                {"type": "function", "function": {
                    "name": "calculator",
                    "description": "Perform mathematical calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                        },
                        "required": ["expression"],
                    },
                }},
            ],
            "expected": [],
        },
        {
            "prompt": "Write me a poem about the ocean.",
            "functions": [
                {"type": "function", "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                }},
                {"type": "function", "function": {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                }},
            ],
            "expected": [],
        },
    ]

    for i, case in enumerate(relevance_cases):
        examples.append({
            "id": f"relevance_{i}",
            "category": "relevance",
            **case,
        })

    return examples


# ---------------------------------------------------------------------------
# Try to load from HuggingFace, fall back to built-in data
# ---------------------------------------------------------------------------

def _try_load_hf_dataset(
    categories: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Attempt to download BFCL v3 from HuggingFace.

    Returns None if the ``datasets`` library is not installed or the
    download fails for any reason.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        return None

    categories = categories or ["simple", "multiple"]
    examples: List[Dict[str, Any]] = []

    try:
        ds = load_dataset(HF_DATASET_ID, split="train")
    except Exception:
        return None

    per_cat_limit = (limit // len(categories)) if limit else None

    for cat in categories:
        count = 0
        for row in ds:
            # BFCL rows may have varying schema depending on version.
            # Common keys: "id", "question", "function", "ground_truth"
            row_id = str(row.get("id", ""))
            if cat == "simple" and "simple" not in row_id:
                continue
            if cat == "multiple" and "multiple" not in row_id:
                continue
            if cat == "parallel" and "parallel" not in row_id:
                continue
            if cat == "relevance" and "relevance" not in row_id:
                continue

            question = row.get("question")
            functions = row.get("function")
            ground_truth = row.get("ground_truth")
            if not question or not functions:
                continue

            # Parse JSON strings if necessary
            if isinstance(question, str):
                try:
                    question = json.loads(question)
                except (json.JSONDecodeError, TypeError):
                    pass
            if isinstance(functions, str):
                try:
                    functions = json.loads(functions)
                except (json.JSONDecodeError, TypeError):
                    pass
            if isinstance(ground_truth, str):
                try:
                    ground_truth = json.loads(ground_truth)
                except (json.JSONDecodeError, TypeError):
                    pass

            # Normalize question to a prompt string
            if isinstance(question, list) and question:
                # Typically [{"role": "user", "content": "..."}]
                prompt = question[-1].get("content", str(question))
            elif isinstance(question, dict):
                prompt = question.get("content", str(question))
            else:
                prompt = str(question)

            # Normalize functions to OpenAI tool format
            tool_defs = []
            if isinstance(functions, list):
                for fn in functions:
                    if isinstance(fn, dict):
                        if "type" in fn and fn["type"] == "function":
                            tool_defs.append(fn)
                        elif "name" in fn:
                            tool_defs.append({"type": "function", "function": fn})
            elif isinstance(functions, dict):
                if "type" in functions and functions["type"] == "function":
                    tool_defs = [functions]
                elif "name" in functions:
                    tool_defs = [{"type": "function", "function": functions}]

            if not tool_defs:
                continue

            # Normalize ground truth to list of {name, arguments}
            expected = []
            if isinstance(ground_truth, list):
                for gt in ground_truth:
                    if isinstance(gt, dict) and "name" in gt:
                        expected.append({
                            "name": gt["name"],
                            "arguments": gt.get("arguments", gt.get("parameters", {})),
                        })
                    elif isinstance(gt, str):
                        # Some BFCL entries encode calls as Python strings
                        pass  # skip unparseable
            elif isinstance(ground_truth, dict) and "name" in ground_truth:
                expected = [{
                    "name": ground_truth["name"],
                    "arguments": ground_truth.get("arguments",
                                                  ground_truth.get("parameters", {})),
                }]

            examples.append({
                "id": row_id or f"{cat}_{count}",
                "category": cat,
                "prompt": prompt,
                "functions": tool_defs,
                "expected": expected,
            })
            count += 1
            if per_cat_limit and count >= per_cat_limit:
                break

    return examples if examples else None


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

@DatasetRegistry.register("bfcl")
class BFCLDataset(Dataset):
    """BFCL (Berkeley Function Calling Leaderboard) dataset.

    Evaluates whether a model can produce the correct function call given a
    user request and a set of available function definitions.

    Categories:
        - simple: single function, correct arguments
        - multiple: pick the right function from several options
        - parallel: call multiple functions in one response
        - relevance: detect when NO function should be called
    """

    name = "bfcl"
    description = "Tool-calling proficiency benchmark (function name + argument matching)"
    url = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard"

    def __init__(self, data_dir: Path):
        super().__init__(data_dir)
        self._cached_data: Optional[List[Dict[str, Any]]] = None

    @property
    def problem_count(self) -> int:
        """Number of built-in problems (may be more if HF data is available)."""
        return len(_build_bfcl_test_data())

    @property
    def available_splits(self) -> List[str]:
        return ["test", "sample"]

    # ---- persistence helpers ------------------------------------------------

    def _data_file(self) -> Path:
        return self.data_dir / "bfcl_data.json"

    def is_downloaded(self, data_dir: Optional[Path] = None) -> bool:
        d = Path(data_dir) if data_dir else self.data_dir
        return (d / "bfcl_data.json").exists()

    def download(self, data_dir: Optional[Path] = None) -> bool:
        """Build / download BFCL data and save to disk.

        Tries HuggingFace first; if that fails, uses the built-in curated
        examples.
        """
        d = Path(data_dir) if data_dir else self.data_dir
        d.mkdir(parents=True, exist_ok=True)

        print("Attempting to load BFCL from HuggingFace...")
        hf_data = _try_load_hf_dataset(limit=400)

        if hf_data:
            data = hf_data
            print(f"Loaded {len(data)} examples from HuggingFace")
        else:
            print("HuggingFace load failed or datasets library not installed.")
            print("Using built-in curated examples.")
            data = _build_bfcl_test_data()
            print(f"Built {len(data)} curated examples")

        out_path = d / "bfcl_data.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved to {out_path}")

        # Also write a sample file (first 5)
        sample_path = d / "bfcl_sample.json"
        with open(sample_path, "w") as f:
            json.dump(data[:5], f, indent=2)

        self._cached_data = data
        return True

    # ---- loading ------------------------------------------------------------

    def _load_raw(self) -> List[Dict[str, Any]]:
        if self._cached_data is not None:
            return self._cached_data

        data_file = self._data_file()
        if data_file.exists():
            with open(data_file) as f:
                self._cached_data = json.load(f)
            return self._cached_data

        # Fall back to built-in data (no download needed)
        self._cached_data = _build_bfcl_test_data()
        return self._cached_data

    def load(self, split: str = "test", limit: Optional[int] = None) -> List[Problem]:
        """Load BFCL problems as Problem objects.

        The Problem wrapper stores BFCL-specific fields in ``metadata``:
            - category, functions, expected, original_id
        """
        if split not in self.available_splits:
            raise ValueError(
                f"Invalid split '{split}'. Available: {self.available_splits}"
            )

        raw = self._load_raw()

        if split == "sample":
            raw = raw[:5]

        if limit is not None:
            raw = raw[:limit]

        return [self._to_problem(item) for item in raw]

    def load_raw(self, split: str = "test", limit: Optional[int] = None,
                 categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Load raw BFCL examples (dicts) with optional category filter.

        This is the preferred interface for the benchmark runner since it
        preserves function definitions and expected calls without the
        Problem wrapper overhead.
        """
        raw = self._load_raw()

        if split == "sample":
            raw = raw[:5]

        if categories:
            raw = [r for r in raw if r["category"] in categories]

        if limit is not None:
            raw = raw[:limit]

        return raw

    # ---- conversion ---------------------------------------------------------

    @staticmethod
    def _to_problem(item: Dict[str, Any]) -> Problem:
        """Wrap a BFCL example as a Problem for registry compatibility."""
        expected_names = [e["name"] for e in item.get("expected", [])]
        return Problem(
            task_id=item["id"],
            prompt=item["prompt"],
            canonical_solution=json.dumps(item["expected"]),
            test_cases=[json.dumps(e) for e in item.get("expected", [])],
            entry_point=expected_names[0] if expected_names else "",
            metadata={
                "source": "bfcl",
                "category": item["category"],
                "functions": item["functions"],
                "expected": item["expected"],
                "original_id": item["id"],
            },
        )
