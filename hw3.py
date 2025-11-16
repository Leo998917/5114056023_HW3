"""
HW3 Spam Demo - Simple rule-based spam classifier
"""


def classify_message(text):
    """
    Classify a message as spam or ham using rule-based detection.
    
    Args:
        text (str): The message text to classify
        
    Returns:
        str: "spam" if message contains spam keywords, "ham" otherwise
    """
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Spam detection rules
    spam_keywords = ["buy now", "free", "win"]
    
    # Check if any spam keyword is present
    for keyword in spam_keywords:
        if keyword in text_lower:
            return "spam"
    
    return "ham"


if __name__ == "__main__":
    # Demo test cases
    test_messages = [
        "Click here to buy now!",
        "You won! Click to win a prize",
        "Get free money instantly",
        "Hello, how are you today?",
        "Let's meet for coffee tomorrow",
        "LIMITED TIME: Free offer inside!"
    ]
    
    print("Spam Classification Demo")
    print("-" * 50)
    
    for message in test_messages:
        classification = classify_message(message)
        print(f"Message: {message}")
        print(f"Classification: {classification}")
        print()
