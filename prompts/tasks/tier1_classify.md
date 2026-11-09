# Task (tier 1): classify the complaint
Label the complaint. Use exactly one value per label:
- "type": delivery | damage | billing | refund | staff | repeat (repeat = the customer says they already complained about this before)
- "language": nl | fr | en
- "urgency": low | normal | high (high = legal or press threat, health or safety risk, or a deadline within 2 days)
- "sentiment": positive | neutral | negative | very_negative (very_negative = angry, insulting or threatening)

Return: {"type": "...", "language": "...", "urgency": "...", "sentiment": "..."}
