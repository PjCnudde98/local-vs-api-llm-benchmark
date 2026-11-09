# Task (tier 2): extract fields
Extract these fields from the complaint. Use null when the complaint does not state the value. Copy values as written; do not guess.
- "order_number": the order number exactly as written
- "product": the product the complaint is about, in a few words
- "requested_remedy": refund | replacement | redelivery | voucher | apology | information | other (what the customer asks for)
- "deadline": a date the customer sets for a response or solution, as YYYY-MM-DD

Return: {"order_number": ..., "product": ..., "requested_remedy": ..., "deadline": ...}
