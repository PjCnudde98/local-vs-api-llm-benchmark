# Task (tier 5): decide the remedy with lookup tools
Decide which remedy the policy allows for this complaint and explain it to the agent. Check the facts with the tools before you decide. Call one tool per turn; you get its result in the next message. You have at most 5 tool calls.

Tools:
- get_order: {"order_number": "<order number as written in the complaint>"} returns status, dates, amount, items, photo and payment details, and the customer id
- get_contact_history: {"customer_id": "<customer id from get_order>"} returns earlier contacts with this customer

To call a tool, return: {"tool": "<tool name>", "args": {...}}
When you have decided, return: {"final": {"decision": "refund | replacement | redelivery | voucher | decline | escalate", "justification": "<which facts and which policy rule lead to this decision>", "agent_note": "<what the agent should do next>"}}
