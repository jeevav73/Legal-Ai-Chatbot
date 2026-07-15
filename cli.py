"""
cli.py
Terminal-only version of Nyaya Sahayak — useful for quick testing without
Streamlit, or on machines where you just want to verify the pipeline works.

Run with:
    python cli.py
"""

from rag_engine import answer_question

BANNER = """
==================================================
  Nyaya Sahayak - Indian Law Legal-Aid Chatbot
  (local models only, no external API)
==================================================
Type your legal question, or 'exit' to quit.
Reminder: general legal information only, not a
substitute for advice from a licensed advocate.
--------------------------------------------------
"""


def main():
    print(BANNER)
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye. Please consult an advocate before acting on any legal matter.")
            break
        if not question:
            continue

        print("\n(thinking...)")
        result = answer_question(question)
        print(f"\nNyaya Sahayak: {result['answer']}")
        if result["sources"]:
            print(f"\n[Sources: {', '.join(result['sources'])}]")


if __name__ == "__main__":
    main()
