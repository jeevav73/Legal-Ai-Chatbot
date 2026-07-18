"""
app.py
Streamlit UI for RightsGuard — a local, no-API Indian legal-aid chatbot.

Run with:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="Legal Ai Chatbot", page_icon="⚖️", layout="wide")

st.title("Indian Law Legal-Ai Chatbot")
st.caption(
    "Runs entirely on local, open-source models (InLegalBERT + Aalap-Mistral-7B) — "
    "no external API required."
)
st.warning(
    "This tool gives general legal information and draft documents, not legal advice "
    "from a licensed advocate. Please have any draft reviewed by a lawyer before filing.",
    icon="⚠️",
)

tab_chat, tab_complaint = st.tabs(["💬 Ask a Legal Question", "📝 Generate a Complaint Draft"])

# ---------------------------------------------------------------------
# TAB 1 — Chat
# ---------------------------------------------------------------------
with tab_chat:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    user_question = st.chat_input("e.g. Police filed a false case on me under a bailable offence — what can I do?")
    if user_question:
        st.session_state.chat_history.append(("user", user_question))
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Looking up relevant law and drafting an answer..."):
                try:
                    from rag_engine import answer_question

                    result = answer_question(user_question)
                except Exception as e:
                    st.error("Could not generate an answer — see details below.")
                    st.exception(e)
                    result = {"answer": "", "sources": []}

            if result["answer"]:
                st.markdown(result["answer"])
            if result["sources"]:
                st.caption(f"Sources: {', '.join(result['sources'])}")

        st.session_state.chat_history.append(("assistant", result["answer"]))

# ---------------------------------------------------------------------
# TAB 2 — Complaint generator
# ---------------------------------------------------------------------
with tab_complaint:
    st.subheader("Fill in the details below to generate a draft complaint (.docx)")

    with st.form("complaint_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Your full name")
            address = st.text_area("Your address", height=70)
            contact = st.text_input("Your contact number")
            addressee = st.text_input(
                "Complaint addressed to",
                placeholder="e.g. The Superintendent of Police, XYZ District",
            )
        with col2:
            incident_date = st.text_input("Date of incident (DD-MM-YYYY)")
            location = st.text_input("Location / Police station concerned")
            accused = st.text_input("Accused / opposite party (if known)")
            witnesses = st.text_input("Witnesses (names, if any)")

        incident_summary = st.text_area(
            "Describe what happened (facts only, in your own words)", height=150
        )
        evidence = st.text_area("Evidence available (documents, messages, photos, etc.)", height=70)
        relief_requested = st.text_area(
            "What action are you requesting?",
            placeholder="e.g. Kindly ensure a fair investigation into the FIR registered against me.",
            height=70,
        )

        submitted = st.form_submit_button("Generate Complaint Draft")

    if submitted:
        if not name or not incident_summary:
            st.error("Please fill in at least your name and the incident description.")
        else:
            details = {
                "name": name,
                "address": address,
                "contact": contact,
                "addressee": addressee,
                "incident_date": incident_date,
                "location": location,
                "incident_summary": incident_summary,
                "accused": accused,
                "witnesses": witnesses,
                "evidence": evidence,
                "relief_requested": relief_requested,
            }
            with st.spinner("Drafting your complaint..."):
                try:
                    from complaint_generator import generate_complaint

                    file_path = generate_complaint(details)
                except Exception as e:
                    st.error("Could not generate complaint draft — see details below.")
                    st.exception(e)
                    file_path = None

            if file_path:
                st.success("Draft generated below. Please review it carefully — and have a "
                           "licensed advocate check it — before filing.")
                with open(file_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download complaint (.docx)",
                        data=f,
                        file_name=file_path.split("/")[-1],
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
