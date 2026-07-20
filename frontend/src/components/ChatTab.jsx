import { useState, useRef, useEffect } from 'react';

const initialMessages = [
  {
    role: 'assistant',
    message:
      'Welcome to Nyaya Sahayak. Ask a legal question and I will answer using local legal context and models.',
  },
];

function ChatTab() {
  const [messages, setMessages] = useState(initialMessages);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const submitQuestion = async () => {
    const trimmed = question.trim();
    if (!trimmed) return;

    const API_BASE_URL = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? '' : 'https://jeevavj-nyaya-sahayak-backend.hf.space');

    setError('');
    setMessages((prev) => [...prev, { role: 'user', message: trimmed }]);
    setQuestion('');
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmed }),
      });

      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload?.error || 'Unable to fetch answer');
      }

      const data = await response.json();
      setMessages((prev) => [...prev, { role: 'assistant', message: data.answer, sources: data.sources }]);
    } catch (err) {
      setError(err.message || 'Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    submitQuestion();
  };

  return (
    <div className="chat-tab">
      <div className="chat-window">
        {messages.map((item, index) => (
          <div key={index} className={`chat-message ${item.role}`}>
            <div className="chat-bubble">
              <p>{item.message}</p>
              {item.sources?.length ? <p className="chat-sources">Sources: {item.sources.join(', ')}</p> : null}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {error ? <div className="error-banner">{error}</div> : null}

      <form className="chat-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. Police filed a false case on me under a bailable offence — what can I do?"
          disabled={loading}
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Thinking...' : 'Send'}
        </button>
      </form>
    </div>
  );
}

export default ChatTab;
