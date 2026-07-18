import { useState } from 'react';
import ChatTab from './components/ChatTab';
import ComplaintTab from './components/ComplaintTab';
import WarningBanner from './components/WarningBanner';

function App() {
  const [activeTab, setActiveTab] = useState('chat');

  return (
    <div className="app-shell">
      <header className="page-header">
        <div>
          <p className="eyebrow">Nyaya Sahayak</p>
          <h1>Local legal guidance and complaint drafting</h1>
          <p className="subtitle">
            Ask a legal question or generate a complaint draft using local models and
            trusted legal context — no external API required.
          </p>
        </div>
      </header>

      <WarningBanner />
      <div className="tab-bar">
        <button
          className={activeTab === 'chat' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('chat')}
        >
          Ask a Legal Question
        </button>
        <button
          className={activeTab === 'complaint' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('complaint')}
        >
          Generate a Complaint Draft
        </button>
      </div>
      <main className="tab-content">
        {activeTab === 'chat' ? <ChatTab /> : <ComplaintTab />}
      </main>
    </div>
  );
}

export default App;
