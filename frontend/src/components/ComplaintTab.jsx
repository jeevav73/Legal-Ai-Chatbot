import { useState } from 'react';

const initialForm = {
  name: '',
  address: '',
  contact: '',
  addressee: '',
  incident_date: '',
  location: '',
  accused: '',
  witnesses: '',
  evidence: '',
  incident_summary: '',
  relief_requested: '',
};

function ComplaintTab() {
  const [form, setForm] = useState(initialForm);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleChange = (field) => (event) => {
    setForm((prev) => ({ ...prev, [field]: event.target.value }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError('');
    setSuccess('');

    if (!form.name.trim() || !form.incident_summary.trim()) {
      setError('Please provide your name and a summary of the incident.');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/complaint', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });

      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload?.error || 'Unable to generate complaint draft');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'complaint_draft.docx';
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);

      setSuccess('Complaint draft generated successfully. Please have it reviewed by a licensed advocate before filing.');
    } catch (err) {
      setError(err.message || 'Something went wrong while generating the complaint draft.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="complaint-tab">
      {error ? <div className="error-banner">{error}</div> : null}
      {success ? <div className="success-banner">{success}</div> : null}

      <form className="complaint-form" onSubmit={handleSubmit}>
        <div className="grid-two-column">
          <label>
            Full name
            <input value={form.name} onChange={handleChange('name')} disabled={loading} />
          </label>
          <label>
            Contact number
            <input value={form.contact} onChange={handleChange('contact')} disabled={loading} />
          </label>
          <label className="full-width">
            Address
            <textarea value={form.address} onChange={handleChange('address')} disabled={loading} rows={3} />
          </label>
          <label>
            Addressed to
            <input value={form.addressee} onChange={handleChange('addressee')} disabled={loading} />
          </label>
          <label>
            Date of incident
            <input value={form.incident_date} onChange={handleChange('incident_date')} disabled={loading} placeholder="DD-MM-YYYY" />
          </label>
          <label>
            Location / Police station
            <input value={form.location} onChange={handleChange('location')} disabled={loading} />
          </label>
          <label>
            Accused / opposite party
            <input value={form.accused} onChange={handleChange('accused')} disabled={loading} />
          </label>
          <label>
            Witnesses
            <input value={form.witnesses} onChange={handleChange('witnesses')} disabled={loading} />
          </label>
        </div>

        <label>
          Incident summary
          <textarea value={form.incident_summary} onChange={handleChange('incident_summary')} disabled={loading} rows={5} />
        </label>

        <label>
          Evidence available
          <textarea value={form.evidence} onChange={handleChange('evidence')} disabled={loading} rows={4} />
        </label>

        <label>
          Relief requested
          <textarea value={form.relief_requested} onChange={handleChange('relief_requested')} disabled={loading} rows={4} />
        </label>

        <button type="submit" disabled={loading}>
          {loading ? 'Generating...' : 'Generate Complaint Draft'}
        </button>
      </form>
    </div>
  );
}

export default ComplaintTab;
