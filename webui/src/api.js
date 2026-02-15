const API_BASE = import.meta.env.VITE_LLM_API_URL || 'http://localhost:8000';

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || JSON.stringify(err));
  }
  return res.json();
}

async function post(path, body) {
  return request(path, { method: 'POST', body: JSON.stringify(body) });
}

export async function listModels() {
  return request('/api/models');
}

export async function loadModel(model) {
  return post('/api/models/load', { model });
}

export async function getActiveModel() {
  return request('/api/models/active');
}

export async function setActiveModel(model) {
  return post('/api/models/active', { model });
}

export async function deleteModel(model) {
  return request(`/api/models/${encodeURIComponent(model)}`, { method: 'DELETE' });
}

export async function searchLibrary(q = '') {
  const path = q ? `/api/library/search?q=${encodeURIComponent(q)}` : '/api/library/search';
  return request(path);
}

export async function pullModel(model, onProgress) {
  const res = await fetch(`${API_BASE}/api/library/pull`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || JSON.stringify(err));
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        onProgress(JSON.parse(line));
      } catch (_) {}
    }
  }
  if (buffer.trim()) {
    try {
      onProgress(JSON.parse(buffer));
    } catch (_) {}
  }
}

export async function getCapabilities() {
  return request('/api/models/active/capabilities');
}

export async function sendPrompt(prompt, stream = false) {
  if (stream) {
    const res = await fetch(`${API_BASE}/api/prompt`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, stream: true }),
    });
    if (!res.ok) throw new Error(await res.text());
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let full = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const data = JSON.parse(line);
          if (data.content) full += data.content;
        } catch (_) {}
      }
    }
    return { response: full };
  }
  return post('/api/prompt', { prompt, stream: false });
}

export async function getContext() {
  return request('/api/context');
}

export async function setContext(context) {
  return post('/api/context', { context });
}
