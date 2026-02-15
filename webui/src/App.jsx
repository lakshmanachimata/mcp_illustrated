import { useState, useEffect, useCallback, useMemo } from 'react'
import * as api from './api'
import './App.css'

const DEBOUNCE_MS = 400

export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('llm-theme') || 'dark')
  const [localModels, setLocalModels] = useState([])
  const [libraryModels, setLibraryModels] = useState([])
  const [searchQuery, setSearchQuery] = useState('')
  const [searchInput, setSearchInput] = useState('')
  const [activeModel, setActiveModel] = useState(null)
  const [downloadingModel, setDownloadingModel] = useState(null)
  const [downloadProgress, setDownloadProgress] = useState(null)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [contextModalOpen, setContextModalOpen] = useState(false)
  const [contextValue, setContextValue] = useState('')
  const [contextSaving, setContextSaving] = useState(false)
  const [error, setError] = useState(null)
  const [loadingList, setLoadingList] = useState(false)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('llm-theme', theme)
  }, [theme])

  const loadLocalModels = useCallback(async () => {
    setLoadingList(true)
    setError(null)
    try {
      const data = await api.listModels()
      const list = Array.isArray(data) ? data : (data?.models ?? [])
      setLocalModels(Array.isArray(list) ? list : [])
    } catch (e) {
      setError(e.message || 'Failed to load models')
    } finally {
      setLoadingList(false)
    }
  }, [])

  const loadActive = useCallback(async () => {
    try {
      const data = await api.getActiveModel()
      setActiveModel(data.active_model || null)
    } catch (_) {}
  }, [])

  useEffect(() => {
    loadLocalModels()
    loadActive()
  }, [loadLocalModels, loadActive])

  useEffect(() => {
    if (!searchQuery.trim()) {
      setLibraryModels([])
      return
    }
    const t = setTimeout(async () => {
      try {
        const data = await api.searchLibrary(searchQuery)
        setLibraryModels(data.models || [])
      } catch (_) {
        setLibraryModels([])
      }
    }, DEBOUNCE_MS)
    return () => clearTimeout(t)
  }, [searchQuery])

  const filteredLocal = useMemo(() => {
    if (!searchQuery.trim()) return localModels
    const q = searchQuery.toLowerCase()
    return localModels.filter((m) => (m.name || m.model || '').toLowerCase().includes(q))
  }, [localModels, searchQuery])

  const localNames = useMemo(() => new Set(localModels.map((m) => m.name || m.model)), [localModels])
  const libraryOnly = useMemo(
    () => libraryModels.filter((m) => !localNames.has(m.name || m.model)),
    [libraryModels, localNames]
  )

  const setActive = async (model) => {
    setError(null)
    try {
      await api.setActiveModel(model)
      await api.loadModel(model)
      await loadActive()
    } catch (e) {
      setError(e.message)
    }
  }

  const deleteModel = async (model) => {
    if (!window.confirm(`Delete "${model}"?`)) return
    setError(null)
    try {
      await api.deleteModel(model)
      await loadActive()
      await loadLocalModels()
    } catch (e) {
      setError(e.message)
    }
  }

  const downloadModel = async (model) => {
    if (downloadingModel) return
    setDownloadingModel(model)
    setDownloadProgress({ status: 'Starting…', completed: 0, total: 0 })
    setError(null)
    try {
      await api.pullModel(model, (p) => setDownloadProgress(p))
      await loadLocalModels()
    } catch (e) {
      setError(e.message)
    } finally {
      setDownloadingModel(null)
      setDownloadProgress(null)
    }
  }

  const sendMessage = async () => {
    const text = input.trim()
    if (!text || sending) return
    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: text }])
    setSending(true)
    setError(null)
    try {
      const res = await api.sendPrompt(text, false)
      const reply = res.response || ''
      setMessages((prev) => [...prev, { role: 'assistant', content: reply }])
    } catch (e) {
      setError(e.message)
      setMessages((prev) => [...prev, { role: 'assistant', content: `Error: ${e.message}` }])
    } finally {
      setSending(false)
    }
  }

  const openContextModal = async () => {
    setContextModalOpen(true)
    try {
      const data = await api.getContext()
      setContextValue(data.context_prompt || '')
    } catch (_) {
      setContextValue('')
    }
  }

  const saveContext = async () => {
    setContextSaving(true)
    try {
      await api.setContext(contextValue)
      setContextModalOpen(false)
    } catch (e) {
      setError(e.message)
    } finally {
      setContextSaving(false)
    }
  }

  const toggleTheme = () => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-header">
          <span className="logo">LLM</span>
          <button type="button" className="icon-btn" onClick={toggleTheme} title={theme === 'dark' ? 'Switch to light' : 'Switch to dark'}>
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>
        </div>
        <div className="search-wrap">
          <input
            type="text"
            className="search-input"
            placeholder="Search local & library…"
            value={searchInput}
            onChange={(e) => {
              setSearchInput(e.target.value)
              setSearchQuery(e.target.value)
            }}
          />
        </div>
        <div className="model-list-wrap">
          {loadingList && <div className="list-loading">Loading…</div>}
          {filteredLocal.map((m) => {
            const name = m.name || m.model
            const isActive = activeModel === name
            return (
              <div key={name} className={`model-row ${isActive ? 'active' : ''}`}>
                <span className="model-name" title={name}>{name}</span>
                <span className="model-badges">
                  <button type="button" className="btn-pill" onClick={() => setActive(name)} disabled={isActive}>
                    {isActive ? 'Active' : 'Activate'}
                  </button>
                  <button type="button" className="btn-pill danger" onClick={() => deleteModel(name)}>Delete</button>
                </span>
              </div>
            )
          })}
          {searchQuery.trim() && libraryOnly.slice(0, 20).map((m) => {
            const name = m.name || m.model
            const downloading = downloadingModel === name
            const progress = downloading && downloadProgress
            return (
              <div key={`lib-${name}`} className="model-row library">
                <span className="model-name" title={name}>{name}</span>
                <span className="model-badges">
                  {downloading ? (
                    <span className="download-progress">
                      {progress?.status || '…'} {progress?.total ? `${Math.round((100 * (progress.completed || 0)) / progress.total)}%` : ''}
                    </span>
                  ) : (
                    <button
                      type="button"
                      className="btn-pill primary"
                      onClick={() => downloadModel(name)}
                      disabled={!!downloadingModel}
                    >
                      Download
                    </button>
                  )}
                </span>
              </div>
            )
          })}
          {!loadingList && !filteredLocal.length && !(searchQuery.trim() && libraryOnly.length) && (
            <div className="list-empty">No models. Search above to download from library.</div>
          )}
        </div>
      </aside>

      <main className="main">
        <header className="main-header">
          <span className="active-label">{activeModel ? `Active: ${activeModel}` : 'No model selected'}</span>
          <button type="button" className="btn-context" onClick={openContextModal}>
            Set context prompt
          </button>
        </header>

        {error && (
          <div className="banner error" role="alert">
            {error}
          </div>
        )}

        <div className="chat">
          <div className="messages">
            {messages.length === 0 && (
              <div className="welcome">
                <p>Send a message below. Use “Set context prompt” to define system behavior.</p>
              </div>
            )}
            {messages.map((msg, i) => (
              <div key={i} className={`message ${msg.role}`}>
                <div className="message-content">{msg.content}</div>
              </div>
            ))}
            {sending && (
              <div className="message assistant">
                <div className="message-content typing">Thinking…</div>
              </div>
            )}
          </div>
          <div className="chat-input-wrap">
            <textarea
              className="chat-input"
              placeholder="Message…"
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  sendMessage()
                }
              }}
              disabled={sending}
            />
            <button type="button" className="btn-send" onClick={sendMessage} disabled={!input.trim() || sending}>
              Send
            </button>
          </div>
        </div>
      </main>

      {contextModalOpen && (
        <div className="modal-backdrop" onClick={() => setContextModalOpen(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Context prompt (system prompt)</h3>
              <button type="button" className="icon-btn" onClick={() => setContextModalOpen(false)}>×</button>
            </div>
            <textarea
              className="modal-textarea"
              value={contextValue}
              onChange={(e) => setContextValue(e.target.value)}
              placeholder="e.g. You are a helpful assistant."
              rows={6}
            />
            <div className="modal-actions">
              <button type="button" className="btn-secondary" onClick={() => setContextModalOpen(false)}>Cancel</button>
              <button type="button" className="btn-primary" onClick={saveContext} disabled={contextSaving}>
                {contextSaving ? 'Saving…' : 'Save'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
