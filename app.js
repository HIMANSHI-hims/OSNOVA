/**
 * app.js - OSTutor Frontend Application Logic
 * Supports SSE streaming, multi-turn chat, side-by-side comparison,
 * IME-safe input, and dynamic ChromaDB/Ollama status polling.
 */

// Configure Marked.js options
if (window.marked) {
  window.marked.setOptions({
    breaks: true,
    gfm: true,
    highlight: function(code, lang) {
      if (window.hljs) {
        const language = hljs.getLanguage(lang) ? lang : 'plaintext';
        return hljs.highlight(code, { language }).value;
      }
      return code;
    }
  });
}

// State
const state = {
  activeMode: 'rag', // 'rag' | 'plain' | 'compare' | 'mcq' | 'viva'
  selectedModel: 'qwen3:4b',
  isGenerating: false,
  activeAbortController: null,
  activeSessionId: null,
  sessions: [], // [{ id, title, mode, messages: [] }]
};

// Mode definitions and banners
const MODE_CONFIG = {
  rag: {
    title: '🎓 Ask Tutor (RAG)',
    desc: 'Answering strictly from your OS notes with cited pages and sources.',
    placeholder: "Ask an OS concept, e.g. 'Explain Peterson's algorithm' or 'What is a TLB?'...",
    hint: 'Ask Tutor (RAG)'
  },
  plain: {
    title: '🤖 Plain Qwen',
    desc: 'General LLM answers without referencing your notes in ./data.',
    placeholder: "Ask anything to plain Qwen, e.g. 'Compare Linux vs Windows kernel'...",
    hint: 'Plain Qwen'
  },
  compare: {
    title: '⚖️ Side-by-Side Compare',
    desc: 'Running Plain Qwen vs OSTutor RAG simultaneously to compare grounding.',
    placeholder: "Ask a question to see Plain vs Notes side by side, e.g. 'Explain Round Robin'...",
    hint: 'Compare Mode'
  },
  mcq: {
    title: '📝 MCQ Quiz Generator',
    desc: 'Generating 5 practice multiple-choice questions strictly from your course notes.',
    placeholder: "Enter topic for MCQs, e.g. 'process scheduling' or 'memory paging'...",
    hint: 'MCQ Generator'
  },
  viva: {
    title: '🎯 Viva Prep',
    desc: 'Generating 5 oral exam viva questions ordered easy-to-hard with model answers.',
    placeholder: "Enter topic for Viva questions, e.g. 'deadlocks' or 'system calls'...",
    hint: 'Viva Prep'
  }
};

// DOM Elements
const elements = {
  body: document.body,
  themeToggleBtn: document.getElementById('themeToggleBtn'),
  sunIcon: document.getElementById('sunIcon'),
  moonIcon: document.getElementById('moonIcon'),
  sidebar: document.getElementById('sidebar'),
  sidebarOpenBtn: document.getElementById('sidebarOpenBtn'),
  sidebarToggleBtn: document.getElementById('sidebarToggleBtn'),
  newChatBtn: document.getElementById('newChatBtn'),
  historyList: document.getElementById('historyList'),
  clearHistoryBtn: document.getElementById('clearHistoryBtn'),
  statusDot: document.getElementById('statusDot'),
  statusTitle: document.getElementById('statusTitle'),
  statusSub: document.getElementById('statusSub'),
  modePills: document.querySelectorAll('.mode-pill'),
  bannerBadge: document.getElementById('bannerBadge'),
  bannerDesc: document.getElementById('bannerDesc'),
  modelSelect: document.getElementById('modelSelect'),
  notesManagerBtn: document.getElementById('notesManagerBtn'),
  topChunkBadge: document.getElementById('topChunkBadge'),
  chatViewport: document.getElementById('chatViewport'),
  welcomeScreen: document.getElementById('welcomeScreen'),
  messagesContainer: document.getElementById('messagesContainer'),
  compareContainer: document.getElementById('compareContainer'),
  comparePlainOutput: document.getElementById('comparePlainOutput'),
  compareRagOutput: document.getElementById('compareRagOutput'),
  chatForm: document.getElementById('chatForm'),
  messageInput: document.getElementById('messageInput'),
  sendBtn: document.getElementById('sendBtn'),
  stopBtn: document.getElementById('stopBtn'),
  hintModeText: document.getElementById('hintModeText'),
  notesModal: document.getElementById('notesModal'),
  closeModalBtn: document.getElementById('closeModalBtn'),
  modalChunkCount: document.getElementById('modalChunkCount'),
  modalFileCount: document.getElementById('modalFileCount'),
  modalFileList: document.getElementById('modalFileList'),
  triggerIngestBtn: document.getElementById('triggerIngestBtn'),
  ingestSpinner: document.getElementById('ingestSpinner'),
  ingestBtnText: document.getElementById('ingestBtnText'),
  ingestResultBox: document.getElementById('ingestResultBox'),
};

// Initialize Application
function init() {
  loadTheme();
  loadSessions();
  setupEventListeners();
  checkStatus();
  updateModeUI();
}

// -------------------------------------------------------------
// Event Listeners Setup
// -------------------------------------------------------------
function setupEventListeners() {
  // Theme Toggle
  elements.themeToggleBtn.addEventListener('click', toggleTheme);

  // Mobile Sidebar
  elements.sidebarOpenBtn?.addEventListener('click', () => elements.sidebar.classList.add('open'));
  elements.sidebarToggleBtn?.addEventListener('click', () => elements.sidebar.classList.remove('open'));

  // New Chat
  elements.newChatBtn.addEventListener('click', () => startNewSession());
  elements.clearHistoryBtn.addEventListener('click', clearAllSessions);

  // Mode Selection
  elements.modePills.forEach(pill => {
    pill.addEventListener('click', () => {
      const mode = pill.getAttribute('data-mode');
      setMode(mode);
    });
  });

  // Model Selection
  elements.modelSelect.addEventListener('change', (e) => {
    state.selectedModel = e.target.value;
  });

  // Auto-expanding textarea & IME-safe Enter Submit
  elements.messageInput.addEventListener('input', autoResizeTextarea);
  elements.messageInput.addEventListener('keydown', handleTextareaKeydown);

  // Form submit
  elements.chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    handleSubmit();
  });

  // Stop generation
  elements.stopBtn.addEventListener('click', stopGeneration);

  // Quick Topic Chips
  document.querySelectorAll('.topic-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const prompt = chip.getAttribute('data-prompt');
      elements.messageInput.value = prompt;
      autoResizeTextarea();
      elements.messageInput.focus();
    });
  });

  // Welcome Starter Cards
  document.querySelectorAll('.starter-card').forEach(card => {
    card.addEventListener('click', () => {
      const prompt = card.getAttribute('data-prompt');
      elements.messageInput.value = prompt;
      handleSubmit();
    });
  });

  // Notes Ingestion Modal
  elements.notesManagerBtn.addEventListener('click', () => {
    elements.notesModal.classList.remove('hidden');
    checkStatus();
  });
  elements.closeModalBtn.addEventListener('click', () => {
    elements.notesModal.classList.add('hidden');
  });
  elements.notesModal.addEventListener('click', (e) => {
    if (e.target === elements.notesModal) {
      elements.notesModal.classList.add('hidden');
    }
  });

  // Trigger Re-ingest
  elements.triggerIngestBtn.addEventListener('click', triggerIngestion);
}

// -------------------------------------------------------------
// Auto-resize Textarea & IME-safe Submit
// -------------------------------------------------------------
function autoResizeTextarea() {
  const input = elements.messageInput;
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 160) + 'px';
}

function handleTextareaKeydown(e) {
  // Guard against IME (Input Method Editor) premature submission
  if (e.isComposing || e.keyCode === 229) {
    return;
  }
  // Enter without shift -> submit
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSubmit();
  }
}

// -------------------------------------------------------------
// Mode Switching
// -------------------------------------------------------------
function setMode(mode) {
  if (state.isGenerating) return;
  state.activeMode = mode;
  updateModeUI();

  // If in compare mode, switch view
  if (mode === 'compare') {
    elements.messagesContainer.classList.add('hidden');
    elements.welcomeScreen.classList.add('hidden');
    elements.compareContainer.classList.remove('hidden');
  } else {
    elements.compareContainer.classList.add('hidden');
    const currentSession = getCurrentSession();
    if (currentSession && currentSession.messages.length > 0) {
      elements.welcomeScreen.classList.add('hidden');
      elements.messagesContainer.classList.remove('hidden');
    } else {
      elements.welcomeScreen.classList.remove('hidden');
      elements.messagesContainer.classList.add('hidden');
    }
  }
}

function updateModeUI() {
  elements.modePills.forEach(pill => {
    const isCurrent = pill.getAttribute('data-mode') === state.activeMode;
    pill.classList.toggle('active', isCurrent);
  });

  const config = MODE_CONFIG[state.activeMode] || MODE_CONFIG.rag;
  elements.bannerBadge.textContent = config.title;
  elements.bannerDesc.textContent = config.desc;
  elements.messageInput.placeholder = config.placeholder;
  elements.hintModeText.textContent = config.hint;
}

// -------------------------------------------------------------
// Server Status & Ingestion
// -------------------------------------------------------------
async function checkStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) throw new Error('Status response not ok');
    const data = await res.json();

    // Update status indicator
    elements.statusDot.className = 'status-indicator online';
    elements.statusTitle.textContent = `${data.model || 'Qwen'} Ready`;
    elements.statusSub = `${data.total_chunks} chunks in DB`;
    elements.statusSub.textContent = `${data.total_chunks} note chunks indexed`;

    // Badges
    elements.topChunkBadge.textContent = `${data.total_chunks} chunks`;
    elements.modalChunkCount.textContent = data.total_chunks;
    elements.modalFileCount.textContent = data.data_files ? data.data_files.length : 0;

    // Populate model dropdown
    if (data.available_models && data.available_models.length > 0) {
      elements.modelSelect.innerHTML = '';
      data.available_models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m;
        opt.textContent = m;
        if (m === data.model) opt.selected = true;
        elements.modelSelect.appendChild(opt);
      });
      state.selectedModel = data.model;
    }

    // Populate file list in modal
    elements.modalFileList.innerHTML = '';
    if (data.data_files && data.data_files.length > 0) {
      data.data_files.forEach(f => {
        const li = document.createElement('li');
        li.className = 'file-item';
        li.innerHTML = `<span>📄</span> <span>${f}</span>`;
        elements.modalFileList.appendChild(li);
      });
    } else {
      elements.modalFileList.innerHTML = `<li class="empty-list">No files in ./data/. Put your OS PDFs there!</li>`;
    }
  } catch (err) {
    elements.statusDot.className = 'status-indicator warning';
    elements.statusTitle.textContent = 'Server Offline';
    elements.statusSub.textContent = 'Run python app_server.py';
  }
}

async function triggerIngestion() {
  elements.ingestSpinner.classList.remove('hidden');
  elements.ingestBtnText.textContent = 'Indexing notes...';
  elements.triggerIngestBtn.disabled = true;
  elements.ingestResultBox.className = 'ingest-result hidden';

  try {
    const res = await fetch('/api/ingest', { method: 'POST' });
    const data = await res.json();
    elements.ingestResultBox.classList.remove('hidden');
    if (data.success) {
      elements.ingestResultBox.className = 'ingest-result success';
      elements.ingestResultBox.textContent = `✅ ${data.message}`;
    } else {
      elements.ingestResultBox.className = 'ingest-result error';
      elements.ingestResultBox.textContent = `⚠️ ${data.message}`;
    }
    await checkStatus();
  } catch (err) {
    elements.ingestResultBox.classList.remove('hidden');
    elements.ingestResultBox.className = 'ingest-result error';
    elements.ingestResultBox.textContent = `❌ Ingestion error: ${err.message}`;
  } finally {
    elements.ingestSpinner.classList.add('hidden');
    elements.ingestBtnText.textContent = '⚡ Re-Index Notes Now';
    elements.triggerIngestBtn.disabled = false;
  }
}

// -------------------------------------------------------------
// Chat Submission & Streaming
// -------------------------------------------------------------
async function handleSubmit() {
  const text = elements.messageInput.value.trim();
  if (!text || state.isGenerating) return;

  // Clear input
  elements.messageInput.value = '';
  autoResizeTextarea();

  // If on compare mode, run side-by-side stream
  if (state.activeMode === 'compare') {
    handleCompareSubmit(text);
    return;
  }

  // Ensure active session
  let session = getCurrentSession();
  if (!session) {
    session = createSession(text.slice(0, 30));
  } else if (session.messages.length === 0) {
    session.title = text.slice(0, 30);
    renderSessionsList();
  }

  // Add User Message
  addMessageToDOM('user', text);
  session.messages.push({ role: 'user', content: text, mode: state.activeMode });

  // Prepare Assistant message container in DOM
  const { bubbleEl, sourcesEl, messageId } = createAssistantMessageInDOM();
  setGeneratingState(true);
  state.activeAbortController = new AbortController();

  let fullResponse = '';
  let sources = [];

  try {
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: text,
        mode: state.activeMode,
        model: state.selectedModel
      }),
      signal: state.activeAbortController.signal
    });

    if (!response.ok) {
      throw new Error(`Server returned error status ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    streamLoop: while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep partial line in buffer

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith('data: ')) {
          try {
            const eventData = JSON.parse(trimmed.slice(6));
            
            if (eventData.type === 'sources') {
              sources = eventData.sources || [];
              renderSources(sourcesEl, sources);
            } else if (eventData.type === 'token') {
              fullResponse += eventData.token;
              renderMarkdown(bubbleEl, fullResponse, true);
              scrollToBottom();
            } else if (eventData.type === 'done') {
              await reader.cancel();
              break streamLoop;
            } else if (eventData.type === 'error') {
              fullResponse += `\n\n*Error: ${eventData.error}*`;
              renderMarkdown(bubbleEl, fullResponse, false);
            }
          } catch (e) {
            console.error('Error parsing SSE event', e, trimmed);
          }
        }
      }
    }

    // Final render without cursor
    renderMarkdown(bubbleEl, fullResponse, false);
    session.messages.push({
      role: 'assistant',
      content: fullResponse,
      sources: sources,
      mode: state.activeMode
    });
    saveSessions();
  } catch (err) {
    if (err.name === 'AbortError') {
      fullResponse += '\n\n*(Generation stopped by user)*';
    } else {
      fullResponse += `\n\n*(Error: ${err.message}. Make sure Ollama and app_server.py are running)*`;
    }
    renderMarkdown(bubbleEl, fullResponse, false);
  } finally {
    setGeneratingState(false);
    scrollToBottom();
  }
}

// -------------------------------------------------------------
// Side-by-Side Compare Streaming
// -------------------------------------------------------------
async function handleCompareSubmit(query) {
  elements.comparePlainOutput.innerHTML = `<div class="pane-placeholder">Query: "${escapeHtml(query)}"</div><div class="message-bubble"></div>`;
  elements.compareRagOutput.innerHTML = `<div class="pane-placeholder">Query: "${escapeHtml(query)}"</div><div class="sources-container"></div><div class="message-bubble"></div>`;

  const plainBubble = elements.comparePlainOutput.querySelector('.message-bubble');
  const ragBubble = elements.compareRagOutput.querySelector('.message-bubble');
  const ragSourcesEl = elements.compareRagOutput.querySelector('.sources-container');

  setGeneratingState(true);
  state.activeAbortController = new AbortController();

  let plainText = '';
  let ragText = '';

  // Stream Plain
  const streamPlain = async () => {
    try {
      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: query, mode: 'plain', model: state.selectedModel }),
        signal: state.activeAbortController.signal
      });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() || '';

        for (const line of lines) {
          if (line.trim().startsWith('data: ')) {
            const data = JSON.parse(line.trim().slice(6));
            if (data.type === 'token') {
              plainText += data.token;
              renderMarkdown(plainBubble, plainText, true);
            }
          }
        }
      }
      renderMarkdown(plainBubble, plainText, false);
    } catch (e) {
      if (e.name !== 'AbortError') {
        plainBubble.innerHTML = `<em>Error: ${e.message}</em>`;
      }
    }
  };

  // Stream RAG
  const streamRag = async () => {
    try {
      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: query, mode: 'rag', model: state.selectedModel }),
        signal: state.activeAbortController.signal
      });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() || '';

        for (const line of lines) {
          if (line.trim().startsWith('data: ')) {
            const data = JSON.parse(line.trim().slice(6));
            if (data.type === 'sources') {
              renderSources(ragSourcesEl, data.sources);
            } else if (data.type === 'token') {
              ragText += data.token;
              renderMarkdown(ragBubble, ragText, true);
            }
          }
        }
      }
      renderMarkdown(ragBubble, ragText, false);
    } catch (e) {
      if (e.name !== 'AbortError') {
        ragBubble.innerHTML = `<em>Error: ${e.message}</em>`;
      }
    }
  };

  try {
    await Promise.all([streamPlain(), streamRag()]);
  } finally {
    setGeneratingState(false);
  }
}

// -------------------------------------------------------------
// Markdown Rendering & Syntax Highlighting
// -------------------------------------------------------------
function renderMarkdown(targetEl, text, isStreaming) {
  let html = window.marked ? window.marked.parse(text) : escapeHtml(text);
  if (isStreaming) {
    html += '<span class="streaming-cursor"></span>';
  }
  targetEl.innerHTML = html;

  // Add Copy buttons and headers to code blocks
  targetEl.querySelectorAll('pre').forEach(pre => {
    if (pre.parentElement.classList.contains('code-block-wrapper')) return;

    const code = pre.querySelector('code');
    let lang = 'code';
    if (code && code.className) {
      const match = code.className.match(/language-([a-z0-9_+-]+)/i);
      if (match) lang = match[1];
    }

    const wrapper = document.createElement('div');
    wrapper.className = 'code-block-wrapper';
    const header = document.createElement('div');
    header.className = 'code-header';
    header.innerHTML = `
      <span>${lang}</span>
      <button class="copy-btn" title="Copy snippet">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
        <span>Copy</span>
      </button>
    `;

    header.querySelector('.copy-btn').addEventListener('click', (e) => {
      const btn = e.currentTarget;
      navigator.clipboard.writeText(code ? code.innerText : pre.innerText);
      btn.querySelector('span').textContent = 'Copied!';
      setTimeout(() => { btn.querySelector('span').textContent = 'Copy'; }, 2000);
    });

    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.appendChild(header);
    wrapper.appendChild(pre);

    if (window.hljs && code) {
      hljs.highlightElement(code);
    }
  });
}

function renderSources(containerEl, sources) {
  if (!sources || sources.length === 0) {
    containerEl.innerHTML = '';
    return;
  }
  const card = document.createElement('div');
  card.className = 'sources-card';
  let listHtml = '';
  sources.forEach((s, idx) => {
    listHtml += `
      <div class="source-item">
        <span class="source-badge">📄 ${escapeHtml(s.source)} (p. ${s.page})</span>
        <span class="source-snippet">${escapeHtml(s.snippet || '')}</span>
      </div>
    `;
  });

  card.innerHTML = `
    <details>
      <summary class="sources-header">
        <span>📚 Sources Used from Your Notes (${sources.length})</span>
      </summary>
      <div class="sources-list">${listHtml}</div>
    </details>
  `;
  containerEl.innerHTML = '';
  containerEl.appendChild(card);
}

// -------------------------------------------------------------
// DOM Message Creation
// -------------------------------------------------------------
function addMessageToDOM(role, content, sources = []) {
  elements.welcomeScreen.classList.add('hidden');
  elements.messagesContainer.classList.remove('hidden');

  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  avatar.textContent = role === 'user' ? '👤' : '🎓';

  const body = document.createElement('div');
  body.className = 'message-body';

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  if (role === 'user') {
    bubble.textContent = content;
  } else {
    renderMarkdown(bubble, content, false);
  }

  body.appendChild(bubble);

  if (sources && sources.length > 0) {
    const sourcesContainer = document.createElement('div');
    renderSources(sourcesContainer, sources);
    body.appendChild(sourcesContainer);
  }

  msgDiv.appendChild(avatar);
  msgDiv.appendChild(body);
  elements.messagesContainer.appendChild(msgDiv);
  scrollToBottom();
}

function createAssistantMessageInDOM() {
  elements.welcomeScreen.classList.add('hidden');
  elements.messagesContainer.classList.remove('hidden');

  const msgDiv = document.createElement('div');
  msgDiv.className = 'message assistant';

  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  avatar.textContent = '🎓';

  const body = document.createElement('div');
  body.className = 'message-body';

  const sourcesContainer = document.createElement('div');
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  bubble.innerHTML = '<span class="streaming-cursor"></span>';

  body.appendChild(sourcesContainer);
  body.appendChild(bubble);

  msgDiv.appendChild(avatar);
  msgDiv.appendChild(body);
  elements.messagesContainer.appendChild(msgDiv);
  scrollToBottom();

  return { bubbleEl: bubble, sourcesEl: sourcesContainer };
}

function scrollToBottom() {
  elements.chatViewport.scrollTop = elements.chatViewport.scrollHeight;
}

function stopGeneration() {
  if (state.activeAbortController) {
    state.activeAbortController.abort();
    state.activeAbortController = null;
  }
  setGeneratingState(false);
}

function setGeneratingState(isGenerating) {
  state.isGenerating = isGenerating;
  if (isGenerating) {
    elements.sendBtn.classList.add('hidden');
    elements.stopBtn.classList.remove('hidden');
  } else {
    elements.sendBtn.classList.remove('hidden');
    elements.stopBtn.classList.add('hidden');
  }
}

// -------------------------------------------------------------
// Sessions & Storage
// -------------------------------------------------------------
function loadSessions() {
  try {
    const raw = localStorage.getItem('ostutor_sessions');
    state.sessions = raw ? JSON.parse(raw) : [];
  } catch (e) {
    state.sessions = [];
  }

  if (state.sessions.length > 0) {
    selectSession(state.sessions[0].id);
  } else {
    startNewSession();
  }
}

function saveSessions() {
  localStorage.setItem('ostutor_sessions', JSON.stringify(state.sessions));
  renderSessionsList();
}

function getCurrentSession() {
  return state.sessions.find(s => s.id === state.activeSessionId);
}

function startNewSession() {
  if (state.isGenerating) return;
  const newSession = createSession('New OS Discussion');
  selectSession(newSession.id);
}

function createSession(title) {
  const id = 'sess_' + Date.now();
  const session = {
    id,
    title,
    messages: []
  };
  state.sessions.unshift(session);
  saveSessions();
  return session;
}

function selectSession(id) {
  state.activeSessionId = id;
  const session = getCurrentSession();
  renderSessionsList();

  // Clear current viewport
  elements.messagesContainer.innerHTML = '';
  if (!session || session.messages.length === 0) {
    elements.welcomeScreen.classList.remove('hidden');
    elements.messagesContainer.classList.add('hidden');
  } else {
    elements.welcomeScreen.classList.add('hidden');
    elements.messagesContainer.classList.remove('hidden');
    session.messages.forEach(m => {
      addMessageToDOM(m.role, m.content, m.sources);
    });
  }
  scrollToBottom();
}

function deleteSession(id, e) {
  e.stopPropagation();
  state.sessions = state.sessions.filter(s => s.id !== id);
  saveSessions();
  if (state.activeSessionId === id) {
    if (state.sessions.length > 0) {
      selectSession(state.sessions[0].id);
    } else {
      startNewSession();
    }
  }
}

function clearAllSessions() {
  if (confirm('Clear all conversation history?')) {
    state.sessions = [];
    localStorage.removeItem('ostutor_sessions');
    startNewSession();
  }
}

function renderSessionsList() {
  elements.historyList.innerHTML = '';
  state.sessions.forEach(sess => {
    const item = document.createElement('div');
    item.className = `history-item ${sess.id === state.activeSessionId ? 'active' : ''}`;
    item.innerHTML = `
      <span class="history-item-title">${escapeHtml(sess.title || 'Untitled Session')}</span>
      <button class="history-item-del" title="Delete session">&times;</button>
    `;
    item.addEventListener('click', () => selectSession(sess.id));
    item.querySelector('.history-item-del').addEventListener('click', (e) => deleteSession(sess.id, e));
    elements.historyList.appendChild(item);
  });
}

// -------------------------------------------------------------
// Theme Management
// -------------------------------------------------------------
function toggleTheme() {
  const currentTheme = elements.body.getAttribute('data-theme') || 'dark';
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
  elements.body.setAttribute('data-theme', newTheme);
  localStorage.setItem('ostutor_theme', newTheme);
  updateThemeIcons(newTheme);
}

function loadTheme() {
  const saved = localStorage.getItem('ostutor_theme') || 'dark';
  elements.body.setAttribute('data-theme', saved);
  updateThemeIcons(saved);
}

function updateThemeIcons(theme) {
  if (theme === 'light') {
    elements.moonIcon.classList.add('hidden');
    elements.sunIcon.classList.remove('hidden');
  } else {
    elements.moonIcon.classList.remove('hidden');
    elements.sunIcon.classList.add('hidden');
  }
}

// -------------------------------------------------------------
// Utilities
// -------------------------------------------------------------
function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/[&<>"']/g, function(m) {
    return {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    }[m];
  });
}

function renderMath(element) {
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetPromise([element]).catch((error) => {
      console.error("Math rendering error:", error);
    });
  }
}

// Run on load
document.addEventListener('DOMContentLoaded', init);