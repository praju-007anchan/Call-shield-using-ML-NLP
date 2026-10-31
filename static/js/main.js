/**
 * CallShield Frontend Logic
 * Interactive Audio Waveform, OpenAI Whisper STT, ML Predictions, and Session History
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const modeTabs = document.querySelectorAll('.mode-tab-btn');
  const uploadPanel = document.getElementById('panel-upload');
  const micPanel = document.getElementById('panel-mic');
  const textPanel = document.getElementById('panel-text');

  const audioFileInput = document.getElementById('audio-file-input');
  const dropzone = document.getElementById('dropzone');
  const selectedFileBox = document.getElementById('selected-file-box');
  const selectedFileName = document.getElementById('selected-file-name');
  const selectedFileSize = document.getElementById('selected-file-size');
  const removeFileBtn = document.getElementById('remove-file-btn');

  // Waveform Audio Player Elements
  const audioPreview = document.getElementById('audio-preview');
  const audioPlayer = document.getElementById('audio-player');
  const waveformCanvasBox = document.getElementById('waveform-canvas-box');
  const waveformCanvas = document.getElementById('waveform-canvas');
  const waveformCursor = document.getElementById('waveform-cursor');
  const waveformHoverTooltip = document.getElementById('waveform-hover-tooltip');
  const waveformLoading = document.getElementById('waveform-loading');
  const waveformTimeDisplay = document.getElementById('waveform-time-display');
  const waveformPlayBtn = document.getElementById('waveform-play-btn');
  const waveformPlayIcon = document.getElementById('waveform-play-icon');
  const waveformTrackName = document.getElementById('waveform-track-name');
  const waveformMuteBtn = document.getElementById('waveform-mute-btn');

  const micBtn = document.getElementById('mic-btn');
  const micStatus = document.getElementById('mic-status');
  const micTimer = document.getElementById('mic-timer');
  const micLevelBox = document.getElementById('mic-level-box');
  const micLevelFill = document.getElementById('mic-level-fill');
  const micLevelLabel = document.getElementById('mic-level-label');

  const transcriptInput = document.getElementById('transcript-input');
  const analyzeBtn = document.getElementById('analyze-btn');
  const resetBtn = document.getElementById('reset-btn');
  const analyzeBtnText = document.getElementById('analyze-btn-text');
  const analyzeSpinner = document.getElementById('analyze-spinner');

  const alertContainer = document.getElementById('alert-container');
  const resultSurface = document.getElementById('result-surface');
  const placeholderSurface = document.getElementById('placeholder-surface');
  const verdictBanner = document.getElementById('verdict-banner');
  const verdictTag = document.getElementById('verdict-tag');
  const verdictTitle = document.getElementById('verdict-title');
  const verdictConfidence = document.getElementById('verdict-confidence');
  const verdictSubrisk = document.getElementById('verdict-subrisk');

  const transcriptResult = document.getElementById('transcript-result');
  const languageTag = document.getElementById('language-tag');
  const durationTag = document.getElementById('duration-tag');
  const segmentsContainer = document.getElementById('segments-container');
  const segmentsList = document.getElementById('segments-list');

  const urgencyCount = document.getElementById('urgency-count');
  const urgencyMatches = document.getElementById('urgency-matches');
  const moneyCount = document.getElementById('money-count');
  const moneyMatches = document.getElementById('money-matches');
  const securityCount = document.getElementById('security-count');
  const securityMatches = document.getElementById('security-matches');

  const modelsTableBody = document.getElementById('models-table-body');
  const historyList = document.getElementById('history-list');
  const historyTableBody = document.getElementById('history-table-body');
  const historySearchInput = document.getElementById('history-search-input');
  const historyFilterSelect = document.getElementById('history-filter-select');
  const historyEmptyState = document.getElementById('history-empty-state');
  const clearHistoryBtn = document.getElementById('clear-history-btn');
  const resultModelName = document.getElementById('result-model-name');
  const resultConfRepeat = document.getElementById('result-conf-repeat');

  // Live Transcription & Review Elements
  const transcribedPreviewBox = document.getElementById('transcribed-preview-box');
  const transcribeStatusPill = document.getElementById('transcribe-status-pill');
  const liveTranscriptEditor = document.getElementById('live-transcript-editor');
  const retranscribeBtn = document.getElementById('retranscribe-btn');
  const copyToTextTabBtn = document.getElementById('copy-to-text-tab-btn');
  const editFromResultsBtn = document.getElementById('edit-from-results-btn');
  const textCharCount = document.getElementById('text-char-count');

  // Application State
  let currentMode = 'upload'; // 'upload' | 'mic' | 'text'
  let currentFile = null;
  let mediaRecorder = null;
  let recordedChunks = [];
  let recordStartTime = null;
  let recordTimerInterval = null;
  let callHistory = JSON.parse(localStorage.getItem('callshield_history') || '[]');

  // Live Studio Microphone Recording State (Unthrottled 16kHz WAV)
  let recordingStream = null;
  let recordAudioCtx = null;
  let recordScriptNode = null;
  let recordPcmChunks = [];
  let recordSampleRate = 16000;
  let isRecordingActive = false;

  function resampleFloat32(source, fromRate, toRate = 16000) {
    if (fromRate === toRate || !source || source.length === 0) return source;
    const ratio = fromRate / toRate;
    const targetLength = Math.round(source.length / ratio);
    const result = new Float32Array(targetLength);
    for (let i = 0; i < targetLength; i++) {
      const originPos = i * ratio;
      const i1 = Math.floor(originPos);
      const i2 = Math.min(i1 + 1, source.length - 1);
      const weight = originPos - i1;
      result[i] = source[i1] * (1 - weight) + source[i2] * weight;
    }
    return result;
  }

  function encodeWAV(samples, sampleRate = 16000) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    function writeString(offset, string) {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    }

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM format
    view.setUint16(22, 1, true); // Mono channel
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // Byte rate
    view.setUint16(32, 2, true); // Block align
    view.setUint16(34, 16, true); // 16-bit
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([view], { type: 'audio/wav' });
  }

  // =========================================================================
  // IndexedDB & Session State Persistence
  // Retains uploaded audio, interactive waveform, and ML analysis results
  // when navigating to Model Bench and back, until explicitly removed.
  // =========================================================================
  const DB_NAME = 'CallShieldAudioDB';
  const DB_VERSION = 1;
  const STORE_NAME = 'active_session';

  function openStateDB() {
    return new Promise((resolve) => {
      if (!window.indexedDB) {
        resolve(null);
        return;
      }
      try {
        const req = window.indexedDB.open(DB_NAME, DB_VERSION);
        req.onupgradeneeded = (e) => {
          const db = e.target.result;
          if (!db.objectStoreNames.contains(STORE_NAME)) {
            db.createObjectStore(STORE_NAME);
          }
        };
        req.onsuccess = (e) => resolve(e.target.result);
        req.onerror = (e) => {
          console.warn('CallShield IndexedDB open error:', e);
          resolve(null);
        };
      } catch (err) {
        console.warn('CallShield IndexedDB exception:', err);
        resolve(null);
      }
    });
  }

  async function setPersistedItem(key, value) {
    if (key !== 'active_audio') {
      try {
        sessionStorage.setItem(`callshield_${key}`, JSON.stringify(value));
      } catch (_) {}
    }
    try {
      const db = await openStateDB();
      if (!db) return;
      return new Promise((resolve) => {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);
        store.put(value, key);
        tx.oncomplete = () => resolve(true);
        tx.onerror = () => resolve(false);
      });
    } catch (err) {
      console.warn('setPersistedItem error:', err);
    }
  }

  async function getPersistedItem(key) {
    try {
      const db = await openStateDB();
      if (db) {
        const result = await new Promise((resolve) => {
          const tx = db.transaction(STORE_NAME, 'readonly');
          const store = tx.objectStore(STORE_NAME);
          const req = store.get(key);
          req.onsuccess = () => resolve(req.result || null);
          req.onerror = () => resolve(null);
        });
        if (result !== null && result !== undefined) return result;
      }
    } catch (err) {
      console.warn('getPersistedItem IndexedDB error:', err);
    }

    if (key !== 'active_audio') {
      try {
        const item = sessionStorage.getItem(`callshield_${key}`);
        return item ? JSON.parse(item) : null;
      } catch (_) {}
    }
    return null;
  }

  async function clearPersistedItem(key) {
    try {
      sessionStorage.removeItem(`callshield_${key}`);
      const db = await openStateDB();
      if (!db) return;
      return new Promise((resolve) => {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);
        store.delete(key);
        tx.oncomplete = () => resolve(true);
        tx.onerror = () => resolve(false);
      });
    } catch (err) {
      console.warn('clearPersistedItem error:', err);
    }
  }

  async function clearAllPersistedState() {
    await clearPersistedItem('active_audio');
    await clearPersistedItem('active_analysis');
    await clearPersistedItem('active_mode');
    await clearPersistedItem('active_text');
    sessionStorage.removeItem('callshield_active_audio');
    sessionStorage.removeItem('callshield_active_analysis');
    sessionStorage.removeItem('callshield_active_mode');
    sessionStorage.removeItem('callshield_active_text');
  }

  // =========================================================================
  // Web Audio API & Waveform Engine
  // =========================================================================
  let audioCtx = null;
  let analyserNode = null;
  let audioSourceConnected = false;
  let currentWaveformPeaks = []; // Normalized actual PCM peaks [0.08, 0.95]
  let isAudioPlaying = false;
  let isScrubbing = false;
  let animFrameId = null;

  function initWebAudio() {
    if (!audioCtx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      audioCtx = new AudioContextClass();
      analyserNode = audioCtx.createAnalyser();
      analyserNode.fftSize = 128;
      analyserNode.smoothingTimeConstant = 0.8;
    }

    if (!audioSourceConnected && audioPlayer) {
      try {
        const source = audioCtx.createMediaElementSource(audioPlayer);
        source.connect(analyserNode);
        analyserNode.connect(audioCtx.destination);
        audioSourceConnected = true;
      } catch (e) {
        console.warn('MediaElementSource already connected or error:', e);
      }
    }

    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
  }

  /**
   * Decodes the actual audio file into raw PCM audio samples
   * and computes amplitude peaks across 110 bars.
   */
  async function loadAndDecodeWaveform(file) {
    if (!waveformCanvas) return;

    waveformLoading.classList.remove('hidden');
    waveformTrackName.textContent = file.name;
    waveformTimeDisplay.textContent = '00:00 / 00:00';

    try {
      initWebAudio();
      const arrayBuffer = await file.arrayBuffer();

      // Decode audio data using browser Web Audio API
      const decodedBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
      const channelData = decodedBuffer.getChannelData(0); // Primary mono/left channel
      const totalBars = 110;
      const step = Math.floor(channelData.length / totalBars);
      const rawPeaks = [];
      let maxVal = 0.0001;

      // Calculate peak amplitude for each waveform bucket
      for (let i = 0; i < totalBars; i++) {
        const start = i * step;
        let peak = 0;
        const subStep = Math.max(1, Math.floor(step / 48));
        for (let j = 0; j < step; j += subStep) {
          const val = Math.abs(channelData[start + j] || 0);
          if (val > peak) peak = val;
        }
        rawPeaks.push(peak);
        if (peak > maxVal) maxVal = peak;
      }

      // Normalize peaks so the highest voice peak scales nicely to 0.94
      currentWaveformPeaks = rawPeaks.map(p => {
        const norm = p / maxVal;
        return Math.max(0.08, Math.min(0.94, norm));
      });

      waveformLoading.classList.add('hidden');
      drawWaveform();

      if (audioPlayer.duration && !isNaN(audioPlayer.duration)) {
        waveformTimeDisplay.textContent = `00:00 / ${formatTime(audioPlayer.duration)}`;
      }
    } catch (err) {
      console.warn('Web Audio decode warning:', err);
      // Fallback: create speech-like baseline if format is partially unsupported by decodeAudioData
      generateFallbackPeaks();
      waveformLoading.classList.add('hidden');
      drawWaveform();
    }
  }

  function generateFallbackPeaks() {
    currentWaveformPeaks = [];
    const count = 110;
    for (let i = 0; i < count; i++) {
      const envelope = Math.exp(-Math.pow((i - 30) / 18, 2)) + Math.exp(-Math.pow((i - 75) / 22, 2));
      const val = Math.max(0.1, Math.min(0.88, envelope * 0.75 + Math.random() * 0.15));
      currentWaveformPeaks.push(val);
    }
  }

  /**
   * Draws the actual audio waveform on the canvas.
   * Differentiates between played and unplayed portions.
   * Reacts dynamically to live audio frequency during playback.
   */
  function drawWaveform() {
    if (!waveformCanvas) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = waveformCanvas.getBoundingClientRect();
    let cssWidth = rect.width;
    if (!cssWidth && waveformCanvasBox) {
      cssWidth = waveformCanvasBox.clientWidth;
    }
    if (!cssWidth) {
      requestAnimationFrame(drawWaveform);
      return;
    }

    const targetW = Math.floor(cssWidth * dpr);
    const targetH = Math.floor(96 * dpr);

    if (waveformCanvas.width !== targetW || waveformCanvas.height !== targetH) {
      waveformCanvas.width = targetW;
      waveformCanvas.height = targetH;
    }

    const w = waveformCanvas.width;
    const h = waveformCanvas.height;
    const ctx = waveformCanvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);

    const barCount = currentWaveformPeaks.length || 110;
    const barWidth = Math.max(2 * dpr, (w / barCount) * 0.52);
    const gap = (w - barCount * barWidth) / (barCount - 1);
    const centerY = h / 2;

    const duration = audioPlayer.duration || 1;
    const currentTime = audioPlayer.currentTime || 0;
    const progress = Math.max(0, Math.min(1, currentTime / duration));
    const cursorPixelX = progress * w;

    // Update cursor position in CSS
    waveformCursor.style.left = `${progress * 100}%`;

    // Query live frequency data from AnalyserNode when playing
    let freqData = null;
    if (analyserNode && isAudioPlaying) {
      freqData = new Uint8Array(analyserNode.frequencyBinCount);
      analyserNode.getByteFrequencyData(freqData);
    }

    // Render bars
    for (let i = 0; i < barCount; i++) {
      const x = i * (barWidth + gap);
      let peak = currentWaveformPeaks[i] || 0.12;

      // Real-time audio amplitude modulation around the playhead
      if (freqData) {
        const distFromCursor = Math.abs(x - cursorPixelX) / w;
        if (distFromCursor < 0.18) {
          const freqIndex = Math.floor((i / barCount) * freqData.length * 0.65);
          const liveAmp = (freqData[freqIndex] || 0) / 255;
          const factor = (1 - distFromCursor / 0.18);
          peak = Math.min(0.96, peak + liveAmp * 0.22 * factor);
        }
      }

      const barH = peak * (h * 0.86);
      const topY = centerY - barH / 2;
      const isPlayed = (x <= cursorPixelX);

      if (isPlayed) {
        // PLAYED STATE: Vibrant Lime to Teal Gradient with subtle glow
        const grad = ctx.createLinearGradient(0, topY, 0, topY + barH);
        grad.addColorStop(0, '#c4f122');
        grad.addColorStop(1, '#2a8b8c');
        ctx.fillStyle = grad;
        ctx.shadowColor = 'rgba(196, 241, 34, 0.45)';
        ctx.shadowBlur = 4 * dpr;
      } else {
        // UNPLAYED STATE: Crisp muted spruce-silver
        ctx.fillStyle = 'rgba(235, 242, 235, 0.26)';
        ctx.shadowBlur = 0;
      }

      const radius = Math.min(barWidth / 2, 2 * dpr);
      ctx.beginPath();
      ctx.roundRect(x, topY, barWidth, barH, radius);
      ctx.fill();
    }

    // Loop animation while playing
    if (isAudioPlaying) {
      animFrameId = requestAnimationFrame(drawWaveform);
    }
  }

  // =========================================================================
  // Playback & Seek Interactivity
  // =========================================================================
  waveformPlayBtn.addEventListener('click', () => {
    initWebAudio();
    if (audioPlayer.paused) {
      audioPlayer.play().catch(err => {
        showAlert('Could not start audio playback: ' + err.message, 'error');
      });
    } else {
      audioPlayer.pause();
    }
  });

  audioPlayer.addEventListener('play', () => {
    isAudioPlaying = true;
    waveformPlayIcon.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor" style="width: 14px; height: 14px;"><rect x="6" y="4" width="4" height="16" rx="1"/><rect x="14" y="4" width="4" height="16" rx="1"/></svg>';
    cancelAnimationFrame(animFrameId);
    animFrameId = requestAnimationFrame(drawWaveform);
  });

  audioPlayer.addEventListener('pause', () => {
    isAudioPlaying = false;
    waveformPlayIcon.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor" style="width: 14px; height: 14px;"><polygon points="5 3 19 12 5 21 5 3"/></svg>';
    cancelAnimationFrame(animFrameId);
    drawWaveform();
  });

  audioPlayer.addEventListener('ended', () => {
    isAudioPlaying = false;
    waveformPlayIcon.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor" style="width: 14px; height: 14px;"><polygon points="5 3 19 12 5 21 5 3"/></svg>';
    audioPlayer.currentTime = 0;
    cancelAnimationFrame(animFrameId);
    drawWaveform();
    updateWaveformTime();
  });

  audioPlayer.addEventListener('timeupdate', () => {
    updateWaveformTime();
    if (!isAudioPlaying) {
      drawWaveform();
    }
  });

  audioPlayer.addEventListener('loadedmetadata', () => {
    updateWaveformTime();
    drawWaveform();
  });

  function updateWaveformTime() {
    const cur = audioPlayer.currentTime || 0;
    const dur = audioPlayer.duration || 0;
    waveformTimeDisplay.textContent = `${formatTime(cur)} / ${formatTime(dur)}`;
  }

  function formatTime(sec) {
    if (isNaN(sec) || !isFinite(sec)) return '00:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  }

  // Click & Drag on Waveform to Seek
  function seekToPosition(e) {
    if (!audioPlayer.duration) return;
    const rect = waveformCanvasBox.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    audioPlayer.currentTime = ratio * audioPlayer.duration;
    updateWaveformTime();
    drawWaveform();
  }

  waveformCanvasBox.addEventListener('pointerdown', (e) => {
    isScrubbing = true;
    initWebAudio();
    seekToPosition(e);
  });

  window.addEventListener('pointermove', (e) => {
    if (isScrubbing) {
      seekToPosition(e);
    }
  });

  window.addEventListener('pointerup', () => {
    isScrubbing = false;
  });

  // Hover Tooltip Timestamp
  waveformCanvasBox.addEventListener('pointermove', (e) => {
    if (!audioPlayer.duration) return;
    const rect = waveformCanvasBox.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const hoverSeconds = ratio * audioPlayer.duration;
    waveformHoverTooltip.textContent = formatTime(hoverSeconds);
    waveformHoverTooltip.style.left = `${ratio * 100}%`;
    waveformHoverTooltip.classList.remove('hidden');
  });

  waveformCanvasBox.addEventListener('pointerleave', () => {
    waveformHoverTooltip.classList.add('hidden');
  });

  // Volume Mute Toggle
  waveformMuteBtn.addEventListener('click', () => {
    audioPlayer.muted = !audioPlayer.muted;
    if (audioPlayer.muted) {
      waveformMuteBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width: 15px; height: 15px;"><line x1="1" y1="1" x2="23" y2="23"/><path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6"/><path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2a7 7 0 0 1-.11 1.23"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>';
      waveformMuteBtn.title = 'Unmute Audio';
    } else {
      waveformMuteBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width: 15px; height: 15px;"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>';
      waveformMuteBtn.title = 'Mute Audio';
    }
  });

  // Window Resize
  window.addEventListener('resize', () => {
    if (currentWaveformPeaks.length > 0) {
      drawWaveform();
    }
  });

  // =========================================================================
  // File Upload & Demo Handling
  // =========================================================================
  renderHistory();

  // Mode Tab Switching
  modeTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      modeTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      currentMode = tab.dataset.mode;

      uploadPanel.classList.add('hidden');
      micPanel.classList.add('hidden');
      textPanel.classList.add('hidden');

      if (currentMode === 'upload') uploadPanel.classList.remove('hidden');
      else if (currentMode === 'mic') micPanel.classList.remove('hidden');
      else if (currentMode === 'text') textPanel.classList.remove('hidden');

      clearAlert();
    });
  });

  // Drag and Drop
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelection(files[0]);
    }
  });

  dropzone.addEventListener('click', () => {
    audioFileInput.click();
  });

  audioFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileSelection(e.target.files[0]);
    }
  });

  removeFileBtn.addEventListener('click', async (e) => {
    e.stopPropagation();
    await clearSelectedFile();
  });

  async function handleFileSelection(file, isRestoring = false) {
    const validExtensions = ['wav', 'mp3', 'm4a', 'ogg', 'webm', 'flac', 'aac'];
    const ext = file.name.split('.').pop().toLowerCase();

    if (!validExtensions.includes(ext)) {
      showAlert('Unsupported audio format. Please upload WAV, MP3, M4A, OGG, or WEBM.', 'error');
      return;
    }

    currentFile = file;
    selectedFileName.textContent = file.name;
    selectedFileSize.textContent = formatBytes(file.size);
    selectedFileBox.classList.remove('hidden');
    dropzone.classList.add('hidden');

    // Create object URL for local audio playback
    const audioUrl = URL.createObjectURL(file);
    audioPlayer.src = audioUrl;
    audioPreview.classList.remove('hidden');

    // Decode and display real audio waveform
    loadAndDecodeWaveform(file);

    clearAlert();

    if (!isRestoring) {
      // Persist uploaded audio file into IndexedDB
      try {
        await setPersistedItem('active_audio', {
          blob: file,
          name: file.name,
          type: file.type || 'audio/mpeg',
          size: file.size,
          lastModified: file.lastModified || Date.now()
        });
        await setPersistedItem('active_mode', 'upload');
        // Clear previous analysis results when a brand new file is chosen
        await clearPersistedItem('active_analysis');
      } catch (err) {
        console.warn('Could not persist audio file:', err);
      }

      resultSurface.classList.add('hidden');
      if (placeholderSurface) {
        placeholderSurface.classList.remove('hidden');
      }
    }
  }

  async function clearSelectedFile() {
    currentFile = null;
    audioFileInput.value = '';
    selectedFileBox.classList.add('hidden');
    dropzone.classList.remove('hidden');
    audioPreview.classList.add('hidden');
    if (transcribedPreviewBox) transcribedPreviewBox.classList.add('hidden');
    if (liveTranscriptEditor) liveTranscriptEditor.value = '';
    audioPlayer.pause();
    audioPlayer.src = '';
    currentWaveformPeaks = [];
    isAudioPlaying = false;
    cancelAnimationFrame(animFrameId);
    waveformPlayIcon.textContent = '▶';
    waveformTimeDisplay.textContent = '00:00 / 00:00';
    resultSurface.classList.add('hidden');
    if (placeholderSurface) {
      placeholderSurface.classList.remove('hidden');
    }
    await clearAllPersistedState();
  }

  // Preloaded Demo Samples
  window.loadSampleAudio = function(sampleName) {
    const sampleUrl = `/static/samples/${sampleName}`;
    showAlert(`Loading demo sample "${sampleName}"...`, 'info');

    // Switch to upload tab
    document.querySelector('[data-mode="upload"]').click();

    fetch(sampleUrl)
      .then(res => {
        if (!res.ok) throw new Error('Could not load sample file');
        return res.blob();
      })
      .then(blob => {
        const file = new File([blob], sampleName, { type: blob.type || 'audio/mpeg' });
        handleFileSelection(file);
        showAlert(`Loaded sample: ${sampleName}. Click "Analyze Call" to evaluate.`, 'info');
      })
      .catch(err => {
        showAlert('Could not load sample audio: ' + err.message, 'error');
      });
  };

  // =========================================================================
  // Studio Microphone Recording Engine (Direct 16kHz WAV + Dual Pipeline)
  // Prevents browser 3-second noise-gating ducking and maintains crystal clarity.
  // =========================================================================
  micBtn.addEventListener('click', async () => {
    if (isRecordingActive) {
      stopRecording();
    } else {
      startRecording();
    }
  });

  async function startRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showAlert('Microphone recording is not supported in this browser.', 'error');
      return;
    }

    try {
      // 1. Request unthrottled microphone audio without aggressive Chrome noise-gating or AGC ducking
      let stream = null;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
          }
        });
      } catch (_) {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      }

      recordingStream = stream;
      recordedChunks = [];
      recordPcmChunks = [];
      isRecordingActive = true;

      // 2. Setup Web Audio API PCM capture for crystal-clear uncompressed 16kHz WAV
      const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
      if (AudioCtxClass) {
        try {
          recordAudioCtx = new AudioCtxClass();
          recordSampleRate = recordAudioCtx.sampleRate;
          const source = recordAudioCtx.createMediaStreamSource(stream);
          recordScriptNode = recordAudioCtx.createScriptProcessor(4096, 1, 1);

          recordScriptNode.onaudioprocess = (e) => {
            if (!isRecordingActive) return;
            const inputData = e.inputBuffer.getChannelData(0);
            recordPcmChunks.push(new Float32Array(inputData));

            // Compute real-time RMS audio energy for live visual VU meter
            let sumSq = 0;
            for (let i = 0; i < inputData.length; i++) {
              sumSq += inputData[i] * inputData[i];
            }
            const rms = Math.sqrt(sumSq / inputData.length);
            if (micLevelFill) {
              const pct = Math.min(100, Math.max(4, Math.round(rms * 450)));
              micLevelFill.style.width = `${pct}%`;
            }
          };

          const silentGain = recordAudioCtx.createGain();
          silentGain.gain.value = 0; // Prevent speaker feedback loop
          source.connect(recordScriptNode);
          recordScriptNode.connect(silentGain);
          silentGain.connect(recordAudioCtx.destination);
        } catch (ctxErr) {
          console.warn('Web Audio PCM capture init warning:', ctxErr);
        }
      }

      // 3. Setup parallel MediaRecorder with timeslice as reliable secondary stream
      try {
        const mimeCandidates = [
          'audio/webm;codecs=opus',
          'audio/webm',
          'audio/ogg;codecs=opus',
          'audio/mp4'
        ];
        let chosenMime = '';
        for (const m of mimeCandidates) {
          if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) {
            chosenMime = m;
            break;
          }
        }
        mediaRecorder = new MediaRecorder(stream, chosenMime ? { mimeType: chosenMime, audioBitsPerSecond: 128000 } : {});
        mediaRecorder.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) recordedChunks.push(e.data);
        };
        // 200ms timeslice ensures chunks flush continuously without buffer overflow
        mediaRecorder.start(200);
      } catch (recErr) {
        console.warn('MediaRecorder init warning:', recErr);
      }

      // 4. Update UI State
      micBtn.classList.add('recording');
      micStatus.textContent = 'Recording live call... Speak clearly into microphone. Click to finish.';
      recordStartTime = Date.now();
      micTimer.textContent = '00:00';
      micTimer.classList.remove('hidden');
      if (micLevelBox) micLevelBox.classList.remove('hidden');
      if (micLevelFill) micLevelFill.style.width = '5%';
      recordTimerInterval = setInterval(updateTimer, 500);

    } catch (err) {
      console.error('Microphone access failed:', err);
      showAlert('Microphone permission denied or device not found. Please allow microphone access in your browser settings.', 'error');
    }
  }

  async function stopRecording() {
    isRecordingActive = false;
    micBtn.classList.remove('recording');
    micStatus.textContent = 'Processing call recording...';
    micTimer.classList.add('hidden');
    if (micLevelBox) micLevelBox.classList.add('hidden');
    clearInterval(recordTimerInterval);

    // Stop MediaRecorder if running
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      try { mediaRecorder.stop(); } catch (_) {}
    }

    // Disconnect and release Web Audio nodes
    if (recordScriptNode) {
      try { recordScriptNode.disconnect(); } catch (_) {}
      recordScriptNode = null;
    }
    if (recordAudioCtx) {
      try { await recordAudioCtx.close(); } catch (_) {}
      recordAudioCtx = null;
    }

    // Stop and release all microphone tracks
    if (recordingStream) {
      try {
        recordingStream.getTracks().forEach(track => track.stop());
      } catch (_) {}
      recordingStream = null;
    }

    // Generate high-fidelity WAV file from PCM chunks (or fallback to MediaRecorder blob)
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(11, 19);
    let recordedFile = null;

    if (recordPcmChunks.length > 0) {
      try {
        let totalSamples = 0;
        for (const chunk of recordPcmChunks) totalSamples += chunk.length;
        const allSamples = new Float32Array(totalSamples);
        let offset = 0;
        for (const chunk of recordPcmChunks) {
          allSamples.set(chunk, offset);
          offset += chunk.length;
        }

        // Resample directly to 16kHz for native Whisper input
        const resampledSamples = resampleFloat32(allSamples, recordSampleRate, 16000);
        const wavBlob = encodeWAV(resampledSamples, 16000);
        recordedFile = new File([wavBlob], `recorded-call-${timestamp}.wav`, { type: 'audio/wav' });
      } catch (wavErr) {
        console.warn('WAV generation fallback:', wavErr);
      }
    }

    if (!recordedFile) {
      const mime = mediaRecorder ? (mediaRecorder.mimeType || 'audio/webm') : 'audio/webm';
      const blob = new Blob(recordedChunks, { type: mime });
      recordedFile = new File([blob], `recorded-call-${timestamp}.webm`, { type: blob.type });
    }

    micStatus.textContent = 'Click microphone to record call audio';

    // Pass the recorded audio into the file handler & waveform visualizer
    await handleFileSelection(recordedFile);

    // Switch view to upload mode so user can see preview, playback, and live transcript review
    const uploadTabBtn = document.querySelector('[data-mode="upload"]');
    if (uploadTabBtn) uploadTabBtn.click();

    // Automatically trigger Whisper STT transcription so user can review/edit words
    transcribeCurrentAudio();
  }

  function updateTimer() {
    const elapsed = Math.floor((Date.now() - recordStartTime) / 1000);
    const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const secs = String(elapsed % 60).padStart(2, '0');
    micTimer.textContent = `${mins}:${secs}`;
  }

  // =========================================================================
  // Call Analysis & Machine Learning Prediction
  // =========================================================================
  analyzeBtn.addEventListener('click', async () => {
    clearAlert();

    const formData = new FormData();

    if (currentMode === 'upload' || currentMode === 'mic') {
      if (!currentFile) {
        showAlert('Please upload or record an audio file.', 'error');
        return;
      }
      formData.append('audio', currentFile);
      if (liveTranscriptEditor && liveTranscriptEditor.value.trim()) {
        formData.append('edited_transcript', liveTranscriptEditor.value.trim());
      }
    } else if (currentMode === 'text') {
      const text = transcriptInput.value.trim();
      if (!text) {
        showAlert('Please enter transcript text to analyze.', 'error');
        return;
      }
      formData.append('text', text);
    }

    setLoading(true);

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (!res.ok || !data.success) {
        if (res.status === 401) {
          showAlert('Session expired. Please <a href="/login" style="color: inherit; text-decoration: underline;">log in</a> again.', 'error');
        } else {
          showAlert(data.error || 'Prediction failed. Please try again.', 'error');
        }
        return;
      }

      displayResults(data);
      saveToHistory(data);

      try {
        await setPersistedItem('active_analysis', data);
        await setPersistedItem('active_mode', currentMode);
        if (currentMode === 'text') {
          await setPersistedItem('active_text', transcriptInput.value.trim());
        }
      } catch (err) {
        console.warn('Could not persist analysis state:', err);
      }

    } catch (err) {
      showAlert('Network error or server unavailable. Please try again.', 'error');
    } finally {
      setLoading(false);
    }
  });

  resetBtn.addEventListener('click', async () => {
    await clearSelectedFile();
    transcriptInput.value = '';
    if (liveTranscriptEditor) liveTranscriptEditor.value = '';
    if (transcribedPreviewBox) transcribedPreviewBox.classList.add('hidden');
    if (textCharCount) textCharCount.textContent = '0 characters';
    resultSurface.classList.add('hidden');
    if (placeholderSurface) {
      placeholderSurface.classList.remove('hidden');
    }
    clearAlert();
    await clearAllPersistedState();
  });

  // =========================================================================
  // Live Audio Transcription & Manual Review Engine
  // =========================================================================
  async function transcribeCurrentAudio(triggerBtn = null) {
    if (!currentFile) return;

    if (transcribedPreviewBox) transcribedPreviewBox.classList.remove('hidden');
    if (transcribeStatusPill) {
      transcribeStatusPill.textContent = 'Transcribing with Whisper...';
      transcribeStatusPill.style.background = 'rgba(255, 170, 0, 0.2)';
      transcribeStatusPill.style.color = '#ffaa00';
    }
    if (triggerBtn) triggerBtn.disabled = true;

    try {
      const fd = new FormData();
      fd.append('audio', currentFile);
      const res = await fetch('/api/transcribe', {
        method: 'POST',
        body: fd
      });
      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.error || 'Transcription failed');
      }

      if (liveTranscriptEditor) {
        liveTranscriptEditor.value = data.transcript;
      }
      if (transcriptInput && !transcriptInput.value.trim()) {
        transcriptInput.value = data.transcript;
        if (textCharCount) textCharCount.textContent = `${data.transcript.length} characters`;
      }

      if (transcribeStatusPill) {
        transcribeStatusPill.textContent = 'Transcribed (Editable)';
        transcribeStatusPill.style.background = 'rgba(196, 241, 34, 0.15)';
        transcribeStatusPill.style.color = 'var(--accent)';
      }
      showAlert('Speech transcribed with OpenAI Whisper! You can review or manually edit words below, then click "Analyze Call".', 'info');
    } catch (err) {
      console.warn('Transcription error:', err);
      if (transcribeStatusPill) {
        transcribeStatusPill.textContent = 'Transcription failed';
        transcribeStatusPill.style.background = 'rgba(255, 68, 68, 0.2)';
        transcribeStatusPill.style.color = '#ff4444';
      }
      showAlert(`Audio transcription note: ${err.message}. You can also type or edit the transcript manually in the "Paste Text" tab.`, 'error');
    } finally {
      if (triggerBtn) triggerBtn.disabled = false;
    }
  }

  if (retranscribeBtn) {
    retranscribeBtn.addEventListener('click', () => {
      transcribeCurrentAudio(retranscribeBtn);
    });
  }

  if (copyToTextTabBtn) {
    copyToTextTabBtn.addEventListener('click', () => {
      const text = (liveTranscriptEditor ? liveTranscriptEditor.value.trim() : '');
      if (!text) {
        showAlert('No transcribed text to copy yet.', 'error');
        return;
      }
      transcriptInput.value = text;
      if (textCharCount) textCharCount.textContent = `${text.length} characters`;
      document.querySelector('[data-mode="text"]').click();
      transcriptInput.focus();
      showAlert('Transcript copied to manual text tab! You can edit it here.', 'info');
    });
  }

  if (editFromResultsBtn) {
    editFromResultsBtn.addEventListener('click', () => {
      const resultText = transcriptResult.textContent.replace(/^"|"$/g, '').trim();
      transcriptInput.value = resultText;
      if (textCharCount) textCharCount.textContent = `${resultText.length} characters`;
      document.querySelector('[data-mode="text"]').click();
      transcriptInput.scrollIntoView({ behavior: 'smooth' });
      transcriptInput.focus();
      showAlert('Transcript loaded into manual editor! Adjust words and click Analyze Call to see how the model responds.', 'info');
    });
  }

  if (transcriptInput && textCharCount) {
    transcriptInput.addEventListener('input', () => {
      textCharCount.textContent = `${transcriptInput.value.length} characters`;
    });
  }

  window.quickFillScript = function(scriptType) {
    document.querySelector('[data-mode="text"]').click();
    const scripts = {
      digital_arrest: "Hello good morning sir, this is Officer Kumar from the Police Department. You are under digital arrest regarding an illegal parcel and cheque bounced in the bank. You must make an immediate payment of penalty funds to this bank account to avoid arrest.",
      bank_otp: "Greetings, this is an urgent security notification from your bank account department. Your debit card has been suspended. Please provide your card number, CVV, and one time password immediately to verify your account.",
      clinic: "Greetings, this is City Dental Clinic calling to confirm your routine checkup and cleaning appointment scheduled for tomorrow at 10 AM. Please call us back to confirm or reschedule."
    };
    const sample = scripts[scriptType] || "";
    transcriptInput.value = sample;
    if (textCharCount) textCharCount.textContent = `${sample.length} characters`;
    transcriptInput.focus();
    showAlert(`Loaded "${scriptType.replace('_', ' ').toUpperCase()}" test script into manual editor. Click "Analyze Call" to evaluate!`, 'info');
  };

  function displayResults(data, isRestore = false) {
    resultSurface.classList.remove('hidden');
    if (placeholderSurface) {
      placeholderSurface.classList.add('hidden');
    }

    if (!isRestore && window.innerWidth < 1080) {
      resultSurface.scrollIntoView({ behavior: 'smooth' });
    }

    // Verdict Banner - Binary Classification: SPAM or NOT SPAM
    const isSpam = data.prediction === 'SPAM' || data.prediction === 'SPAM / FRAUD CALL';
    verdictBanner.className = `verdict-banner ${isSpam ? 'spam' : 'legit'}`;
    verdictTag.textContent = isSpam ? 'CRITICAL ALERT' : 'VERIFIED LEGITIMATE';
    verdictTitle.textContent = isSpam ? 'SPAM' : 'NOT SPAM';
    verdictConfidence.textContent = `${(data.confidence * 100).toFixed(1)}%`;

    const verdictIcon = document.getElementById('verdict-icon');
    if (verdictIcon) {
      if (isSpam) {
        verdictIcon.innerHTML = `
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" style="width: 26px; height: 26px;">
            <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
            <line x1="12" y1="9" x2="12" y2="13"/>
            <line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
        `;
      } else {
        verdictIcon.innerHTML = `
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" style="width: 26px; height: 26px;">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            <path d="m9 12 2 2 4-4"/>
          </svg>
        `;
      }
    }

    if (verdictSubrisk) {
      if (isSpam && data.fraud_risk && data.fraud_risk !== 'N/A (Legitimate)') {
        verdictSubrisk.className = 'secondary-risk-pill';
        verdictSubrisk.innerHTML = `<span style="opacity: 0.8;">Supporting Intelligence:</span> <strong>${data.fraud_risk}</strong>`;
        verdictSubrisk.classList.remove('hidden');
      } else {
        verdictSubrisk.classList.add('hidden');
      }
    }

    if (resultModelName) {
      resultModelName.textContent = data.model_name ? `${data.model_name} (C=10, L2)` : 'Logistic Regression (C=10, L2)';
    }
    if (resultConfRepeat) {
      resultConfRepeat.textContent = `${(data.confidence * 100).toFixed(1)}%`;
    }

    // Transcript
    transcriptResult.textContent = `"${data.transcript}"`;
    languageTag.textContent = (data.language || 'en').toUpperCase();
    durationTag.textContent = data.duration_seconds ? `${data.duration_seconds.toFixed(1)}s` : 'N/A';

    // Segments
    if (data.segments && data.segments.length > 0) {
      segmentsContainer.classList.remove('hidden');
      segmentsList.innerHTML = data.segments.map(seg => `
        <div class="segment-item">
          <span>${seg.text}</span>
          <span class="segment-time mono">${seg.start.toFixed(1)}s - ${seg.end.toFixed(1)}s</span>
        </div>
      `).join('');
    } else {
      segmentsContainer.classList.add('hidden');
    }

    // Signals
    const signals = data.signals || {};
    renderSignal('urgency', urgencyCount, urgencyMatches, signals.urgency_keywords);
    renderSignal('financial', moneyCount, moneyMatches, signals.financial_keywords);
    renderSignal('security', securityCount, securityMatches, signals.security_keywords);

    // Model comparison table
    if (data.model_results && data.model_results.length > 0) {
      modelsTableBody.innerHTML = data.model_results.map(m => {
        const mSpam = m.prediction === 'SPAM' || m.prediction === 'SPAM / FRAUD CALL';
        return `
          <tr class="${m.is_best ? 'winner-row' : ''}">
            <td>
              <strong>${m.model}</strong>
              ${m.is_best ? '<span class="tag-badge winner" style="margin-left: 6px;">Best</span>' : ''}
            </td>
            <td>
              <span class="tag-badge ${mSpam ? 'spam' : 'legit'}">
                ${mSpam ? 'Spam' : 'Not Spam'}
              </span>
            </td>
            <td class="mono font-semibold">${(m.confidence * 100).toFixed(1)}%</td>
            <td class="mono text-muted">${(m.accuracy * 100).toFixed(1)}%</td>
            <td class="mono text-muted">${(m.f1_score * 100).toFixed(1)}%</td>
          </tr>
        `;
      }).join('');
    }
  }

  function renderSignal(name, countEl, matchesEl, signalData) {
    if (!signalData) {
      countEl.textContent = '0';
      matchesEl.innerHTML = '<span class="mono" style="font-size: 10px; color: var(--text-muted);">None detected</span>';
      return;
    }
    countEl.textContent = signalData.count;
    if (signalData.matches && signalData.matches.length > 0) {
      matchesEl.innerHTML = signalData.matches.map(kw => `<span class="signal-pill">${kw}</span>`).join('');
    } else {
      matchesEl.innerHTML = '<span class="mono" style="font-size: 10px; color: var(--text-muted);">None detected</span>';
    }
  }

  function saveToHistory(data) {
    const now = new Date();
    const item = {
      id: Date.now(),
      date: now.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' + now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      filename: currentFile ? currentFile.name : (currentMode === 'text' ? 'Pasted Text' : 'Microphone Recording'),
      prediction: data.prediction,
      confidence: data.confidence,
      model: data.model_name || 'Logistic Regression',
      transcript: data.transcript ? (data.transcript.slice(0, 90) + (data.transcript.length > 90 ? '...' : '')) : 'N/A',
    };

    callHistory.unshift(item);
    if (callHistory.length > 20) callHistory.pop();
    localStorage.setItem('callshield_history', JSON.stringify(callHistory));
    renderHistory();
  }

  function renderHistory() {
    if (!historyTableBody && !historyList) return;

    const searchTerm = (historySearchInput ? historySearchInput.value.trim().toLowerCase() : '');
    const filterVerdict = (historyFilterSelect ? historyFilterSelect.value : 'all');

    let filtered = callHistory.filter(item => {
      const isSpam = item.prediction === 'SPAM' || item.prediction === 'SPAM / FRAUD CALL';
      if (filterVerdict === 'spam' && !isSpam) return false;
      if (filterVerdict === 'legit' && isSpam) return false;
      if (searchTerm) {
        const hay = ((item.filename || '') + ' ' + (item.transcript || '') + ' ' + (item.model || '')).toLowerCase();
        if (!hay.includes(searchTerm)) return false;
      }
      return true;
    });

    if (historyTableBody) {
      if (callHistory.length === 0) {
        historyTableBody.innerHTML = `
          <tr>
            <td colspan="6" style="text-align: center; padding: 24px 16px; color: var(--text-muted); font-size: 13px;">
              No calls analyzed yet in this session. Upload audio or select a demo sample above to begin.
            </td>
          </tr>
        `;
        if (historyEmptyState) historyEmptyState.classList.add('hidden');
        return;
      }

      if (filtered.length === 0) {
        historyTableBody.innerHTML = '';
        if (historyEmptyState) historyEmptyState.classList.remove('hidden');
        return;
      }

      if (historyEmptyState) historyEmptyState.classList.add('hidden');

      historyTableBody.innerHTML = filtered.map(item => {
        const isSpam = item.prediction === 'SPAM' || item.prediction === 'SPAM / FRAUD CALL';
        return `
          <tr>
            <td class="mono text-muted" style="white-space: nowrap; font-size: 11px;">${item.date}</td>
            <td class="font-semibold" style="max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${item.filename}">
              ${item.filename}
            </td>
            <td style="max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-muted); font-size: 12px;" title="${item.transcript}">
              "${item.transcript}"
            </td>
            <td class="mono" style="font-size: 11px;">${item.model || 'Logistic Regression'}</td>
            <td>
              <span class="tag-badge ${isSpam ? 'spam' : 'legit'}">
                ${isSpam ? 'Spam' : 'Not Spam'}
              </span>
            </td>
            <td class="mono font-semibold">${(item.confidence * 100).toFixed(1)}%</td>
          </tr>
        `;
      }).join('');
    } else if (historyList) {
      // Fallback
      if (filtered.length === 0) {
        historyList.innerHTML = '<p style="font-size: 13px; color: var(--text-muted); padding: 12px 0;">No calls match filter.</p>';
        return;
      }
      historyList.innerHTML = filtered.map(item => {
        const isSpam = item.prediction === 'SPAM' || item.prediction === 'SPAM / FRAUD CALL';
        return `
          <div style="display: flex; align-items: center; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid var(--border-card); font-size: 13px;">
            <div>
              <div style="display: flex; align-items: center; gap: 8px;">
                <strong style="max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${item.filename}</strong>
                <span class="mono" style="font-size: 11px; color: var(--text-muted);">${item.date}</span>
              </div>
              <p style="font-size: 12px; color: var(--text-muted); margin-top: 2px;">${item.transcript}</p>
            </div>
            <div style="text-align: right; flex-shrink: 0; margin-left: 12px;">
              <span class="tag-badge ${isSpam ? 'spam' : 'legit'}">${isSpam ? 'Spam' : 'Not Spam'}</span>
              <div class="mono" style="font-size: 11px; margin-top: 2px; font-weight: 600;">${(item.confidence * 100).toFixed(1)}%</div>
            </div>
          </div>
        `;
      }).join('');
    }
  }

  if (historySearchInput) {
    historySearchInput.addEventListener('input', renderHistory);
  }
  if (historyFilterSelect) {
    historyFilterSelect.addEventListener('change', renderHistory);
  }

  if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener('click', () => {
      callHistory = [];
      localStorage.removeItem('callshield_history');
      renderHistory();
    });
  }

  async function loadInitialServerHistory() {
    if (callHistory.length === 0) {
      try {
        const res = await fetch('/api/history');
        if (res.ok) {
          const sHistory = await res.json();
          if (Array.isArray(sHistory) && sHistory.length > 0) {
            callHistory = sHistory.map(entry => ({
              id: entry.id,
              date: entry.timestamp,
              filename: entry.filename || 'Recorded Call',
              prediction: entry.prediction,
              confidence: entry.confidence,
              model: entry.model || 'Logistic Regression',
              transcript: entry.transcript ? (entry.transcript.slice(0, 90) + (entry.transcript.length > 90 ? '...' : '')) : 'N/A',
            }));
            localStorage.setItem('callshield_history', JSON.stringify(callHistory));
            renderHistory();
          }
        }
      } catch (_) {}
    }
  }
  loadInitialServerHistory();

  function showAlert(msg, type = 'error') {
    const iconSvg = type === 'error'
      ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width: 18px; height: 18px; flex-shrink: 0;"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`
      : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width: 18px; height: 18px; flex-shrink: 0;"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`;
    alertContainer.innerHTML = `
      <div class="alert-box ${type}">
        ${iconSvg}
        <div>${msg}</div>
      </div>
    `;
    alertContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function clearAlert() {
    alertContainer.innerHTML = '';
  }

  function setLoading(loading) {
    if (loading) {
      analyzeBtn.disabled = true;
      analyzeSpinner.classList.remove('hidden');
      analyzeBtnText.textContent = 'Analyzing with Whisper & ML...';
    } else {
      analyzeBtn.disabled = false;
      analyzeSpinner.classList.add('hidden');
      analyzeBtnText.textContent = 'Analyze Call';
    }
  }

  function formatBytes(bytes, decimals = 1) {
    if (!+bytes) return '0 B';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
  }

  // =========================================================================
  // Restore Active Session State on Page Load
  // Preserves uploaded audio, interactive waveform, and ML analysis results
  // when navigating across the workspace (e.g. Model Bench) until removed.
  // =========================================================================
  async function restoreSessionState() {
    try {
      const savedMode = await getPersistedItem('active_mode');
      const savedAudio = await getPersistedItem('active_audio');
      const savedAnalysis = await getPersistedItem('active_analysis');
      const savedText = await getPersistedItem('active_text');

      if (savedMode === 'text' && !savedAudio) {
        const textTab = document.querySelector('[data-mode="text"]');
        if (textTab) textTab.click();
        if (savedText) {
          transcriptInput.value = savedText;
        }
        if (savedAnalysis) {
          displayResults(savedAnalysis, true);
        }
        return;
      }

      if (savedAudio && savedAudio.blob) {
        const uploadTab = document.querySelector('[data-mode="upload"]');
        if (uploadTab && !uploadTab.classList.contains('active')) {
          uploadTab.click();
        }

        const restoredFile = new File([savedAudio.blob], savedAudio.name, {
          type: savedAudio.type || 'audio/mpeg',
          lastModified: savedAudio.lastModified || Date.now()
        });

        await handleFileSelection(restoredFile, true);

        if (savedAnalysis) {
          displayResults(savedAnalysis, true);
        }
      }
    } catch (err) {
      console.warn('CallShield state restoration failed:', err);
    }
  }

  // Restore active session on load
  restoreSessionState();

  // Auto-trigger sample analysis if query parameter present (for preview / demo / documentation)
  const urlParams = new URLSearchParams(window.location.search);
  const sampleParam = urlParams.get('sample');
  if (sampleParam === 'spam') {
    quickFillScript('digital_arrest');
    setTimeout(() => {
      const btn = document.getElementById('analyze-btn');
      if (btn) btn.click();
    }, 300);
  } else if (sampleParam === 'legit') {
    quickFillScript('clinic');
    setTimeout(() => {
      const btn = document.getElementById('analyze-btn');
      if (btn) btn.click();
    }, 300);
  }
});
