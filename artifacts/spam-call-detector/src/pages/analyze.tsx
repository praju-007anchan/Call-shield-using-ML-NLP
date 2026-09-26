import { useMemo, useRef, useState } from 'react';
import { AlertTriangle, ArrowUpRight, Check, FileAudio, FileText, Info, LoaderCircle, Mic2, Paperclip, RotateCcw, ShieldAlert, Sparkles, UploadCloud } from 'lucide-react';
import { useGetDashboard, usePredictCall, useTranscribeAudio, type PredictionResult, type TranscriptionResult } from '@workspace/api-client-react';

type SourceMode = 'recording' | 'transcript';

export default function AnalyzePage() {
  const [sourceMode, setSourceMode] = useState<SourceMode>('recording');
  const [file, setFile] = useState<File | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [pastedText, setPastedText] = useState('');
  const [transcript, setTranscript] = useState<TranscriptionResult | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [error, setError] = useState('');
  const fileInput = useRef<HTMLInputElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recordingChunks = useRef<Blob[]>([]);
  const transcribe = useTranscribeAudio();
  const predict = usePredictCall();
  const { data: dashboard } = useGetDashboard();

  const isWorking = transcribe.isPending || predict.isPending;
  const canAnalyze = sourceMode === 'recording' ? Boolean(file) : pastedText.trim().length > 0;

  const runPrediction = (text: string, durationSeconds?: number) => {
    predict.mutate({ data: { text, durationSeconds } }, {
      onSuccess: (result) => { setPrediction(result); setError(''); },
      onError: () => setError('The signal could not be scored. Try again in a moment.'),
    });
  };

  const analyze = () => {
    setError('');
    setPrediction(null);
    if (sourceMode === 'transcript') {
      runPrediction(pastedText.trim());
      return;
    }
    if (!file) return;
    transcribe.mutate({ data: { audio: file, model: 'small' } }, {
      onSuccess: (result) => {
        setTranscript(result);
        runPrediction(result.transcript, result.durationSeconds);
      },
      onError: () => setError('Transcription did not complete. Check the recording and try again.'),
    });
  };

  const reset = () => {
    if (recorderRef.current) recorderRef.current.stop();
    setIsRecording(false);
    setFile(null);
    setPastedText('');
    setTranscript(null);
    setPrediction(null);
    setError('');
    if (fileInput.current) fileInput.current.value = '';
  };

  const startRecording = async () => {
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      setError('Microphone recording is not available in this browser. Upload an audio file instead.');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      recordingChunks.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) recordingChunks.current.push(event.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        const recording = new Blob(recordingChunks.current, { type: recorder.mimeType || 'audio/webm' });
        setFile(new File([recording], `call-recording-${new Date().toISOString().slice(11, 19).replaceAll(':', '-')}.webm`, { type: recording.type }));
        setIsRecording(false);
      };
      recorderRef.current = recorder;
      recorder.start();
      setIsRecording(true);
      setError('');
    } catch {
      setError('Microphone permission was not granted. Upload an audio file instead.');
    }
  };

  const stopRecording = () => recorderRef.current?.stop();

  return (
    <div className="signal-grid min-h-[calc(100dvh-68px)]">
      <div className="mx-auto max-w-[1420px] px-5 py-7 md:px-9 md:py-10">
        <div className="mb-8 flex flex-col justify-between gap-5 lg:flex-row lg:items-end">
          <div className="animate-rise-in">
            <div className="mb-3 flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.18em] text-accent"><span className="h-1.5 w-1.5 rounded-full bg-accent" /> decision surface</div>
            <h1 className="max-w-2xl text-[clamp(2rem,4vw,3.55rem)] font-semibold leading-[.98] tracking-[-0.065em]">Turn a noisy call<br /><span className="text-accent">into a clear next step.</span></h1>
            <p className="mt-4 max-w-xl text-sm leading-6 text-muted-foreground">Review a recording or transcript. CallShield surfaces what it sees, how sure it is, and what still needs a human ear.</p>
          </div>
          <div className="flex gap-5 border-l-2 border-primary pl-4 text-xs text-muted-foreground animate-rise-in delay-1">
            <div><span className="block font-mono text-[10px] uppercase tracking-[.12em] text-foreground/50">spam rate</span><strong className="mt-1 block text-lg text-foreground">{dashboard ? `${(dashboard.spamRate * 100).toFixed(1)}%` : '—'}</strong></div>
            <div><span className="block font-mono text-[10px] uppercase tracking-[.12em] text-foreground/50">avg confidence</span><strong className="mt-1 block text-lg text-foreground">{dashboard ? `${(dashboard.avgConfidence * 100).toFixed(1)}%` : '—'}</strong></div>
          </div>
        </div>

        <div className="grid items-start gap-5 xl:grid-cols-[minmax(360px,0.76fr)_minmax(540px,1.24fr)]">
          <section className="animate-rise-in delay-1">
            <div className="overflow-hidden rounded-xl border border-card-border bg-card shadow-[var(--shadow-soft)]">
              <div className="flex items-center justify-between border-b border-card-border px-5 py-4">
                <div><h2 className="font-semibold tracking-[-0.02em]">New review</h2><p className="mt-0.5 text-xs text-muted-foreground">Start with the source you have.</p></div>
                <span className="font-mono text-[10px] text-muted-foreground">01 / 02</span>
              </div>
              <div className="flex border-b border-card-border px-5 pt-3">
                <button type="button" data-testid="button-mode-recording" onClick={() => setSourceMode('recording')} className={`border-b-2 px-1 pb-3 text-xs font-medium ${sourceMode === 'recording' ? 'border-primary text-foreground' : 'border-transparent text-muted-foreground'}`}><FileAudio size={14} className="mr-2 inline" />Recording</button>
                <button type="button" data-testid="button-mode-transcript" onClick={() => setSourceMode('transcript')} className={`ml-5 border-b-2 px-1 pb-3 text-xs font-medium ${sourceMode === 'transcript' ? 'border-primary text-foreground' : 'border-transparent text-muted-foreground'}`}><FileText size={14} className="mr-2 inline" />Transcript</button>
              </div>
              <div className="p-5">
                {sourceMode === 'recording' ? <RecordingInput file={file} fileInput={fileInput} isRecording={isRecording} onFile={setFile} onStartRecording={startRecording} onStopRecording={stopRecording} /> : (
                  <div>
                    <label htmlFor="transcript-input" className="mb-2 block text-xs font-medium">Paste call transcript</label>
                    <textarea id="transcript-input" data-testid="input-transcript" value={pastedText} onChange={(event) => setPastedText(event.target.value)} placeholder="Paste the conversation here…" className="min-h-[194px] w-full resize-y rounded-lg border border-input bg-background px-3.5 py-3 text-sm leading-6 outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/20" />
                    <p className="mt-2 text-right font-mono text-[10px] text-muted-foreground">{pastedText.length.toLocaleString()} characters</p>
                  </div>
                )}
                <div className="mt-5 flex items-center justify-between gap-4">
                  <div className="flex items-center gap-2 text-[11px] text-muted-foreground"><Sparkles size={14} className="text-accent" /> Model: <span className="font-mono">small</span></div>
                  <button type="button" disabled={!canAnalyze || isWorking} data-testid="button-analyze" onClick={analyze} className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-semibold text-primary-foreground transition hover:brightness-95 disabled:cursor-not-allowed disabled:opacity-40">
                    {isWorking ? <><LoaderCircle size={16} className="animate-spin" /> {transcribe.isPending ? 'Transcribing' : 'Scoring'}</> : <>Analyze <ArrowUpRight size={16} /></>}
                  </button>
                </div>
                {error && <div className="mt-4 flex items-start gap-2 rounded-lg border border-destructive/20 bg-destructive/5 p-3 text-xs text-destructive" data-testid="status-analysis-error"><AlertTriangle size={15} className="mt-0.5 shrink-0" /> <span>{error}</span><button type="button" data-testid="button-dismiss-error" onClick={() => setError('')} className="ml-auto underline">Dismiss</button></div>}
              </div>
            </div>
            <div className="mt-4 flex items-start gap-3 px-1 text-xs leading-5 text-muted-foreground"><Info size={15} className="mt-0.5 shrink-0 text-accent" /><p>Results are decision support, not a verdict. Listen for context when confidence is below 70%.</p></div>
          </section>

          <section className="animate-rise-in delay-2">
            {isWorking ? <ResultSkeleton /> : prediction ? <Results prediction={prediction} transcript={transcript} sourceText={sourceMode === 'transcript' ? pastedText : undefined} onReset={reset} /> : <EmptyResult />}
          </section>
        </div>
      </div>
    </div>
  );
}

function RecordingInput({ file, fileInput, isRecording, onFile, onStartRecording, onStopRecording }: { file: File | null; fileInput: React.RefObject<HTMLInputElement | null>; isRecording: boolean; onFile: (file: File | null) => void; onStartRecording: () => void; onStopRecording: () => void }) {
  const [isDragging, setIsDragging] = useState(false);
  const audioUrl = useMemo(() => file ? URL.createObjectURL(file) : '', [file]);
  return (
    <div>
      <input ref={fileInput} type="file" accept="audio/*" className="hidden" data-testid="input-audio-file" onChange={(event) => onFile(event.target.files?.[0] ?? null)} />
      <div onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }} onDragLeave={() => setIsDragging(false)} onDrop={(event) => { event.preventDefault(); setIsDragging(false); onFile(event.dataTransfer.files?.[0] ?? null); }} className={`flex min-h-[194px] w-full flex-col items-center justify-center rounded-lg border border-dashed bg-background px-5 text-center transition ${isDragging ? 'border-accent bg-accent/[.08]' : 'border-input hover:border-accent hover:bg-accent/[.03]'}`}>
        {file ? <><span className="mb-3 grid h-11 w-11 place-items-center rounded-full bg-primary/20 text-accent"><Check size={21} /></span><span className="max-w-full truncate text-sm font-medium">{file.name}</span><span className="mt-1 font-mono text-[10px] text-muted-foreground">{(file.size / 1024 / 1024).toFixed(2)} MB · ready to review</span><div className="mt-3 flex h-8 items-end gap-1" aria-label="Audio waveform preview">{[18, 27, 12, 30, 20, 34, 15, 25, 18, 29, 11, 23, 31, 15, 26, 19, 32, 14].map((height, index) => <span key={index} className="w-1 rounded-full bg-accent/70" style={{ height }} />)}</div>{audioUrl && <audio className="mt-3 h-7 w-full max-w-[260px]" controls src={audioUrl} />}</> : <><span className="mb-3 grid h-11 w-11 place-items-center rounded-full bg-secondary text-accent transition group-hover:scale-105"><UploadCloud size={21} /></span><button type="button" data-testid="button-upload-recording" onClick={() => fileInput.current?.click()} className="text-sm font-medium hover:text-accent">Drop a recording here or browse</button><span className="mt-1 text-xs text-muted-foreground">WAV, MP3, M4A, or a browser recording</span></>}
      </div>
      <div className="mt-3 flex items-center justify-between gap-3">
        <span className="font-mono text-[10px] uppercase tracking-[.12em] text-muted-foreground">{isRecording ? 'recording live' : 'microphone input'}</span>
        <button type="button" onClick={isRecording ? onStopRecording : onStartRecording} className={`inline-flex items-center gap-2 rounded-md border px-2.5 py-1.5 text-xs font-medium transition ${isRecording ? 'border-destructive/40 text-destructive hover:bg-destructive/5' : 'border-input hover:border-accent hover:text-accent'}`}>
          <Mic2 size={13} /> {isRecording ? 'Stop recording' : 'Record from mic'}
        </button>
      </div>
    </div>
  );
}

function EmptyResult() {
  return <div className="flex min-h-[425px] flex-col items-center justify-center rounded-xl border border-dashed border-border bg-card/50 px-8 text-center"><span className="mb-4 grid h-12 w-12 place-items-center rounded-full border border-border bg-card text-muted-foreground"><Mic2 size={20} /></span><h2 className="text-base font-semibold">Your review will appear here</h2><p className="mt-2 max-w-xs text-sm leading-5 text-muted-foreground">Upload a recording or paste a transcript to see the transcript, signals, and confidence in one view.</p><span className="mt-5 font-mono text-[10px] uppercase tracking-[.16em] text-muted-foreground/70">waiting for input</span></div>;
}

function ResultSkeleton() {
  return <div className="space-y-4" data-testid="status-analysis-loading"><div className="h-[180px] rounded-xl skeleton" /><div className="h-[245px] rounded-xl skeleton" /></div>;
}

function Results({ prediction, transcript, sourceText, onReset }: { prediction: PredictionResult; transcript: TranscriptionResult | null; sourceText?: string; onReset: () => void }) {
  const isSpam = prediction.label === 'spam';
  const confidence = Math.round(prediction.confidence * 100);
  const probability = Math.round(prediction.probability * 100);
  const features = useMemo(() => Object.entries(prediction.features ?? {}).slice(0, 6), [prediction.features]);
  return (
    <div className="space-y-4" data-testid="panel-analysis-results">
      <div className={`relative overflow-hidden rounded-xl border bg-card p-5 shadow-[var(--shadow-soft)] ${isSpam ? 'border-destructive/30' : 'border-primary/50'}`}>
        <div className={`absolute right-0 top-0 h-24 w-24 translate-x-8 -translate-y-8 rounded-full blur-2xl ${isSpam ? 'bg-destructive/15' : 'bg-primary/25'}`} />
        <div className="relative flex items-start justify-between gap-4">
          <div><div className="mb-3 flex items-center gap-2 font-mono text-[10px] uppercase tracking-[.16em] text-muted-foreground"><span className={`h-1.5 w-1.5 rounded-full ${isSpam ? 'bg-destructive' : 'bg-primary'}`} /> final call decision</div><h2 data-testid="text-call-verdict" className="text-[2rem] font-semibold leading-none tracking-[-.06em]">{isSpam ? 'SPAM CALL' : 'NOT A SPAM CALL'}</h2><p className="mt-2 max-w-sm text-sm leading-5 text-muted-foreground">{isSpam ? 'The selected model found enough evidence to classify this call as spam.' : 'The selected model did not find enough evidence to classify this call as spam.'}</p></div>
          <div className={`grid h-16 w-16 shrink-0 place-items-center rounded-full border-4 text-lg font-semibold ${isSpam ? 'border-destructive/35 text-destructive' : 'border-primary/60 text-foreground'}`} data-testid="text-confidence">{confidence}%</div>
        </div>
        <div className="relative mt-6 border-t border-card-border pt-3"><div className="flex justify-between font-mono text-[10px] uppercase tracking-[.1em] text-muted-foreground"><span>confidence</span><span>{prediction.mode === 'heuristic' ? 'heuristic mode' : prediction.model}</span></div><div className="mt-2 h-1.5 overflow-hidden rounded-full bg-muted"><div className={`h-full rounded-full ${isSpam ? 'bg-destructive' : 'bg-primary'}`} style={{ width: `${confidence}%` }} /></div></div>
      </div>
      <div className="overflow-hidden rounded-xl border border-card-border bg-card shadow-[var(--shadow-soft)]">
        <div className="flex items-center justify-between border-b border-card-border px-5 py-4"><div><h3 className="font-semibold">Evidence</h3><p className="mt-0.5 text-xs text-muted-foreground">Decision: {isSpam ? 'Spam call' : 'Not a spam call'} · Probability {probability}% · {prediction.signals.length} signals</p></div><button type="button" data-testid="button-new-review" onClick={onReset} className="inline-flex items-center gap-1.5 rounded-md border border-input px-2.5 py-1.5 text-xs font-medium hover:bg-muted"><RotateCcw size={13} /> New review</button></div>
        <div className="grid gap-0 md:grid-cols-2"><div className="border-b border-card-border p-5 md:border-b-0 md:border-r"><h4 className="mb-3 font-mono text-[10px] uppercase tracking-[.14em] text-muted-foreground">Signals detected</h4>{prediction.signals.length ? <ul className="space-y-3">{prediction.signals.map((signal, index) => <li key={`${signal}-${index}`} className="flex items-start gap-2 text-sm"><span className="mt-1 grid h-4 w-4 shrink-0 place-items-center rounded-full bg-destructive/10 text-destructive"><ShieldAlert size={11} /></span><span>{signal}</span></li>)}</ul> : <p className="text-sm text-muted-foreground">No notable signals detected.</p>}</div><div className="p-5"><h4 className="mb-3 font-mono text-[10px] uppercase tracking-[.14em] text-muted-foreground">Feature summary</h4>{features.length ? <div className="space-y-2.5">{features.map(([name, value]) => <div key={name} className="flex items-center justify-between gap-3 text-xs"><span className="truncate text-muted-foreground">{name.replaceAll('_', ' ')}</span><span className="font-mono text-foreground">{typeof value === 'number' ? value.toFixed(2) : String(value)}</span></div>)}</div> : <p className="text-sm text-muted-foreground">Feature details are unavailable for this result.</p>}</div></div>
      </div>
      {(transcript || sourceText) && <TranscriptCard transcript={transcript} fallbackText={transcript ? undefined : sourceText} />}
    </div>
  );
}

function TranscriptCard({ transcript, fallbackText }: { transcript: TranscriptionResult | null; fallbackText?: string }) {
  return <div className="overflow-hidden rounded-xl border border-card-border bg-card shadow-[var(--shadow-soft)]"><div className="flex items-center justify-between border-b border-card-border px-5 py-4"><div><h3 className="font-semibold">Transcript</h3><p className="mt-0.5 text-xs text-muted-foreground">{transcript ? `${transcript.language.toUpperCase()} · ${formatSeconds(transcript.durationSeconds)} · ${transcript.mode} mode` : 'Pasted source'}</p></div><Paperclip size={15} className="text-muted-foreground" /></div>{transcript ? <div className="max-h-[300px] space-y-3 overflow-auto p-5">{transcript.segments.map((segment) => <div key={segment.id} className="flex gap-3 text-sm leading-5"><span className="w-10 shrink-0 pt-0.5 font-mono text-[10px] text-muted-foreground">{formatSeconds(segment.start)}</span><p>{segment.text}</p></div>)}</div> : <p className="p-5 text-sm leading-6 text-muted-foreground">{fallbackText}</p>}</div>;
}

function formatSeconds(seconds: number) { const minutes = Math.floor(seconds / 60); const remainder = Math.floor(seconds % 60).toString().padStart(2, '0'); return `${minutes}:${remainder}`; }