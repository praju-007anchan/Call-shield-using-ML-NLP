import { createContext, useContext, useMemo, useState, type ReactNode } from 'react';
import type { PredictionResult, TranscriptionResult } from '@workspace/api-client-react';

export type SourceMode = 'recording' | 'transcript';

type ReviewSessionValue = {
  sourceMode: SourceMode;
  setSourceMode: (mode: SourceMode) => void;
  file: File | null;
  setFile: (file: File | null) => void;
  pastedText: string;
  setPastedText: (text: string) => void;
  transcript: TranscriptionResult | null;
  setTranscript: (transcript: TranscriptionResult | null) => void;
  prediction: PredictionResult | null;
  setPrediction: (prediction: PredictionResult | null) => void;
  clearReview: () => void;
};

const ReviewSessionContext = createContext<ReviewSessionValue | null>(null);

export function ReviewSessionProvider({ children }: { children: ReactNode }) {
  const [sourceMode, setSourceMode] = useState<SourceMode>('recording');
  const [file, setFile] = useState<File | null>(null);
  const [pastedText, setPastedText] = useState('');
  const [transcript, setTranscript] = useState<TranscriptionResult | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);

  const value = useMemo(() => ({
    sourceMode,
    setSourceMode,
    file,
    setFile,
    pastedText,
    setPastedText,
    transcript,
    setTranscript,
    prediction,
    setPrediction,
    clearReview: () => {
      setFile(null);
      setPastedText('');
      setTranscript(null);
      setPrediction(null);
    },
  }), [sourceMode, file, pastedText, transcript, prediction]);

  return <ReviewSessionContext.Provider value={value}>{children}</ReviewSessionContext.Provider>;
}

export function useReviewSession() {
  const value = useContext(ReviewSessionContext);
  if (!value) throw new Error('useReviewSession must be used inside ReviewSessionProvider');
  return value;
}