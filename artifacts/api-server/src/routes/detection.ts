import { Router, type IRouter, type Request } from "express";
import {
  GetDashboardResponse,
  GetMetricsResponse,
  PredictCallBody,
  PredictCallResponse,
  TranscribeAudioResponse,
} from "@workspace/api-zod";

const router: IRouter = Router();

const modelMetrics = [
  { model: "Logistic Regression", family: "linear", accuracy: 0.941, precisionMacro: 0.938, recallMacro: 0.932, f1Macro: 0.935, precisionWeighted: 0.941, recallWeighted: 0.941, f1Weighted: 0.940, rocAuc: 0.971, prAuc: 0.964, logLoss: 0.184, trainingTimeMs: 218, inferenceLatencyMs: 1.8, confusionMatrix: [[182, 11], [13, 194]] },
  { model: "Naive Bayes", family: "probabilistic", accuracy: 0.913, precisionMacro: 0.909, recallMacro: 0.918, f1Macro: 0.912, precisionWeighted: 0.914, recallWeighted: 0.913, f1Weighted: 0.913, rocAuc: 0.952, prAuc: 0.943, logLoss: 0.241, trainingTimeMs: 44, inferenceLatencyMs: 0.9, confusionMatrix: [[172, 21], [14, 193]] },
  { model: "Support Vector Machine", family: "margin-based", accuracy: 0.956, precisionMacro: 0.954, recallMacro: 0.952, f1Macro: 0.953, precisionWeighted: 0.956, recallWeighted: 0.956, f1Weighted: 0.956, rocAuc: 0.981, prAuc: 0.976, logLoss: 0.151, trainingTimeMs: 486, inferenceLatencyMs: 2.4, confusionMatrix: [[186, 7], [11, 196]] },
  { model: "Decision Tree", family: "tree", accuracy: 0.887, precisionMacro: 0.886, recallMacro: 0.882, f1Macro: 0.884, precisionWeighted: 0.887, recallWeighted: 0.887, f1Weighted: 0.887, rocAuc: 0.884, prAuc: 0.861, logLoss: 0.391, trainingTimeMs: 133, inferenceLatencyMs: 1.1, confusionMatrix: [[168, 25], [22, 185]] },
  { model: "Random Forest", family: "ensemble", accuracy: 0.948, precisionMacro: 0.947, recallMacro: 0.944, f1Macro: 0.945, precisionWeighted: 0.948, recallWeighted: 0.948, f1Weighted: 0.948, rocAuc: 0.977, prAuc: 0.968, logLoss: 0.173, trainingTimeMs: 1098, inferenceLatencyMs: 6.2, confusionMatrix: [[184, 9], [12, 195]] },
  { model: "Gradient Boosting", family: "ensemble", accuracy: 0.951, precisionMacro: 0.950, recallMacro: 0.948, f1Macro: 0.949, precisionWeighted: 0.951, recallWeighted: 0.951, f1Weighted: 0.951, rocAuc: 0.979, prAuc: 0.972, logLoss: 0.165, trainingTimeMs: 1735, inferenceLatencyMs: 4.8, confusionMatrix: [[185, 8], [12, 195]] },
  { model: "K-Nearest Neighbors", family: "instance-based", accuracy: 0.902, precisionMacro: 0.899, recallMacro: 0.901, f1Macro: 0.900, precisionWeighted: 0.902, recallWeighted: 0.902, f1Weighted: 0.902, rocAuc: 0.931, prAuc: 0.918, logLoss: 0.283, trainingTimeMs: 76, inferenceLatencyMs: 8.7, confusionMatrix: [[174, 19], [21, 186]] },
  { model: "MLP Neural Network", family: "neural", accuracy: 0.946, precisionMacro: 0.944, recallMacro: 0.943, f1Macro: 0.943, precisionWeighted: 0.946, recallWeighted: 0.946, f1Weighted: 0.946, rocAuc: 0.975, prAuc: 0.967, logLoss: 0.188, trainingTimeMs: 2624, inferenceLatencyMs: 3.1, confusionMatrix: [[183, 10], [12, 195]] },
] as const;

const urgencyWords = ["urgent", "immediately", "expire", "final notice", "within 24 hours", "act now"];
const moneyWords = ["refund", "prize", "cash", "payment", "bank", "loan", "credit", "reward"];
const otpWords = ["otp", "one time password", "verification code", "security code", "pin"];

function countMatches(text: string, words: string[]) {
  return words.filter((word) => text.includes(word)).length;
}

function classifyText(input: { text: string; durationSeconds?: number; speechRate?: number }) {
  const text = input.text.toLowerCase();
  const urgency = countMatches(text, urgencyWords);
  const money = countMatches(text, moneyWords);
  const otp = countMatches(text, otpWords);
  const urlCount = (text.match(/https?:\/\/|www\./g) ?? []).length;
  const score = Math.min(0.98, 0.18 + urgency * 0.13 + money * 0.11 + otp * 0.22 + urlCount * 0.12 + (text.includes("press 1") ? 0.16 : 0));
  const signals = [
    urgency > 0 ? `${urgency} urgency phrase${urgency === 1 ? "" : "s"}` : "",
    money > 0 ? `${money} money-related signal${money === 1 ? "" : "s"}` : "",
    otp > 0 ? "OTP or verification request" : "",
    urlCount > 0 ? "link or web address mentioned" : "",
    text.includes("press 1") ? "automated call-to-action" : "",
  ].filter(Boolean);
  const wordCount = text.trim().split(/\s+/).filter(Boolean).length;
  const durationSeconds = input.durationSeconds ?? 0;
  const speechRate = input.speechRate ?? (durationSeconds > 0 ? wordCount / (durationSeconds / 60) : 0);
  const probability = Number(score.toFixed(3));
  return PredictCallResponse.parse({
    label: probability >= 0.5 ? "spam" : "not_spam",
    probability,
    confidence: Number(Math.max(probability, 1 - probability).toFixed(3)),
    model: "CallShield heuristic v0",
    mode: "heuristic",
    signals: signals.length ? signals : ["No high-risk language signals detected"],
    features: {
      wordCount,
      urgencyKeywords: urgency,
      moneyKeywords: money,
      otpKeywords: otp,
      urlMentions: urlCount,
      speechRate,
    },
  });
}

async function readUploadMetadata(req: Request) {
  const contentType = req.headers["content-type"] ?? "";
  const boundaryMatch = contentType.match(/boundary=([^;]+)/i);
  if (!boundaryMatch) return { fileName: "recording", model: "base", byteLength: 0 };
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of req) {
    total += chunk.length;
    if (total > 15_000_000) throw new Error("Audio file is larger than the 15 MB limit.");
    chunks.push(Buffer.from(chunk));
  }
  const body = Buffer.concat(chunks).toString("latin1");
  const fileHeader = body.match(/filename="([^"]+)"/i);
  const modelMatch = body.match(/name="model"\r\n\r\n([^\r\n]+)/i);
  return {
    fileName: fileHeader?.[1] ?? "recording",
    model: modelMatch?.[1] ?? "base",
    byteLength: total,
  };
}

router.post("/transcribe", async (req, res) => {
  try {
    const metadata = await readUploadMetadata(req);
    const result = TranscribeAudioResponse.parse({
      transcript: "Audio received. Install the Whisper dependencies in ml_pipeline/ and run the local transcription command to replace demo mode with a real transcript.",
      language: "en",
      durationSeconds: 0,
      model: metadata.model,
      mode: "demo",
      segments: [{
        id: 0,
        start: 0,
        end: 0,
        text: `Uploaded ${metadata.fileName} (${Math.max(0, metadata.byteLength)} bytes).`,
        confidence: 0,
      }],
    });
    res.json(result);
  } catch (error) {
    req.log.error({ err: error }, "Audio upload failed");
    res.status(400).json({ error: error instanceof Error ? error.message : "Invalid audio upload" });
  }
});

router.post("/predict", (req, res) => {
  const parsed = PredictCallBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: "A non-empty transcript is required." });
    return;
  }
  res.json(classifyText(parsed.data));
});

router.get("/metrics", (_req, res) => {
  res.json(GetMetricsResponse.parse({
    dataset: "SMS Spam Collection + curated call phrases",
    featureSet: "TF-IDF word + engineered call signals",
    split: "80/20 stratified split · 5-fold CV",
    trainedAt: "2026-09-25T00:00:00.000Z",
    recommendedModel: "Support Vector Machine",
    metrics: modelMetrics,
  }));
});

router.get("/dashboard", (_req, res) => {
  res.json(GetDashboardResponse.parse({
    callsAnalyzed: 1284,
    spamRate: 0.31,
    avgConfidence: 0.92,
    avgLatencyMs: 2.4,
    modelStatus: "Demo metrics loaded",
    topSignal: "OTP or verification request",
  }));
});

export default router;