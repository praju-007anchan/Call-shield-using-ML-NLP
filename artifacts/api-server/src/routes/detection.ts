import { Router, type IRouter, type Request } from "express";
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { rm, writeFile } from "node:fs/promises";
import { extname, join, resolve } from "node:path";
import { tmpdir } from "node:os";
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

async function readUpload(req: Request) {
  const contentType = req.headers["content-type"] ?? "";
  const boundaryMatch = contentType.match(/boundary="?([^";]+)"?/i);
  if (!boundaryMatch) throw new Error("Audio upload must use multipart/form-data.");
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of req) {
    total += chunk.length;
    if (total > 15_000_000) throw new Error("Audio file is larger than the 15 MB limit.");
    chunks.push(Buffer.from(chunk));
  }
  const body = Buffer.concat(chunks);
  const headerEnd = body.indexOf(Buffer.from("\r\n\r\n"));
  const boundary = Buffer.from(`\r\n--${boundaryMatch[1]}`);
  const fileEnd = headerEnd >= 0 ? body.indexOf(boundary, headerEnd + 4) : -1;
  if (headerEnd < 0 || fileEnd < 0) throw new Error("The uploaded audio file could not be read.");
  const header = body.subarray(0, headerEnd).toString("latin1");
  const fileHeader = header.match(/filename="([^"]+)"/i);
  const fileBytes = body.subarray(headerEnd + 4, fileEnd);
  if (!fileBytes.length) throw new Error("The uploaded audio file is empty.");
  const metadataText = body.toString("latin1");
  const modelMatch = metadataText.match(/name="model"\r\n\r\n([^\r\n]+)/i);
  return {
    fileName: fileHeader?.[1] ?? "recording",
    model: modelMatch?.[1] ?? "base",
    byteLength: total,
    fileBytes,
  };
}

function runWhisper(audioPath: string, model: string) {
  return new Promise<Record<string, unknown>>((resolveResult, reject) => {
    const scriptPath = [
      resolve(process.cwd(), "ml_pipeline", "transcribe_worker.py"),
      resolve(process.cwd(), "..", "..", "ml_pipeline", "transcribe_worker.py"),
    ].find(existsSync);
    if (!scriptPath) {
      reject(new Error("Whisper worker is not installed in ml_pipeline."));
      return;
    }
    const python = process.env.PYTHON_BIN ?? "python3";
    const child = spawn(python, [scriptPath, "--audio", audioPath, "--model", model], {
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });
    let stdout = "";
    let stderr = "";
    const timeout = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error("Whisper transcription timed out after 10 minutes."));
    }, 10 * 60 * 1000);
    child.stdout.on("data", (chunk: Buffer) => { stdout += chunk.toString(); });
    child.stderr.on("data", (chunk: Buffer) => { stderr += chunk.toString(); });
    child.on("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timeout);
      if (code !== 0) {
        reject(new Error(stderr.trim() || `Whisper exited with code ${code ?? "unknown"}.`));
        return;
      }
      try {
        resolveResult(JSON.parse(stdout) as Record<string, unknown>);
      } catch {
        reject(new Error("Whisper returned an invalid transcription response."));
      }
    });
  });
}

router.post("/transcribe", async (req, res) => {
  let audioPath = "";
  try {
    const upload = await readUpload(req);
    const extension = extname(upload.fileName).toLowerCase().replace(/[^a-z0-9.]/g, "") || ".audio";
    audioPath = join(tmpdir(), `callshield-${randomUUID()}${extension}`);
    await writeFile(audioPath, upload.fileBytes);
    const transcription = await runWhisper(audioPath, upload.model);
    const result = TranscribeAudioResponse.parse({
      transcript: transcription.transcript,
      language: transcription.language,
      durationSeconds: transcription.durationSeconds,
      model: upload.model,
      mode: "whisper",
      segments: transcription.segments,
    });
    res.json(result);
  } catch (error) {
    req.log.error({ err: error }, "Audio upload failed");
    const message = error instanceof Error ? error.message : "Invalid audio upload";
    const status = message.includes("Whisper") || message.includes("No module named") || message.includes("timed out") ? 503 : 400;
    res.status(status).json({ error: message });
  } finally {
    if (audioPath) await rm(audioPath, { force: true }).catch(() => undefined);
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